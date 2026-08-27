"""
FastAPI backend for AskMyBook RAG.

Routes:
  POST   /workspaces                     -> create workspace
  GET    /workspaces                     -> list workspaces
  DELETE /workspaces/{workspace_id}      -> delete workspace
  PATCH  /workspaces/{workspace_id}      -> rename workspace

  GET    /workspaces/{workspace_id}/documents        -> list uploaded docs
  POST   /workspaces/{workspace_id}/documents         -> upload + ingest a file
  DELETE /workspaces/{workspace_id}/documents/{name}  -> remove a doc + its vectors

  GET    /workspaces/{workspace_id}/chat              -> chat history
  DELETE /workspaces/{workspace_id}/chat              -> clear chat history
  POST   /chat/stream                                 -> SSE token stream (the core endpoint)
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import config
import workspace_manager as wm
from document_loader import load_document
from models import (
    WorkspaceCreate, WorkspaceInfo, WorkspaceListResponse,
    ChatHistoryResponse, QueryRequest, UploadResponse,
)
from rag_engine import ingest_documents, delete_document_vectors, stream_answer, invalidate_vectorstore_cache

app = FastAPI(title="AskMyBook RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Workspace routes -----------------------------------------------------

@app.post("/workspaces", response_model=WorkspaceInfo)
def create_workspace(payload: WorkspaceCreate):
    entry = wm.create_workspace(payload.name)
    return WorkspaceInfo(**entry, document_count=0, message_count=0)


@app.get("/workspaces", response_model=WorkspaceListResponse)
def list_workspaces():
    return WorkspaceListResponse(workspaces=wm.list_workspaces())


@app.delete("/workspaces/{workspace_id}")
def delete_workspace(workspace_id: str):
    if not wm.workspace_exists(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")
    wm.delete_workspace(workspace_id)
    invalidate_vectorstore_cache(workspace_id)
    return {"status": "deleted", "workspace_id": workspace_id}


@app.patch("/workspaces/{workspace_id}")
def rename_workspace(workspace_id: str, payload: WorkspaceCreate):
    if not wm.workspace_exists(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")
    wm.rename_workspace(workspace_id, payload.name)
    return {"status": "renamed", "workspace_id": workspace_id, "name": payload.name}


# --- Document routes --------------------------------------------------

@app.get("/workspaces/{workspace_id}/documents")
def list_documents(workspace_id: str):
    if not wm.workspace_exists(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"documents": wm.get_documents(workspace_id)}


@app.post("/workspaces/{workspace_id}/documents", response_model=UploadResponse)
async def upload_document(workspace_id: str, file: UploadFile = File(...)):
    if not wm.workspace_exists(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")

    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {config.MAX_UPLOAD_SIZE_MB}MB limit ({size_mb:.1f}MB)",
        )

    # Persist the raw upload to disk — the loaders (PyPDFLoader, Docx2txtLoader,
    # openpyxl) all need a filesystem path, not an in-memory buffer.
    uploads_dir = wm.get_uploads_dir(workspace_id)
    uploads_dir.mkdir(exist_ok=True)
    dest_path = uploads_dir / file.filename
    dest_path.write_bytes(contents)

    try:
        documents = load_document(str(dest_path))
        chunk_count = await ingest_documents(workspace_id, documents, file.filename)
    except ValueError as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

    wm.add_document_meta(workspace_id, file.filename, chunk_count, ext)

    return UploadResponse(
        filename=file.filename,
        chunks_created=chunk_count,
        status="ingested",
    )


@app.delete("/workspaces/{workspace_id}/documents/{filename}")
async def delete_document(workspace_id: str, filename: str):
    if not wm.workspace_exists(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")

    await delete_document_vectors(workspace_id, filename)
    wm.remove_document_meta(workspace_id, filename)

    upload_path = wm.get_uploads_dir(workspace_id) / filename
    upload_path.unlink(missing_ok=True)

    return {"status": "deleted", "filename": filename}


# --- Chat history routes ---------------------------------------------

@app.get("/workspaces/{workspace_id}/chat", response_model=ChatHistoryResponse)
def get_chat_history(workspace_id: str):
    if not wm.workspace_exists(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ChatHistoryResponse(
        workspace_id=workspace_id,
        messages=wm.get_chat_history(workspace_id),
    )


@app.delete("/workspaces/{workspace_id}/chat")
def clear_chat_history(workspace_id: str):
    if not wm.workspace_exists(workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")
    wm.clear_chat_history(workspace_id)
    return {"status": "cleared", "workspace_id": workspace_id}


# --- Streaming chat endpoint (the core deliverable) --------------------

@app.post("/chat/stream")
async def chat_stream(payload: QueryRequest):
    """
    Server-Sent Events endpoint. Each event is a JSON payload:
        {"type": "token", "content": "..."}   -- one per generated token
        {"type": "done"}                       -- terminal event
        {"type": "error", "detail": "..."}    -- on failure

    The frontend (Streamlit, via httpx) reads this stream line-by-line
    and appends each token's content to the in-progress assistant message.
    """
    if not wm.workspace_exists(payload.workspace_id):
        raise HTTPException(status_code=404, detail="Workspace not found")

    wm.append_chat_message(payload.workspace_id, "user", payload.question)

    async def event_generator():
        full_response = ""
        try:
            async for token in stream_answer(payload.workspace_id, payload.question):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            wm.append_chat_message(payload.workspace_id, "assistant", full_response)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            # If generation fails partway, persist whatever was produced
            # so the chat history doesn't silently lose the exchange,
            # then surface the error to the client.
            if full_response:
                wm.append_chat_message(payload.workspace_id, "assistant", full_response)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disables nginx buffering if deployed behind it
        },
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}