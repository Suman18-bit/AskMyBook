"""
Core RAG engine — refactored from your original CLI script.

Key changes from the original:
  1. `vectorstore` is no longer a single global — it's built per-workspace
     via `get_vectorstore(workspace_id)`, pointing Chroma at that
     workspace's own persist directory. This is what gives each workspace
     isolated document storage.
  2. `llm.invoke(...)` -> `llm.astream(...)`. Your original blocked until
     the full response was ready; this yields tokens as they arrive so
     the frontend can render them in real time over SSE.
  3. The retriever's `.invoke()` stays synchronous (Chroma's similarity
     search is fast and CPU-bound, not worth the async overhead), but
     runs inside a thread via `run_in_threadpool` so it doesn't block
     the FastAPI event loop while other requests are in flight.

Everything else — the MMR search params, the prompt template, the model
choice — is preserved exactly as you had it.
"""
from typing import AsyncGenerator

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from starlette.concurrency import run_in_threadpool

import config
from workspace_manager import get_chroma_path

# --- Shared singletons ---------------------------------------------------
# Embeddings and the LLM are stateless w.r.t. workspace, so one instance
# is reused across all workspaces/requests. Only the vectorstore (and its
# retriever) is workspace-specific.
embeddings = MistralAIEmbeddings(model=config.EMBEDDING_MODEL)
llm = ChatMistralAI(model=config.LLM_MODEL, streaming=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
)

# Preserved verbatim from your original script.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a precise, reliable AI assistant designed for Retrieval-Augmented Generation (RAG). Your responsibility is to answer questions strictly using the provided context. Core Rules:
    - Use only the information in {context} to generate answers.
    - If the answer is not present, reply exactly: "I don't know based on the given context."
    - Never invent, assume, or add external knowledge.
    - Keep responses clear, concise, and directly relevant.
    - When appropriate, format answers with bullet points, numbered lists, or short paragraphs for readability.
    - If multiple possible answers exist in the context, present them all neutrally without prioritizing or guessing.
    """),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

# Simple in-memory cache so we don't reconstruct a Chroma client on every
# single request for the same workspace. Keyed by workspace_id.
_vectorstore_cache: dict[str, Chroma] = {}


def get_vectorstore(workspace_id: str) -> Chroma:
    """
    Returns (and caches) the Chroma vectorstore scoped to a single
    workspace's persist directory. This is the mechanism that keeps
    each workspace's documents fully isolated from every other workspace.
    """
    if workspace_id not in _vectorstore_cache:
        _vectorstore_cache[workspace_id] = Chroma(
            persist_directory=get_chroma_path(workspace_id),
            embedding_function=embeddings,
        )
    return _vectorstore_cache[workspace_id]


def invalidate_vectorstore_cache(workspace_id: str) -> None:
    """Call after deleting a workspace so a stale client isn't reused."""
    _vectorstore_cache.pop(workspace_id, None)


def get_retriever(workspace_id: str):
    vectorstore = get_vectorstore(workspace_id)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.RETRIEVAL_K,
            "fetch_k": config.RETRIEVAL_FETCH_K,
            "lambda_mult": config.RETRIEVAL_LAMBDA_MULT,
        },
    )


async def ingest_documents(workspace_id: str, documents: list, filename: str) -> int:
    """
    Splits documents into chunks and adds them to the workspace's
    vectorstore. Returns the number of chunks created.

    Runs the split + add_documents call in a thread since both are
    CPU/IO-bound and would otherwise block the event loop during a
    large file upload.
    """
    def _split_and_add():
        chunks = text_splitter.split_documents(documents)
        # Tag every chunk with its source filename so we can later
        # delete just this file's vectors without touching the rest
        # of the workspace's data.
        for chunk in chunks:
            chunk.metadata["source_file"] = filename
        vectorstore = get_vectorstore(workspace_id)
        vectorstore.add_documents(chunks)
        return len(chunks)

    return await run_in_threadpool(_split_and_add)


async def delete_document_vectors(workspace_id: str, filename: str) -> None:
    """
    Removes all chunks belonging to a specific uploaded file from the
    workspace's vectorstore, using the source_file metadata tag set
    during ingestion.
    """
    def _delete():
        vectorstore = get_vectorstore(workspace_id)
        # Chroma's `get` with a where-filter lets us find matching IDs,
        # then `delete` removes exactly those — nothing else in the
        # workspace's collection is touched.
        existing = vectorstore.get(where={"source_file": filename})
        ids = existing.get("ids", [])
        if ids:
            vectorstore.delete(ids=ids)

    await run_in_threadpool(_delete)


async def stream_answer(workspace_id: str, question: str) -> AsyncGenerator[str, None]:
    """
    The async-streaming replacement for your original:

        docs = retriever.invoke(question)
        context = "\\n\\n".join([doc.page_content for doc in docs])
        final_prompt = prompt_template.format(context=context, question=question)
        result = llm.invoke(final_prompt)
        print(result)

    Retrieval still happens once, up front, exactly as before — MMR
    search over the workspace's collection. The only structural change
    is `llm.invoke()` -> `async for chunk in llm.astream()`, which lets
    the FastAPI route yield each token to the client as it's generated
    instead of waiting for the full completion.
    """
    retriever = get_retriever(workspace_id)

    # Retrieval is sync but fast; thread it so it doesn't block the loop.
    docs = await run_in_threadpool(retriever.invoke, question)

    if not docs:
        # No documents in this workspace, or nothing relevant retrieved —
        # short-circuit rather than sending an empty context to the model.
        yield "I don't know based on the given context."
        return

    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt_template.format(context=context, question=question)

    async for chunk in llm.astream(final_prompt):
        # ChatMistralAI streams AIMessageChunk objects; `.content` holds
        # the incremental text delta for this chunk.
        if chunk.content:
            yield chunk.content