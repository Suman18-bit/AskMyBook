"""
Thin HTTP client the Streamlit app uses to talk to the FastAPI backend.

The one non-obvious piece is `stream_chat()`: Streamlit has no native SSE
client, so we open a streaming httpx request against the FastAPI endpoint
and manually parse the `data: {...}\n\n` frames ourselves. This is the
standard pattern for consuming SSE from a non-browser client — the same
thing the browser's EventSource does under the hood, just written out.
"""
import json
from typing import Generator

import httpx

BACKEND_URL = "http://localhost:8000"

# A shared client with a generous read timeout — LLM generation can take
# a while, and the default httpx timeout (5s) would kill the stream mid-answer.
_client = httpx.Client(base_url=BACKEND_URL, timeout=httpx.Timeout(120.0, connect=5.0))


class BackendError(Exception):
    """Raised when the backend returns a non-2xx response with a detail message."""
    pass


def _raise_for_status(response: httpx.Response):
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise BackendError(detail)


# --- Workspace operations --------------------------------------------

def list_workspaces() -> list[dict]:
    resp = _client.get("/workspaces")
    _raise_for_status(resp)
    return resp.json()["workspaces"]


def create_workspace(name: str) -> dict:
    resp = _client.post("/workspaces", json={"name": name})
    _raise_for_status(resp)
    return resp.json()


def delete_workspace(workspace_id: str) -> None:
    resp = _client.delete(f"/workspaces/{workspace_id}")
    _raise_for_status(resp)


def rename_workspace(workspace_id: str, new_name: str) -> None:
    resp = _client.patch(f"/workspaces/{workspace_id}", json={"name": new_name})
    _raise_for_status(resp)


# --- Document operations ---------------------------------------------

def list_documents(workspace_id: str) -> list[dict]:
    resp = _client.get(f"/workspaces/{workspace_id}/documents")
    _raise_for_status(resp)
    return resp.json()["documents"]


def upload_document(workspace_id: str, filename: str, file_bytes: bytes) -> dict:
    files = {"file": (filename, file_bytes)}
    # File processing (parsing + embedding) can take a while for large
    # files, so this call uses the client's full 120s timeout.
    resp = _client.post(f"/workspaces/{workspace_id}/documents", files=files)
    _raise_for_status(resp)
    return resp.json()


def delete_document(workspace_id: str, filename: str) -> None:
    resp = _client.delete(f"/workspaces/{workspace_id}/documents/{filename}")
    _raise_for_status(resp)


# --- Chat history operations -------------------------------------------

def get_chat_history(workspace_id: str) -> list[dict]:
    resp = _client.get(f"/workspaces/{workspace_id}/chat")
    _raise_for_status(resp)
    return resp.json()["messages"]


def clear_chat_history(workspace_id: str) -> None:
    resp = _client.delete(f"/workspaces/{workspace_id}/chat")
    _raise_for_status(resp)


# --- Streaming chat (the core interaction) ------------------------------

def stream_chat(workspace_id: str, question: str) -> Generator[dict, None, None]:
    """
    Yields parsed SSE event dicts as they arrive from the backend:
        {"type": "token", "content": "..."}
        {"type": "done"}
        {"type": "error", "detail": "..."}

    Uses httpx's streaming context manager so the connection stays open
    and we read/yield chunks as they're written by the server, rather
    than waiting for the full response body.
    """
    with _client.stream(
        "POST",
        "/chat/stream",
        json={"workspace_id": workspace_id, "question": question},
    ) as response:
        if response.status_code >= 400:
            response.read()
            _raise_for_status(response)

        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw = line[len("data: "):]
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            yield event