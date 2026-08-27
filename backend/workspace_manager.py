"""
Workspace lifecycle management.

Each workspace gets:
  - its own directory under data/workspaces/<workspace_id>/
  - its own Chroma persist directory (chroma/) -> vector isolation
  - its own chat_history.json -> conversation isolation
  - its own documents.json -> tracks uploaded file metadata

A JSON registry file (_registry.json) tracks workspace id -> name -> created_at
so the frontend can list workspaces without scanning the filesystem each time.
"""
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from config import WORKSPACES_DIR, WORKSPACE_REGISTRY_FILE

# A process-wide lock guards registry read/write. Streamlit + FastAPI both
# run single-process in the default dev setup, but multiple requests can
# still race on the same file, so we serialize registry mutations.
_registry_lock = Lock()


def _load_registry() -> dict:
    if not WORKSPACE_REGISTRY_FILE.exists():
        return {}
    with open(WORKSPACE_REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: dict) -> None:
    with open(WORKSPACE_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str)


def _workspace_dir(workspace_id: str) -> Path:
    return WORKSPACES_DIR / workspace_id


def get_chroma_path(workspace_id: str) -> str:
    """Returns the Chroma persist directory for a given workspace."""
    return str(_workspace_dir(workspace_id) / "chroma")


def get_chat_history_path(workspace_id: str) -> Path:
    return _workspace_dir(workspace_id) / "chat_history.json"


def get_documents_meta_path(workspace_id: str) -> Path:
    return _workspace_dir(workspace_id) / "documents.json"


def get_uploads_dir(workspace_id: str) -> Path:
    return _workspace_dir(workspace_id) / "uploads"


def create_workspace(name: str) -> dict:
    workspace_id = str(uuid.uuid4())
    ws_dir = _workspace_dir(workspace_id)
    ws_dir.mkdir(parents=True, exist_ok=True)
    (ws_dir / "chroma").mkdir(exist_ok=True)
    get_uploads_dir(workspace_id).mkdir(exist_ok=True)

    get_chat_history_path(workspace_id).write_text("[]", encoding="utf-8")
    get_documents_meta_path(workspace_id).write_text("[]", encoding="utf-8")

    entry = {
        "id": workspace_id,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with _registry_lock:
        registry = _load_registry()
        registry[workspace_id] = entry
        _save_registry(registry)

    return entry


def list_workspaces() -> list[dict]:
    with _registry_lock:
        registry = _load_registry()

    results = []
    for ws_id, entry in registry.items():
        doc_count = len(_read_json(get_documents_meta_path(ws_id), default=[]))
        msg_count = len(_read_json(get_chat_history_path(ws_id), default=[]))
        results.append({
            **entry,
            "document_count": doc_count,
            "message_count": msg_count,
        })

    # Most recently created first
    results.sort(key=lambda w: w["created_at"], reverse=True)
    return results


def workspace_exists(workspace_id: str) -> bool:
    with _registry_lock:
        registry = _load_registry()
    return workspace_id in registry


def delete_workspace(workspace_id: str) -> bool:
    with _registry_lock:
        registry = _load_registry()
        if workspace_id not in registry:
            return False
        del registry[workspace_id]
        _save_registry(registry)

    ws_dir = _workspace_dir(workspace_id)
    if ws_dir.exists():
        shutil.rmtree(ws_dir)
    return True


def rename_workspace(workspace_id: str, new_name: str) -> bool:
    with _registry_lock:
        registry = _load_registry()
        if workspace_id not in registry:
            return False
        registry[workspace_id]["name"] = new_name
        _save_registry(registry)
    return True


# --- Chat history helpers -----------------------------------------------

def _read_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_chat_history(workspace_id: str) -> list[dict]:
    return _read_json(get_chat_history_path(workspace_id), default=[])


def append_chat_message(workspace_id: str, role: str, content: str) -> None:
    path = get_chat_history_path(workspace_id)
    history = _read_json(path, default=[])
    history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)


def clear_chat_history(workspace_id: str) -> None:
    get_chat_history_path(workspace_id).write_text("[]", encoding="utf-8")


# --- Document metadata helpers -------------------------------------------

def get_documents(workspace_id: str) -> list[dict]:
    return _read_json(get_documents_meta_path(workspace_id), default=[])


def add_document_meta(workspace_id: str, filename: str, chunk_count: int, file_type: str) -> None:
    path = get_documents_meta_path(workspace_id)
    docs = _read_json(path, default=[])
    docs.append({
        "filename": filename,
        "chunk_count": chunk_count,
        "file_type": file_type,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, default=str)


def remove_document_meta(workspace_id: str, filename: str) -> None:
    path = get_documents_meta_path(workspace_id)
    docs = _read_json(path, default=[])
    docs = [d for d in docs if d["filename"] != filename]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, default=str)