"""
Central configuration for the AskMyBook RAG backend.
All paths and tunables live here so nothing is hardcoded downstream.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Base paths -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACES_DIR = BASE_DIR / "data" / "workspaces"
WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)

WORKSPACE_REGISTRY_FILE = WORKSPACES_DIR / "_registry.json"

# --- API keys -----------------------------------------------------------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not set. Add it to your .env file before starting the backend."
    )

# --- Model configuration -------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small")

# --- Chunking -------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval --------------------------------------------------------
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "10"))
RETRIEVAL_LAMBDA_MULT = float(os.getenv("RETRIEVAL_LAMBDA_MULT", "0.5"))

# --- Uploads ------------------------------------------------------------
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx"}
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))

# --- CORS (Streamlit runs on a different port than FastAPI) -------------
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")