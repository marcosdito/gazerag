"""
Central configuration for GazeRAG.

All tunable knobs live here. Values can be overridden via environment variables
(or a .env file at the project root) so that local development, Docker, and CI
can each use their own settings without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (silently does nothing if missing)
load_dotenv()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = Path(os.environ.get("PAPERS_DIR", DATA_DIR / "papers"))
GAZE_DIR = Path(os.environ.get("GAZE_DIR", DATA_DIR / "gaze_samples"))
CHROMA_DB_PATH = Path(os.environ.get("CHROMA_DB_PATH", DATA_DIR / "chroma_db"))

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
# all-MiniLM-L6-v2: small, fast, strong baseline for English scientific text
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Groq's fastest open model — perfectly capable for RAG question answering
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")

# -----------------------------------------------------------------------------
# API keys
# -----------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# -----------------------------------------------------------------------------
# Chroma collection name
# -----------------------------------------------------------------------------
CHROMA_COLLECTION = "gazerag_papers"

# -----------------------------------------------------------------------------
# RAG hyperparameters
# -----------------------------------------------------------------------------
CHUNK_SIZE = 800           # characters per chunk
CHUNK_OVERLAP = 100        # overlap between consecutive chunks
TOP_K_RETRIEVAL = 5        # number of chunks pulled for each query


def validate() -> None:
    """Raise a clear error if required configuration is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a .env file at the project root "
            "with `GROQ_API_KEY=gsk_...` (get yours at https://console.groq.com), "
            "or export it in your shell."
        )
