import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local-no-key")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-4-e4b")

DOCS_BASE_URL = os.getenv("DOCS_BASE_URL", "https://docs.example.com/getting-started")
DOCS_DOMAIN = os.getenv("DOCS_DOMAIN", "https://docs.example.com")

RAW_DOCS_DIR = Path(os.getenv("RAW_DOCS_DIR", PROJECT_ROOT / "data" / "raw_docs"))
CLEAN_DOCS_DIR = Path(os.getenv("CLEAN_DOCS_DIR", PROJECT_ROOT / "data" / "clean_docs"))
DB_DIR = str(Path(os.getenv("LANCEDB_DIR", PROJECT_ROOT / "lancedb_store")))
TABLE_NAME = os.getenv("LANCEDB_TABLE", "technical_docs")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
CACHE_DIR = str(Path(os.getenv("FASTEMBED_CACHE_DIR", PROJECT_ROOT / ".fastembed_cache")))
