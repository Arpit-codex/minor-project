"""
Configuration module for the Local Memory-Based Contextual AI Assistant.
Reads settings from environment variables with sensible defaults.
Automatically ensures directory structures exist.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")

# Storage Directories
DATA_DIR = BASE_DIR / "data"
MEMORY_DIR = BASE_DIR / "memory"
LOGS_DIR = DATA_DIR / "logs"
MEMORY_INDEX_DIR = DATA_DIR / "memory_index"
DB_PATH = DATA_DIR / "conversations.db"

# Memory Categories
DEFAULT_CATEGORIES: List[str] = [
    "personal",
    "college",
    "family",
    "career",
    "exams",
    "projects",
    "preferences",
    "technical_knowledge",
    "general",
]

class Config:
    """Application configuration."""

    # API Keys & AI Provider
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "auto").strip().lower()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()

    # FreeLLMAPI / OpenAI-compatible proxy settings
    FREELLMAPI_KEY: str = os.getenv("FREELLMAPI_KEY", "").strip()
    FREELLMAPI_BASE_URL: str = os.getenv("FREELLMAPI_BASE_URL", "http://localhost:3001/v1").strip().rstrip("/")
    FREELLMAPI_MODEL: str = os.getenv("FREELLMAPI_MODEL", "gemini-3.1-flash-lite").strip()
    FREELLMAPI_EMBEDDING_MODEL: str = os.getenv("FREELLMAPI_EMBEDDING_MODEL", "auto").strip()
    FREELLMAPI_TIMEOUT: int = int(os.getenv("FREELLMAPI_TIMEOUT", "90"))

    # Embeddings: 'local' (zero-latency TF-IDF/BM25), 'freellmapi', or 'gemini'
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004").strip()

    # Thresholds & Limits
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
    MEMORY_RELEVANCE_THRESHOLD: float = float(os.getenv("MEMORY_RELEVANCE_THRESHOLD", "0.30"))
    MAX_RECENT_MESSAGES: int = int(os.getenv("MAX_RECENT_MESSAGES", "8"))
    MAX_RETRIEVED_MEMORIES: int = int(os.getenv("MAX_RETRIEVED_MEMORIES", "5"))
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
    MEMORY_AUTO_STORE_CONFIDENCE: float = float(os.getenv("MEMORY_AUTO_STORE_CONFIDENCE", "0.65"))

    # Active Categories
    CATEGORIES: List[str] = DEFAULT_CATEGORIES

    @classmethod
    def ensure_directories(cls):
        """Create all required directories and category subfolders if missing."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        MEMORY_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        for cat in cls.CATEGORIES:
            (MEMORY_DIR / cat).mkdir(parents=True, exist_ok=True)

        # Place .gitkeep files where necessary
        for keep_dir in [LOGS_DIR, MEMORY_INDEX_DIR]:
            gitkeep = keep_dir / ".gitkeep"
            if not gitkeep.exists():
                try:
                    gitkeep.touch()
                except OSError:
                    pass

    @classmethod
    def is_api_configured(cls) -> bool:
        """Check if a valid API key (FreeLLMAPI or Gemini) is configured."""
        has_freellm = bool(cls.FREELLMAPI_KEY and cls.FREELLMAPI_KEY != "your_freellmapi_key_here")
        has_gemini = bool(cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != "your_gemini_api_key_here")
        return has_freellm or has_gemini

# Initialize directories on module load
Config.ensure_directories()
