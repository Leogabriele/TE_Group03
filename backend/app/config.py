# backend/app/config.py

"""
Configuration Management using Pydantic Settings
"""

from typing import Literal, List, Tuple
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path

# Absolute project root — works from ANY working directory
_PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # ── API Keys ──────────────────────────────────────────────────────────────
    GROQ_API_KEY:   str = Field(..., description="Groq API key")
    NVIDIA_API_KEY: str = Field(..., description="NVIDIA NIM API key")

    # ── MongoDB ───────────────────────────────────────────────────────────────
    MONGODB_URI:     str = Field(..., description="MongoDB connection URI")
    MONGODB_DB_NAME: str = Field(default="llm_security_auditor")

    # ── App Settings ──────────────────────────────────────────────────────────
    ENVIRONMENT: Literal["development", "production", "testing"] = Field(default="development")
    LOG_LEVEL:   Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    API_PORT:    int = Field(default=8000, ge=1000, le=65535)

    # ── Attacker Model ────────────────────────────────────────────────────────
    ATTACKER_PROVIDER: Literal["groq", "nvidia", "ollama"] = Field(
        default="groq",
        description="Provider for the attacker/red-team LLM"
    )
    ATTACKER_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Model name for the attacker LLM"
    )
    ATTACKER_OLLAMA_MODEL: str = Field(
        default="llama3.2:latest",
        description="Ollama model to use as last-resort attacker fallback"
    )
    ATTACKER_FALLBACKS: str = Field(
        default="groq/llama-3.1-8b-instant,groq/llama3-70b-8192",
        description=(
            "Comma-separated list of provider/model pairs for attacker fallback chain. "
            "Example: groq/llama-3.1-8b-instant,nvidia/meta/llama3-70b-instruct,ollama/llama3.2:latest"
        )
    )

    # ── Judge Model ───────────────────────────────────────────────────────────
    JUDGE_MODEL:       str   = Field(default="llama-3.3-70b-versatile")
    JUDGE_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── Target Model ──────────────────────────────────────────────────────────
    TARGET_MODEL_PROVIDER: Literal["groq", "nvidia", "ollama"] = Field(default="nvidia")
    TARGET_MODEL_NAME:     str = Field(default="meta/llama3-70b-instruct")

    # ── Ollama ────────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama server"
    )

    # ── Generation Parameters ─────────────────────────────────────────────────
    ATTACKER_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    MAX_TOKENS:           int   = Field(default=1024, ge=100, le=4096)

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    MAX_REQUESTS_PER_MINUTE: int = Field(default=30, ge=1)
    REQUEST_TIMEOUT:         int = Field(default=60, ge=10)

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR: str = Field(default="./data")
    LOGS_DIR: str = Field(default="./logs")

    # ── Validators ────────────────────────────────────────────────────────────

    @validator("MONGODB_URI")
    def validate_mongodb_uri(cls, v):
        if not v.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError("MongoDB URI must start with mongodb:// or mongodb+srv://")
        return v

    @validator("DATA_DIR", "LOGS_DIR")
    def create_directories(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    # ── Helper Methods ────────────────────────────────────────────────────────

    def get_attacker_fallback_list(self) -> List[Tuple[str, str]]:
        """
        Parse ATTACKER_FALLBACKS into a list of (provider, model) tuples.

        Example:
            "groq/llama-3.1-8b-instant,nvidia/meta/llama3-70b-instruct"
            → [("groq", "llama-3.1-8b-instant"), ("nvidia", "meta/llama3-70b-instruct")]
        """
        result = []
        if not self.ATTACKER_FALLBACKS:
            return result
        for entry in self.ATTACKER_FALLBACKS.split(","):
            entry = entry.strip()
            if not entry:
                continue
            # Split on first "/" only so model paths like "meta/llama3-70b" stay intact
            parts = entry.split("/", 1)
            if len(parts) == 2:
                provider, model = parts[0].strip(), parts[1].strip()
                if provider and model:
                    result.append((provider, model))
            else:
                # No provider prefix — skip with warning
                pass
        return result

    class Config:
        # ✅ Absolute path — works from ANY CWD (scripts/, root, etc.)
        env_file          = str(_PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        case_sensitive    = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings (for FastAPI dependency injection)"""
    return settings
