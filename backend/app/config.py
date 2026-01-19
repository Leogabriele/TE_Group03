"""
Configuration Management using Pydantic Settings
"""

from typing import Literal
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # LLM API Keys
    GROQ_API_KEY: str = Field(..., description="Groq API key")
    NVIDIA_API_KEY: str = Field(..., description="NVIDIA NIM API key")
    
    # MongoDB Configuration
    MONGODB_URI: str = Field(..., description="MongoDB connection URI")
    MONGODB_DB_NAME: str = Field(default="llm_security_auditor")
    
    # Application Settings
    ENVIRONMENT: Literal["development", "production", "testing"] = Field(default="development")
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    API_PORT: int = Field(default=8000, ge=1000, le=65535)
    
    # Model Configuration
    ATTACKER_MODEL: str = Field(default="llama-3.1-8b-instant")
    JUDGE_MODEL: str = Field(default="llama-3.3-70b-versatile")
    TARGET_MODEL_PROVIDER: Literal["groq", "nvidia"] = Field(default="nvidia")
    TARGET_MODEL_NAME: str = Field(default="meta/llama3-70b-instruct")
    
    # LLM Generation Parameters
    ATTACKER_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    JUDGE_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=1.0)
    MAX_TOKENS: int = Field(default=1024, ge=100, le=4096)
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = Field(default=30, ge=1)
    REQUEST_TIMEOUT: int = Field(default=60, ge=10)
    
    # Paths
    DATA_DIR: str = Field(default="./data")
    LOGS_DIR: str = Field(default="./logs")
    
    @validator("MONGODB_URI")
    def validate_mongodb_uri(cls, v):
        """Validate MongoDB URI format"""
        if not v.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError("MongoDB URI must start with mongodb:// or mongodb+srv://")
        return v
    
    @validator("DATA_DIR", "LOGS_DIR")
    def create_directories(cls, v):
        """Create directories if they don't exist"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings (for FastAPI dependency injection)"""
    return settings
