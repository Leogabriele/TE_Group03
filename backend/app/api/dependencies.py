"""
FastAPI dependencies for dependency injection
"""

from backend.app.config import settings, get_settings
from backend.app.models.database import db, get_database
from backend.app.core.orchestrator import Orchestrator


async def get_orchestrator() -> Orchestrator:
    """Get orchestrator instance"""
    return Orchestrator()


__all__ = [
    "get_settings",
    "get_database",
    "get_orchestrator"
]
