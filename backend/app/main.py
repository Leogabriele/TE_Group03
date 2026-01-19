"""
FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from backend.app.config import settings
from backend.app.models.database import db
from backend.app.api.routes import router
from backend.app.utils.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting LLM Security Auditor API...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Connect to database
    try:
        await db.connect()
        logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    await db.disconnect()
    logger.info("✅ Shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="LLM Security Auditor",
    description="Automated adversarial testing framework for LLMs",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Security Auditor API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health"
    }


# Run with: uvicorn backend.app.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower()
    )
