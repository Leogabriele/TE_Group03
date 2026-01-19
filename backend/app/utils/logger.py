"""
Logging configuration using Loguru
"""

import sys
from loguru import logger
from pathlib import Path
from backend.app.config import settings


def setup_logging():
    """Configure application logging"""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True
    )
    
    # File handler for all logs
    log_file = Path(settings.LOGS_DIR) / "app.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # Error file handler
    error_log_file = Path(settings.LOGS_DIR) / "errors.log"
    logger.add(
        error_log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
        compression="zip"
    )
    
    logger.info(f"✅ Logging configured - Level: {settings.LOG_LEVEL}")
    return logger


# Initialize logging
app_logger = setup_logging()
