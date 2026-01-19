"""
Utility helper functions
"""

import time
import json
from typing import Any, Dict
from datetime import datetime
from functools import wraps


def timing_decorator(func):
    """Decorator to measure execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = int((time.time() - start) * 1000)
        return result, elapsed
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = int((time.time() - start) * 1000)
        return result, elapsed
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def sanitize_text(text: str, max_length: int = 1000) -> str:
    """Sanitize and truncate text"""
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text.strip()


def format_timestamp(dt: datetime) -> str:
    """Format datetime for display"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON with fallback"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def calculate_percentage(part: int, total: int) -> float:
    """Calculate percentage safely"""
    if total == 0:
        return 0.0
    return round((part / total) * 100, 2)


import asyncio  # For timing_decorator
