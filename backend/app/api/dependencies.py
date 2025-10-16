"""
FastAPI dependencies - shared resources and dependency injection.

Dependency Injection = A way to share resources (like database connections)
across multiple endpoints without recreating them every time.
"""

from typing import Generator  # For type hints
from functools import lru_cache  # For caching
import logging  # For logging

# Import our services
from main import DocumentChatbot  # Our main chatbot class
from app.config.settings import settings, Settings

# ========== Setup Logging ==========
logger = logging.getLogger(__name__)


# ========== Chatbot Singleton ==========
# Singleton = Only one instance exists, shared by all requests
# This is important because loading models is expensive

_chatbot_instance = None  # Global variable to store the chatbot


def get_chatbot() -> DocumentChatbot:
    """
    Get the chatbot instance (creates it if it doesn't exist).
    
    This is a dependency that endpoints can use:
    
    @app.post("/query")
    async def query_endpoint(chatbot: DocumentChatbot = Depends(get_chatbot)):
        response = chatbot.query("test")
        ...
    
    Returns:
        DocumentChatbot instance
    """
    global _chatbot_instance
    
    # Create instance if it doesn't exist
    if _chatbot_instance is None:
        logger.info("Initializing DocumentChatbot instance...")
        
        try:
            # Initialize with default providers
            _chatbot_instance = DocumentChatbot(
                embedding_provider="sentence_transformers",
                vector_store_provider="pinecone"
            )
            logger.info("✓ Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize chatbot: {e}")
            raise RuntimeError(f"Chatbot initialization failed: {e}")
    
    # Return the singleton instance
    return _chatbot_instance


# ========== Settings Dependency ==========
@lru_cache()  # Cache the result (settings don't change during runtime)
def get_settings() -> Settings:
    """
    Get application settings.
    
    This is a dependency for endpoints that need settings:
    
    @app.get("/config")
    async def config_endpoint(settings: Settings = Depends(get_settings)):
        return {"chunk_size": settings.CHUNK_SIZE}
    
    Returns:
        Settings instance
    """
    return settings


# ========== File Upload Validator ==========
def validate_file_upload(
    file_size: int,
    file_extension: str
) -> None:
    """
    Validate uploaded file.
    
    Checks:
    1. File size is within limits
    2. File extension is allowed
    
    Args:
        file_size: Size of file in bytes
        file_extension: File extension (without dot)
        
    Raises:
        ValueError: If validation fails
    """
    # Check file size
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        raise ValueError(
            f"File too large: {file_size / 1024 / 1024:.2f} MB. "
            f"Maximum allowed: {settings.MAX_FILE_SIZE_MB} MB"
        )
    
    # Check file extension
    if file_extension.lower() not in settings.ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File type '.{file_extension}' not allowed. "
            f"Allowed types: {settings.ALLOWED_EXTENSIONS}"
        )
    
    logger.debug(f"File validation passed: {file_extension}, {file_size} bytes")


# ========== Request ID Generator ==========
import uuid

def generate_request_id() -> str:
    """
    Generate unique request ID for tracking.
    
    Useful for:
    - Logging
    - Debugging
    - Request tracing
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())


# ========== Rate Limiting (Simple In-Memory) ==========
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict

# Store request counts: {ip_address: [(timestamp, count), ...]}
_request_tracker: Dict[str, list] = defaultdict(list)


def check_rate_limit(
    client_ip: str,
    max_requests: int = 100,
    time_window: int = 60  # seconds
) -> bool:
    """
    Simple rate limiting check.
    
    Limits requests per IP address.
    
    Args:
        client_ip: Client IP address
        max_requests: Maximum requests allowed
        time_window: Time window in seconds
        
    Returns:
        True if within limit, False if exceeded
    """
    now = datetime.now()
    cutoff_time = now - timedelta(seconds=time_window)
    
    # Get requests for this IP
    requests = _request_tracker[client_ip]
    
    # Remove old requests (outside time window)
    requests = [req for req in requests if req > cutoff_time]
    
    # Update tracker
    _request_tracker[client_ip] = requests
    
    # Check if limit exceeded
    if len(requests) >= max_requests:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return False
    
    # Add current request
    requests.append(now)
    
    return True


# ========== Authentication (Placeholder) ==========
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(None)) -> str:
    """
    Verify API key from request header.
    
    Usage:
    @app.get("/protected")
    async def protected_endpoint(api_key: str = Depends(verify_api_key)):
        return {"message": "Access granted"}
    
    Client must send:
    Headers: {
        "X-API-Key": "your-api-key"
    }
    
    Args:
        x_api_key: API key from request header
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    # For now, this is a placeholder
    # In production, check against database
    
    # Example: Simple key check (not secure for production!)
    # valid_keys = ["your-secret-key-123", "another-key-456"]
    
    # if x_api_key not in valid_keys:
    #     raise HTTPException(
    #         status_code=401,
    #         detail="Invalid API key"
    #     )
    
    # For development, allow all requests
    return x_api_key or "development"


# ========== Request Context ==========
from contextvars import ContextVar

# Context variables for storing request-specific data
# Useful for logging, tracing, etc.
request_id_var: ContextVar[str] = ContextVar('request_id', default=None)
user_id_var: ContextVar[str] = ContextVar('user_id', default=None)


def set_request_context(request_id: str, user_id: str = None):
    """
    Set request context variables.
    
    Args:
        request_id: Unique request ID
        user_id: User ID (optional)
    """
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get()


def get_user_id() -> str:
    """Get current user ID from context."""
    return user_id_var.get()


# ========== Cleanup on Shutdown ==========
def cleanup_resources():
    """
    Cleanup resources when API shuts down.
    
    Call this in the shutdown event.
    """
    global _chatbot_instance
    
    logger.info("Cleaning up resources...")
    
    # Clear chatbot instance
    _chatbot_instance = None
    
    # Clear rate limit tracker
    _request_tracker.clear()
    
    logger.info("✓ Resources cleaned up")