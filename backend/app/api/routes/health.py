"""
Health check endpoint - verify API and services are working.
"""

from fastapi import APIRouter, Depends  # APIRouter groups related endpoints
from typing import Dict, Any  # Type hints
import logging  # Logging

# Import dependencies
from app.api.dependencies import get_chatbot, get_settings
from main import DocumentChatbot
from app.config.settings import Settings

# ========== Setup ==========
logger = logging.getLogger(__name__)

# Create router
# APIRouter is like a mini FastAPI app for organizing endpoints
router = APIRouter()


# ========== Health Check Endpoint ==========
@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check - is the API running?
    
    **How to test:**
    ```bash
    curl http://localhost:8000/api/v1/health
    ```
    
    **Response:**
    ```json
    {
        "status": "healthy",
        "service": "Documentation Chatbot API",
        "version": "1.0.0"
    }
    ```
    
    Returns:
        Dictionary with health status
    """
    return {
        "status": "healthy",
        "service": "Documentation Chatbot API",
        "version": "1.0.0"
    }


# ========== Detailed Health Check ==========
@router.get("/health/detailed")
async def detailed_health_check(
    chatbot: DocumentChatbot = Depends(get_chatbot),
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Detailed health check - test all services.
    
    Checks:
    - Vector store connection
    - Embedding generator
    - LLM service
    
    **How to test:**
    ```bash
    curl http://localhost:8000/api/v1/health/detailed
    ```
    
    **Response:**
    ```json
    {
        "status": "healthy",
        "services": {
            "vector_store": true,
            "embedding_generator": true,
            "llm_service": true
        },
        "configuration": {
            "chunk_size": 1000,
            "top_k": 5,
            "similarity_threshold": 0.7
        }
    }
    ```
    
    Args:
        chatbot: Injected DocumentChatbot instance
        settings: Injected Settings instance
        
    Returns:
        Dictionary with detailed health information
    """
    logger.info("Performing detailed health check...")
    
    # Check all services
    service_health = chatbot.health_check()
    
    # Determine overall status
    all_healthy = all(service_health.values())
    overall_status = "healthy" if all_healthy else "degraded"
    
    # Build response
    response = {
        "status": overall_status,
        "services": service_health,
        "configuration": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": settings.TOP_K_RESULTS,
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "embedding_dimension": settings.EMBEDDING_DIMENSION
        }
    }
    
    logger.info(f"Health check complete: {overall_status}")
    return response


# ========== Readiness Check ==========
@router.get("/health/ready")
async def readiness_check(
    chatbot: DocumentChatbot = Depends(get_chatbot)
) -> Dict[str, bool]:
    """
    Kubernetes-style readiness probe.
    
    Checks if the API is ready to handle requests.
    
    **How to test:**
    ```bash
    curl http://localhost:8000/api/v1/health/ready
    ```
    
    **Response:**
    ```json
    {
        "ready": true
    }
    ```
    
    Args:
        chatbot: Injected DocumentChatbot instance
        
    Returns:
        Dictionary with readiness status
    """
    try:
        # Quick check: can we access the chatbot?
        _ = chatbot.vector_store.check_connection()
        return {"ready": True}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"ready": False}


# ========== Liveness Check ==========
@router.get("/health/live")
async def liveness_check() -> Dict[str, bool]:
    """
    Kubernetes-style liveness probe.
    
    Simple check that the API process is alive.
    
    **How to test:**
    ```bash
    curl http://localhost:8000/api/v1/health/live
    ```
    
    **Response:**
    ```json
    {
        "alive": true
    }
    ```
    
    Returns:
        Dictionary with liveness status
    """
    # If this endpoint responds, the API is alive
    return {"alive": True}