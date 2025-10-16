"""
Documents endpoint - manage uploaded documents.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

# Import dependencies
from app.api.dependencies import get_chatbot
from main import DocumentChatbot

# Setup
logger = logging.getLogger(__name__)
router = APIRouter()


# ========== Get Statistics ==========
@router.get("/documents/stats")
async def get_statistics(
    chatbot: DocumentChatbot = Depends(get_chatbot)
) -> Dict[str, Any]:
    """
    Get system statistics.
    
    **How to test:**
    ```bash
    curl http://localhost:8000/api/v1/documents/stats
    ```
    
    **Response:**
    ```json
    {
        "total_vectors": 150,
        "embedding_dimension": 384,
        "settings": {...}
    }
    ```
    """
    try:
        stats = chatbot.get_statistics()
        logger.info("Statistics retrieved")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Delete Document ==========
@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    chatbot: DocumentChatbot = Depends(get_chatbot)
) -> Dict[str, Any]:
    """
    Delete a document by ID.
    
    **How to test:**
    ```bash
    curl -X DELETE "http://localhost:8000/api/v1/documents/abc-123"
    ```
    
    **Response:**
    ```json
    {
        "success": true,
        "message": "Document deleted",
        "document_id": "abc-123"
    }
    ```
    """
    try:
        logger.info(f"Deleting document: {document_id}")
        
        result = chatbot.delete_document(document_id)
        
        logger.info(f"✓ Document deleted: {document_id}")
        
        return {
            "success": True,
            "message": "Document deleted successfully",
            "document_id": document_id
        }
        
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))