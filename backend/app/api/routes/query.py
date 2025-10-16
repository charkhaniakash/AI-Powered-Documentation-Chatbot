"""
Query endpoint - ask questions about documents.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

# Import dependencies
from app.api.dependencies import get_chatbot
from main import DocumentChatbot

# Setup
logger = logging.getLogger(__name__)
router = APIRouter()


# ========== Request Model ==========
class QueryRequest(BaseModel):
    """Request body for query endpoint."""
    query: str                                    # The question
    top_k: Optional[int] = 5                     # Number of results
    document_ids: Optional[List[str]] = None     # Filter by documents


# ========== Response Model ==========
class QueryResponse(BaseModel):
    """Response from query endpoint."""
    answer: str                    # The answer
    sources: List[Dict[str, Any]]  # Source documents
    confidence: float              # Confidence score


# ========== Query Endpoint ==========
@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    chatbot: DocumentChatbot = Depends(get_chatbot)
) -> QueryResponse:
    """
    Ask a question about your documents.
    
    **How to test with curl:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/query" \
         -H "Content-Type: application/json" \
         -d '{"query": "What is this document about?"}'
    ```
    
    **How to test with Python:**
    ```python
    import requests
    
    data = {"query": "What is RAG?"}
    response = requests.post('http://localhost:8000/api/v1/query', json=data)
    print(response.json())
    ```
    
    **Response:**
    ```json
    {
        "answer": "RAG stands for...",
        "sources": [
            {
                "filename": "document.pdf",
                "score": 0.89,
                "text": "..."
            }
        ],
        "confidence": 0.85
    }
    ```
    """
    try:
        logger.info(f"Query received: {request.query[:50]}...")
        
        # Query chatbot
        response = chatbot.query(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids
        )
        
        # Format sources
        sources = []
        for source in response.sources:
            sources.append({
                "filename": source.chunk.metadata.filename,
                "document_id": source.chunk.metadata.document_id,
                "score": source.score,
                "text": source.chunk.text[:200]  # First 200 chars
            })
        
        logger.info(f"✓ Query completed: {len(sources)} sources")
        
        return QueryResponse(
            answer=response.answer,
            sources=sources,
            confidence=response.confidence
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")