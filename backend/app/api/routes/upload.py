"""
Upload endpoint - upload and ingest documents.
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import Dict, Any
import os
import logging

# Import dependencies
from app.api.dependencies import get_chatbot, validate_file_upload
from main import DocumentChatbot

# Setup
logger = logging.getLogger(__name__)
router = APIRouter()


# ========== Upload Single Document ==========
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),  # File from form-data
    chatbot: DocumentChatbot = Depends(get_chatbot)
) -> Dict[str, Any]:
    """
    Upload and ingest a document.
    
    **How to test with curl:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/upload" \
         -F "file=@your_document.pdf"
    ```
    
    **How to test with Python:**
    ```python
    import requests
    
    files = {'file': open('document.pdf', 'rb')}
    response = requests.post('http://localhost:8000/api/v1/upload', files=files)
    print(response.json())
    ```
    
    **Response:**
    ```json
    {
        "success": true,
        "document_id": "abc-123",
        "filename": "document.pdf",
        "message": "Document uploaded and processed successfully"
    }
    ```
    """
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        # Get file extension
        file_extension = file.filename.split('.')[-1]
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Validate file
        validate_file_upload(file_size, file_extension)
        
        # Save file temporarily
        temp_path = f"data/uploads/{file.filename}"
        os.makedirs("data/uploads", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Ingest document
        result = chatbot.ingest_document(
            file_path=temp_path,
            filename=file.filename
        )
        
        # Clean up temp file
        os.remove(temp_path)
        
        logger.info(f"✓ Document uploaded: {result.document_id}")
        
        return {
            "success": True,
            "document_id": result.document_id,
            "filename": file.filename,
            "message": "Document uploaded and processed successfully"
        }
        
    except ValueError as e:
        # Validation error
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        # Other errors
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")