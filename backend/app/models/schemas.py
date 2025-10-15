"""
Data models and schemas for the Documentation Chatbot.
These define the structure of data flowing through the application.
"""

# Import necessary libraries
from pydantic import BaseModel, Field, validator  # For data validation
from typing import List, Optional, Dict, Any  # For type hints
from datetime import datetime  # For timestamps
from enum import Enum  # For enumeration types


# ========== Enums ==========
class FileType(str, Enum):
    """
    Enumeration of supported file types.
    
    Enum ensures only valid file types are used throughout the app.
    str inheritance allows it to be serialized as string in JSON.
    """
    PDF = "pdf"      # PDF documents
    MARKDOWN = "md"  # Markdown files
    TEXT = "txt"     # Plain text files


class ProcessingStatus(str, Enum):
    """
    Status of document processing pipeline.
    
    Tracks where a document is in the ingestion pipeline.
    """
    PENDING = "pending"          # Waiting to be processed
    UPLOADING = "uploading"      # File is being uploaded
    PARSING = "parsing"          # Extracting text from file
    CHUNKING = "chunking"        # Splitting text into chunks
    EMBEDDING = "embedding"      # Generating embeddings
    STORING = "storing"          # Storing in vector database
    COMPLETED = "completed"      # Successfully processed
    FAILED = "failed"            # Processing failed


# ========== Document Models ==========
class DocumentMetadata(BaseModel):
    """
    Metadata associated with a document.
    
    This information helps with filtering, tracking, and auditing.
    """
    # Original filename
    filename: str = Field(
        ...,
        description="Original filename of the uploaded document"
    )
    
    # File type
    file_type: FileType = Field(
        ...,
        description="Type of the document (pdf, md, txt)"
    )
    
    # File size in bytes
    file_size: int = Field(
        ...,
        gt=0,  # Must be greater than 0
        description="Size of the file in bytes"
    )
    
    # Upload timestamp
    uploaded_at: datetime = Field(
        default_factory=datetime.now,  # Automatically set to current time
        description="Timestamp when document was uploaded"
    )
    
    # Unique document ID
    document_id: str = Field(
        ...,
        description="Unique identifier for the document"
    )
    
    # Number of pages (for PDFs) or lines (for text files)
    page_count: Optional[int] = Field(
        default=None,
        description="Number of pages/sections in document"
    )
    
    # User who uploaded (optional, for multi-user systems)
    uploaded_by: Optional[str] = Field(
        default=None,
        description="User ID who uploaded the document"
    )
    
    # Custom metadata fields
    custom_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional custom metadata"
    )
    
    class Config:
        """Pydantic configuration"""
        # Allow the model to be used with ORM objects
        from_attributes = True
        # Use enum values in JSON
        use_enum_values = True


class TextChunk(BaseModel):
    """
    Represents a chunk of text from a document.
    
    Documents are split into chunks for better retrieval.
    Each chunk is embedded and stored in the vector database.
    """
    # Unique identifier for this chunk
    chunk_id: str = Field(
        ...,
        description="Unique identifier for the chunk"
    )
    
    # The actual text content
    text: str = Field(
        ...,
        min_length=1,  # Must have at least 1 character
        description="Text content of the chunk"
    )
    
    # Metadata about the source document
    metadata: DocumentMetadata = Field(
        ...,
        description="Metadata of the source document"
    )
    
    # Position of this chunk in the document
    chunk_index: int = Field(
        ...,
        ge=0,  # Must be >= 0
        description="Index of this chunk in the document"
    )
    
    # Start character position in original document
    start_char: int = Field(
        ...,
        ge=0,
        description="Starting character position in original document"
    )
    
    # End character position in original document
    end_char: int = Field(
        ...,
        gt=0,  # Must be > 0
        description="Ending character position in original document"
    )
    
    # Embedding vector (optional, set after embedding)
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Embedding vector for this chunk"
    )
    
    @validator('end_char')
    def validate_char_positions(cls, end_char, values):
        """
        Validate that end_char is greater than start_char.
        
        Args:
            end_char: Ending character position
            values: Other field values
            
        Returns:
            end_char if valid
            
        Raises:
            ValueError: If end_char <= start_char
        """
        # Check if start_char exists in values
        if 'start_char' in values:
            # Ensure end position is after start position
            if end_char <= values['start_char']:
                raise ValueError("end_char must be greater than start_char")
        return end_char
    
    def get_text_preview(self, max_length: int = 100) -> str:
        """
        Get a preview of the chunk text.
        
        Args:
            max_length: Maximum length of preview
            
        Returns:
            Preview string
        """
        # If text is short, return as-is
        if len(self.text) <= max_length:
            return self.text
        # Otherwise, truncate and add ellipsis
        return self.text[:max_length] + "..."


class EmbeddedChunk(TextChunk):
    """
    A text chunk with its embedding vector.
    
    Extends TextChunk to require an embedding.
    This is used after the embedding step.
    """
    # Override embedding to make it required
    embedding: List[float] = Field(
        ...,
        min_items=1,  # Must have at least 1 dimension
        description="Embedding vector for this chunk"
    )
    
    @validator('embedding')
    def validate_embedding_dimension(cls, embedding, values):
        """
        Validate embedding dimension matches expected size.
        
        Args:
            embedding: The embedding vector
            values: Other field values
            
        Returns:
            embedding if valid
            
        Raises:
            ValueError: If dimension doesn't match
        """
        # Import settings to check dimension
        from app.config.settings import settings
        
        expected_dim = settings.EMBEDDING_DIMENSION
        actual_dim = len(embedding)
        
        # Check if dimensions match
        if actual_dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, "
                f"got {actual_dim}"
            )
        return embedding


# ========== Query Models ==========
class QueryRequest(BaseModel):
    """
    Request model for querying the chatbot.
    
    This is what the user sends when asking a question.
    """
    # The user's question
    query: str = Field(
        ...,
        min_length=1,  # Must have at least 1 character
        max_length=1000,  # Reasonable maximum length
        description="User's question"
    )
    
    # Optional: Filter by document IDs
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter results to specific documents"
    )
    
    # Optional: Number of results to retrieve
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,  # Max 20 results
        description="Number of results to retrieve"
    )
    
    # Optional: Include source chunks in response
    include_sources: bool = Field(
        default=True,
        description="Include source chunks in response"
    )
    
    # Optional: Conversation history for context
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Previous conversation messages"
    )


class RetrievedChunk(BaseModel):
    """
    A chunk retrieved from the vector database.
    
    Includes the chunk data and its similarity score.
    """
    # The chunk data
    chunk: TextChunk = Field(
        ...,
        description="Retrieved text chunk"
    )
    
    # Similarity score (0.0 to 1.0)
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score of the chunk"
    )
    
    # Rank in the results
    rank: int = Field(
        ...,
        ge=1,
        description="Rank of this chunk in retrieval results"
    )


class QueryResponse(BaseModel):
    """
    Response model for a query.
    
    Contains the LLM's answer and supporting information.
    """
    # The generated answer
    answer: str = Field(
        ...,
        description="Generated answer to the query"
    )
    
    # Source chunks used to generate the answer
    sources: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Source chunks used for answer generation"
    )
    
    # Confidence score (optional)
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer"
    )
    
    # Response generation metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response"
    )
    
    # Timestamp
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when answer was generated"
    )


# ========== Processing Models ==========
class ProcessingJob(BaseModel):
    """
    Tracks the status of a document processing job.
    
    Used to monitor and update the progress of document ingestion.
    """
    # Job ID
    job_id: str = Field(
        ...,
        description="Unique identifier for the processing job"
    )
    
    # Document metadata
    document_metadata: DocumentMetadata = Field(
        ...,
        description="Metadata of the document being processed"
    )
    
    # Current status
    status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current processing status"
    )
    
    # Progress percentage (0-100)
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Processing progress percentage"
    )
    
    # Start time
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="When processing started"
    )
    
    # Completion time (optional)
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When processing completed"
    )
    
    # Error message (if failed)
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    
    # Number of chunks created
    chunk_count: Optional[int] = Field(
        default=None,
        description="Number of chunks created from document"
    )


# ========== Upload Models ==========
class UploadResponse(BaseModel):
    """
    Response after uploading a document.
    
    Returns the job ID and initial status.
    """
    # Job ID for tracking
    job_id: str = Field(
        ...,
        description="Job ID for tracking processing status"
    )
    
    # Document ID
    document_id: str = Field(
        ...,
        description="Unique identifier for the document"
    )
    
    # Message
    message: str = Field(
        default="Document uploaded successfully",
        description="Status message"
    )
    
    # Initial status
    status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Initial processing status"
    )


# ========== Error Models ==========
class ErrorResponse(BaseModel):
    """
    Standard error response model.
    
    Used for consistent error reporting across the API.
    """
    # Error message
    error: str = Field(
        ...,
        description="Error message"
    )
    
    # Error code (optional)
    code: Optional[str] = Field(
        default=None,
        description="Error code for programmatic handling"
    )
    
    # Additional details (optional)
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    
    # Timestamp
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the error occurred"
    )