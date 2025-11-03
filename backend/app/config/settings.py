"""
Configuration management for the Documentation Chatbot.
This module handles all environment variables and application settings.
"""

# Import necessary libraries
from pydantic_settings import BaseSettings  # Base class for settings management with validation
from pydantic import Field  # For field validation and default values
from typing import Optional  # For optional type hints
import os  # For environment variable access


class Settings(BaseSettings):
    """
    Application settings class using Pydantic for validation.
    
    Pydantic automatically:
    - Reads from .env file
    - Validates types
    - Provides defaults
    - Raises errors for missing required fields
    """
    
    # ========== API Keys ==========
    # GROQ API key for embeddings and LLM
    # Field(...) means this is REQUIRED - app won't start without it
    GROQ_API_KEY: str = Field(
        ...,  # Required field
        description="Groq API key for embeddings and chat completions"
    )
    
    # Pinecone API key for vector database
    PINECONE_API_KEY: str = Field(
        ...,  # Required field
        description="Pinecone API key for vector storage"
    )
    
    # Pinecone environment (e.g., 'us-west1-gcp', 'eu-west1-gcp')
    PINECONE_ENVIRONMENT: str = Field(
        default="us-west1-gcp",  # Default value if not provided
        description="Pinecone environment/region"
    )
    
    # ========== Vector Database Settings ==========
    # Name of the Pinecone index where vectors will be stored
    PINECONE_INDEX_NAME: str = Field(
        default="documentation-chatbot",
        description="Name of the Pinecone index"
    )
    
    # Dimension of embedding vectors (Groq's embedding model dimension)
    # This MUST match the embedding model's output dimension
    EMBEDDING_DIMENSION: int = Field(
        default=1536,
        description="Dimension of embedding vectors"
    )
    
    # ========== LLM Settings ==========
    # Model name for Groq chat completions
    LLM_MODEL_NAME: str = Field(
        default="llama-3.1-8b-instant",  # Mixtral is good for reasoning
        description="Groq LLM model name"
    )
    
    # Model name for Groq embeddings
    EMBEDDING_MODEL_NAME: str = Field(
        default="llama-3.1-8b-instant",  # Note: Groq doesn't have dedicated embedding models
        description="Model for generating embeddings"
    )
    
    # Maximum tokens in LLM response
    MAX_TOKENS: int = Field(
        default=1024,
        description="Maximum tokens in LLM response"
    )
    
    # Temperature for LLM (0.0 = deterministic, 1.0 = creative)
    TEMPERATURE: float = Field(
        default=0.3,  # Low temperature for factual responses
        ge=0.0,  # Greater than or equal to 0
        le=2.0,  # Less than or equal to 2
        description="LLM temperature for response generation"
    )
    
    # ========== Chunking Settings ==========
    # Size of each text chunk in characters
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Size of text chunks for processing"
    )
    
    # Overlap between chunks to maintain context
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Overlap between consecutive chunks"
    )
    
    # ========== Retrieval Settings ==========
    # Number of relevant chunks to retrieve for context
    TOP_K_RESULTS: int = Field(
        default=5,
        description="Number of top results to retrieve from vector DB"
    )
    
    # Minimum similarity score to consider a chunk relevant (0.0 to 1.0)
    SIMILARITY_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval"
    )
    
    # ========== File Processing Settings ==========
    # Maximum file size in MB
    MAX_FILE_SIZE_MB: int = Field(
        default=10,
        description="Maximum allowed file size in megabytes"
    )
    
    # Allowed file extensions
    # Now includes Excel (.xlsx, .xls), CSV (.csv), and Word documents (.docx, .doc)
    ALLOWED_EXTENSIONS: list[str] = Field(
        default=["pdf", "md", "txt", "xlsx", "xls", "csv", "docx", "doc"],
        description="Allowed file extensions for upload"
    )
    
    # Directory for temporary file uploads
    UPLOAD_DIR: str = Field(
        default="data/uploads",
        description="Directory for storing uploaded files"
    )
    
    # ========== Application Settings ==========
    # Application name
    APP_NAME: str = Field(
        default="Documentation Chatbot",
        description="Application name"
    )
    
    # Debug mode
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # Logging level
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    # Add to Settings class (around line 90)

    # ========== Hybrid Search Settings ==========
    # Enable hybrid search (BM25 + Vector)
    ENABLE_HYBRID_SEARCH: bool = Field(
        default=True,
        description="Enable hybrid search with BM25"
    )

    # Use RRF or weighted fusion
    USE_RRF: bool = Field(
        default=True,
        description="Use Reciprocal Rank Fusion (True) or weighted fusion (False)"
    )

    # Vector search weight (for weighted fusion)
    VECTOR_WEIGHT: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in weighted fusion"
    )

    # BM25 weight (for weighted fusion)
    BM25_WEIGHT: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 search in weighted fusion"
    )

    # ========== Contextual Compression Settings ==========
    ENABLE_CONTEXTUAL_COMPRESSION: bool = Field(
        default=True,
        description="Enable contextual compression"
    )
    COMPRESSION_RELEVANCE_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relevance threshold for compression"
    )
    COMPRESSION_MAX_SENTENCES: int = Field(
        default=5,
        ge=1,
        description="Maximum sentences per chunk for compression"
    )

    # ========== Cache Settings ==========
    ENABLE_CACHING: bool = Field(
        default=True,
        description="Enable multi-level caching system"
    )

    # L1 Cache (In-Memory)
    L1_CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable L1 in-memory cache"
    )
    L1_CACHE_MAX_SIZE: int = Field(
        default=1000,
        description="Maximum L1 cache entries"
    )
    L1_CACHE_TTL_SECONDS: int = Field(
        default=3600,  # 1 hour
        description="L1 cache time-to-live in seconds"
    )

    # L3 Cache (Redis)
    L3_CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable L3 Redis cache"
    )
    REDIS_HOST: str = Field(
        default="localhost",
        description="Redis host"
    )
    REDIS_PORT: int = Field(
        default=6379,
        description="Redis port"
    )
    REDIS_DB: int = Field(
        default=0,
        description="Redis database number"
    )
    REDIS_PASSWORD: Optional[str] = Field(
        default=None,
        description="Redis password (optional)"
    )
    REDIS_TTL_SECONDS: int = Field(
        default=86400,  # 24 hours
        description="Redis cache TTL in seconds"
    )
    REDIS_MAX_CONNECTIONS: int = Field(
        default=10,
        description="Maximum Redis connection pool size"
    )

    # Cache Key Prefix
    CACHE_KEY_PREFIX: str = Field(
        default="rag_cache",
        description="Prefix for all cache keys"
    )

    ENABLE_DUPLICATE_DETECTION: bool = Field(
    default=True,
    description="Enable duplicate document detection"
)

    DUPLICATE_DETECTION_METHOD: str = Field(
        default="both",  # "file_hash", "content_hash", or "both"
        description="Duplicate detection method"
    )

    ALLOW_DUPLICATE_FILENAMES: bool = Field(
        default=False,
        description="Allow same filename if content differs"
    )
    
    class Config:
        """
        Pydantic configuration class.
        Tells Pydantic where to find environment variables.
        """
        # Look for .env file in the project root
        env_file = ".env"
        # Don't ignore extra fields in .env (strict validation)
        extra = "ignore"
        # Make the settings case-insensitive for env vars
        case_sensitive = False
        


# ========== Global Settings Instance ==========
# Create a single instance of settings that will be used throughout the app
# This is a singleton pattern - only one settings object exists
settings = Settings()


# ========== Helper Functions ==========
def get_settings() -> Settings:
    """
    Dependency function to get settings instance.
    
    This function will be used in FastAPI as a dependency:
    def some_endpoint(settings: Settings = Depends(get_settings)):
        ...
    
    Returns:
        Settings: The global settings instance
    """
    return settings


def create_upload_directory() -> None:
    """
    Create the upload directory if it doesn't exist.
    
    This ensures the directory is ready before the app starts
    processing files.
    """
    # os.makedirs creates the directory and all parent directories
    # exist_ok=True prevents errors if directory already exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Print confirmation message
    print(f"✓ Upload directory ensured: {settings.UPLOAD_DIR}")


# ========== Validation Functions ==========
def validate_settings() -> None:
    """
    Validate critical settings on startup.
    
    This catches configuration errors early before the app
    tries to use invalid settings.
    
    Raises:
        ValueError: If any critical setting is invalid
    """
    # Check if chunk overlap is less than chunk size
    if settings.CHUNK_OVERLAP >= settings.CHUNK_SIZE:
        raise ValueError(
            f"CHUNK_OVERLAP ({settings.CHUNK_OVERLAP}) must be less than "
            f"CHUNK_SIZE ({settings.CHUNK_SIZE})"
        )
    
    # Check if API keys are not empty
    if not settings.GROQ_API_KEY.strip():
        raise ValueError("GROQ_API_KEY cannot be empty")
    
    if not settings.PINECONE_API_KEY.strip():
        raise ValueError("PINECONE_API_KEY cannot be empty")
    
    # Check if file size is reasonable
    if settings.MAX_FILE_SIZE_MB <= 0:
        raise ValueError("MAX_FILE_SIZE_MB must be greater than 0")
    
    print("✓ Settings validation passed")


# ========== Initialization ==========
# Run these functions when the module is imported
if __name__ != "__main__":
    # Create upload directory on import
    create_upload_directory()
    # Validate settings on import
    validate_settings()