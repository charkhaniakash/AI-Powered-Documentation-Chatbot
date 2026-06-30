"""
Embedding service for generating vector representations of text.
Uses Groq API to create embeddings for semantic search.
"""

# Import necessary libraries
import logging  # For logging
import os  # For environment variables
from typing import List, Dict, Any, Optional  # For type hints
import time  # For retry delays
import json  # For JSON operations

# Import Groq client
from groq import Groq  # Groq API client

# Internal imports
from app.config.settings import settings  # Application settings
from app.models.schemas import TextChunk, EmbeddedChunk  # Data models

# ========== Setup Logging ==========
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


# ========== Embedding Generator Class ==========
class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using Groq API.
    
    What are embeddings?
    - Vector representations of text (lists of numbers)
    - Capture semantic meaning
    - Similar texts have similar embeddings
    - Enable semantic search (find by meaning, not just keywords)
    
    Why embeddings are crucial for RAG:
    1. Convert text to numbers for mathematical comparison
    2. Enable finding relevant context based on meaning
    3. Support vector similarity search in databases
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            api_key: Groq API key (uses settings default if None)
        """
        # Use provided API key or get from settings
        self.api_key = api_key or settings.GROQ_API_KEY
        
        # Initialize Groq client
        # The client handles authentication and API communication
        self.client = Groq(api_key=self.api_key)
        
        # Model configuration
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        
        # Rate limiting configuration
        # Prevents hitting API rate limits
        self.max_retries = 3  # Number of retry attempts
        self.retry_delay = 1.0  # Initial delay in seconds
        self.backoff_factor = 2.0  # Multiply delay by this each retry
        
        # Batch processing configuration
        # Process multiple chunks efficiently
        self.batch_size = 10  # Number of chunks to process at once
        
        logger.info(
            f"EmbeddingGenerator initialized with model: {self.model_name}"
        )
    
    
    def embed_chunks(
        self,
        chunks: List[TextChunk],
        show_progress: bool = True
    ) -> List[EmbeddedChunk]:
        """
        Generate embeddings for a list of text chunks.
        
        This is the main entry point for embedding generation.
        Processes chunks in batches for efficiency.
        
        Args:
            chunks: List of TextChunk objects to embed
            show_progress: Whether to log progress
            
        Returns:
            List of EmbeddedChunk objects with embeddings
            
        Raises:
            ValueError: If chunks list is empty
            RuntimeError: If embedding generation fails
        """
        # Validate input
        if not chunks:
            raise ValueError("Cannot embed empty chunks list")
        
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        embedded_chunks = []
        total_chunks = len(chunks)
        
        # Process chunks in batches to avoid overwhelming the API
        # and to implement efficient rate limiting
        for i in range(0, total_chunks, self.batch_size):
            # Get current batch
            batch = chunks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
            
            if show_progress:
                logger.info(
                    f"Processing batch {batch_num}/{total_batches} "
                    f"({len(batch)} chunks)"
                )
            
            # Process batch with retry logic
            try:
                batch_embedded = self._embed_batch(batch)
                embedded_chunks.extend(batch_embedded)
            except Exception as e:
                logger.error(f"Failed to embed batch {batch_num}: {str(e)}")
                raise RuntimeError(f"Embedding failed at batch {batch_num}: {str(e)}")
            
            # Small delay between batches to respect rate limits
            if i + self.batch_size < total_chunks:
                time.sleep(0.1)  # 100ms delay
        
        logger.info(f"Successfully embedded {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    
    def _embed_batch(self, chunks: List[TextChunk]) -> List[EmbeddedChunk]:
        """
        Embed a batch of chunks with retry logic.
        
        Implements exponential backoff for handling rate limits
        and transient errors.
        
        Args:
            chunks: Batch of chunks to embed
            
        Returns:
            List of EmbeddedChunk objects
            
        Raises:
            RuntimeError: If all retries fail
        """
        # Try embedding with retries
        for attempt in range(self.max_retries):
            try:
                # Attempt to generate embeddings
                return self._generate_embeddings(chunks)
                
            except Exception as e:
                # Check if this is the last attempt
                if attempt == self.max_retries - 1:
                    # No more retries, raise error
                    logger.error(
                        f"Failed after {self.max_retries} attempts: {str(e)}"
                    )
                    raise RuntimeError(f"Embedding failed: {str(e)}")
                
                # Calculate delay with exponential backoff
                # First retry: 1s, second: 2s, third: 4s, etc.
                delay = self.retry_delay * (self.backoff_factor ** attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. "
                    f"Retrying in {delay}s..."
                )
                
                # Wait before retrying
                time.sleep(delay)
        
        # Should never reach here due to raise in loop
        raise RuntimeError("Unexpected embedding failure")
    
    
    def _generate_embeddings(
        self,
        chunks: List[TextChunk]
    ) -> List[EmbeddedChunk]:
        """
        Generate embeddings using Groq API.
        
        Note: Groq doesn't have a dedicated embedding API endpoint yet.
        This implementation uses a workaround - we'll use the chat completion
        API with a special prompt to extract semantic representations,
        or use a placeholder approach until Groq releases embedding endpoints.
        
        For production, you might want to use:
        - OpenAI's text-embedding-ada-002
        - Sentence Transformers (local)
        - Cohere embeddings
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of EmbeddedChunk objects
        """
        embedded_chunks = []
        
        # IMPORTANT NOTE:
        # Groq currently doesn't provide embedding endpoints.
        # Here are options:
        
        # Option 1: Use Sentence Transformers (local, free)
        # Option 2: Use OpenAI embeddings (paid, high quality)
        # Option 3: Use Cohere embeddings (paid)
        # Option 4: Simulate with chat completion (not recommended)
        
        # For this demo, I'll show Option 1 (Sentence Transformers)
        # which is the most practical for production
        
        try:
            # Import sentence transformers
            from sentence_transformers import SentenceTransformer
            
            # Initialize model (happens once, cached)
            if not hasattr(self, '_transformer_model'):
                logger.info("Loading Sentence Transformer model...")
                # all-MiniLM-L6-v2: Fast, good quality, 384 dimensions
                # You can use other models for different dimensions
                self._transformer_model = SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2'
                )
                logger.info("Model loaded successfully")
            
            # Extract texts from chunks
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings for all texts at once
            # encode() returns numpy array of shape (num_texts, embedding_dim)
            embeddings = self._transformer_model.encode(
                texts,
                show_progress_bar=False,  # Disable progress bar for cleaner logs
                convert_to_numpy=True      # Return as numpy array
            )
            
            # Create EmbeddedChunk objects
            for chunk, embedding in zip(chunks, embeddings):
                # Convert numpy array to list for JSON serialization
                embedding_list = embedding.tolist()
                
                # Create EmbeddedChunk by copying chunk data and adding embedding
                embedded_chunk = EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    chunk_index=chunk.chunk_index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    embedding=embedding_list
                )
                embedded_chunks.append(embedded_chunk)
            
            return embedded_chunks
            
        except ImportError:
            # Sentence Transformers not installed
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise RuntimeError(
                "Embedding model not available. Install sentence-transformers."
            )
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Queries need to be embedded the same way as documents
        for accurate similarity comparison.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ValueError: If query is empty
            RuntimeError: If embedding fails
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Cannot embed empty query")
        
        logger.debug(f"Embedding query: {query[:50]}...")
        
        try:
            # Import sentence transformers
            from sentence_transformers import SentenceTransformer
            
            # Load model if not already loaded
            if not hasattr(self, '_transformer_model'):
                logger.info("Loading Sentence Transformer model...")
                self._transformer_model = SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2'
                )
            
            # Generate embedding for query
            # encode() with single string returns 1D array
            embedding = self._transformer_model.encode(
                query,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            logger.debug(
                f"Generated query embedding with {len(embedding_list)} dimensions"
            )
            
            return embedding_list
            
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise RuntimeError("Embedding model not available")
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise RuntimeError(f"Query embedding failed: {str(e)}")
    
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this generator.
        
        Important for:
        - Vector database configuration
        - Validation
        - Model compatibility checks
        
        Returns:
            Embedding dimension
        """
        # For all-MiniLM-L6-v2, dimension is 384
        # For all-mpnet-base-v2, dimension is 768
        # Adjust based on your model choice
        
        # If model is loaded, get dimension from it
        if hasattr(self, '_transformer_model'):
            return self._transformer_model.get_sentence_embedding_dimension()
        
        # Otherwise, return configured dimension
        # Note: Update settings.EMBEDDING_DIMENSION to match your model!
        return 384  # Default for all-MiniLM-L6-v2
    
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Cosine similarity measures the angle between vectors:
        - 1.0: Identical direction (most similar)
        - 0.0: Orthogonal (no similarity)
        - -1.0: Opposite direction (most dissimilar)
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
            
        Raises:
            ValueError: If embeddings have different dimensions
        """
        # Validate dimensions match
        if len(embedding1) != len(embedding2):
            raise ValueError(
                f"Embedding dimensions don't match: "
                f"{len(embedding1)} vs {len(embedding2)}"
            )
        
        # Compute dot product
        # Sum of element-wise multiplication
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Compute magnitudes (L2 norms)
        # Square root of sum of squares
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Cosine similarity formula
        # cos(θ) = (A · B) / (||A|| × ||B||)
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return similarity
    
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate an embedding vector.
        
        Checks:
        1. Correct dimension
        2. All values are finite numbers
        3. Not all zeros (degenerate embedding)
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check dimension
        expected_dim = self.get_embedding_dimension()
        if len(embedding) != expected_dim:
            logger.warning(
                f"Invalid dimension: expected {expected_dim}, got {len(embedding)}"
            )
            return False
        
        # Check for finite values
        # Ensures no NaN or Inf values
        if not all(isinstance(x, (int, float)) and 
                   -1e10 < x < 1e10 for x in embedding):
            logger.warning("Embedding contains invalid values")
            return False
        
        # Check not all zeros
        # A zero vector is meaningless
        if all(x == 0 for x in embedding):
            logger.warning("Embedding is all zeros")
            return False
        
        return True


# ========== Alternative Implementation Using OpenAI ==========
class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """
    Alternative embedding generator using OpenAI's API.
    
    More expensive but potentially higher quality.
    Use this if you have OpenAI API access.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key."""
        import openai
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        openai.api_key = self.api_key
        self.client = openai
        self.model_name = "text-embedding-ada-002"
        self.embedding_dimension = 1536  # Ada-002 dimension
        
        # Batch configuration
        self.batch_size = 100  # OpenAI can handle larger batches
        
        logger.info("OpenAIEmbeddingGenerator initialized")
    
    
    def _generate_embeddings(
        self,
        chunks: List[TextChunk]
    ) -> List[EmbeddedChunk]:
        """Generate embeddings using OpenAI API."""
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Call OpenAI API
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        
        # Extract embeddings from response
        embedded_chunks = []
        for chunk, embedding_obj in zip(chunks, response.data):
            embedding = embedding_obj.embedding
            
            embedded_chunk = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                chunk_index=chunk.chunk_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                embedding=embedding
            )
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    
    def embed_query(self, query: str) -> List[float]:
        """Embed query using OpenAI."""
        response = self.client.embeddings.create(
            input=[query],
            model=self.model_name
        )
        return response.data[0].embedding


# ========== Utility Functions ==========
def get_embedding_generator(
    provider: str = "sentence_transformers"
) -> EmbeddingGenerator:
    """
    Factory function to get embedding generator.
    
    Args:
        provider: 'sentence_transformers' or 'openai'
        
    Returns:
        EmbeddingGenerator instance
    """
    if provider == "openai":
        return OpenAIEmbeddingGenerator()
    else:
        return EmbeddingGenerator()


def compute_embeddings_stats(embeddings: List[List[float]]) -> Dict[str, Any]:
    """
    Compute statistics about a set of embeddings.
    
    Useful for debugging and quality checks.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Dictionary with statistics
    """
    import numpy as np
    
    # Convert to numpy array for easier computation
    emb_array = np.array(embeddings)
    
    stats = {
        "count": len(embeddings),
        "dimension": len(embeddings[0]) if embeddings else 0,
        "mean_magnitude": float(np.mean(np.linalg.norm(emb_array, axis=1))),
        "std_magnitude": float(np.std(np.linalg.norm(emb_array, axis=1))),
        "min_value": float(np.min(emb_array)),
        "max_value": float(np.max(emb_array)),
        "mean_value": float(np.mean(emb_array)),
    }
    
    return stats