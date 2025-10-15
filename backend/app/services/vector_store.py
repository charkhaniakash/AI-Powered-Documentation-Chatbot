"""
Vector store service for storing and retrieving embeddings using Pinecone.
Provides semantic search capabilities for the RAG system.
"""

# Import necessary libraries
import logging  # For logging
from typing import List, Dict, Any, Optional, Tuple  # For type hints
import time  # For delays
import json  # For JSON operations

# Import Pinecone
from pinecone import Pinecone, ServerlessSpec  # Pinecone client and config

# Internal imports
from app.config.settings import settings  # Application settings
from app.models.schemas import EmbeddedChunk, RetrievedChunk, TextChunk  # Data models

# ========== Setup Logging ==========
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


# ========== Vector Store Class ==========
class VectorStore:
    """
    Manages vector storage and retrieval using Pinecone.
    
    What is a vector database?
    - Specialized database for storing embeddings (vectors)
    - Optimized for similarity search
    - Finds "nearest neighbors" in vector space
    - Much faster than comparing all vectors manually
    
    Why Pinecone?
    - Fully managed (no infrastructure to maintain)
    - Scales automatically
    - Fast similarity search
    - Good free tier for testing
    
    Core operations:
    1. Upsert: Store vectors with metadata
    2. Query: Find similar vectors
    3. Delete: Remove vectors
    4. Update: Modify vectors or metadata
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            api_key: Pinecone API key (uses settings if None)
            environment: Pinecone environment (uses settings if None)
            index_name: Name of index to use (uses settings if None)
            dimension: Embedding dimension (uses settings if None)
        """
        # Use provided values or fall back to settings
        self.api_key = api_key or settings.PINECONE_API_KEY
        self.environment = environment or settings.PINECONE_ENVIRONMENT
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        
        # Initialize Pinecone client
        # The client manages connection to Pinecone's API
        self.pc = Pinecone(api_key=self.api_key)
        
        # Index object (will be set in _get_or_create_index)
        self.index = None
        
        # Initialize index
        self._get_or_create_index()
        
        logger.info(
            f"VectorStore initialized: index={self.index_name}, "
            f"dimension={self.dimension}"
        )
    
    
    def _get_or_create_index(self) -> None:
        """
        Get existing index or create new one.
        
        This ensures the index exists before we try to use it.
        Creates index with appropriate configuration if it doesn't exist.
        
        Raises:
            RuntimeError: If index creation fails
        """
        try:
            # List all existing indexes
            # list_indexes() returns list of index names
            existing_indexes = self.pc.list_indexes()
            
            # Check if our index exists
            if self.index_name not in [idx.name for idx in existing_indexes]:
                logger.info(f"Creating new index: {self.index_name}")
                
                # Create index with serverless configuration
                # Serverless is cost-effective for variable workloads
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,  # Must match embedding dimension
                    metric="cosine",  # Similarity metric (cosine, euclidean, or dotproduct)
                    spec=ServerlessSpec(
                        cloud="aws",  # Cloud provider
                        region=self.environment  # Region/environment
                    )
                )
                
                # Wait for index to be ready
                # Index creation is asynchronous
                logger.info("Waiting for index to be ready...")
                time.sleep(10)  # Give it 10 seconds to initialize
                
                logger.info(f"Index {self.index_name} created successfully")
            else:
                logger.info(f"Using existing index: {self.index_name}")
            
            # Connect to the index
            # This returns an Index object for operations
            self.index = self.pc.Index(self.index_name)
            
            # Verify index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
        except Exception as e:
            logger.error(f"Failed to initialize index: {str(e)}")
            raise RuntimeError(f"Index initialization failed: {str(e)}")
    
    
    def upsert_chunks(
        self,
        chunks: List[EmbeddedChunk],
        namespace: str = "",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Store embedded chunks in vector database.
        
        "Upsert" = Update or Insert
        - If vector ID exists, updates it
        - If vector ID doesn't exist, inserts it
        
        Args:
            chunks: List of embedded chunks to store
            namespace: Namespace for organizing vectors (optional)
            batch_size: Number of vectors to upsert per batch
            
        Returns:
            Dictionary with upsert results
            
        Raises:
            ValueError: If chunks list is empty
            RuntimeError: If upsert fails
        """
        # Validate input
        if not chunks:
            raise ValueError("Cannot upsert empty chunks list")
        
        logger.info(f"Upserting {len(chunks)} chunks to Pinecone...")
        
        # Prepare vectors for upsert
        # Pinecone expects tuples of (id, values, metadata)
        vectors = []
        for chunk in chunks:
            # Create vector tuple
            vector = (
                chunk.chunk_id,  # Unique ID
                chunk.embedding,  # Vector values
                self._prepare_metadata(chunk)  # Metadata dict
            )
            vectors.append(vector)
        
        # Upsert in batches for efficiency
        total_upserted = 0
        failed_batches = []
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            try:
                # Upsert batch to Pinecone
                # upsert() returns response with upserted_count
                response = self.index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
                
                # Track success
                upserted_count = response.get('upserted_count', len(batch))
                total_upserted += upserted_count
                
                logger.info(
                    f"Batch {batch_num}/{total_batches}: "
                    f"upserted {upserted_count} vectors"
                )
                
            except Exception as e:
                logger.error(f"Failed to upsert batch {batch_num}: {str(e)}")
                failed_batches.append(batch_num)
        
        # Check if any batches failed
        if failed_batches:
            logger.warning(
                f"Some batches failed: {failed_batches}. "
                f"Successfully upserted {total_upserted}/{len(vectors)} vectors."
            )
        else:
            logger.info(f"Successfully upserted all {total_upserted} vectors")
        
        # Return results
        return {
            "total_chunks": len(chunks),
            "upserted_count": total_upserted,
            "failed_batches": failed_batches,
            "success": len(failed_batches) == 0
        }
    
    
    def _prepare_metadata(self, chunk: EmbeddedChunk) -> Dict[str, Any]:
        """
        Prepare metadata for storage in Pinecone.
        
        Pinecone metadata has restrictions:
        - Only specific types allowed (string, number, boolean, list)
        - Nested objects must be flattened
        - Max size limits
        
        Args:
            chunk: Embedded chunk with metadata
            
        Returns:
            Flattened metadata dictionary
        """
        # Extract document metadata
        doc_meta = chunk.metadata
        
        # Handle file_type - could be enum or string
        if hasattr(doc_meta.file_type, 'value'):
            file_type_str = doc_meta.file_type.value
        else:
            file_type_str = str(doc_meta.file_type)
        
        # Create flattened metadata
        metadata = {
            # Chunk information
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            
            # Document information (flattened)
            "filename": doc_meta.filename,
            "file_type": file_type_str,  # Use the safely converted string
            "file_size": doc_meta.file_size,
            "document_id": doc_meta.document_id,
            "uploaded_at": doc_meta.uploaded_at.isoformat(),
        }
        
        # Add optional fields if present
        if doc_meta.page_count is not None:
            metadata["page_count"] = doc_meta.page_count
        
        if doc_meta.uploaded_by is not None:
            metadata["uploaded_by"] = doc_meta.uploaded_by
        
        # Add custom metadata if present
        if doc_meta.custom_metadata:
            for key, value in doc_meta.custom_metadata.items():
                metadata[f"custom_{key}"] = value
        
        return metadata
    
    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = None,
        namespace: str = "",
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[RetrievedChunk]:
        """
        Query for similar vectors.
        
        This is the core retrieval operation for RAG.
        Finds chunks most similar to the query embedding.
        
        Args:
            query_embedding: Query vector to search for
            top_k: Number of results to return (uses settings if None)
            namespace: Namespace to search in
            filter_dict: Metadata filters (optional)
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of RetrievedChunk objects, sorted by relevance
            
        Raises:
            ValueError: If query_embedding is invalid
            RuntimeError: If query fails
        """
        # Use configured top_k if not provided
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        logger.info(f"Querying Pinecone for top {top_k} similar vectors...")
        
        try:
            # Query Pinecone
            # query() performs similarity search
            results = self.index.query(
                vector=query_embedding,  # Query vector
                top_k=top_k,  # Number of results
                namespace=namespace,  # Namespace to search
                filter=filter_dict,  # Metadata filters
                include_metadata=include_metadata  # Include metadata in response
            )
            
            # Process results into RetrievedChunk objects
            retrieved_chunks = []
            
            # results.matches is a list of Match objects
            for rank, match in enumerate(results.matches, start=1):
                # Extract match data
                score = match.score  # Similarity score (0-1 for cosine)
                chunk_id = match.id  # Vector ID (same as chunk_id)
                metadata = match.metadata if include_metadata else {}
                
                # Filter by similarity threshold
                # Only return chunks above minimum similarity
                if score < settings.SIMILARITY_THRESHOLD:
                    logger.debug(
                        f"Skipping chunk {chunk_id} with score {score:.4f} "
                        f"(below threshold {settings.SIMILARITY_THRESHOLD})"
                    )
                    continue
                
                # Reconstruct TextChunk from metadata
                # We stored the full text and metadata in Pinecone
                text_chunk = self._metadata_to_chunk(chunk_id, metadata)
                
                # Create RetrievedChunk with score and rank
                retrieved_chunk = RetrievedChunk(
                    chunk=text_chunk,
                    score=score,
                    rank=rank
                )
                retrieved_chunks.append(retrieved_chunk)
            
            logger.info(
                f"Retrieved {len(retrieved_chunks)} chunks "
                f"(above threshold {settings.SIMILARITY_THRESHOLD})"
            )
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise RuntimeError(f"Vector query failed: {str(e)}")
    
    
    def _metadata_to_chunk(
        self,
        chunk_id: str,
        metadata: Dict[str, Any]
    ) -> TextChunk:
        """
        Reconstruct TextChunk from Pinecone metadata.
        
        Converts the flattened metadata back to structured objects.
        
        Args:
            chunk_id: Chunk ID
            metadata: Metadata dictionary from Pinecone
            
        Returns:
            TextChunk object
        """
        from datetime import datetime
        from app.models.schemas import DocumentMetadata, FileType
        
        # Reconstruct DocumentMetadata
        doc_metadata = DocumentMetadata(
            filename=metadata.get("filename", "unknown"),
            file_type=FileType(metadata.get("file_type", "txt")),
            file_size=metadata.get("file_size", 0),
            document_id=metadata.get("document_id", "unknown"),
            page_count=metadata.get("page_count"),
            uploaded_by=metadata.get("uploaded_by"),
            # Parse ISO datetime string back to datetime object
            uploaded_at=datetime.fromisoformat(
                metadata.get("uploaded_at", datetime.now().isoformat())
            )
        )
        
        # Reconstruct TextChunk
        text_chunk = TextChunk(
            chunk_id=chunk_id,
            text=metadata.get("text", ""),
            metadata=doc_metadata,
            chunk_index=metadata.get("chunk_index", 0),
            start_char=metadata.get("start_char", 0),
            end_char=metadata.get("end_char", 0)
        )
        
        return text_chunk
    
    
    def delete_by_document_id(
        self,
        document_id: str,
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Delete all chunks belonging to a document.
        
        Useful for:
        - Removing outdated documents
        - Cleaning up after errors
        - Document updates (delete then re-ingest)
        
        Args:
            document_id: ID of document to delete
            namespace: Namespace to delete from
            
        Returns:
            Dictionary with deletion results
            
        Raises:
            RuntimeError: If deletion fails
        """
        logger.info(f"Deleting all chunks for document: {document_id}")
        
        try:
            # Delete by metadata filter
            # Pinecone supports deletion by filter
            self.index.delete(
                filter={
                    "document_id": {"$eq": document_id}
                },
                namespace=namespace
            )
            
            logger.info(f"Successfully deleted chunks for document {document_id}")
            
            return {
                "document_id": document_id,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            raise RuntimeError(f"Deletion failed: {str(e)}")
    
    
    def delete_by_chunk_ids(
        self,
        chunk_ids: List[str],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Delete specific chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            namespace: Namespace to delete from
            
        Returns:
            Dictionary with deletion results
        """
        if not chunk_ids:
            raise ValueError("Cannot delete empty chunk_ids list")
        
        logger.info(f"Deleting {len(chunk_ids)} chunks...")
        
        try:
            # Delete by IDs
            # Pinecone accepts list of IDs for batch deletion
            self.index.delete(
                ids=chunk_ids,
                namespace=namespace
            )
            
            logger.info(f"Successfully deleted {len(chunk_ids)} chunks")
            
            return {
                "deleted_count": len(chunk_ids),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to delete chunks: {str(e)}")
            raise RuntimeError(f"Deletion failed: {str(e)}")
    
    
    def get_index_stats(self, namespace: str = "") -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Useful for:
        - Monitoring storage usage
        - Debugging
        - Understanding data distribution
        
        Args:
            namespace: Namespace to get stats for (empty = all namespaces)
            
        Returns:
            Dictionary with index statistics
        """
        try:
            # Get index statistics
            # describe_index_stats() returns count, dimensions, etc.
            stats = self.index.describe_index_stats()
            
            # Extract relevant information
            result = {
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", self.dimension),
                "index_fullness": stats.get("index_fullness", 0.0),
            }
            
            # Add namespace-specific stats if available
            if "namespaces" in stats:
                result["namespaces"] = stats["namespaces"]
            
            logger.info(f"Index stats: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    
    def fetch_chunks(
        self,
        chunk_ids: List[str],
        namespace: str = ""
    ) -> List[TextChunk]:
        """
        Fetch specific chunks by their IDs.
        
        Different from query - this retrieves by exact ID,
        not by similarity.
        
        Args:
            chunk_ids: List of chunk IDs to fetch
            namespace: Namespace to fetch from
            
        Returns:
            List of TextChunk objects
        """
        if not chunk_ids:
            raise ValueError("Cannot fetch empty chunk_ids list")
        
        logger.info(f"Fetching {len(chunk_ids)} chunks...")
        
        try:
            # Fetch vectors by ID
            # fetch() returns vectors with metadata
            results = self.index.fetch(
                ids=chunk_ids,
                namespace=namespace
            )
            
            # Process results
            chunks = []
            for chunk_id, vector_data in results.vectors.items():
                # Extract metadata
                metadata = vector_data.metadata
                
                # Reconstruct chunk
                chunk = self._metadata_to_chunk(chunk_id, metadata)
                chunks.append(chunk)
            
            logger.info(f"Successfully fetched {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to fetch chunks: {str(e)}")
            raise RuntimeError(f"Fetch failed: {str(e)}")
    
    
    def update_metadata(
        self,
        chunk_id: str,
        metadata: Dict[str, Any],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Update metadata for a specific chunk.
        
        Note: You cannot update the vector itself, only metadata.
        To update the vector, you must upsert again.
        
        Args:
            chunk_id: ID of chunk to update
            metadata: New metadata values
            namespace: Namespace
            
        Returns:
            Dictionary with update results
        """
        logger.info(f"Updating metadata for chunk: {chunk_id}")
        
        try:
            # Update metadata
            # update() modifies metadata without changing the vector
            self.index.update(
                id=chunk_id,
                set_metadata=metadata,
                namespace=namespace
            )
            
            logger.info(f"Successfully updated metadata for {chunk_id}")
            
            return {
                "chunk_id": chunk_id,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to update metadata: {str(e)}")
            raise RuntimeError(f"Update failed: {str(e)}")
    
    
    def clear_namespace(self, namespace: str) -> Dict[str, Any]:
        """
        Delete all vectors in a namespace.
        
        Warning: This is destructive and cannot be undone!
        
        Args:
            namespace: Namespace to clear
            
        Returns:
            Dictionary with deletion results
        """
        logger.warning(f"Clearing entire namespace: {namespace}")
        
        try:
            # Delete all vectors in namespace
            # delete_all=True removes everything
            self.index.delete(
                delete_all=True,
                namespace=namespace
            )
            
            logger.info(f"Successfully cleared namespace: {namespace}")
            
            return {
                "namespace": namespace,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to clear namespace: {str(e)}")
            raise RuntimeError(f"Clear failed: {str(e)}")
    
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        metadata_filter: Dict[str, Any],
        top_k: int = None,
        namespace: str = ""
    ) -> List[RetrievedChunk]:
        """
        Perform hybrid search: vector similarity + metadata filtering.
        
        Combines:
        1. Semantic similarity (vector search)
        2. Structured filtering (metadata)
        
        Example filters:
        - {"document_id": {"$eq": "doc123"}}
        - {"file_type": {"$in": ["pdf", "md"]}}
        - {"uploaded_at": {"$gte": "2024-01-01"}}
        
        Args:
            query_embedding: Query vector
            metadata_filter: Metadata filter dictionary
            top_k: Number of results
            namespace: Namespace to search
            
        Returns:
            List of RetrievedChunk objects
        """
        logger.info(f"Performing hybrid search with filter: {metadata_filter}")
        
        # Use query_similar with filter
        return self.query_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter_dict=metadata_filter
        )
    
    
    def check_connection(self) -> bool:
        """
        Check if connection to Pinecone is working.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            # Try to get index stats as a health check
            self.index.describe_index_stats()
            logger.info("Pinecone connection OK")
            return True
        except Exception as e:
            logger.error(f"Pinecone connection failed: {str(e)}")
            return False


# ========== Alternative: ChromaDB Implementation ==========
class ChromaVectorStore:
    """
    Alternative vector store using ChromaDB (local, open-source).
    
    Advantages over Pinecone:
    - Free and open-source
    - Runs locally (no API costs)
    - Good for development and testing
    - Supports metadata filtering
    
    Disadvantages:
    - Not managed (you handle scaling)
    - Slower for large datasets
    - Less enterprise features
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of collection (like Pinecone index)
            persist_directory: Directory to store data
        """
        import chromadb
        from chromadb.config import Settings
        
        # Use settings or default
        self.collection_name = collection_name or settings.PINECONE_INDEX_NAME
        
        # Initialize ChromaDB client
        # PersistentClient saves data to disk
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        # Collections in Chroma are like indexes in Pinecone
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"ChromaVectorStore initialized: {self.collection_name}")
    
    
    def upsert_chunks(
        self,
        chunks: List[EmbeddedChunk],
        **kwargs
    ) -> Dict[str, Any]:
        """Upsert chunks to ChromaDB."""
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]
        
        # Upsert to collection
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return {
            "total_chunks": len(chunks),
            "upserted_count": len(chunks),
            "success": True
        }
    
    
    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = None,
        **kwargs
    ) -> List[RetrievedChunk]:
        """Query ChromaDB for similar vectors."""
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Process results
        retrieved_chunks = []
        for rank, (chunk_id, score, metadata, document) in enumerate(
            zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0],
                results['documents'][0]
            ),
            start=1
        ):
            # Convert distance to similarity (Chroma returns distance)
            similarity = 1 - score
            
            # Skip if below threshold
            if similarity < settings.SIMILARITY_THRESHOLD:
                continue
            
            # Reconstruct chunk
            text_chunk = self._metadata_to_chunk(chunk_id, metadata, document)
            
            retrieved_chunk = RetrievedChunk(
                chunk=text_chunk,
                score=similarity,
                rank=rank
            )
            retrieved_chunks.append(retrieved_chunk)
        
        return retrieved_chunks
    
    
    def _prepare_metadata(self, chunk: EmbeddedChunk) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB."""
        # Similar to Pinecone but Chroma has different restrictions
        doc_meta = chunk.metadata
        return {
            "chunk_index": chunk.chunk_index,
            "document_id": doc_meta.document_id,
            "filename": doc_meta.filename,
            "file_type": doc_meta.file_type.value,
        }
    
    
    def _metadata_to_chunk(
        self,
        chunk_id: str,
        metadata: Dict[str, Any],
        text: str
    ) -> TextChunk:
        """Reconstruct TextChunk from ChromaDB data."""
        from datetime import datetime
        from app.models.schemas import DocumentMetadata, FileType
        
        doc_metadata = DocumentMetadata(
            filename=metadata.get("filename", "unknown"),
            file_type=FileType(metadata.get("file_type", "txt")),
            file_size=0,
            document_id=metadata.get("document_id", "unknown"),
            uploaded_at=datetime.now()
        )
        
        text_chunk = TextChunk(
            chunk_id=chunk_id,
            text=text,
            metadata=doc_metadata,
            chunk_index=metadata.get("chunk_index", 0),
            start_char=0,
            end_char=len(text)
        )
        
        return text_chunk


# ========== Utility Functions ==========
def get_vector_store(provider: str = "pinecone") -> VectorStore:
    """
    Factory function to get vector store.
    
    Args:
        provider: 'pinecone' or 'chroma'
        
    Returns:
        VectorStore instance
    """
    if provider == "chroma":
        return ChromaVectorStore()
    else:
        return VectorStore()