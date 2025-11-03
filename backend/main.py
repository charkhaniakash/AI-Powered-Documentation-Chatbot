"""
Main pipeline orchestrator for the Documentation Chatbot.
Coordinates all services to implement the complete RAG system with Hybrid Search.
"""

# Import necessary libraries
import logging
from typing import List, Dict, Any, Optional
import os
import sys
import json

# Import all service modules
from app.config.settings import settings, validate_settings, create_upload_directory
from app.services.file_processor import FileProcessor, get_file_processor
from app.services.chunking import TextChunker, get_text_chunker
from app.services.embedding import EmbeddingGenerator, get_embedding_generator
from app.services.vector_store import VectorStore, get_vector_store
from app.services.llm import LLMService, get_llm_service
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    ProcessingJob,
    ProcessingStatus,
    UploadResponse,
    FileType
)
from app.utils.helpers import (
    timing_decorator,
    calculate_file_hash,
    ProgressTracker,
    setup_logging
)
from app.services.duplicate_detector import get_duplicate_detector
from app.services.cache_manager import get_cache_manager

# ========== Setup Logging ==========
setup_logging(log_level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


# ========== Main RAG Pipeline Class ==========
class DocumentChatbot:
    """
    Complete RAG pipeline for document Q&A with Hybrid Search.
    
    This class orchestrates all components:
    1. File Processing
    2. Text Chunking
    3. Embedding Generation
    4. Vector Storage (Pinecone)
    5. BM25 Indexing (for hybrid search)
    6. Query Processing (Vector + BM25 + RRF)
    7. Response Generation (LLM)
    
    Usage:
        chatbot = DocumentChatbot()
        chatbot.ingest_document("document.pdf")
        response = chatbot.query("What is this document about?")
    """
    
    def __init__(
        self,
        embedding_provider: str = "sentence_transformers",
        vector_store_provider: str = "pinecone"
    ):
        """
        Initialize the chatbot with all necessary components.
        
        Args:
            embedding_provider: 'sentence_transformers' or 'openai'
            vector_store_provider: 'pinecone' or 'chroma'
        """
        logger.info("=" * 80)
        logger.info("Initializing DocumentChatbot with Hybrid Search...")
        logger.info("=" * 80)
        
        # Validate settings
        validate_settings()
        create_upload_directory()
        
        # Initialize all services
        logger.info("Loading services...")
        self.file_processor = get_file_processor()
        logger.info("✓ File processor ready")
        
        self.text_chunker = get_text_chunker()
        logger.info("✓ Text chunker ready")
        
        self.embedding_generator = get_embedding_generator(embedding_provider)
        logger.info("✓ Embedding generator ready")
        
        self.vector_store = get_vector_store(vector_store_provider)
        logger.info("✓ Vector store ready")
        
        self.llm_service = get_llm_service()
        logger.info("✓ LLM service ready")

        # Initialize cache manager
        if settings.ENABLE_CACHING:
            self.cache_manager = get_cache_manager()
            logger.info("✓ Cache manager ready")
        else:
            self.cache_manager = None
            logger.info("○ Caching disabled")
        
        # Log hybrid search status
        if settings.ENABLE_HYBRID_SEARCH:
            fusion_method = "RRF" if settings.USE_RRF else f"Weighted ({settings.VECTOR_WEIGHT}/{settings.BM25_WEIGHT})"
            logger.info(f"✓ Hybrid Search ENABLED (Fusion: {fusion_method})")
        else:
            logger.info("⚠ Hybrid Search DISABLED (Vector-only mode)")
        
        # Storage for tracking processing jobs
        self.processing_jobs: Dict[str, ProcessingJob] = {}
        
        logger.info("=" * 80)
        logger.info("DocumentChatbot initialized successfully")
        logger.info("=" * 80)
    
    
    @timing_decorator
    def ingest_document(
        self,
        file_path: str,
        filename: Optional[str] = None,
        chunking_strategy: str = "recursive"
    ) -> UploadResponse:
        """
        Ingest a document into the system.
        
        Complete ingestion pipeline:
        1. Process file (extract text)
        2. Chunk text
        3. Generate embeddings
        4. Store in vector database (Pinecone)
        5. Index in BM25 (if hybrid search enabled)
        
        Args:
            file_path: Path to document file
            filename: Original filename (uses file_path if None)
            chunking_strategy: Strategy for chunking ('simple', 'sentence', 'recursive')
            
        Returns:
            UploadResponse with job details
            
        Raises:
            ValueError: If file is invalid
            RuntimeError: If ingestion fails
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"INGESTING DOCUMENT: {file_path}")
        logger.info("=" * 80)
        
        # Use filename from path if not provided
        if filename is None:
            filename = os.path.basename(file_path)

        
        # ========== DUPLICATE DETECTION ==========

        duplicate_detector = get_duplicate_detector()

        # Read file content for duplicate check
        with open(file_path, 'rb') as f:
            file_content = f.read()

        # Check if document is duplicate BEFORE processing
        is_duplicate, existing_doc_id, detection_method = duplicate_detector.check_duplicate_document(
            file_path=file_path,
            text_content=""  # We'll check after extraction
        )

        # If exact file duplicate found, skip processing
        if is_duplicate and detection_method == "file_hash":
            logger.warning(
                f"⚠️  DUPLICATE FILE DETECTED!\n"
                f"   File: {filename}\n"
                f"   Existing Document ID: {existing_doc_id}\n"
                f"   Action: Skipping upload (returning existing document)"
            )
            
            # Get existing metadata
            existing_metadata = duplicate_detector.get_document_metadata(existing_doc_id)
            
            return UploadResponse(
                job_id=existing_doc_id,
                document_id=existing_doc_id,
                message=f"Document '{filename}' already exists (duplicate file detected)",
                status=ProcessingStatus.COMPLETED,
                metadata=existing_metadata
            )
        
        try:
            # Step 1: Validate and process file
            logger.info("📄 Step 1/4: Processing file...")
            file_type = self.file_processor.validate_file(file_path, filename)
            text, metadata = self.file_processor.process_file(
                file_path=file_path,
                filename=filename,
                file_type=file_type
            )
            logger.info(f"   ✓ Extracted {len(text):,} characters from {filename}")

            # ========== CHECK CONTENT DUPLICATE (after text extraction) ==========
            is_duplicate, existing_doc_id, detection_method = duplicate_detector.check_duplicate_document(
                file_path=file_path,
                text_content=text
            )

            if is_duplicate:
                logger.warning(
                    f"⚠️  DUPLICATE CONTENT DETECTED!\n"
                    f"   File: {filename}\n"
                    f"   Existing Document ID: {existing_doc_id}\n"
                    f"   Detection Method: {detection_method}\n"
                    f"   Action: Skipping upload (returning existing document)"
                )
                
                existing_metadata = duplicate_detector.get_document_metadata(existing_doc_id)
                
                return UploadResponse(
                    job_id=existing_doc_id,
                    document_id=existing_doc_id,
                    message=f"Document '{filename}' already exists (duplicate content detected)",
                    status=ProcessingStatus.COMPLETED,
                    metadata=existing_metadata
                )


            
            # Step 2: Chunk the text
            logger.info(f"✂️  Step 2/4: Chunking text (strategy: {chunking_strategy})...")
            chunks = self.text_chunker.chunk_text(
                text=text,
                metadata=metadata,
                strategy=chunking_strategy
            )
            logger.info(f"   ✓ Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            logger.info("🧮 Step 3/4: Generating embeddings...")
            embedded_chunks = self.embedding_generator.embed_chunks(
                chunks=chunks,
                show_progress=True
            )
            logger.info(f"   ✓ Generated {len(embedded_chunks)} embeddings")
            
            # Step 4: Store in vector database (+ BM25 if enabled)
            logger.info("💾 Step 4/4: Storing in vector database...")
            upsert_result = self.vector_store.upsert_chunks(
                chunks=embedded_chunks,
                namespace=""
            )
            
            logger.info(f"   ✓ Stored {upsert_result['upserted_count']} vectors in Pinecone")
            
            if upsert_result.get('hybrid_search_enabled'):
                logger.info(f"   ✓ Added {len(chunks)} chunks to BM25 index")
            
            # Create upload response
            upload_response = UploadResponse(
                job_id=metadata.document_id,
                document_id=metadata.document_id,
                message=f"Document '{filename}' ingested successfully",
                status=ProcessingStatus.COMPLETED
            )
            
            logger.info("=" * 80)
            logger.info(f"✅ INGESTION COMPLETE: {metadata.document_id}")
            logger.info("=" * 80 + "\n")


            # ========== REGISTER NEW DOCUMENT ==========
            duplicate_detector.register_new_document(
                file_path=file_path,
                text_content=text,
                document_id=metadata.document_id,
                metadata={
                    "filename": filename,
                    "file_type": file_type.value,
                    "file_size": metadata.file_size,
                    "chunk_count": len(chunks)
                }
            )

            logger.info(f"✓ Document registered in duplicate detector")
            
            return upload_response
            
        except Exception as e:
            logger.error(f"❌ Document ingestion failed: {str(e)}")
            raise RuntimeError(f"Ingestion failed: {str(e)}")
    
    
    @timing_decorator
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        include_sources: bool = True,
        rephrase_query: bool = False,
        streaming: bool = False
    ) -> QueryResponse:
        """
        Query the chatbot with a question.
        
        Complete RAG query pipeline with Hybrid Search:
        1. (Optional) Rephrase query
        2. Generate query embedding
        3. Retrieve similar chunks (Vector + BM25 + Fusion)
        4. Generate response with LLM
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            document_ids: Filter by specific documents
            include_sources: Include source chunks in response
            rephrase_query: Whether to rephrase query for better retrieval
            streaming: Use streaming response (prints in real-time)
            
        Returns:
            QueryResponse with answer and sources
            
        Raises:
            ValueError: If query is empty
            RuntimeError: If query fails
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"PROCESSING QUERY: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info("=" * 80)
        
        # Use default top_k if not provided
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        try:
            # Optional: Rephrase query for better retrieval
            original_query = query
            if rephrase_query:
                logger.info("🔄 Rephrasing query for better retrieval...")
                query = self.llm_service.rephrase_query(query)
                logger.info(f"   Rephrased: {query}")
            
            # Step 1: Generate query embedding
            logger.info("🧮 Step 1/3: Generating query embedding...")
            query_embedding = self.embedding_generator.embed_query(query)
            logger.info(f"   ✓ Generated {len(query_embedding)}-dimensional embedding")
            
            # Step 2: Retrieve similar chunks (HYBRID SEARCH)
            search_mode = "Hybrid (Vector + BM25)" if settings.ENABLE_HYBRID_SEARCH else "Vector-only"
            logger.info(f"🔍 Step 2/3: Retrieving relevant chunks ({search_mode})...")
            
            # Build metadata filter if document_ids provided
            filter_dict = None
            if document_ids:
                filter_dict = {
                    "document_id": {"$in": document_ids}
                }
                logger.info(f"   Filtering by document IDs: {document_ids}")
            
            # THIS IS WHERE HYBRID SEARCH HAPPENS
            retrieved_chunks = self.vector_store.query_similar(
                query_embedding=query_embedding,
                query_text=query,  # IMPORTANT: Pass query text for BM25
                top_k=top_k,
                filter_dict=filter_dict,
                include_metadata=True
            )
            
            logger.info(f"   ✓ Retrieved {len(retrieved_chunks)} relevant chunks")
            
            if settings.ENABLE_HYBRID_SEARCH:
                fusion_method = "RRF" if settings.USE_RRF else "Weighted"
                logger.info(f"   ℹ Fusion method: {fusion_method}")
            
            # Show top results
            if retrieved_chunks:
                logger.info("   Top 3 results:")
                for i, rc in enumerate(retrieved_chunks[:3], 1):
                    logger.info(f"     [{i}] {rc.chunk.metadata.filename} (score: {rc.score:.3f})")
            
            # ========== NEW: Step 3 - Contextual Compression ==========
            if settings.ENABLE_CONTEXTUAL_COMPRESSION and retrieved_chunks:
                logger.info("🗜️  Step 3/4: Applying contextual compression...")
                
                # Import compressor (lazy import)
                from app.services.contextual_compressor import get_contextual_compressor
                
                # Get compressor instance
                compressor = get_contextual_compressor()
                
                # Store original for comparison
                original_chunks = retrieved_chunks.copy()
                
                # Compress chunks
                retrieved_chunks = compressor.compress_retrieved_chunks(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    preserve_order=True
                )
                
                # Log compression stats
                if retrieved_chunks:
                    stats = compressor.get_compression_stats(original_chunks, retrieved_chunks)
                    logger.info(
                        f"   ✓ Compression: {stats['compression_ratio_percent']:.1f}% reduction "
                        f"({stats['original_total_chars']:,} → {stats['compressed_total_chars']:,} chars)"
                    )
                else:
                    logger.warning("   ⚠️  All chunks filtered out by compression!")
                    # Fallback to original chunks
                    retrieved_chunks = original_chunks
                    logger.info("   ℹ  Using original uncompressed chunks")
            else:
                if not settings.ENABLE_CONTEXTUAL_COMPRESSION:
                    logger.info("ℹ️  Step 3/4: Contextual compression disabled (skipped)")
                else:
                    logger.info("ℹ️  Step 3/4: No chunks to compress (skipped)")
            
            # Check if we have any results
            if not retrieved_chunks:
                logger.warning("⚠️  No relevant chunks found")
                return QueryResponse(
                    answer="I couldn't find any relevant information in the documents to answer your question.",
                    sources=[],
                    confidence=0.0,
                    metadata={
                        "query": original_query,
                        "chunks_found": 0,
                        "hybrid_search": settings.ENABLE_HYBRID_SEARCH
                    }
                )
            
            # Step 3: Generate response with LLM
            logger.info("🤖 Step 3/3: Generating response with LLM...")
            
            if streaming:
                # Streaming response (prints in real-time)
                logger.info("   Streaming response:")
                print("\n" + "-" * 80)
                print("Assistant: ", end='', flush=True)
                
                full_answer = ""
                for chunk in self.llm_service.generate_streaming_response(
                    query=original_query,
                    context_chunks=retrieved_chunks,
                    conversation_history=None
                ):
                    print(chunk, end='', flush=True)
                    full_answer += chunk
                
                print("\n" + "-" * 80)
                
                # Create response object
                response = QueryResponse(
                    answer=full_answer,
                    sources=retrieved_chunks if include_sources else [],
                    confidence=self._calculate_confidence(retrieved_chunks, full_answer),
                    metadata={
                        "original_query": original_query,
                        "mode": "streaming",
                        "hybrid_search": settings.ENABLE_HYBRID_SEARCH,
                        "fusion_method": "RRF" if settings.USE_RRF else "weighted",
                        "compression_enabled": settings.ENABLE_CONTEXTUAL_COMPRESSION  # ADD THIS LINE
                    }
                )
            else:
                # Non-streaming response
                response = self.llm_service.generate_response(
                    query=original_query,
                    context_chunks=retrieved_chunks,
                    conversation_history=None
                )
                
                # Add metadata
                response.metadata["hybrid_search"] = settings.ENABLE_HYBRID_SEARCH
                response.metadata["fusion_method"] = "RRF" if settings.USE_RRF else "weighted"
                response.metadata["compression_enabled"] = settings.ENABLE_CONTEXTUAL_COMPRESSION  # ADD THIS LINE
                
                if not include_sources:
                    response.sources = []
            
            logger.info(f"   ✓ Generated response ({len(response.answer)} chars)")
            logger.info(f"   ✓ Confidence: {response.confidence:.2f}")
            
            # Add query metadata
            response.metadata["original_query"] = original_query
            if rephrase_query:
                response.metadata["rephrased_query"] = query
            
            logger.info("=" * 80)
            logger.info("✅ QUERY COMPLETE")
            logger.info("=" * 80 + "\n")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {str(e)}")
            raise RuntimeError(f"Query failed: {str(e)}")
    
    
    def _calculate_confidence(self, chunks: List, answer: str) -> float:
        """Calculate confidence score for answer."""
        if not chunks:
            return 0.0
        
        # Average similarity score
        avg_score = sum(c.score for c in chunks) / len(chunks)
        
        # Boost if answer is substantial
        length_factor = min(len(answer) / 200, 1.0) * 0.2
        
        confidence = min(avg_score + length_factor, 1.0)
        return confidence
    
    
    def query_with_citations(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query with explicit inline citations.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and citation mapping
        """
        logger.info(f"Processing query with citations: {query[:50]}...")
        
        # Generate embedding and retrieve chunks
        query_embedding = self.embedding_generator.embed_query(query)
        retrieved_chunks = self.vector_store.query_similar(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k or settings.TOP_K_RESULTS
        )
        
        # Generate answer with citations
        result = self.llm_service.answer_with_citations(
            query=query,
            context_chunks=retrieved_chunks
        )
        
        return result
    
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document from the system.
        
        Removes all chunks associated with the document.
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            Dictionary with deletion results
        """
        logger.info(f"Deleting document: {document_id}")
        
        try:
            result = self.vector_store.delete_by_document_id(
                document_id=document_id
            )
            logger.info(f"✓ Document {document_id} deleted")
            return result
            
        except Exception as e:
            logger.error(f"Document deletion failed: {str(e)}")
            raise RuntimeError(f"Deletion failed: {str(e)}")
    
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the system.
        
        Returns:
            Dictionary with various statistics
        """
        logger.info("Gathering system statistics...")
        
        stats = {
            "vector_store": self.vector_store.get_index_stats(),
            "embedding_dimension": self.embedding_generator.get_embedding_dimension(),
            "hybrid_search_enabled": settings.ENABLE_HYBRID_SEARCH,
            "fusion_method": "RRF" if settings.USE_RRF else f"Weighted ({settings.VECTOR_WEIGHT}/{settings.BM25_WEIGHT})",
            "settings": {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "top_k": settings.TOP_K_RESULTS,
                "similarity_threshold": settings.SIMILARITY_THRESHOLD,
                "llm_model": settings.LLM_MODEL_NAME,
                "embedding_model": settings.EMBEDDING_MODEL_NAME
            }
        }
        
        return stats
    
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all services.
        
        Returns:
            Dictionary with health status of each service
        """
        logger.info("Performing health check...")
        
        health = {
            "vector_store": False,
            "embedding_generator": False,
            "llm_service": False,
            "hybrid_search": False
        }
        
        # Check vector store
        try:
            health["vector_store"] = self.vector_store.check_connection()
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
        
        # Check embedding generator
        try:
            test_embedding = self.embedding_generator.embed_query("test")
            health["embedding_generator"] = len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Embedding generator health check failed: {e}")
        
        # Check LLM service
        try:
            test_response = self.llm_service.generate_response(
                query="test",
                context_chunks=[]
            )
            health["llm_service"] = len(test_response.answer) > 0
        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
        
        # Check hybrid search
        if settings.ENABLE_HYBRID_SEARCH and self.vector_store.hybrid_search:
            try:
                bm25_stats = self.vector_store.hybrid_search.get_index_stats()
                health["hybrid_search"] = bm25_stats.get("index_exists", False)
            except Exception as e:
                logger.error(f"Hybrid search health check failed: {e}")
        else:
            health["hybrid_search"] = None  # Not enabled
        
        all_healthy = all(v for v in health.values() if v is not None)
        
        logger.info(f"Health check: {'✅ All systems operational' if all_healthy else '⚠️ Some systems failing'}")
        
        return health


# ========== Convenience Functions ==========
def ingest_multiple_documents(
    chatbot: DocumentChatbot,
    file_paths: List[str],
    show_progress: bool = True
) -> List[UploadResponse]:
    """
    Ingest multiple documents.
    
    Args:
        chatbot: DocumentChatbot instance
        file_paths: List of file paths
        show_progress: Whether to show progress
        
    Returns:
        List of UploadResponse objects
    """
    logger.info(f"Ingesting {len(file_paths)} documents...")
    
    results = []
    
    if show_progress:
        tracker = ProgressTracker(total=len(file_paths), description="Ingesting documents")
    
    for file_path in file_paths:
        try:
            result = chatbot.ingest_document(file_path)
            results.append(result)
            
            if show_progress:
                tracker.update(1)
                
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {str(e)}")
            if show_progress:
                tracker.update(1)
    
    if show_progress:
        tracker.finish()
    
    logger.info(f"Completed: {len(results)}/{len(file_paths)} documents ingested successfully")
    return results


def interactive_chat(chatbot: DocumentChatbot) -> None:
    """
    Start an interactive chat session.
    
    Args:
        chatbot: DocumentChatbot instance
    """
    print("\n" + "=" * 80)
    print("DOCUMENTATION CHATBOT - Interactive Mode")
    if settings.ENABLE_HYBRID_SEARCH:
        fusion = "RRF" if settings.USE_RRF else f"Weighted {settings.VECTOR_WEIGHT}/{settings.BM25_WEIGHT}"
        print(f"Hybrid Search: ENABLED ({fusion})")
    else:
        print("Search Mode: Vector-only")
    print("=" * 80)
    print("Type your questions below. Type 'exit' or 'quit' to end the session.")
    print("\nCommands:")
    print("  /stats   - Show system statistics")
    print("  /health  - Check system health")
    print("  /hybrid  - Toggle hybrid search on/off")
    print("=" * 80 + "\n")
    
    # Conversation history
    conversation_history = []
    
    while True:
        # Get user input
        try:
            query = input("\n🤔 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break
        
        # Check for exit
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
        
        # Check for commands
        if query.startswith('/'):
            if query == '/stats':
                stats = chatbot.get_statistics()
                print("\n📊 System Statistics:")
                print(json.dumps(stats, indent=2))
                continue
            elif query == '/health':
                health = chatbot.health_check()
                print("\n🏥 System Health:")
                for service, status in health.items():
                    if status is None:
                        status_icon = "⊘"
                        status_text = "DISABLED"
                    elif status:
                        status_icon = "✓"
                        status_text = "OK"
                    else:
                        status_icon = "✗"
                        status_text = "FAILED"
                    print(f"  {status_icon} {service}: {status_text}")
                continue
            elif query == '/hybrid':
                current = settings.ENABLE_HYBRID_SEARCH
                print(f"\n🔀 Hybrid Search currently: {'ENABLED' if current else 'DISABLED'}")
                print("   (Modify .env file and restart to change)")
                continue
            else:
                print("❌ Unknown command. Available: /stats, /health, /hybrid")
                continue
        
        # Skip empty queries
        if not query:
            continue
        
        # Process query
        try:
            print("\n⏳ Processing...\n")
            response = chatbot.query(
                query=query,
                include_sources=True,
                streaming=False  # Set to True for real-time streaming
            )
            
            # Display answer
            print("\n" + "=" * 80)
            print("🤖 Assistant:")
            print("-" * 80)
            print(response.answer)
            
            # Display sources
            if response.sources:
                print("\n📚 Sources:")
                print("-" * 80)
                for i, retrieved_chunk in enumerate(response.sources[:3], start=1):
                    chunk = retrieved_chunk.chunk
                    score = retrieved_chunk.score
                    print(f"\n[{i}] {chunk.metadata.filename} (score: {score:.3f})")
                    preview = chunk.text[:150].replace('\n', ' ')
                    print(f"    {preview}...")
            
            # Display metadata
            print(f"\n💡 Confidence: {response.confidence:.2f}")
            if settings.ENABLE_HYBRID_SEARCH:
                fusion = response.metadata.get('fusion_method', 'unknown')
                print(f"🔀 Fusion: {fusion}")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            logger.error(f"Query failed: {str(e)}")


# ========== Main Entry Point ==========
def main():
    """Main function for running the chatbot."""
    
    print("\n" + "=" * 80)
    print("DOCUMENTATION CHATBOT - RAG System with Hybrid Search")
    print("=" * 80)
    
    # Initialize chatbot
    print("\n📦 Initializing chatbot...")
    try:
        chatbot = DocumentChatbot(
            embedding_provider="sentence_transformers",
            vector_store_provider="pinecone"
        )
        print("✅ Chatbot initialized\n")
    except Exception as e:
        print(f"❌ Failed to initialize: {str(e)}")
        sys.exit(1)
    
    # Check health
    print("🏥 Checking system health...")
    health = chatbot.health_check()
    all_healthy = all(v for v in health.values() if v is not None)
    
    if not all_healthy:
        print("⚠️  Warning: Some services are not healthy!")
        for service, status in health.items():
            if status is False:
                print(f"   ✗ {service} is not responding")
        
        proceed = input("\nDo you want to continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
    else:
        print("✅ All systems operational\n")
    
    # Main menu
    while True:
        print("\n" + "=" * 80)
        print("Main Menu:")
        print("  1. Ingest document(s)")
        print("  2. Query chatbot")
        print("  3. Interactive chat")
        print("  4. View statistics")
        print("  5. Rebuild BM25 index")
        print("  6. Exit")
        print("=" * 80)
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            # Ingest documents
            file_path = input("Enter file path (or directory): ").strip()
            
            if os.path.isfile(file_path):
                try:
                    result = chatbot.ingest_document(file_path)
                    print(f"✅ Document ingested: {result.document_id}")
                except Exception as e:
                    print(f"❌ Ingestion failed: {str(e)}")
                    
            elif os.path.isdir(file_path):
                files = [
                    os.path.join(file_path, f) 
                    for f in os.listdir(file_path) 
                    if any(f.endswith(ext) for ext in settings.ALLOWED_EXTENSIONS)
                ]
                print(f"Found {len(files)} documents")
                if files:
                    results = ingest_multiple_documents(chatbot, files)
                    print(f"✅ Ingested {len(results)}/{len(files)} documents")
                else:
                    print("❌ No valid documents found")
            else:
                print("❌ Invalid path")
        
        elif choice == '2':
            # Single query
            query = input("Enter your question: ").strip()
            if query:
                try:
                    response = chatbot.query(query, streaming=False)
                    print("\n" + "=" * 80)
                    print("Answer:")
                    print("-" * 80)
                    print(response.answer)
                    print("=" * 80)
                except Exception as e:
                    print(f"❌ Query failed: {str(e)}")
        
        elif choice == '3':
            # Interactive chat
            interactive_chat(chatbot)
        
        elif choice == '4':
            # Statistics
            stats = chatbot.get_statistics()
            print("\n📊 System Statistics:")
            print(json.dumps(stats, indent=2))
        
        elif choice == '5':
            # Rebuild BM25
            if settings.ENABLE_HYBRID_SEARCH:
                print("\n🔄 Rebuilding BM25 index...")
                result = chatbot.vector_store.rebuild_bm25_index()
                print(json.dumps(result, indent=2))
            else:
                print("❌ Hybrid search is not enabled")
        
        elif choice == '6':
            # Exit
            print("\nGoodbye!")
            break
        
        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()