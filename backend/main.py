"""
Main pipeline orchestrator for the Documentation Chatbot.
Coordinates all services to implement the complete RAG system.
"""

# Import necessary libraries
import logging  # For logging
from typing import List, Dict, Any, Optional  # For type hints
import os  # For environment variables

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

# ========== Setup Logging ==========
setup_logging(log_level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


# ========== Main RAG Pipeline Class ==========
class DocumentChatbot:
    """
    Complete RAG pipeline for document Q&A.
    
    This class orchestrates all components:
    1. File Processing
    2. Text Chunking
    3. Embedding Generation
    4. Vector Storage
    5. Query Processing
    6. Response Generation
    
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
        logger.info("Initializing DocumentChatbot...")
        
        # Validate settings
        validate_settings()
        create_upload_directory()
        
        # Initialize all services
        # Each service is a specialized component
        self.file_processor = get_file_processor()
        self.text_chunker = get_text_chunker()
        self.embedding_generator = get_embedding_generator(embedding_provider)
        self.vector_store = get_vector_store(vector_store_provider)
        self.llm_service = get_llm_service()
        
        # Storage for tracking processing jobs
        # In production, this would be a database
        self.processing_jobs: Dict[str, ProcessingJob] = {}
        
        logger.info("DocumentChatbot initialized successfully")
    
    
    @timing_decorator
    def ingest_document(
        self,
        file_path: str,
        filename: Optional[str] = None,
        chunking_strategy: str = "recursive"
    ) -> UploadResponse:
        """
        Ingest a document into the system.
        
        This is the complete ingestion pipeline:
        1. Process file (extract text)
        2. Chunk text
        3. Generate embeddings
        4. Store in vector database
        
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
        logger.info(f"Starting document ingestion: {file_path}")
        
        # Use filename from path if not provided
        if filename is None:
            filename = os.path.basename(file_path)
        
        try:
            # Step 1: Validate and process file
            logger.info("Step 1: Processing file...")
            file_type = self.file_processor.validate_file(file_path, filename)
            text, metadata = self.file_processor.process_file(
                file_path=file_path,
                filename=filename,
                file_type=file_type
            )
            logger.info(f"✓ Extracted {len(text)} characters")
            
            # Step 2: Chunk the text
            logger.info(f"Step 2: Chunking text using {chunking_strategy} strategy...")
            chunks = self.text_chunker.chunk_text(
                text=text,
                metadata=metadata,
                strategy=chunking_strategy
            )
            logger.info(f"✓ Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            logger.info("Step 3: Generating embeddings...")
            embedded_chunks = self.embedding_generator.embed_chunks(
                chunks=chunks,
                show_progress=True
            )
            logger.info(f"✓ Generated {len(embedded_chunks)} embeddings")
            
            # Step 4: Store in vector database
            logger.info("Step 4: Storing in vector database...")
            upsert_result = self.vector_store.upsert_chunks(
                chunks=embedded_chunks,
                namespace=""  # Default namespace
            )
            logger.info(f"✓ Stored {upsert_result['upserted_count']} vectors")
            
            # Create upload response
            upload_response = UploadResponse(
                job_id=metadata.document_id,
                document_id=metadata.document_id,
                message="Document ingested successfully",
                status=ProcessingStatus.COMPLETED
            )
            
            logger.info(f"✓ Document ingestion completed: {metadata.document_id}")
            return upload_response
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            raise RuntimeError(f"Ingestion failed: {str(e)}")
    
    
    @timing_decorator
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        include_sources: bool = True,
        rephrase_query: bool = False
    ) -> QueryResponse:
        """
        Query the chatbot with a question.
        
        This is the complete RAG query pipeline:
        1. (Optional) Rephrase query
        2. Generate query embedding
        3. Retrieve similar chunks
        4. Generate response with LLM
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            document_ids: Filter by specific documents
            include_sources: Include source chunks in response
            rephrase_query: Whether to rephrase query for better retrieval
            
        Returns:
            QueryResponse with answer and sources
            
        Raises:
            ValueError: If query is empty
            RuntimeError: If query fails
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Optional: Rephrase query for better retrieval
            original_query = query
            if rephrase_query:
                logger.info("Rephrasing query...")
                query = self.llm_service.rephrase_query(query)
                logger.info(f"Rephrased to: {query}")
            
            # Step 1: Generate query embedding
            logger.info("Step 1: Generating query embedding...")
            query_embedding = self.embedding_generator.embed_query(query)
            logger.info(f"✓ Generated embedding with {len(query_embedding)} dimensions")
            
            # Step 2: Retrieve similar chunks
            logger.info("Step 2: Retrieving similar chunks...")
            
            # Build metadata filter if document_ids provided
            filter_dict = None
            if document_ids:
                filter_dict = {
                    "document_id": {"$in": document_ids}
                }
            
            retrieved_chunks = self.vector_store.query_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict,
                include_metadata=True
            )
            logger.info(f"✓ Retrieved {len(retrieved_chunks)} relevant chunks")
            
            # Check if we have any results
            if not retrieved_chunks:
                logger.warning("No relevant chunks found")
                # Return a response indicating no results
                return QueryResponse(
                    answer="I couldn't find any relevant information in the documents to answer your question.",
                    sources=[],
                    confidence=0.0,
                    metadata={
                        "query": original_query,
                        "chunks_found": 0
                    }
                )
            
            # Step 3: Generate response with LLM (using streaming for real-time output)
            logger.info("Step 3: Generating response...")
            full_answer = ""
            for chunk in self.llm_service.generate_streaming_response(
                query=original_query,  # Use original query for answer generation
                context_chunks=retrieved_chunks,
                conversation_history=None
            ):
                print(chunk, end='', flush=True)  # Print chunks in real-time for demo
                full_answer += chunk
            
            # Reconstruct QueryResponse for consistency (streaming doesn't return full object)
            # Note: Confidence and metadata are approximated here for demo purposes
            response = QueryResponse(
                answer=full_answer,
                sources=retrieved_chunks,
                confidence=0.8,  # Placeholder; in full implementation, calculate properly
                metadata={
                    "original_query": original_query,
                    "note": "Generated via streaming_response"
                }
            )
            logger.info(f"✓ Generated streaming response with confidence {response.confidence:.2f}")
            
            # Add query metadata
            response.metadata["original_query"] = original_query
            if rephrase_query:
                response.metadata["rephrased_query"] = query
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise RuntimeError(f"Query failed: {str(e)}")
    
    
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
        logger.info("Processing query with citations...")
        
        # Generate embedding and retrieve chunks
        query_embedding = self.embedding_generator.embed_query(query)
        retrieved_chunks = self.vector_store.query_similar(
            query_embedding=query_embedding,
            top_k=top_k
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
            "settings": {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "top_k": settings.TOP_K_RESULTS,
                "similarity_threshold": settings.SIMILARITY_THRESHOLD
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
            "llm_service": False
        }
        
        # Check vector store
        try:
            health["vector_store"] = self.vector_store.check_connection()
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
        
        # Check embedding generator
        try:
            # Try to embed a test string
            test_embedding = self.embedding_generator.embed_query("test")
            health["embedding_generator"] = len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Embedding generator health check failed: {e}")
        
        # Check LLM service
        try:
            # Try a simple generation
            test_response = self.llm_service.generate_response(
                query="test",
                context_chunks=[]
            )
            health["llm_service"] = len(test_response.answer) > 0
        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
        
        all_healthy = all(health.values())
        logger.info(f"Health check: {'✓ All systems operational' if all_healthy else '✗ Some systems failing'}")
        
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
        show_progress: Whether to show progress bar
        
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
            # Continue with next document
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
    print("=" * 80)
    print("Type your questions below. Type 'exit' or 'quit' to end the session.")
    print("Commands:")
    print("  - /stats : Show system statistics")
    print("  - /health : Check system health")
    print("=" * 80 + "\n")
    
    # Conversation history for context
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
                    status_icon = "✓" if status else "✗"
                    print(f"  {status_icon} {service}: {'OK' if status else 'FAILED'}")
                continue
            else:
                print("Unknown command. Available commands: /stats, /health")
                continue
        
        # Skip empty queries
        if not query:
            continue
        
        # Process query
        try:
            print("\n⏳ Processing...")
            response = chatbot.query(query, include_sources=True)
            
            # Display answer
            print("\n" + "=" * 80)
            print("🤖 Assistant:")
            print("-" * 80)
            print(response.answer)
            
            # Display sources if available
            if response.sources:
                print("\n📚 Sources:")
                print("-" * 80)
                for i, retrieved_chunk in enumerate(response.sources[:3], start=1):  # Show top 3
                    chunk = retrieved_chunk.chunk
                    score = retrieved_chunk.score
                    print(f"\n[{i}] {chunk.metadata.filename} (relevance: {score:.2f})")
                    print(f"    {chunk.text[:150]}...")
            
            # Display metadata
            print(f"\n💡 Confidence: {response.confidence:.2f}")
            print("=" * 80)
            
            # Add to conversation history
            conversation_history.append({
                "role": "user",
                "content": query
            })
            conversation_history.append({
                "role": "assistant",
                "content": response.answer
            })
            
            # Keep only last 10 messages for context
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            logger.error(f"Query failed: {str(e)}")


# ========== Main Entry Point ==========
def main():
    """
    Main function for running the chatbot.
    """
    import sys
    import json
    
    print("\n" + "=" * 80)
    print("DOCUMENTATION CHATBOT - RAG System")
    print("=" * 80)
    
    # Initialize chatbot
    print("\n📦 Initializing chatbot...")
    chatbot = DocumentChatbot(
        embedding_provider="sentence_transformers",  # or "openai"
        vector_store_provider="pinecone"  # or "chroma"
    )
    print("✓ Chatbot initialized\n")
    
    # Check health
    print("🏥 Checking system health...")
    health = chatbot.health_check()
    all_healthy = all(health.values())
    
    if not all_healthy:
        print("⚠️  Warning: Some services are not healthy!")
        for service, status in health.items():
            if not status:
                print(f"   ✗ {service} is not responding")
        
        proceed = input("\nDo you want to continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
    else:
        print("✓ All systems operational\n")
    
    # Main menu
    while True:
        print("\n" + "=" * 80)
        print("Main Menu:")
        print("  1. Ingest document(s)")
        print("  2. Query chatbot")
        print("  3. Interactive chat")
        print("  4. View statistics")
        print("  5. Exit")
        print("=" * 80)
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            # Ingest documents
            file_path = input("Enter file path (or directory): ").strip()
            
            if os.path.isfile(file_path):
                # Single file
                try:
                    result = chatbot.ingest_document(file_path)
                    print(f"✓ Document ingested: {result.document_id}")
                except Exception as e:
                    print(f"✗ Ingestion failed: {str(e)}")
                    
            elif os.path.isdir(file_path):
                # Directory
                files = [
                    os.path.join(file_path, f) 
                    for f in os.listdir(file_path) 
                    if f.endswith(tuple(settings.ALLOWED_EXTENSIONS))
                ]
                print(f"Found {len(files)} documents")
                results = ingest_multiple_documents(chatbot, files)
                print(f"✓ Ingested {len(results)} documents")
            else:
                print("✗ Invalid path")
        
        elif choice == '2':
            # Single query
            query = input("Enter your question: ").strip()
            if query:
                try:
                    response = chatbot.query(query)
                    print("\n" + "=" * 80)
                    print("Answer:")
                    print(response.answer)
                    print("=" * 80)
                except Exception as e:
                    print(f"✗ Query failed: {str(e)}")
        
        elif choice == '3':
            # Interactive chat
            interactive_chat(chatbot)
        
        elif choice == '4':
            # Statistics
            stats = chatbot.get_statistics()
            print("\n📊 System Statistics:")
            print(json.dumps(stats, indent=2))
        
        elif choice == '5':
            # Exit
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()