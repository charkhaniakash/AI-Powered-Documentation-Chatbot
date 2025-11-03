"""
Duplicate Document Detection Service
Intelligently detects and handles duplicate documents to prevent:
- Duplicate chunks in vector database
- Wasted storage and costs
- Duplicate search results
"""

import logging
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from app.config.settings import settings
from app.models.schemas import DocumentMetadata

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


class DuplicateDetector:
    """
    Detects and manages duplicate documents.
    
    Detection Methods:
    1. File Hash - Exact file match (MD5)
    2. Content Hash - Same content, different file (SHA256 of text)
    3. Chunk Hash - Prevent duplicate chunks
    
    Storage:
    - Uses Redis cache for fast lookups
    - Falls back to in-memory if Redis unavailable
    """
    
    def __init__(self):
        """Initialize duplicate detector."""
        # Try to use cache manager for storage
        try:
            from app.services.cache_manager import get_cache_manager
            self.cache_manager = get_cache_manager()
            logger.info("✓ Duplicate detector using Redis/Cache storage")
        except ImportError:
            self.cache_manager = None
            logger.warning("○ Cache manager not available, using in-memory storage")
        
        # In-memory fallback
        self._file_hashes: Dict[str, str] = {}  # file_hash -> document_id
        self._content_hashes: Dict[str, str] = {}  # content_hash -> document_id
        self._document_metadata: Dict[str, Dict] = {}  # document_id -> metadata
        
        logger.info("DuplicateDetector initialized")
    
    
    # ========== File Hash Detection ==========
    
    def calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of file content.
        
        Fast and efficient for exact file matching.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash string
        """
        md5_hash = hashlib.md5()
        
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b""):
                    md5_hash.update(chunk)
            
            file_hash = md5_hash.hexdigest()
            logger.debug(f"File hash calculated: {file_hash}")
            return file_hash
            
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {str(e)}")
            raise
    
    
    def check_file_hash_exists(self, file_hash: str) -> Optional[str]:
        """
        Check if file hash already exists in system.
        
        Args:
            file_hash: File MD5 hash
            
        Returns:
            Existing document_id if found, None otherwise
        """
        # Check cache first
        if self.cache_manager:
            cache_key = f"file_hash:{file_hash}"
            cached_doc_id = self.cache_manager.get(cache_key)
            if cached_doc_id:
                logger.info(f"✓ Duplicate file detected (hash: {file_hash[:8]}...)")
                return cached_doc_id
        
        # Check in-memory fallback
        if file_hash in self._file_hashes:
            doc_id = self._file_hashes[file_hash]
            logger.info(f"✓ Duplicate file detected (hash: {file_hash[:8]}...)")
            return doc_id
        
        return None
    
    
    def store_file_hash(self, file_hash: str, document_id: str) -> None:
        """
        Store file hash mapping.
        
        Args:
            file_hash: File MD5 hash
            document_id: Document ID
        """
        # Store in cache
        if self.cache_manager:
            cache_key = f"file_hash:{file_hash}"
            # Store permanently (or with very long TTL)
            self.cache_manager.set(cache_key, document_id, ttl=31536000)  # 1 year
        
        # Store in memory (fallback)
        self._file_hashes[file_hash] = document_id
        
        logger.debug(f"Stored file hash: {file_hash[:8]}... -> {document_id}")
    
    
    # ========== Content Hash Detection ==========
    
    def calculate_content_hash(self, text: str) -> str:
        """
        Calculate SHA256 hash of extracted text content.
        
        Detects same content even if filename/metadata differs.
        
        Args:
            text: Extracted text content
            
        Returns:
            SHA256 hash string
        """
        # Normalize text for consistent hashing
        normalized_text = self._normalize_text_for_hashing(text)
        
        content_hash = hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
        logger.debug(f"Content hash calculated: {content_hash[:8]}...")
        return content_hash
    
    
    def _normalize_text_for_hashing(self, text: str) -> str:
        """
        Normalize text for consistent hashing.
        
        Removes:
        - Extra whitespace
        - Leading/trailing spaces
        - Normalizes line endings
        
        Args:
            text: Raw text
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        normalized = " ".join(text.split())
        
        # Lowercase for case-insensitive matching
        normalized = normalized.lower()
        
        # Remove common formatting differences
        normalized = normalized.replace('\r\n', '\n')
        
        return normalized
    
    
    def check_content_hash_exists(self, content_hash: str) -> Optional[str]:
        """
        Check if content hash already exists.
        
        Args:
            content_hash: Content SHA256 hash
            
        Returns:
            Existing document_id if found, None otherwise
        """
        # Check cache
        if self.cache_manager:
            cache_key = f"content_hash:{content_hash}"
            cached_doc_id = self.cache_manager.get(cache_key)
            if cached_doc_id:
                logger.info(f"✓ Duplicate content detected (hash: {content_hash[:8]}...)")
                return cached_doc_id
        
        # Check in-memory
        if content_hash in self._content_hashes:
            doc_id = self._content_hashes[content_hash]
            logger.info(f"✓ Duplicate content detected (hash: {content_hash[:8]}...)")
            return doc_id
        
        return None
    
    
    def store_content_hash(self, content_hash: str, document_id: str) -> None:
        """
        Store content hash mapping.
        
        Args:
            content_hash: Content SHA256 hash
            document_id: Document ID
        """
        # Store in cache
        if self.cache_manager:
            cache_key = f"content_hash:{content_hash}"
            self.cache_manager.set(cache_key, document_id, ttl=31536000)  # 1 year
        
        # Store in memory
        self._content_hashes[content_hash] = document_id
        
        logger.debug(f"Stored content hash: {content_hash[:8]}... -> {document_id}")
    
    
    # ========== Chunk Deduplication ==========
    
    def calculate_chunk_hash(self, chunk_text: str) -> str:
        """
        Calculate hash for individual chunk.
        
        Args:
            chunk_text: Chunk text
            
        Returns:
            MD5 hash string
        """
        normalized = self._normalize_text_for_hashing(chunk_text)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    
    def deduplicate_chunks(self, chunks: List[Any]) -> List[Any]:
        """
        Remove duplicate chunks from list.
        
        Keeps first occurrence of each unique chunk.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            Deduplicated list of chunks
        """
        seen_hashes = set()
        unique_chunks = []
        duplicates_removed = 0
        
        for chunk in chunks:
            chunk_hash = self.calculate_chunk_hash(chunk.text)
            
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                unique_chunks.append(chunk)
            else:
                duplicates_removed += 1
                logger.debug(f"Removed duplicate chunk: {chunk.chunk_id}")
        
        if duplicates_removed > 0:
            logger.info(
                f"Removed {duplicates_removed} duplicate chunks "
                f"({len(unique_chunks)} unique chunks remain)"
            )
        
        return unique_chunks
    
    
    # ========== Document Metadata Storage ==========
    
    def store_document_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store document metadata for reference.
        
        Args:
            document_id: Document ID
            metadata: Document metadata dict
        """
        # Store in cache
        if self.cache_manager:
            cache_key = f"doc_metadata:{document_id}"
            self.cache_manager.set(cache_key, metadata, ttl=31536000)
        
        # Store in memory
        self._document_metadata[document_id] = metadata
    
    
    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored document metadata.
        
        Args:
            document_id: Document ID
            
        Returns:
            Metadata dict if found, None otherwise
        """
        # Check cache
        if self.cache_manager:
            cache_key = f"doc_metadata:{document_id}"
            cached_metadata = self.cache_manager.get(cache_key)
            if cached_metadata:
                return cached_metadata
        
        # Check in-memory
        return self._document_metadata.get(document_id)
    
    
    # ========== High-Level Detection ==========
    
    def check_duplicate_document(
        self,
        file_path: str,
        text_content: str
    ) -> Tuple[bool, Optional[str], str]:
        """
        Comprehensive duplicate detection.
        
        Checks both file hash and content hash.
        
        Args:
            file_path: Path to file
            text_content: Extracted text content
            
        Returns:
            Tuple of (is_duplicate, existing_doc_id, detection_method)
        """
        logger.info("Checking for duplicate document...")
        
        # Method 1: Check file hash
        file_hash = self.calculate_file_hash(file_path)
        existing_doc_id = self.check_file_hash_exists(file_hash)
        
        if existing_doc_id:
            logger.warning(
                f"⚠️  DUPLICATE DETECTED: Exact file match found "
                f"(document_id: {existing_doc_id})"
            )
            return True, existing_doc_id, "file_hash"
        
        # Method 2: Check content hash
        content_hash = self.calculate_content_hash(text_content)
        existing_doc_id = self.check_content_hash_exists(content_hash)
        
        if existing_doc_id:
            logger.warning(
                f"⚠️  DUPLICATE DETECTED: Same content found "
                f"(document_id: {existing_doc_id})"
            )
            return True, existing_doc_id, "content_hash"
        
        # Not a duplicate
        logger.info("✓ No duplicate detected - document is unique")
        return False, None, "none"
    
    
    def register_new_document(
        self,
        file_path: str,
        text_content: str,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Register a new document after ingestion.
        
        Stores hashes and metadata for future duplicate detection.
        
        Args:
            file_path: Path to file
            text_content: Extracted text
            document_id: Generated document ID
            metadata: Document metadata
        """
        logger.info(f"Registering new document: {document_id}")
        
        # Store file hash
        file_hash = self.calculate_file_hash(file_path)
        self.store_file_hash(file_hash, document_id)
        
        # Store content hash
        content_hash = self.calculate_content_hash(text_content)
        self.store_content_hash(content_hash, document_id)
        
        # Store metadata
        metadata_dict = {
            "document_id": document_id,
            "file_hash": file_hash,
            "content_hash": content_hash,
            "filename": metadata.get("filename", "unknown"),
            "uploaded_at": datetime.now().isoformat(),
            **metadata
        }
        self.store_document_metadata(document_id, metadata_dict)
        
        logger.info(f"✓ Document registered: {document_id}")
    
    
    
    # ========== Statistics ==========
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get duplicate detection statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "tracked_files": len(self._file_hashes),
            "tracked_content": len(self._content_hashes),
            "tracked_documents": len(self._document_metadata),
            "using_cache": self.cache_manager is not None
        }
        
        return stats


# ========== Singleton Instance ==========
_duplicate_detector_instance = None


def get_duplicate_detector() -> DuplicateDetector:
    """
    Get singleton duplicate detector instance.
    
    Returns:
        DuplicateDetector instance
    """
    global _duplicate_detector_instance
    
    if _duplicate_detector_instance is None:
        _duplicate_detector_instance = DuplicateDetector()
    
    return _duplicate_detector_instance