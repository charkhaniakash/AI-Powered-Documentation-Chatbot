"""
Text chunking service for splitting documents into processable chunks.
Implements intelligent chunking strategies to maintain context.
"""

# Import necessary libraries
import uuid  # For generating unique chunk IDs
import logging  # For logging
from typing import List  # For type hints
import re  # For regular expressions

# Import tiktoken for token counting
# Tiktoken is OpenAI's tokenizer, used to count tokens accurately
import tiktoken

# Internal imports
from app.config.settings import settings  # Application settings
from app.models.schemas import TextChunk, DocumentMetadata  # Data models

# ========== Setup Logging ==========
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


# ========== Text Chunker Class ==========
class TextChunker:
    """
    Chunks text documents into smaller pieces for embedding and retrieval.
    
    Why chunking is important:
    1. Embedding models have token limits
    2. Smaller chunks = more precise retrieval
    3. Better relevance matching
    4. Efficient vector search
    
    Chunking strategies:
    - Character-based: Simple splitting by character count
    - Sentence-aware: Respects sentence boundaries
    - Paragraph-aware: Keeps paragraphs together when possible
    - Recursive: Tries larger chunks first, then recursively splits
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk (uses settings default if None)
            chunk_overlap: Overlap between chunks (uses settings default if None)
        """
        # Use provided values or fall back to settings
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # Validate chunk_overlap < chunk_size
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        # Initialize tokenizer for accurate token counting
        # cl100k_base is the encoding used by GPT-3.5 and GPT-4
        # We use this to ensure chunks don't exceed token limits
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoder: {e}")
            self.tokenizer = None
        
        logger.info(
            f"TextChunker initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )
    
    
    def chunk_text(
        self,
        text: str,
        metadata: DocumentMetadata,
        strategy: str = "recursive"
    ) -> List[TextChunk]:
        """
        Chunk text using specified strategy.
        
        Main entry point for text chunking.
        
        Args:
            text: Full text to chunk
            metadata: Document metadata
            strategy: Chunking strategy ('simple', 'sentence', 'recursive')
            
        Returns:
            List of TextChunk objects
            
        Raises:
            ValueError: If text is empty or strategy is invalid
        """
        # Validate input
        if not text or len(text.strip()) == 0:
            raise ValueError("Cannot chunk empty text")
        
        logger.info(f"Chunking text ({len(text)} chars) using {strategy} strategy")
        
        # Route to appropriate chunking method
        if strategy == "simple":
            chunks = self._simple_chunking(text, metadata)
        elif strategy == "sentence":
            chunks = self._sentence_aware_chunking(text, metadata)
        elif strategy == "recursive":
            chunks = self._recursive_chunking(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    
    def _simple_chunking(
        self,
        text: str,
        metadata: DocumentMetadata
    ) -> List[TextChunk]:
        """
        Simple character-based chunking with overlap.
        
        Splits text into fixed-size chunks with specified overlap.
        Fastest method but may split in middle of sentences.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        chunk_index = 0
        start = 0
        
        # Loop until we've processed all text
        while start < len(text):
            # Calculate end position for this chunk
            end = start + self.chunk_size
            
            # Extract chunk text
            chunk_text = text[start:end]
            
            # Skip empty chunks
            if chunk_text.strip():
                # Create TextChunk object
                chunk = TextChunk(
                    chunk_id=str(uuid.uuid4()),  # Unique ID
                    text=chunk_text.strip(),      # Trimmed text
                    metadata=metadata,            # Document metadata
                    chunk_index=chunk_index,      # Position in sequence
                    start_char=start,             # Start position
                    end_char=min(end, len(text))  # End position (clamped to text length)
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position forward
            # Subtract overlap to create overlapping chunks
            start = end - self.chunk_overlap
        
        return chunks
    
    
    def _sentence_aware_chunking(
        self,
        text: str,
        metadata: DocumentMetadata
    ) -> List[TextChunk]:
        """
        Sentence-aware chunking that respects sentence boundaries.
        
        Splits text at sentence boundaries to maintain semantic coherence.
        More intelligent than simple chunking but still may split paragraphs.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of TextChunk objects
        """
        # Split text into sentences using regex
        # This pattern matches: . ! ? followed by space or end of string
        # Look-ahead (?=\s|$) ensures we don't match decimals like "3.14"
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # If splitting failed, fall back to simple chunking
        if len(sentences) <= 1:
            logger.warning("Sentence splitting failed, using simple chunking")
            return self._simple_chunking(text, metadata)
        
        chunks = []
        chunk_index = 0
        current_chunk = []
        current_length = 0
        start_char = 0
        
        # Iterate through sentences
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk)
                chunk = TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata=metadata,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text)
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Handle overlap: keep last few sentences
                # Calculate how many sentences to keep for overlap
                overlap_chars = 0
                overlap_sentences = []
                # Work backwards through current_chunk
                for s in reversed(current_chunk):
                    if overlap_chars + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_chars += len(s)
                    else:
                        break
                
                # Start new chunk with overlap sentences
                current_chunk = overlap_sentences
                current_length = overlap_chars
                start_char = start_char + len(chunk_text) - overlap_chars
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Create final chunk if there's remaining text
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = TextChunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text,
                metadata=metadata,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    
    def _recursive_chunking(
        self,
        text: str,
        metadata: DocumentMetadata
    ) -> List[TextChunk]:
        """
        Recursive chunking that tries to maintain semantic boundaries.
        
        Strategy:
        1. Try to split by paragraphs (double newline)
        2. If paragraphs too large, split by sentences
        3. If sentences too large, split by characters
        
        This preserves document structure while ensuring chunk size limits.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of TextChunk objects
        """
        # Define separators in order of preference
        # We try these in order: paragraph > sentence > word > character
        separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentences
            " ",     # Words
            ""       # Characters (last resort)
        ]
        
        return self._split_text_recursive(
            text=text,
            metadata=metadata,
            separators=separators,
            chunk_index_start=0,
            char_position_start=0
        )
    
    
    def _split_text_recursive(
        self,
        text: str,
        metadata: DocumentMetadata,
        separators: List[str],
        chunk_index_start: int,
        char_position_start: int
    ) -> List[TextChunk]:
        """
        Recursively split text using separator hierarchy.
        
        This is the core recursive function that tries each separator
        in order until finding one that produces appropriate chunk sizes.
        
        Args:
            text: Text to split
            metadata: Document metadata
            separators: List of separators to try
            chunk_index_start: Starting chunk index
            char_position_start: Starting character position in original text
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        
        # Base case: if text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            if text.strip():
                chunk = TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=text.strip(),
                    metadata=metadata,
                    chunk_index=chunk_index_start,
                    start_char=char_position_start,
                    end_char=char_position_start + len(text)
                )
                return [chunk]
            return []
        
        # Get current separator
        if not separators:
            # No more separators, force split by character
            return self._force_split(
                text, metadata, chunk_index_start, char_position_start
            )
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split text by current separator
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means split by character
            splits = list(text)
        
        # Merge small splits into appropriately sized chunks
        current_chunk_parts = []
        current_chunk_length = 0
        chunk_index = chunk_index_start
        current_position = char_position_start
        
        for split in splits:
            # Calculate length including separator
            split_length = len(split) + (len(separator) if separator else 0)
            
            # If this split alone is too large, recursively split it
            if len(split) > self.chunk_size:
                # First, create chunk from accumulated parts
                if current_chunk_parts:
                    chunk_text = separator.join(current_chunk_parts)
                    chunk = TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        text=chunk_text,
                        metadata=metadata,
                        chunk_index=chunk_index,
                        start_char=current_position,
                        end_char=current_position + len(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_position += len(chunk_text) + len(separator)
                    current_chunk_parts = []
                    current_chunk_length = 0
                
                # Recursively split the large part
                recursive_chunks = self._split_text_recursive(
                    text=split,
                    metadata=metadata,
                    separators=remaining_separators,
                    chunk_index_start=chunk_index,
                    char_position_start=current_position
                )
                chunks.extend(recursive_chunks)
                chunk_index += len(recursive_chunks)
                current_position += len(split) + len(separator)
                continue
            
            # Check if adding this split would exceed chunk size
            if current_chunk_length + split_length > self.chunk_size and current_chunk_parts:
                # Create chunk from accumulated parts
                chunk_text = separator.join(current_chunk_parts)
                chunk = TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata=metadata,
                    chunk_index=chunk_index,
                    start_char=current_position,
                    end_char=current_position + len(chunk_text)
                )
                chunks.append(chunk)
                chunk_index += 1
                current_position += len(chunk_text) + len(separator)
                
                # Handle overlap
                overlap_parts = self._get_overlap_parts(
                    current_chunk_parts, separator
                )
                current_chunk_parts = overlap_parts
                current_chunk_length = sum(len(p) for p in overlap_parts)
                if overlap_parts:
                    current_chunk_length += len(separator) * (len(overlap_parts) - 1)
            
            # Add split to current chunk
            current_chunk_parts.append(split)
            current_chunk_length += split_length
        
        # Create final chunk if there's remaining text
        if current_chunk_parts:
            chunk_text = separator.join(current_chunk_parts)
            if chunk_text.strip():
                chunk = TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata=metadata,
                    chunk_index=chunk_index,
                    start_char=current_position,
                    end_char=current_position + len(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks
    
    
    def _get_overlap_parts(
        self,
        parts: List[str],
        separator: str
    ) -> List[str]:
        """
        Get parts for overlap from end of current chunk.
        
        Args:
            parts: List of text parts
            separator: Separator used between parts
            
        Returns:
            List of parts that fit within overlap size
        """
        overlap_parts = []
        overlap_length = 0
        
        # Work backwards through parts
        for part in reversed(parts):
            part_length = len(part) + len(separator)
            if overlap_length + part_length <= self.chunk_overlap:
                overlap_parts.insert(0, part)
                overlap_length += part_length
            else:
                break
        
        return overlap_parts
    
    
    def _force_split(
        self,
        text: str,
        metadata: DocumentMetadata,
        chunk_index_start: int,
        char_position_start: int
    ) -> List[TextChunk]:
        """
        Force split text by character when all separators fail.
        
        Last resort splitting method.
        
        Args:
            text: Text to split
            metadata: Document metadata
            chunk_index_start: Starting chunk index
            char_position_start: Starting character position
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        chunk_index = chunk_index_start
        position = char_position_start
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            if chunk_text.strip():
                chunk = TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text.strip(),
                    metadata=metadata,
                    chunk_index=chunk_index,
                    start_char=position + i,
                    end_char=position + i + len(chunk_text)
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks


# ========== Utility Functions ==========
def get_text_chunker() -> TextChunker:
    """
    Get a TextChunker instance.
    
    Factory function for dependency injection.
    
    Returns:
        TextChunker instance
    """
    return TextChunker()