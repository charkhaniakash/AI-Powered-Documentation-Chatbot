"""
Contextual compression service for filtering and extracting relevant content.
Reduces noise and improves response quality by keeping only relevant sentences.
"""

import logging
import re
from typing import List, Dict, Optional
import numpy as np

from app.config.settings import settings
from app.models.schemas import RetrievedChunk, TextChunk

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


class ContextualCompressor:
    """
    Compresses retrieved chunks by extracting only relevant sentences.
    
    Benefits:
    - Reduces token usage (lower costs)
    - Removes irrelevant context
    - Improves LLM focus on relevant info
    - Allows more relevant chunks in context window
    """
    
    def __init__(
        self,
        relevance_threshold: float = None,
        max_sentences_per_chunk: int = None,
        min_sentence_length: int = 15
    ):
        """
        Initialize the contextual compressor.
        
        Args:
            relevance_threshold: Minimum similarity score to keep a sentence
            max_sentences_per_chunk: Maximum sentences to keep per chunk
            min_sentence_length: Minimum characters for a valid sentence
        """
        self.relevance_threshold = relevance_threshold or settings.COMPRESSION_RELEVANCE_THRESHOLD
        self.max_sentences_per_chunk = max_sentences_per_chunk or settings.COMPRESSION_MAX_SENTENCES
        self.min_sentence_length = min_sentence_length
        
        # Lazy load embedding model (reuse from system)
        self._embedder = None
        
        logger.info(
            f"ContextualCompressor initialized: threshold={self.relevance_threshold}, "
            f"max_sentences={self.max_sentences_per_chunk}"
        )
    
    
    def _get_embedder(self):
        """Lazy load embedding model (reuses existing model)."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer for compression...")
            # Use same model as main embedding generator
            self._embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return self._embedder
    
    
    def compress_retrieved_chunks(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        preserve_order: bool = True
    ) -> List[RetrievedChunk]:
        """
        Compress all retrieved chunks by extracting relevant sentences.
        
        Args:
            query: User's original query
            retrieved_chunks: List of retrieved chunks from vector search
            preserve_order: Whether to maintain original chunk order
            
        Returns:
            List of compressed RetrievedChunk objects
        """
        if not retrieved_chunks:
            logger.warning("No chunks to compress")
            return []
        
        logger.info(f"🗜️  Compressing {len(retrieved_chunks)} retrieved chunks...")
        
        # Get query embedding once
        embedder = self._get_embedder()
        query_embedding = embedder.encode(query, convert_to_numpy=True)
        
        compressed_chunks = []
        total_original_chars = 0
        total_compressed_chars = 0
        
        for retrieved_chunk in retrieved_chunks:
            original_text = retrieved_chunk.chunk.text
            total_original_chars += len(original_text)
            
            # Compress the chunk
            compressed_text, compression_score = self._compress_single_chunk(
                text=original_text,
                query_embedding=query_embedding,
                embedder=embedder
            )
            
            total_compressed_chars += len(compressed_text)
            
            # Skip if compression removed everything
            if not compressed_text.strip():
                logger.debug(f"Chunk {retrieved_chunk.chunk.chunk_id} fully filtered out")
                continue
            
            # Create new compressed chunk
            compressed_chunk = TextChunk(
                chunk_id=retrieved_chunk.chunk.chunk_id,
                text=compressed_text,
                metadata=retrieved_chunk.chunk.metadata,
                chunk_index=retrieved_chunk.chunk.chunk_index,
                start_char=retrieved_chunk.chunk.start_char,
                end_char=retrieved_chunk.chunk.end_char
            )
            
            # Create new RetrievedChunk with updated score
            # Blend original vector score with compression score
            blended_score = (retrieved_chunk.score * 0.7) + (compression_score * 0.3)
            
            compressed_retrieved = RetrievedChunk(
                chunk=compressed_chunk,
                score=blended_score,
                rank=retrieved_chunk.rank
            )
            
            compressed_chunks.append(compressed_retrieved)
        
        # Re-sort by blended score if not preserving order
        if not preserve_order:
            compressed_chunks.sort(key=lambda x: x.score, reverse=True)
            # Update ranks
            for i, chunk in enumerate(compressed_chunks, start=1):
                chunk.rank = i
        
        # Calculate compression ratio
        compression_ratio = (1 - total_compressed_chars / total_original_chars) * 100 if total_original_chars > 0 else 0
        
        logger.info(
            f"   ✓ Compressed to {len(compressed_chunks)} chunks "
            f"({total_compressed_chars:,} chars, {compression_ratio:.1f}% reduction)"
        )
        
        return compressed_chunks
    
    
    def _compress_single_chunk(
        self,
        text: str,
        query_embedding: np.ndarray,
        embedder
    ) -> tuple[str, float]:
        """
        Compress a single chunk by extracting relevant sentences.
        
        Args:
            text: Original chunk text
            query_embedding: Query embedding vector
            embedder: Sentence transformer model
            
        Returns:
            Tuple of (compressed_text, avg_relevance_score)
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return text, 0.0
        
        # If only one sentence, return as-is
        if len(sentences) == 1:
            return text, 1.0
        
        # Calculate relevance for each sentence
        sentence_scores = []
        sentence_embeddings = embedder.encode(sentences, convert_to_numpy=True)
        
        for idx, (sentence, sent_emb) in enumerate(zip(sentences, sentence_embeddings)):
            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, sent_emb)
            
            sentence_scores.append({
                'sentence': sentence,
                'score': float(similarity),
                'position': idx
            })
        
        # Filter by threshold
        relevant_sentences = [
            s for s in sentence_scores
            if s['score'] >= self.relevance_threshold
        ]
        
        # If no sentences pass threshold, keep the top one
        if not relevant_sentences:
            relevant_sentences = [max(sentence_scores, key=lambda x: x['score'])]
        
        # Sort by score and take top N
        relevant_sentences.sort(key=lambda x: x['score'], reverse=True)
        top_sentences = relevant_sentences[:self.max_sentences_per_chunk]
        
        # Re-sort by original position to maintain flow
        top_sentences.sort(key=lambda x: x['position'])
        
        # Join sentences
        compressed_text = ' '.join([s['sentence'] for s in top_sentences])
        
        # Calculate average relevance score
        avg_score = sum(s['score'] for s in top_sentences) / len(top_sentences)
        
        return compressed_text, avg_score
    
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitter using regex
        # Handles common sentence endings
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            # Keep only sentences meeting minimum length
            if len(sent) >= self.min_sentence_length:
                cleaned_sentences.append(sent)
        
        return cleaned_sentences
    
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0 to 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)
    
    
    def get_compression_stats(
        self,
        original_chunks: List[RetrievedChunk],
        compressed_chunks: List[RetrievedChunk]
    ) -> Dict:
        """
        Get statistics about compression.
        
        Args:
            original_chunks: Original retrieved chunks
            compressed_chunks: Compressed chunks
            
        Returns:
            Dictionary with compression statistics
        """
        original_chars = sum(len(c.chunk.text) for c in original_chunks)
        compressed_chars = sum(len(c.chunk.text) for c in compressed_chunks)
        
        stats = {
            'original_chunk_count': len(original_chunks),
            'compressed_chunk_count': len(compressed_chunks),
            'original_total_chars': original_chars,
            'compressed_total_chars': compressed_chars,
            'chars_removed': original_chars - compressed_chars,
            'compression_ratio_percent': ((original_chars - compressed_chars) / original_chars * 100) if original_chars > 0 else 0,
            'chunks_filtered_out': len(original_chunks) - len(compressed_chunks)
        }
        
        return stats


def get_contextual_compressor() -> ContextualCompressor:
    """
    Factory function to get contextual compressor instance.
    
    Returns:
        ContextualCompressor instance
    """
    return ContextualCompressor()