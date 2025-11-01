"""
Hybrid search service - ADD THIS AS NEW FILE
No changes to existing code needed!
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

from app.config.settings import settings
from app.models.schemas import TextChunk, RetrievedChunk

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


class HybridSearchService:
    """Combines vector search with BM25 keyword search."""
    
    def __init__(self):
        self.bm25_index = None
        self.document_chunks = []
        self.bm25_index_path = Path("data/bm25_index.pkl")
        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_bm25_index()
        logger.info("HybridSearchService initialized")
    
    def add_chunks_to_bm25(self, chunks: List[TextChunk]) -> None:
        """Add chunks to BM25 index."""
        if not chunks:
            return
        
        self.document_chunks.extend(chunks)
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in self.document_chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        self._save_bm25_index()
        logger.info(f"BM25 index now has {len(self.document_chunks)} chunks")
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25."""
        if self.bm25_index is None or not self.document_chunks:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.document_chunks[idx]
            score = float(scores[idx])
            results.append((chunk.chunk_id, score))
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        vector_results: List[RetrievedChunk],
        top_k: int = 10,
        use_rrf: bool = True,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[RetrievedChunk]:
        """Perform hybrid search with RRF or weighted fusion."""
        bm25_results = self.search_bm25(query, top_k=top_k * 2)
        
        if not bm25_results:
            return vector_results[:top_k]
        
        if use_rrf:
            merged = self._reciprocal_rank_fusion(vector_results, bm25_results)
        else:
            merged = self._weighted_fusion(vector_results, bm25_results, vector_weight, bm25_weight)
        
        return merged[:top_k]
    
    def _reciprocal_rank_fusion(self, vector_results, bm25_results, k=60):
        """RRF algorithm."""
        rrf_scores = {}
        chunk_map = {}
        
        for rank, retrieved_chunk in enumerate(vector_results, start=1):
            chunk_id = retrieved_chunk.chunk.chunk_id
            rrf_scores[chunk_id] = 1.0 / (k + rank)
            chunk_map[chunk_id] = retrieved_chunk.chunk
        
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += 1.0 / (k + rank)
            else:
                rrf_scores[chunk_id] = 1.0 / (k + rank)
            
            if chunk_id not in chunk_map:
                chunk = self._get_chunk_by_id(chunk_id)
                if chunk:
                    chunk_map[chunk_id] = chunk
        
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        merged = []
        for rank, (chunk_id, score) in enumerate(sorted_ids, start=1):
            if chunk_id in chunk_map:
                merged.append(RetrievedChunk(
                    chunk=chunk_map[chunk_id],
                    score=score,
                    rank=rank
                ))
        
        return merged
    
    def _weighted_fusion(self, vector_results, bm25_results, v_weight, b_weight):
        """Weighted fusion algorithm."""
        scores = {}
        chunk_map = {}
        
        max_v = max(r.score for r in vector_results) if vector_results else 1.0
        for r in vector_results:
            chunk_id = r.chunk.chunk_id
            scores[chunk_id] = (r.score / max_v) * v_weight
            chunk_map[chunk_id] = r.chunk
        
        max_b = max(s for _, s in bm25_results) if bm25_results else 1.0
        if max_b == 0:
            max_b = 1.0
        
        for chunk_id, bm25_score in bm25_results:
            norm_score = (bm25_score / max_b) * b_weight
            if chunk_id in scores:
                scores[chunk_id] += norm_score
            else:
                scores[chunk_id] = norm_score
            
            if chunk_id not in chunk_map:
                chunk = self._get_chunk_by_id(chunk_id)
                if chunk:
                    chunk_map[chunk_id] = chunk
        
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        merged = []
        for rank, (chunk_id, score) in enumerate(sorted_ids, start=1):
            if chunk_id in chunk_map:
                merged.append(RetrievedChunk(
                    chunk=chunk_map[chunk_id],
                    score=score,
                    rank=rank
                ))
        
        return merged
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def _get_chunk_by_id(self, chunk_id: str) -> TextChunk:
        """Get chunk by ID."""
        for chunk in self.document_chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def _save_bm25_index(self):
        """Save index to disk."""
        try:
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump({
                    'bm25_index': self.bm25_index,
                    'document_chunks': self.document_chunks
                }, f)
        except Exception as e:
            logger.error(f"Failed to save BM25: {e}")
    
    def _load_bm25_index(self):
        """Load index from disk."""
        if not self.bm25_index_path.exists():
            return
        
        try:
            with open(self.bm25_index_path, 'rb') as f:
                data = pickle.load(f)
            self.bm25_index = data['bm25_index']
            self.document_chunks = data['document_chunks']
            logger.info(f"Loaded BM25 index with {len(self.document_chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to load BM25: {e}")
    
    def clear_index(self):
        """Clear index."""
        self.bm25_index = None
        self.document_chunks = []
        if self.bm25_index_path.exists():
            self.bm25_index_path.unlink()
    
    def get_index_stats(self):
        """Get stats."""
        return {
            "total_chunks": len(self.document_chunks),
            "index_exists": self.bm25_index is not None
        }


def get_hybrid_search_service():
    """Factory function."""
    return HybridSearchService()