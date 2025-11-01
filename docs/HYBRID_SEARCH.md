# Hybrid Search Implementation

## Overview

This documentation explains the hybrid search system implemented in the AI-Powered Documentation Chatbot. Hybrid search combines **vector search** (semantic similarity) with **BM25 keyword search** (lexical matching) to provide more accurate and relevant search results.

## Table of Contents

1. [Why Hybrid Search?](#why-hybrid-search)
2. [Architecture](#architecture)
3. [How It Works](#how-it-works)
4. [Implementation Details](#implementation-details)
5. [Fusion Algorithms](#fusion-algorithms)
6. [Usage Examples](#usage-examples)
7. [Performance Considerations](#performance-considerations)
8. [Configuration](#configuration)

---

## Why Hybrid Search?

### The Problem with Vector Search Alone

**Vector search** (using embeddings) is excellent at understanding semantic meaning:
- ✅ Finds conceptually similar content
- ✅ Handles synonyms and paraphrasing
- ✅ Understands context and intent

**But it has limitations:**
- ❌ May miss exact keyword matches
- ❌ Struggles with technical terms, acronyms, or product names
- ❌ Can return semantically similar but contextually irrelevant results

### The Problem with Keyword Search Alone

**BM25 keyword search** excels at exact matching:
- ✅ Perfect for specific terms, codes, or identifiers
- ✅ Fast and deterministic
- ✅ No model training required

**But it has limitations:**
- ❌ Doesn't understand synonyms
- ❌ Misses semantically related content
- ❌ Sensitive to exact wording

### The Solution: Hybrid Search

By combining both approaches, we get:
- 🎯 **Best of both worlds** - semantic understanding + exact matching
- 📈 **Higher accuracy** - typically 10-30% improvement in relevance
- 🔍 **Better coverage** - catches results either method might miss alone
- 💪 **Robustness** - works well across diverse query types

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                            │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│  Vector Search   │  │   BM25 Search    │
│   (Pinecone)     │  │   (Local Index)  │
└────────┬─────────┘  └─────────┬────────┘
         │                      │
         │  Top-K Results       │  Top-K Results
         │  (with scores)       │  (with scores)
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Fusion Algorithm   │
         │  (RRF or Weighted)   │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Merged Results     │
         │  (Re-ranked by       │
         │   combined scores)   │
         └──────────────────────┘
```

---

## How It Works

### Step 1: Document Ingestion

When documents are uploaded:

1. **Text Chunking** - Documents are split into manageable chunks
2. **Dual Indexing**:
   - **Vector Store** (Pinecone): Chunks are embedded and stored
   - **BM25 Index** (Local): Chunks are tokenized and indexed for keyword search

### Step 2: Query Processing

When a user searches:

1. **Parallel Search**:
   - Query is sent to both vector store and BM25 index simultaneously
   - Each returns top-K results with relevance scores

2. **Score Fusion**:
   - Results from both sources are merged using a fusion algorithm
   - Duplicate chunks are handled intelligently
   - Final ranking considers both semantic and lexical relevance

3. **Return Results**:
   - Top-N most relevant chunks are returned to the user
   - Scores reflect combined relevance from both methods

---

## Implementation Details

### File Structure

```
backend/app/services/
├── hybrid_search.py      # Main hybrid search service
├── vector_store.py       # Pinecone vector operations
└── rag_service.py        # RAG pipeline integration

backend/data/
└── bm25_index.pkl        # Persisted BM25 index (auto-generated)
```

### Key Components

#### 1. HybridSearchService Class

**Location**: `backend/app/services/hybrid_search.py`

**Responsibilities**:
- Manage BM25 index lifecycle (create, save, load)
- Perform keyword-based search
- Merge vector and BM25 results
- Persist index to disk for fast restarts

**Key Methods**:

```python
# Add documents to BM25 index
add_chunks_to_bm25(chunks: List[TextChunk]) -> None

# Search using BM25 only
search_bm25(query: str, top_k: int = 10) -> List[Tuple[str, float]]

# Perform hybrid search with fusion
hybrid_search(
    query: str,
    vector_results: List[RetrievedChunk],
    top_k: int = 10,
    use_rrf: bool = True,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> List[RetrievedChunk]
```

#### 2. BM25 Index Persistence

**File**: `backend/data/bm25_index.pkl`

**Purpose**:
- Stores the BM25 index and document chunks using Python's `pickle` module
- Enables fast service restarts without rebuilding the index
- Automatically created when documents are indexed

**Contents**:
```python
{
    'bm25_index': BM25Okapi object,
    'document_chunks': List[TextChunk]
}
```

**Lifecycle**:
- **Created**: When first document is added
- **Updated**: Every time new documents are indexed
- **Loaded**: On service initialization
- **Cleared**: When index is explicitly cleared or corrupted

---

## Fusion Algorithms

### 1. Reciprocal Rank Fusion (RRF) - Default

**Formula**: `RRF_score = Σ(1 / (k + rank))`

**How it works**:
- Ranks matter more than absolute scores
- Each result contributes `1/(k+rank)` to the final score
- `k` is a constant (default: 60) that controls rank sensitivity

**Advantages**:
- ✅ Score-agnostic (doesn't require score normalization)
- ✅ Robust to score scale differences
- ✅ Well-tested in information retrieval research
- ✅ Works well when score distributions differ significantly

**Example**:
```
Vector Search Results:        BM25 Results:
1. doc_A (score: 0.95)       1. doc_B (score: 15.2)
2. doc_B (score: 0.87)       2. doc_A (score: 12.8)
3. doc_C (score: 0.82)       3. doc_D (score: 10.1)

RRF Scores:
doc_A: 1/(60+1) + 1/(60+2) = 0.0328
doc_B: 1/(60+2) + 1/(60+1) = 0.0328
doc_C: 1/(60+3) = 0.0159
doc_D: 1/(60+3) = 0.0159

Final Ranking: doc_A, doc_B, doc_C, doc_D
```

### 2. Weighted Fusion

**Formula**: `Score = (norm_vector_score × vector_weight) + (norm_bm25_score × bm25_weight)`

**How it works**:
- Normalizes scores from each method to [0, 1]
- Applies configurable weights to each score
- Sums weighted scores for final ranking

**Advantages**:
- ✅ Fine-grained control over method importance
- ✅ Can favor semantic or lexical matching based on use case
- ✅ Intuitive score interpretation

**Default Weights**:
- Vector: 0.7 (70%) - Prioritizes semantic understanding
- BM25: 0.3 (30%) - Adds keyword precision

**Example**:
```
Vector Search Results:        BM25 Results:
doc_A (score: 0.95)          doc_B (score: 15.2)
doc_B (score: 0.87)          doc_A (score: 12.8)

Normalized:
doc_A: 0.95/0.95 = 1.0       doc_A: 12.8/15.2 = 0.84
doc_B: 0.87/0.95 = 0.92      doc_B: 15.2/15.2 = 1.0

Weighted (0.7 vector, 0.3 bm25):
doc_A: (1.0 × 0.7) + (0.84 × 0.3) = 0.952
doc_B: (0.92 × 0.7) + (1.0 × 0.3) = 0.944

Final Ranking: doc_A, doc_B
```

---

## Usage Examples

### Basic Hybrid Search

```python
from app.services.hybrid_search import HybridSearchService
from app.services.vector_store import VectorStoreService

# Initialize services
hybrid_service = HybridSearchService()
vector_service = VectorStoreService()

# Get vector search results
vector_results = vector_service.search(
    query="How to configure authentication?",
    top_k=10
)

# Perform hybrid search with RRF
final_results = hybrid_service.hybrid_search(
    query="How to configure authentication?",
    vector_results=vector_results,
    top_k=5,
    use_rrf=True  # Use Reciprocal Rank Fusion
)

# Access results
for result in final_results:
    print(f"Rank: {result.rank}")
    print(f"Score: {result.score}")
    print(f"Text: {result.chunk.text}")
```

### Weighted Fusion with Custom Weights

```python
# Favor keyword matching (useful for technical queries)
results = hybrid_service.hybrid_search(
    query="API endpoint /users/login",
    vector_results=vector_results,
    top_k=5,
    use_rrf=False,  # Use weighted fusion
    vector_weight=0.4,  # 40% semantic
    bm25_weight=0.6     # 60% keyword
)
```

### Adding Documents to Index

```python
from app.models.schemas import TextChunk

# Create chunks
chunks = [
    TextChunk(
        chunk_id="doc1_chunk1",
        text="Authentication is configured in the settings file.",
        source="docs/auth.md",
        metadata={"page": 1}
    ),
    # ... more chunks
]

# Add to BM25 index
hybrid_service.add_chunks_to_bm25(chunks)

# Index is automatically saved to disk
```

### Checking Index Stats

```python
stats = hybrid_service.get_index_stats()
print(f"Total chunks indexed: {stats['total_chunks']}")
print(f"Index exists: {stats['index_exists']}")
```

### Clearing the Index

```python
# Clear BM25 index (useful for reindexing)
hybrid_service.clear_index()
```

---

## Performance Considerations

### Memory Usage

**BM25 Index**:
- Stores all document chunks in memory
- Typical size: ~1-2 MB per 1000 chunks
- Persisted to disk as `bm25_index.pkl`

**Recommendations**:
- For large document sets (>100K chunks), consider index sharding
- Monitor memory usage in production
- Implement index cleanup for outdated documents

### Search Speed

**Benchmarks** (approximate, on typical hardware):

| Operation | Time | Notes |
|-----------|------|-------|
| BM25 search (10K chunks) | ~10-50ms | In-memory, very fast |
| Vector search (Pinecone) | ~100-300ms | Network latency included |
| Hybrid fusion (RRF) | ~5-10ms | Minimal overhead |
| Total hybrid search | ~110-360ms | Dominated by vector search |

**Optimization Tips**:
- Use appropriate `top_k` values (don't retrieve more than needed)
- Cache frequently searched queries
- Consider async/parallel processing for multiple queries

### Index Persistence

**Advantages**:
- ✅ Fast service restarts (no reindexing needed)
- ✅ Survives server crashes
- ✅ Consistent search results

**Considerations**:
- File size grows with document count
- Pickle format is Python-specific
- Consider periodic index rebuilds for optimization

---

## Configuration

### Environment Variables

Add to `.env`:

```bash
# Hybrid Search Settings
HYBRID_SEARCH_ENABLED=true
HYBRID_SEARCH_USE_RRF=true
HYBRID_SEARCH_VECTOR_WEIGHT=0.7
HYBRID_SEARCH_BM25_WEIGHT=0.3
HYBRID_SEARCH_TOP_K=10
```

### Code Configuration

In `hybrid_search.py`:

```python
# Adjust RRF constant (higher = less rank-sensitive)
RRF_K = 60  # Default: 60

# Change index storage location
self.bm25_index_path = Path("data/bm25_index.pkl")

# Modify tokenization strategy
def _tokenize(self, text: str) -> List[str]:
    # Current: simple whitespace splitting
    return text.lower().split()
    
    # Alternative: use NLTK or spaCy for better tokenization
    # import nltk
    # return nltk.word_tokenize(text.lower())
```

---

## Benefits Summary

### For Users

1. **Better Search Results**
   - More relevant answers to diverse query types
   - Handles both conceptual and specific searches

2. **Faster Answers**
   - Reduced need for query reformulation
   - Higher first-result accuracy

### For Developers

1. **Improved RAG Quality**
   - Better context retrieval = better LLM responses
   - Reduced hallucinations from irrelevant context

2. **Flexibility**
   - Tune weights based on use case
   - Choose fusion algorithm based on data characteristics

3. **Scalability**
   - BM25 is lightweight and fast
   - Easy to add more ranking signals in the future

---

## Future Enhancements

### Potential Improvements

1. **Advanced Tokenization**
   - Use NLTK or spaCy for better text processing
   - Handle special characters, URLs, code snippets

2. **Query Expansion**
   - Add synonyms and related terms
   - Use LLM for query understanding

3. **Learning to Rank**
   - Train ML model to optimize fusion weights
   - Personalize results based on user feedback

4. **Multi-Index Support**
   - Separate indexes for different document types
   - Domain-specific BM25 configurations

5. **Monitoring & Analytics**
   - Track search quality metrics
   - A/B test different fusion strategies
   - Log slow queries for optimization

---

## Troubleshooting

### Index Not Loading

**Symptom**: Service starts but no BM25 results returned

**Solutions**:
1. Check if `data/bm25_index.pkl` exists
2. Verify file permissions
3. Check logs for pickle errors
4. Try clearing and rebuilding index

### Poor Search Quality

**Symptom**: Irrelevant results returned

**Solutions**:
1. Adjust fusion weights (try 0.5/0.5 for balanced approach)
2. Increase `top_k` to retrieve more candidates
3. Review tokenization strategy
4. Check if documents are properly chunked

### Memory Issues

**Symptom**: High memory usage or OOM errors

**Solutions**:
1. Reduce number of indexed chunks
2. Implement index sharding
3. Use disk-based BM25 alternative (e.g., Elasticsearch)
4. Increase server memory

---

## References

### Academic Papers

1. **BM25**: Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond"
2. **RRF**: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"

### Libraries Used

- **rank-bm25**: Python implementation of BM25 algorithm
- **numpy**: Numerical operations for scoring
- **pickle**: Index serialization

### Related Documentation

- [Vector Store Documentation](./VECTOR_STORE.md)
- [RAG Service Documentation](./RAG_SERVICE.md)
- [API Documentation](./API.md)

---

## Conclusion

Hybrid search significantly improves retrieval quality by combining the strengths of semantic and lexical search. This implementation provides a solid foundation that can be extended and optimized based on your specific use case and requirements.

For questions or issues, please refer to the main project documentation or open an issue on GitHub.
