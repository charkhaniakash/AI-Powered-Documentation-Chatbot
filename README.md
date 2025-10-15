# 📚 AI-Powered Documentation Chatbot

A production-ready RAG (Retrieval Augmented Generation) system that lets users upload documents and query them using natural language.

## 🎯 Features

- **Multi-format Support**: PDF, Markdown, and Text files
- **Intelligent Chunking**: Multiple strategies (simple, sentence-aware, recursive)
- **Semantic Search**: Vector-based similarity search using embeddings
- **LLM Integration**: Groq API for response generation
- **Source Citations**: Answers include references to source documents
- **Conversation Context**: Multi-turn conversations with memory
- **Modular Architecture**: Clean, maintainable, production-ready code

---

## 🏗️ Architecture

```
User Query
    ↓
[1] Query Embedding Generation
    ↓
[2] Vector Similarity Search (Pinecone)
    ↓
[3] Context Retrieval (Top-K Chunks)
    ↓
[4] Prompt Construction (Query + Context)
    ↓
[5] LLM Generation (Groq)
    ↓
Response with Citations
```

---

## 📁 Project Structure

```
documentation-chatbot/
│
├── app/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Configuration management
│   ├── services/
│   │   ├── __init__.py
│   │   ├── file_processor.py    # PDF/Markdown parsing
│   │   ├── chunking.py          # Text chunking
│   │   ├── embedding.py         # Embedding generation
│   │   ├── vector_store.py      # Pinecone operations
│   │   └── llm.py               # LLM query handling
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic models
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Utility functions
│
├── data/
│   └── uploads/                 # Temporary file storage
│
├── main.py                      # Main pipeline orchestrator
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
└── README.md                    # This file
```

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd documentation-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Sentence Transformers Model

```bash
# This will download the model (first time only)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### 5. Setup Environment Variables

Create `.env` file in project root:

```bash
# API Keys
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp

# Vector DB
PINECONE_INDEX_NAME=documentation-chatbot
EMBEDDING_DIMENSION=384

# LLM Settings
LLM_MODEL_NAME=mixtral-8x7b-32768
EMBEDDING_MODEL_NAME=llama-3.1-70b-versatile
MAX_TOKENS=1024
TEMPERATURE=0.3

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# File Processing
MAX_FILE_SIZE_MB=10
UPLOAD_DIR=data/uploads

# App
DEBUG=True
LOG_LEVEL=INFO
```

---

## 📖 Usage

### Option 1: Interactive CLI

```bash
python main.py
```

This launches an interactive menu with options to:
1. Ingest documents
2. Query the chatbot
3. Start interactive chat
4. View statistics

### Option 2: Programmatic Usage

```python
from main import DocumentChatbot

# Initialize chatbot
chatbot = DocumentChatbot()

# Ingest a document
result = chatbot.ingest_document("path/to/document.pdf")
print(f"Document ingested: {result.document_id}")

# Query the chatbot
response = chatbot.query("What is this document about?")
print(response.answer)

# View sources
for source in response.sources:
    print(f"- {source.chunk.metadata.filename} (score: {source.score:.2f})")
```

### Option 3: Batch Document Ingestion

```python
from main import DocumentChatbot, ingest_multiple_documents

chatbot = DocumentChatbot()

# Ingest multiple documents
file_paths = [
    "docs/file1.pdf",
    "docs/file2.md",
    "docs/file3.txt"
]

results = ingest_multiple_documents(chatbot, file_paths, show_progress=True)
print(f"Ingested {len(results)} documents")
```

---

## 🔧 Configuration

### Chunking Strategies

**1. Simple** (`strategy="simple"`)
- Fixed-size chunks with overlap
- Fastest but may split mid-sentence
- Best for: Structured documents with clear sections

**2. Sentence-Aware** (`strategy="sentence"`)
- Respects sentence boundaries
- Better semantic coherence
- Best for: Narrative text, articles

**3. Recursive** (`strategy="recursive"`)
- Hierarchical splitting (paragraphs → sentences → words)
- Maintains document structure
- Best for: General use, mixed content
- **Recommended as default**

Example:
```python
chatbot.ingest_document(
    "document.pdf",
    chunking_strategy="recursive"  # or "simple", "sentence"
)
```

### Embedding Models

**Default: Sentence Transformers (Local, Free)**
- Model: `all-MiniLM-L6-v2`
- Dimension: 384
- Speed: Fast
- Cost: Free
- Quality: Good for most use cases

**Alternative: OpenAI (Paid, Higher Quality)**
```python
chatbot = DocumentChatbot(embedding_provider="openai")
```
- Model: `text-embedding-ada-002`
- Dimension: 1536
- Speed: Fast (API)
- Cost: $0.0001 per 1K tokens
- Quality: Excellent

### Vector Stores

**Default: Pinecone (Cloud, Managed)**
- Fully managed
- Auto-scaling
- Free tier: 1 index, 100K vectors
- Best for: Production

**Alternative: ChromaDB (Local, Open-source)**
```python
chatbot = DocumentChatbot(vector_store_provider="chroma")
```
- Runs locally
- No API costs
- Best for: Development, testing

---

## 📊 Performance Tuning

### Retrieval Quality

Improve retrieval accuracy:

```python
# In .env file
TOP_K_RESULTS=10              # Retrieve more chunks (default: 5)
SIMILARITY_THRESHOLD=0.6      # Lower threshold (default: 0.7)
```

### Chunking Optimization

Balance chunk size for your content:

```python
# Longer chunks = more context but less precise
CHUNK_SIZE=1500
CHUNK_OVERLAP=300

# Shorter chunks = more precise but less context
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

### Response Quality

Adjust LLM parameters:

```python
# More deterministic (factual)
TEMPERATURE=0.1
MAX_TOKENS=512

# More creative (varied responses)
TEMPERATURE=0.7
MAX_TOKENS=2048
```

---

## 🧪 Testing

### Test Document Ingestion

```python
# Test with a sample document
chatbot = DocumentChatbot()
result = chatbot.ingest_document("test_document.pdf")
assert result.status == "completed"
```

### Test Query

```python
# Test basic query
response = chatbot.query("test query")
assert len(response.answer) > 0
assert response.confidence > 0
```

### Health Check

```python
# Check all services
health = chatbot.health_check()
print(health)
# Output: {'vector_store': True, 'embedding_generator': True, 'llm_service': True}
```

---

## 🐛 Troubleshooting

### Issue: "sentence-transformers not installed"

```bash
pip install sentence-transformers
```

### Issue: "Pinecone connection failed"

Check your API key and environment:
```python
# Test Pinecone connection
from pinecone import Pinecone
pc = Pinecone(api_key="your_key")
print(pc.list_indexes())
```

### Issue: "Embedding dimension mismatch"

Ensure your settings match your model:
- `all-MiniLM-L6-v2`: 384 dimensions
- `all-mpnet-base-v2`: 768 dimensions
- OpenAI Ada-002: 1536 dimensions

Update in `.env`:
```bash
EMBEDDING_DIMENSION=384  # Match your model
```

### Issue: "Rate limit exceeded"

Reduce batch size or add delays:
```python
# In embedding.py, reduce batch_size
self.batch_size = 5  # Default is 10
```

### Issue: "Out of memory during embedding"

Process documents in smaller batches:
```python
# Ingest files one at a time
for file_path in file_paths:
    chatbot.ingest_document(file_path)
    time.sleep(1)  # Small delay
```

---

## 📈 Advanced Features

### 1. Query with Citations

Get answers with explicit source references:

```python
result = chatbot.query_with_citations("What are the main features?")
print(result['answer'])
# Output: "The main features include [1] semantic search, [2] multi-format support..."

print(result['citations'])
# Output: {'1': {'document': 'features.pdf', 'relevance_score': 0.89}, ...}
```

### 2. Document Filtering

Query specific documents only:

```python
response = chatbot.query(
    query="What is the pricing model?",
    document_ids=["doc-id-1", "doc-id-2"]
)
```

### 3. Query Rephrasing

Improve retrieval with automatic query enhancement:

```python
response = chatbot.query(
    query="How much does it cost?",
    rephrase_query=True  # Rephrased to "What is the pricing model?"
)
```

### 4. Streaming Responses

Real-time token-by-token generation:

```python
# In your code
for chunk in chatbot.llm_service.generate_streaming_response(
    query="Explain the architecture",
    context_chunks=retrieved_chunks
):
    print(chunk, end='', flush=True)
```

### 5. Follow-up Questions

Generate related questions automatically:

```python
response = chatbot.query("What is RAG?")
follow_ups = chatbot.llm_service.generate_follow_up_questions(
    query="What is RAG?",
    answer=response.answer,
    num_questions=3
)
print("Suggested follow-ups:")
for q in follow_ups:
    print(f"- {q}")
```

### 6. Document Summarization

Summarize document chunks:

```python
# Get chunks for a document
query_embedding = chatbot.embedding_generator.embed_query("summary")
chunks = chatbot.vector_store.query_similar(
    query_embedding=query_embedding,
    top_k=10,
    filter_dict={"document_id": {"$eq": "your-doc-id"}}
)

summary = chatbot.llm_service.summarize_document(chunks, max_length=200)
print(summary)
```

---

## 🔐 Security Best Practices

### 1. Environment Variables

Never commit `.env` file:
```bash
# Add to .gitignore
.env
*.env
```

### 2. API Key Rotation

Rotate keys regularly:
```python
# Update in .env and restart application
GROQ_API_KEY=new_key_here
PINECONE_API_KEY=new_key_here
```

### 3. Input Validation

The system validates:
- File sizes (max 10MB by default)
- File types (PDF, MD, TXT only)
- Query length (max 1000 characters)

### 4. Rate Limiting

Implement rate limiting for production:
```python
from functools import wraps
import time

def rate_limit(calls_per_minute=10):
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator
```

---

## 🚀 Production Deployment

### Option 1: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t doc-chatbot .
docker run -p 8000:8000 --env-file .env doc-chatbot
```

### Option 2: Cloud Deployment (AWS/GCP/Azure)

1. **Package application**:
```bash
zip -r app.zip app/ main.py requirements.txt .env
```

2. **Deploy to cloud function/lambda**

3. **Set environment variables** in cloud console

### Option 3: Kubernetes

Create `deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: doc-chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: doc-chatbot
  template:
    metadata:
      labels:
        app: doc-chatbot
    spec:
      containers:
      - name: doc-chatbot
        image: your-registry/doc-chatbot:latest
        env:
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: groq
```

---

## 📊 Monitoring & Logging

### Enable Detailed Logging

```python
# In .env
LOG_LEVEL=DEBUG
```

### Log to File

```python
from app.utils.helpers import setup_logging

setup_logging(
    log_level="INFO",
    log_file="chatbot.log"
)
```

### Track Metrics

```python
# Get statistics
stats = chatbot.get_statistics()

# Track usage
response = chatbot.query("test")
tokens_used = response.metadata['tokens_used']
cost = calculate_response_cost(response)

print(f"Tokens: {tokens_used}, Cost: ${cost:.4f}")
```

### Error Tracking

```python
try:
    response = chatbot.query("test")
except Exception as e:
    logger.error(f"Query failed: {e}")
    # Send to error tracking service (Sentry, etc.)
```

---

## 🧩 Extending the System

### Add New File Type

In `file_processor.py`:
```python
def _process_docx(self, file_path: str) -> Tuple[str, int]:
    """Process .docx files."""
    import docx
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text, len(doc.paragraphs)
```

### Custom Chunking Strategy

In `chunking.py`:
```python
def _custom_chunking(self, text: str, metadata: DocumentMetadata) -> List[TextChunk]:
    """Your custom chunking logic."""
    # Implement your strategy
    pass
```

### Add New Vector Store

Create new class implementing the same interface:
```python
class MyVectorStore:
    def upsert_chunks(self, chunks: List[EmbeddedChunk]) -> Dict[str, Any]:
        pass
    
    def query_similar(self, query_embedding: List[float]) -> List[RetrievedChunk]:
        pass
```

### Custom LLM Provider

In `llm.py`:
```python
class CustomLLMService(LLMService):
    def _generate_response(self, prompt: str) -> str:
        # Call your LLM API
        pass
```

---

## 📚 API Reference

### DocumentChatbot Class

#### Methods

**`__init__(embedding_provider, vector_store_provider)`**
- Initialize chatbot with specified providers
- Returns: DocumentChatbot instance

**`ingest_document(file_path, filename, chunking_strategy)`**
- Ingest a single document
- Returns: UploadResponse with job details

**`query(query, top_k, document_ids, include_sources, rephrase_query)`**
- Query the chatbot
- Returns: QueryResponse with answer and sources

**`query_with_citations(query, top_k)`**
- Query with explicit citations
- Returns: Dict with answer and citation mapping

**`delete_document(document_id)`**
- Delete a document and all its chunks
- Returns: Dict with deletion results

**`get_statistics()`**
- Get system statistics
- Returns: Dict with various metrics

**`health_check()`**
- Check health of all services
- Returns: Dict with service statuses

---

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

### Code Style

- Follow PEP 8
- Add docstrings to all functions
- Include type hints
- Write unit tests for new features

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Groq** for fast LLM inference
- **Pinecone** for vector database
- **Sentence Transformers** for embeddings
- **LangChain** community for RAG inspiration

---

## 📞 Support

For issues and questions:
- GitHub Issues: [your-repo/issues]
- Email: your-email@example.com
- Documentation: [your-docs-url]

---

## 🗺️ Roadmap

### Upcoming Features

- [ ] Multi-language support
- [ ] Image/diagram extraction from PDFs
- [ ] Web page ingestion
- [ ] Conversation memory persistence
- [ ] Advanced filtering (date ranges, tags)
- [ ] REST API with FastAPI
- [ ] React frontend
- [ ] Docker Compose setup
- [ ] Automated testing suite
- [ ] Performance benchmarks

---

## 📖 Additional Resources

### Learning Materials

- [RAG Explained](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Vector Databases Guide](https://www.pinecone.io/learn/vector-database/)
- [Prompt Engineering](https://www.promptingguide.ai/)

### Related Projects

- LangChain: https://github.com/langchain-ai/langchain
- LlamaIndex: https://github.com/jerryjliu/llama_index
- Haystack: https://github.com/deepset-ai/haystack

---

## 🎓 Example Use Cases

### 1. Technical Documentation Assistant
```python
# Ingest your API docs
chatbot.ingest_document("api_docs.md")

# Query API usage
response = chatbot.query("How do I authenticate API requests?")
```

### 2. Research Paper Analyzer
```python
# Ingest research papers
for paper in ["paper1.pdf", "paper2.pdf", "paper3.pdf"]:
    chatbot.ingest_document(paper)

# Compare methodologies
response = chatbot.query("Compare the methodologies used in these papers")
```

### 3. Legal Document Search
```python
# Ingest contracts
chatbot.ingest_document("contract_2024.pdf")

# Search specific clauses
response = chatbot.query("What are the termination clauses?")
```

### 4. Customer Support Knowledge Base
```python
# Ingest FAQs and support docs
chatbot.ingest_document("faq.md")
chatbot.ingest_document("troubleshooting.pdf")

# Answer customer questions
response = chatbot.query("How do I reset my password?")
```

---

## 💡 Tips & Best Practices

### 1. Document Preparation
- Clean documents before ingestion
- Remove unnecessary formatting
- Ensure text is machine-readable (OCR for scanned PDFs)

### 2. Query Optimization
- Be specific in queries
- Use keywords from documents
- Try rephrasing if results are poor

### 3. Chunking Strategy Selection
- **Technical docs**: Use recursive chunking
- **Narrative text**: Use sentence-aware chunking
- **Structured data**: Use simple chunking

### 4. Performance Optimization
- Batch ingest documents during off-peak hours
- Use appropriate chunk sizes for your content
- Monitor and adjust similarity threshold

### 5. Cost Management
- Use sentence transformers for embeddings (free)
- Cache frequently asked questions
- Implement query deduplication

---

## 🔍 Debugging Guide

### Enable Debug Mode

```python
# In .env
DEBUG=True
LOG_LEVEL=DEBUG
```

### Check Chunk Quality

```python
# After ingestion, inspect chunks
chunks = chatbot.text_chunker.chunk_text(text, metadata)
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i}: {chunk.text[:100]}...")
```

### Verify Embeddings

```python
# Test embedding generation
test_embedding = chatbot.embedding_generator.embed_query("test")
print(f"Embedding dimension: {len(test_embedding)}")
print(f"Sample values: {test_embedding[:5]}")
```

### Test Retrieval

```python
# Check what's being retrieved
query_embedding = chatbot.embedding_generator.embed_query("your query")
results = chatbot.vector_store.query_similar(query_embedding, top_k=5)
for r in results:
    print(f"Score: {r.score:.3f} | Text: {r.chunk.text[:100]}...")
```

---

**Built with ❤️ for better document understanding**