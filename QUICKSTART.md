# 🚀 Quick Start Guide

Get up and running with the Documentation Chatbot in 5 minutes!

---

## Prerequisites

- Python 3.8+
- Groq API key ([Get one here](https://console.groq.com/))
- Pinecone API key ([Get one here](https://www.pinecone.io/))

---

## Installation (3 steps)

python3 -m venv venv
source venv/bin/activate

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Create .env File

Create a `.env` file in the project root:

```bash
# Copy this template
GROQ_API_KEY=gsk_your_actual_groq_key_here
PINECONE_API_KEY=your_actual_pinecone_key_here
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=documentation-chatbot
EMBEDDING_DIMENSION=384
```

### Step 3: Download Embedding Model

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

---

## First Run (2 minutes)

### Option A: Interactive Mode

```bash
python main.py
```

Follow the menu prompts:
1. Select "1" to ingest a document
2. Enter the path to your PDF/MD/TXT file
3. Wait for processing to complete
4. Select "3" for interactive chat
5. Ask questions about your document!

### Option B: Python Script

Create `test_run.py`:

```python
from main import DocumentChatbot

# Initialize
chatbot = DocumentChatbot()

# Ingest a document
print("📄 Ingesting document...")
result = chatbot.ingest_document("your_document.pdf")
print(f"✅ Done! Document ID: {result.document_id}")

# Ask a question
print("\n💬 Asking question...")
response = chatbot.query("What is this document about?")
print(f"\n🤖 Answer:\n{response.answer}")

# Show sources
print(f"\n📚 Based on {len(response.sources)} sources")
for i, source in enumerate(response.sources[:3], 1):
    print(f"{i}. {source.chunk.metadata.filename} (score: {source.score:.2f})")
```

Run it:
```bash
python test_run.py
```

---

## Example Workflow

### 1. Ingest Multiple Documents

```python
from main import DocumentChatbot, ingest_multiple_documents

chatbot = DocumentChatbot()

# Ingest all PDFs from a folder
import os
docs_folder = "path/to/your/documents"
pdf_files = [
    os.path.join(docs_folder, f) 
    for f in os.listdir(docs_folder) 
    if f.endswith('.pdf')
]

results = ingest_multiple_documents(chatbot, pdf_files)
print(f"✅ Ingested {len(results)} documents")
```

### 2. Interactive Q&A Session

```python
from main import DocumentChatbot, interactive_chat

chatbot = DocumentChatbot()
interactive_chat(chatbot)  # Start chatting!
```

### 3. Programmatic Queries

```python
chatbot = DocumentChatbot()

# Query 1
response1 = chatbot.query("What are the main topics?")
print(response1.answer)

# Query 2 with filtering
response2 = chatbot.query(
    "Explain the methodology",
    document_ids=["specific-doc-id"]  # Only search this document
)
print(response2.answer)

# Query 3 with citations
result = chatbot.query_with_citations("List the key findings")
print(result['answer'])
print(f"Citations: {result['citations']}")
```

---

## Common Workflows

### Scenario 1: Research Assistant

```python
# 1. Ingest research papers
chatbot.ingest_document("paper1.pdf")
chatbot.ingest_document("paper2.pdf")
chatbot.ingest_document("paper3.pdf")

# 2. Compare papers
response = chatbot.query("Compare the methodologies used")
print(response.answer)

# 3. Find specific information
response = chatbot.query("What datasets were used?")
print(response.answer)
```

### Scenario 2: Documentation Search

```python
# 1. Ingest your docs
chatbot.ingest_document("api_documentation.md")
chatbot.ingest_document("user_guide.pdf")

# 2. Answer user questions
def answer_user_question(question):
    response = chatbot.query(question, rephrase_query=True)
    return {
        'answer': response.answer,
        'confidence': response.confidence,
        'sources': [s.chunk.metadata.filename for s in response.sources]
    }

# Usage
result = answer_user_question("How do I authenticate?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.0%}")
print(f"Sources: {', '.join(result['sources'])}")
```

### Scenario 3: Knowledge Base Builder

```python
# 1. Batch ingest from directory
import glob

markdown_files = glob.glob("knowledge_base/*.md")
for file in markdown_files:
    try:
        chatbot.ingest_document(file)
        print(f"✅ {file}")
    except Exception as e:
        print(f"❌ {file}: {e}")

# 2. Test retrieval quality
test_questions = [
    "What is the pricing model?",
    "How do I get support?",
    "What are the system requirements?"
]

for question in test_questions:
    response = chatbot.query(question)
    print(f"\nQ: {question}")
    print(f"A: {response.answer[:200]}...")
    print(f"Confidence: {response.confidence:.2f}")
```

---

## Troubleshooting

### Problem: "Module not found"
```bash
# Solution: Install missing module
pip install <module-name>

# Or reinstall all
pip install -r requirements.txt --force-reinstall
```

### Problem: "API key invalid"
```bash
# Solution: Check your .env file
cat .env  # View contents

# Make sure format is correct (no quotes, no spaces)
GROQ_API_KEY=gsk_abc123...
PINECONE_API_KEY=xyz789...
```

### Problem: "Pinecone index not found"
```python
# Solution: Create index manually
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your_key")
pc.create_index(
    name="documentation-chatbot",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-west1-gcp")
)
```

### Problem: "Out of memory"
```python
# Solution: Process smaller batches
# In embedding.py, line 47
self.batch_size = 5  # Reduce from 10 to 5
```

### Problem: "Poor answer quality"
```bash
# Solution 1: Increase chunk retrieval
# In .env
TOP_K_RESULTS=10  # Increase from 5

# Solution 2: Lower similarity threshold
SIMILARITY_THRESHOLD=0.6  # Decrease from 0.7

# Solution 3: Use better chunking strategy
chatbot.ingest_document("doc.pdf", chunking_strategy="recursive")
```

---

## Next Steps

Once you're comfortable with the basics:

1. **Read the full README.md** for advanced features
2. **Customize settings** in `.env` for your use case
3. **Explore the code** - every line is documented
4. **Build the FastAPI wrapper** (next phase)
5. **Create the React frontend** (final phase)

---

## Getting Help

- **Documentation**: See README.md for detailed guide
- **Examples**: Check `examples/` folder for more scripts
- **Issues**: Report bugs on GitHub Issues
- **Questions**: Email support or post in discussions

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────┐
│  DOCUMENTATION CHATBOT - QUICK REFERENCE            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Initialize:                                        │
│    chatbot = DocumentChatbot()                      │
│                                                     │
│  Ingest:                                            │
│    chatbot.ingest_document("file.pdf")              │
│                                                     │
│  Query:                                             │
│    response = chatbot.query("your question")        │
│    print(response.answer)                           │
│                                                     │
│  Delete:                                            │
│    chatbot.delete_document("doc-id")                │
│                                                     │
│  Health:                                            │
│    chatbot.health_check()                           │
│                                                     │
│  Stats:                                             │
│    chatbot.get_statistics()                         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Happy Chatbotting! 🤖**