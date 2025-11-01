Query → Embedding → Vector Search → Retrieved Docs → LLM → Response
```

With compression, it becomes:
```
Query → Embedding → Vector Search → Retrieved Docs → 
→ Compression (filter & extract) → Compressed Docs → LLM → Response





app/services/vector_store.py
No changes needed! The compression happens after vector retrieval, so your vector store stays untouched.


What is Contextual Compression?
Contextual Compression is a technique that filters and compresses retrieved documents to keep only the most relevant parts before sending them to the LLM. Instead of passing entire documents, you extract only the sentences/passages that are actually relevant to the user's query.
Benefits:

Reduces token usage (cost savings)
Improves response quality by removing noise
Allows more relevant documents in the context window
Faster response times