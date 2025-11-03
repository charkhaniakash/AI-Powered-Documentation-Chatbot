##  **How It Works - Complete Flow**
```
User uploads "report.pdf"
    │
    ▼
┌─────────────────────────────────────────┐
│ Calculate File Hash (MD5)               │
│ Hash: a1b2c3d4e5f6...                   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ Check: Does this file hash exist?       │
│ Query: Redis/Cache                      │
└─────────────────┬───────────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
     EXISTS            DOESN'T EXIST
         │                 │
         ▼                 ▼
    ┌─────────┐      ┌──────────────────┐
    │ RETURN  │      │ Extract text     │
    │ "Already│      │ from PDF         │
    │ exists" │      └────────┬─────────┘
    └─────────┘               │
                              ▼
                   ┌──────────────────────┐
                   │ Calculate Content    │
                   │ Hash (SHA256)        │
                   │ Hash: x9y8z7w6...    │
                   └────────┬─────────────┘
                            │
                            ▼
                   ┌──────────────────────┐
                   │ Check: Does content  │
                   │ hash exist?          │
                   └────────┬─────────────┘
                            │
                   ┌────────┴────────┐
                   │                 │
               EXISTS            DOESN'T EXIST
                   │                 │
                   ▼                 ▼
              ┌─────────┐      ┌──────────────┐
              │ RETURN  │      │ Process:     │
              │ "Dup    │      │ - Chunk      │
              │ content"│      │ - Dedupe     │
              └─────────┘      │ - Embed      │
                               │ - Store      │
                               │              │
                               │ Register:    │
                               │ - File hash  │
                               │ - Content    │
                               │ - Metadata   │
                               └──────────────┘

# ========== Chunk Deduplication ==========


When you process a large document (like a PDF or a long article), you don’t store the entire text as one big string in your vector database (like Pinecone, Redis, Chroma, etc).

Instead, you split the text into smaller pieces — called chunks — so that:

Each chunk can be embedded separately (converted into a vector).

Search results can return relevant paragraphs, not entire documents.

Example — imagine your document text:

"This is paragraph one. It talks about AI.
This is paragraph two. It talks about machine learning.
This is paragraph three. It repeats AI again."


When chunked, it might look like:

chunks = [
  "This is paragraph one. It talks about AI.",
  "This is paragraph two. It talks about machine learning.",
  "This is paragraph three. It repeats AI again."
]

🔍 Why “Chunk Hash Detection”?

Sometimes during processing, duplicate chunks can appear by mistake — for example:

The same paragraph appears multiple times in different documents.

Your text splitter accidentally produces overlapping chunks.

The same content was re-uploaded with different metadata.

If you embed every duplicate chunk again, you:

Waste storage in the vector DB.

Pay extra for embedding the same text multiple times.

Get duplicate search results when users query.

