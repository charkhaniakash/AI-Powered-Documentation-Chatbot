# Documentation Chatbot - Codebase Explanation & System Architecture

Welcome to the technical documentation and codebase guide for the **Documentation Chatbot**. This document provides an in-depth breakdown of the Retrieval-Augmented Generation (RAG) architecture, detailed file-by-file explanations of key modules, and flow explanations for ingestion, retrieval, generation, and quality evaluation.

---

## 1. System Architecture Overview

The project implements a complete, modern RAG (Retrieval-Augmented Generation) pipeline that allows users to upload documents of various formats, index them in a high-performance vector store, and perform semantically-grounded interactive Q&A. 

```
                      [ Uploaded Document ]
                                |
                     [ 1. File Processor ]
                     - Parses PDF, MD, Docx, XLS, CSV
                                |
                      [ 2. Text Chunker ]
                      - Splits into logical chunks
                                |
                    [ 3. Embedding Generator ]
                    - SentenceTransformers / OpenAI
                                |
                     [ 4. Vector Database ]
                     - Pinecone / Chroma Storage
                                |
        +-----------------------+-----------------------+
        |                                               |
 [ User Query ]                                         |
        |                                               |
 [ 5. Retrieve Context ] <------------------------------+
        |
 [ 6. LLM Generation ]
 - Specialized Prompt Construction
 - Citation & Source Generation
 - Answer Quality Auto-Evaluation
```

---

## 2. File Processing System (`file_processor.py`)

The `FileProcessor` class (defined in `backend/app/services/file_processor.py`) is responsible for reading, validating, and extracting clean textual content from diverse document formats.

### Core Capabilities
- **PDF Processing (`_process_pdf`)**:
  - Primarily uses **PyMuPDF (`fitz`)** for robust and fast text extraction that excels with multi-column formats and complex layouts.
  - Falls back to **PyPDF2** if PyMuPDF returns insufficient or empty text.
- **Markdown Processing (`_process_markdown`)**:
  - Parses Markdown to HTML using python's `markdown` library with `extra` and `nl2br` extensions.
  - Converts HTML back into structural plain text utilizing **BeautifulSoup4** to remove tag clutter while retaining structural layout.
  - Estimates sections based on header syntax (`#`).
- **Plain Text Processing (`_process_text`)**:
  - Detects encodings gracefully, attempting `utf-8`, `latin-1`, and `cp1252` in sequence to avoid system-level decoding crashes.
- **Excel & CSV Processing (`_process_excel`, `_process_csv`)**:
  - Uses **Pandas** alongside helper libraries like `openpyxl` (for `.xlsx`) and `xlrd` (for older `.xls`).
  - Reads all worksheets into dataframes, formatting them into human-readable tables utilizing `to_string(index=False)` to maintain tabular layout.
- **Word Document Processing (`_process_word`)**:
  - **`.docx` format**: Uses **python-docx** to extract text paragraphs and format structural tables with structural pipe separators (`|`).
  - **`.doc` format**: Uses **Mammoth** to convert legacy binary documents safely into readable structural raw text.

### Text Sanitization & Cleaning
The extracted text is normalized through the internal helper `_clean_text`:
1. Collapses contiguous spaces into a single space.
2. Normalizes line endings and limits excessive blank lines (replaces three or more consecutive newlines with exactly two).
3. Replaces carriage returns (`\r`) and form feeds (`\f`).
4. Strips all leading, trailing, and null-byte (`\x00`) characters to prevent downstream processing errors in tokenizer models.

---

## 3. Chunking & Embedding System (`chunking.py` & `embedding.py`)

### Text Chunking (`chunking.py`)
To ensure context fits within LLM context windows and captures focused meaning, documents must be chunked. The system implements a recursive character splitter (`RecursiveCharacterTextSplitter`) which splits based on priority characters (`["\n\n", "\n", " ", ""]`) to keep paragraphs and sentences whole wherever possible.

### Embedding Generator (`embedding.py`)
The `EmbeddingGenerator` class converts text chunks into mathematical vectors capturing semantic meanings.

- **Providers**:
  - **Sentence Transformers (Default, Local & Free)**: Employs `sentence-transformers/all-MiniLM-L6-v2`. This model produces dense vectors with **384 dimensions**. It is fast and runs completely locally without API dependencies.
  - **OpenAI (Paid, High Performance)**: Utilizes `text-embedding-ada-002` via the official OpenAI API, producing higher-dimension (**1536**) vectors.
- **Batching & Rate Limiting**:
  - Chunks are processed in batches (default: 10) to respect target API rate limits and optimize local model throughput.
  - Implements an exponential backoff retry mechanism (initial delay 1.0s, multiplier 2.0x, up to 3 attempts) to tolerate transient network errors or rate limit ceilings.
- **Cosine Similarity**:
  - Contains built-in arithmetic to compute the cosine similarity between two high-dimensional vectors to measure relative context match:
    $$\text{similarity} = \frac{A \cdot B}{\|A\| \|B\|}$$
- **Vector Validation**:
  - The `validate_embedding` function systematically checks vector dimension length, ensures absence of Infinite/NaN values, and safeguards against empty or degenerate all-zero vectors.

---

## 4. Large Language Model & Citation Flow (`llm.py`)

The `LLMService` (defined in `backend/app/services/llm.py`) utilizes the **Groq API** to perform lightning-fast inference on state-of-the-art models (such as LLaMA-3). It handles the Augmentation and Generation stages of the RAG pipeline.

### RAG Prompt Engineering & Constraints
The class carefully constructs a specialized context prompt (`_build_prompt`) enforcing system-level guardrails:
- Restricts the model from making up answers (combats hallucination).
- Instructs the model to declare if the provided context is insufficient.
- Formats each chunk with explicit source document metadata and relevance scores.

### Citations & Source Attribution Flow
The service can generate formal answers with explicit inline citations using `answer_with_citations`:
1. Organizes context chunks into numbered references: `[1] filenameA`, `[2] filenameB`.
2. Directs the LLM via system instructions to anchor factual claims with explicit inline citations corresponding to the respective index.
3. Returns both the generated raw text answer and a mapped metadata dictionary detailing exactly which document chunks were referenced.

### Automated Answer Quality Evaluation
To guarantee RAG accuracy, the service features an automated evaluator `check_answer_quality`:
- Sends the generated answer, original query, and retrieved document context back to the LLM on a high-determinism temperature (`0.1`).
- Evaluates the answer on a scale of `1-10` across three metrics:
  1. **Relevance**: Does the answer directly address the user's question?
  2. **Accuracy**: Is every statement fully grounded in the provided document context?
  3. **Completeness**: Does it fully capture all required details present in the source materials?
- Returns structured JSON containing numerical quality scores and a list of identified issues or potential hallucinations.

---

## 5. Main Orchestration & CLI Interface (`main.py`)

The complete ingestion and query workflows are orchestrated by the `DocumentChatbot` class in `backend/main.py`.

```
[Ingestion Flow]: FilePath -> FileProcessor -> TextChunker -> EmbeddingGenerator -> VectorStore
[Query Flow]:     Query -> EmbeddingGenerator -> VectorStore.query_similar -> LLMService
```

### Primary Pipelines
- `ingest_document`: Coordinates reading, metadata generation (including calculating unique UUIDs and processing file metadata), chunking, vector generation, and insertion into the target database.
- `query`: Handles optional query rephrasing (optimizing fuzzy user questions into crisp searchable queries), converts the query into a search vector, queries the database, and streams the answer back token-by-token in real-time.

### Interactive Console Interface
The terminal UI provides a user-friendly way to interact with the system:
- **Ingest Document**: Takes a path to a file or directory and ingests supported documents.
- **Query Chatbot**: Processes individual user questions with the integrated RAG pipeline.
- **Interactive Chat**: Keeps a rolling window of conversation history (up to last 10 messages) to support multi-turn conversational follow-up questions.
- **Commands**:
  - `/stats`: Displays index metadata, embedding dimension config, and current chunk settings.
  - `/health`: Performs self-diagnostics on the database connection, local embedding generator, and remote Groq API LLM service.

---

## 6. FastAPI Web Server (`api_server.py`)

The project exposes its core capabilities as a RESTful API via a high-performance **FastAPI** web server (`backend/api_server.py`).

### Core Features
1. **Server Initialization**:
   - Instantiates the central `FastAPI` application utilizing global configuration parameters defined in `app.config.settings`.
   - Generates comprehensive OpenAPI documentation and interactive swagger interfaces exposed at `/docs` and `/redoc`.
2. **CORS Middleware**:
   - Registers standard `CORSMiddleware` to authorize and safely handle cross-origin HTTP requests from front-end applications (such as React), configuring allowed origins, credentials, HTTP methods, and headers.
3. **Route Integration**:
   - Modularizes API routing by registering individual specialized routers under a centralized `/api/v1` prefix to organize endpoints:
     - `/health`: Service check endpoint for verifying system self-diagnostics, database connection, and Groq/LLM status.
     - `/upload`: Facilitates document upload, processing, chunking, and indexation.
     - `/query`: Runs search vectors against the vector database and generates LLM-assisted answers with structured citations.
     - `/documents`: Performs administrative actions including listing indexed items, deleting documents, or viewing details.
4. **Server Lifecycle and Execution**:
   - Uses event decorators `@app.on_event("startup")` and `@app.on_event("shutdown")` to handle setup tasks (such as validating server-side configuration, initialization routines) and safe connection terminations.
   - Deployed on **Uvicorn**, a lightning-fast ASGI web server, leveraging its auto-reload capability during local development.
