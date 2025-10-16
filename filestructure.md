documentation-chatbot/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── settings.py          # Configuration management
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── file_processor.py    # PDF/Markdown parsing
│   │   │   ├── chunking.py          # Text chunking logic
│   │   │   ├── embedding.py         # Groq embeddings
│   │   │   ├── vector_store.py      # Pinecone operations
│   │   │   └── llm.py               # LLM query handling
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py           # Data models
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── helpers.py           # Utility functions
│   ├── requirements.txt
│   └── main.py                      # Entry point (later FastAPI)
│
└── data/
    └── uploads/                     # Temporary file storage





<!-- fast api folder structure -->


    backend/
├── app/
│   ├── api/              # NEW: API endpoints
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── upload.py      # Document upload endpoint
│   │   │   ├── query.py       # Query endpoint
│   │   │   ├── documents.py   # Document management
│   │   │   └── health.py      # Health check
│   │   └── dependencies.py    # Shared dependencies
│   ├── config/           # Existing
│   ├── services/         # Existing (our LLM code)
│   ├── models/           # Existing
│   └── utils/            # Existing
├── main.py              # Existing (CLI version)
└── api_server.py        # NEW: FastAPI server entry point