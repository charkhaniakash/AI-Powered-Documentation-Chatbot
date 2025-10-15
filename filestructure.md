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