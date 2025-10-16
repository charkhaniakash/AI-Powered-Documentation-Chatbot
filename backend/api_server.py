"""
FastAPI server entry point for the Documentation Chatbot API.
This is the main file that starts the web server.
"""

# Import FastAPI framework
from fastapi import FastAPI  # Main FastAPI class
from fastapi.middleware.cors import CORSMiddleware  # For handling CORS (frontend access)
from fastapi.responses import JSONResponse  # For JSON responses
import uvicorn  # ASGI server to run the app
import logging  # For logging

# Import our configuration
from app.config.settings import settings, validate_settings
from app.utils.helpers import setup_logging

# Import API routes (we'll create these next)
from app.api.routes import upload, query, documents, health

# ========== Setup Logging ==========
setup_logging(log_level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


# ========== Create FastAPI Application ==========
# This creates the main API application
app = FastAPI(
    # API metadata (shows up in documentation)
    title=settings.APP_NAME,
    description="AI-Powered Documentation Chatbot with RAG",
    version="1.0.0",
    
    # API documentation URLs
    docs_url="/docs",      # Swagger UI: http://localhost:8000/docs
    redoc_url="/redoc",    # ReDoc UI: http://localhost:8000/redoc
    
    # OpenAPI URL
    openapi_url="/openapi.json"
)


# ========== CORS Middleware ==========
# CORS = Cross-Origin Resource Sharing
# This allows your React frontend to communicate with the API
# Without this, browsers will block requests from frontend to backend

app.add_middleware(
    CORSMiddleware,
    # Allow these origins (websites) to access the API
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://localhost:5173",  # Vite development server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        # Add your production frontend URL here
        # "https://your-frontend.com"
    ],
    
    # Allow credentials (cookies, authorization headers)
    allow_credentials=True,
    
    # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],
    
    # Allow all headers
    allow_headers=["*"],
)


# ========== Include Routers ==========
# Routers organize endpoints into logical groups
# Each router handles a specific domain (upload, query, etc.)

app.include_router(
    health.router,
    prefix="/api/v1",  # All endpoints start with /api/v1
    tags=["Health"]    # Groups in API docs
)

app.include_router(
    upload.router,
    prefix="/api/v1",
    tags=["Upload"]
)

app.include_router(
    query.router,
    prefix="/api/v1",
    tags=["Query"]
)

app.include_router(
    documents.router,
    prefix="/api/v1",
    tags=["Documents"]
)


# ========== Root Endpoint ==========
@app.get("/")
async def root():
    """
    Root endpoint - shows API is running.
    
    Visit: http://localhost:8000/
    """
    return {
        "message": "Documentation Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# ========== Startup Event ==========
@app.on_event("startup")
async def startup_event():
    """
    Runs when the API starts up.
    
    Use this to:
    - Initialize database connections
    - Load ML models
    - Validate configuration
    """
    logger.info("=" * 80)
    logger.info("Starting Documentation Chatbot API")
    logger.info("=" * 80)
    
    # Validate settings
    try:
        validate_settings()
        logger.info("✓ Settings validated")
    except Exception as e:
        logger.error(f"✗ Settings validation failed: {e}")
        raise
    
    # Log configuration
    logger.info(f"Environment: {settings.PINECONE_ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Log level: {settings.LOG_LEVEL}")
    
    logger.info("=" * 80)
    logger.info("API is ready!")
    logger.info(f"Documentation: http://localhost:8000/docs")
    logger.info("=" * 80)


# ========== Shutdown Event ==========
@app.on_event("shutdown")
async def shutdown_event():
    """
    Runs when the API shuts down.
    
    Use this to:
    - Close database connections
    - Save state
    - Cleanup resources
    """
    logger.info("Shutting down API...")
    # Add cleanup code here if needed


# ========== Global Exception Handler ==========
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Catches all unhandled exceptions.
    
    This prevents the API from crashing and returns
    a proper error response to the client.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred",
            "type": type(exc).__name__
        }
    )


# ========== Main Entry Point ==========
if __name__ == "__main__":
    """
    Run the API server when this file is executed directly.
    
    Usage:
        python api_server.py
    """
    # Run the server
    uvicorn.run(
        "api_server:app",  # Module:application
        host="0.0.0.0",    # Listen on all network interfaces
        port=8000,         # Port number
        reload=True,       # Auto-reload on code changes (development only!)
        log_level="info"   # Logging level
    )