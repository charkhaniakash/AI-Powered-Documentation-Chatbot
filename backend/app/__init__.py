"""
Documentation Chatbot Package
AI-powered chatbot for querying documents using RAG (Retrieval Augmented Generation).
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "AI-Powered Documentation Chatbot with RAG"

# Import main classes for easier access
from app.config.settings import settings
from main import DocumentChatbot

__all__ = [
    "DocumentChatbot",
    "settings",
]