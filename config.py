"""
Configuration file for the RAG Chatbot
Copy this file to .env and fill in your API keys
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google AI API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Chatbot Configuration
CHATBOT_NAME = os.getenv("CHATBOT_NAME", "GenAI RAG Assistant")
CHATBOT_DESCRIPTION = os.getenv("CHATBOT_DESCRIPTION", "A helpful AI assistant powered by RAG technology")

# Model Configuration
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.1

# Document Processing Configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5

# Validation
def validate_config():
    """Validate that all required configuration is present"""
    required_vars = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return True
