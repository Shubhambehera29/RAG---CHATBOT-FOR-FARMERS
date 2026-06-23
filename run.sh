#!/bin/bash

echo "🤖 GenAI-Powered RAG Chatbot"
echo "================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if requirements are installed
python3 -c "import streamlit, langchain, google.generativeai, pinecone" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required packages..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install requirements"
        exit 1
    fi
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found"
    echo "Please create a .env file with your API keys"
    echo "Required variables:"
    echo "- GEMINI_API_KEY"
    echo "- PINECONE_API_KEY"
    echo "- PINECONE_INDEX_NAME"
    exit 1
fi

# Run the application
echo "🚀 Starting the application..."
python3 run.py
