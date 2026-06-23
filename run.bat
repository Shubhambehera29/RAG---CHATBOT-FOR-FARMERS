@echo off
echo 🤖 GenAI-Powered RAG Chatbot
echo ================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import streamlit, langchain, google.generativeai, pinecone" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install requirements
        pause
        exit /b 1
    )
)

REM Check if .env file exists
if not exist .env (
    echo ❌ .env file not found
    echo Please create a .env file with your API keys
    echo Required variables:
    echo - GEMINI_API_KEY
    echo - PINECONE_API_KEY
    echo - PINECONE_INDEX_NAME
    pause
    exit /b 1
)

REM Run the application
echo 🚀 Starting the application...
python run.py

pause
