"""
Startup script for the RAG Chatbot
Provides options to run different components of the application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import langchain
        import google.generativeai
        import pinecone
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please create a .env file with your API keys")
        print("Required variables:")
        print("- GEMINI_API_KEY")
        print("- PINECONE_API_KEY") 
        print("- PINECONE_INDEX_NAME")
        return False
    
    # Check if required variables are set
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ["GEMINI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ Environment configuration is valid")
    return True

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting Streamlit application...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

def run_indexing():
    """Run the document indexing script"""
    print("📚 Starting document indexing...")
    subprocess.run([sys.executable, "index_documents.py"])

def run_query():
    """Run the command-line query interface"""
    print("💬 Starting command-line query interface...")
    subprocess.run([sys.executable, "query_documents.py"])

def main():
    """Main function with menu options"""
    print("🤖 GenAI-Powered RAG Chatbot")
    print("=" * 40)
    
    # Check prerequisites
    if not check_requirements():
        return
    
    if not check_env_file():
        return
    
    print("\nWhat would you like to do?")
    print("1. 🚀 Run Streamlit Web App")
    print("2. 📚 Index Documents")
    print("3. 💬 Command Line Query")
    print("4. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                run_streamlit()
                break
            elif choice == "2":
                run_indexing()
                break
            elif choice == "3":
                run_query()
                break
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-4.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
