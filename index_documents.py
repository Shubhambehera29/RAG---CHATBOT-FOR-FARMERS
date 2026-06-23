"""
Document Indexing Script for RAG Chatbot
Processes PDF documents and stores them in Pinecone vector database
"""

import os
import glob
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self):
        """Initialize the document indexer with required services"""
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # PDF files to process (update this list with your PDF files)
        # You can add your PDF files here or use glob patterns
        self.pdf_files = glob.glob("./*.pdf")  # Automatically find all PDF files in current directory
    
    def process_pdf(self, file_path: str):
        """Process a single PDF file and return documents"""
        try:
            logger.info(f"📂 Processing {file_path}...")
            
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata = {
                    "source": file_path,
                    "page": doc.metadata.get("page", 0)
                }
            
            logger.info(f"✅ Loaded {len(documents)} pages from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {str(e)}")
            return []
    
    def chunk_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    
    def index_documents(self, chunk_size=1000, chunk_overlap=200):
        """Index all PDF documents to Pinecone"""
        try:
            # Initialize Pinecone index
            index = self.pc.Index(self.index_name)
            logger.info("✅ Pinecone configured")
            
            all_documents = []
            
            # Process all PDF files
            for file_path in self.pdf_files:
                if os.path.exists(file_path):
                    documents = self.process_pdf(file_path)
                    all_documents.extend(documents)
                else:
                    logger.warning(f"⚠️ File not found: {file_path}")
            
            if not all_documents:
                logger.error("❌ No documents were processed successfully")
                return False
            
            # Chunk all documents
            logger.info("🔄 Chunking documents...")
            chunked_docs = self.chunk_documents(all_documents, chunk_size, chunk_overlap)
            logger.info(f"✅ Created {len(chunked_docs)} chunks")
            
            # Store in Pinecone
            logger.info("🔄 Storing documents in Pinecone...")
            vectorstore = PineconeVectorStore.from_documents(
                chunked_docs,
                self.embeddings,
                index_name=self.index_name
            )
            
            logger.info("🎉 All documents successfully indexed in Pinecone!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during indexing: {str(e)}")
            return False
    
    def clear_index(self):
        """Clear all vectors from the Pinecone index"""
        try:
            index = self.pc.Index(self.index_name)
            index.delete(delete_all=True)
            logger.info("🗑️ Index cleared successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing index: {str(e)}")
            return False

def main():
    """Main function to run the document indexing"""
    print("🚀 Starting Document Indexing Process")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ["GEMINI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        return
    
    # Initialize indexer
    indexer = DocumentIndexer()
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Index documents")
    print("2. Clear index")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        # Get chunking parameters
        try:
            chunk_size = int(input("Enter chunk size (default 1000): ") or "1000")
            chunk_overlap = int(input("Enter chunk overlap (default 200): ") or "200")
        except ValueError:
            chunk_size, chunk_overlap = 1000, 200
            print("Using default values: chunk_size=1000, chunk_overlap=200")
        
        # Index documents
        success = indexer.index_documents(chunk_size, chunk_overlap)
        
        if success:
            print("\n🎉 Document indexing completed successfully!")
        else:
            print("\n❌ Document indexing failed!")
    
    elif choice == "2":
        confirm = input("Are you sure you want to clear the index? (yes/no): ").strip().lower()
        if confirm == "yes":
            indexer.clear_index()
        else:
            print("Index clearing cancelled.")
    
    elif choice == "3":
        print("Goodbye!")
    
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
