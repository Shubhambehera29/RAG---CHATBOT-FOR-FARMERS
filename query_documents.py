"""
Query Script for RAG Chatbot
Command-line interface for querying the document knowledge base
"""

import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentQuery:
    def __init__(self):
        """Initialize the document query system"""
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1
        )
        
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Initialize vector store and QA chain
        self.vectorstore = None
        self.qa_chain = None
        self.setup_qa_chain()
    
    def setup_qa_chain(self):
        """Setup the QA chain with vector store"""
        try:
            # Connect to Pinecone index
            index = self.pc.Index(self.index_name)
            self.vectorstore = PineconeVectorStore(
                index=index,
                embedding=self.embeddings,
                text_key="text"
            )
            
            # Create prompt template
            prompt_template = """
            You are a helpful AI assistant with access to a knowledge base. 
            Use the following pieces of context to answer the user's question.
            If you don't know the answer based on the context, just say that you don't know, 
            don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer: Provide a clear, concise, and helpful answer based on the context above.
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            logger.info("✅ QA chain setup completed")
            
        except Exception as e:
            logger.error(f"❌ Error setting up QA chain: {str(e)}")
            self.qa_chain = None
    
    def query(self, question: str):
        """Query the knowledge base"""
        if not self.qa_chain:
            return "Error: QA chain not properly initialized. Please check your configuration."
        
        try:
            logger.info(f"🔍 Processing query: {question}")
            result = self.qa_chain({"query": question})
            
            answer = result["result"]
            sources = result["source_documents"]
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}", []
    
    def display_results(self, question: str, answer: str, sources: list):
        """Display query results in a formatted way"""
        print("\n" + "="*80)
        print(f"❓ QUESTION: {question}")
        print("="*80)
        print(f"🤖 ANSWER: {answer}")
        print("="*80)
        
        if sources:
            print("📚 SOURCES:")
            print("-"*40)
            for i, source in enumerate(sources, 1):
                source_file = source.metadata.get('source', 'Unknown')
                page = source.metadata.get('page', 'N/A')
                content_preview = source.page_content[:200] + "..." if len(source.page_content) > 200 else source.page_content
                
                print(f"\n{i}. Source: {source_file}")
                print(f"   Page: {page}")
                print(f"   Content: {content_preview}")
        
        print("="*80)

def main():
    """Main function for command-line querying"""
    print("🤖 RAG Chatbot - Command Line Interface")
    print("="*50)
    
    # Check environment variables
    required_vars = ["GEMINI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        return
    
    # Initialize query system
    try:
        query_system = DocumentQuery()
        if not query_system.qa_chain:
            print("❌ Failed to initialize query system. Please check your configuration.")
            return
    except Exception as e:
        print(f"❌ Error initializing query system: {str(e)}")
        return
    
    print("✅ Query system initialized successfully!")
    print("\nType your questions below. Type 'quit' or 'exit' to stop.")
    print("-"*50)
    
    # Interactive query loop
    while True:
        try:
            # Get user input
            question = input("\n❓ Ask me anything: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Skip empty questions
            if not question:
                print("Please enter a question.")
                continue
            
            # Process query
            result = query_system.query(question)
            
            if isinstance(result, tuple):
                answer, sources = result
                query_system.display_results(question, answer, sources)
            else:
                print(f"\n❌ {result}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
