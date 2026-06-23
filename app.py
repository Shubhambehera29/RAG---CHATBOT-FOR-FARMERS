import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
import uuid
from typing import List, Dict

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="GenAI-Powered RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

class RAGChatbot:
    def __init__(self):
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
        
    def setup_vectorstore(self):
        """Initialize or connect to existing Pinecone vector store"""
        try:
            index = self.pc.Index(self.index_name)
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=self.embeddings,
                text_key="text"
            )
            return vectorstore
        except Exception as e:
            st.error(f"Error connecting to Pinecone: {str(e)}")
            return None
    
    def process_pdf(self, uploaded_file) -> List[Dict]:
        """Process uploaded PDF file and return documents"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata = {
                    "source": uploaded_file.name,
                    "page": doc.metadata.get("page", 0)
                }
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return documents
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return []
    
    def chunk_documents(self, documents: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    
    def create_qa_chain(self, vectorstore):
        """Create a QA chain with custom prompt"""
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
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
    
    def query(self, question: str):
        """Query the RAG system"""
        if not st.session_state.qa_chain:
            return "Please upload and process documents first."
        
        try:
            result = st.session_state.qa_chain({"query": question})
            return result["result"], result["source_documents"]
        except Exception as e:
            return f"Error processing query: {str(e)}", []

def main():
    # Header
    st.title("🤖 GenAI-Powered RAG Chatbot")
    st.markdown("**Built with LangChain, Gemini API, Pinecone, and Streamlit**")
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Document upload
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to build your knowledge base"
        )
        
        # Processing settings
        st.subheader("⚙️ Processing Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
        top_k = st.slider("Top K Results", 3, 10, 5)
        
        # Process documents button
        if st.button("🔄 Process Documents", type="primary"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    all_documents = []
                    
                    for uploaded_file in uploaded_files:
                        st.write(f"Processing {uploaded_file.name}...")
                        documents = chatbot.process_pdf(uploaded_file)
                        all_documents.extend(documents)
                    
                    if all_documents:
                        # Chunk documents
                        chunked_docs = chatbot.chunk_documents(
                            all_documents, chunk_size, chunk_overlap
                        )
                        
                        # Create vector store
                        try:
                            index = chatbot.pc.Index(chatbot.index_name)
                            vectorstore = PineconeVectorStore.from_documents(
                                chunked_docs,
                                chatbot.embeddings,
                                index_name=chatbot.index_name
                            )
                            
                            # Create QA chain
                            qa_chain = chatbot.create_qa_chain(vectorstore)
                            
                            # Store in session state
                            st.session_state.vectorstore = vectorstore
                            st.session_state.qa_chain = qa_chain
                            
                            st.success(f"✅ Successfully processed {len(uploaded_files)} file(s) with {len(chunked_docs)} chunks!")
                            
                        except Exception as e:
                            st.error(f"Error creating vector store: {str(e)}")
                    else:
                        st.error("No documents were processed successfully.")
            else:
                st.warning("Please upload at least one PDF file.")
        
        # Connect to existing index
        if st.button("🔗 Connect to Existing Index"):
            with st.spinner("Connecting to existing vector store..."):
                vectorstore = chatbot.setup_vectorstore()
                if vectorstore:
                    qa_chain = chatbot.create_qa_chain(vectorstore)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.qa_chain = qa_chain
                    st.success("✅ Connected to existing vector store!")
                else:
                    st.error("Failed to connect to vector store.")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat with your documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show source documents for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Source Documents"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}:** {source.metadata.get('source', 'Unknown')}")
                        st.write(f"**Page:** {source.metadata.get('page', 'N/A')}")
                        st.write(f"**Content:** {source.page_content[:200]}...")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.qa_chain:
                    response, sources = chatbot.query(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                else:
                    error_msg = "Please upload and process documents first, or connect to an existing vector store."
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🚀 Powered by <strong>LangChain</strong>, <strong>Gemini API</strong>, <strong>Pinecone</strong>, and <strong>Streamlit</strong></p>
            <p>Built with ❤️ for the GenAI community</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
