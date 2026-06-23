# 🤖 GenAI-Powered RAG Chatbot

A modern, domain-specific chatbot built with **Retrieval-Augmented Generation (RAG)** using **LangChain**, **Gemini API**, **Pinecone**, and **Streamlit**. This application allows you to upload PDF documents and chat with them using advanced AI capabilities.

## ✨ Features

- **📄 PDF Document Processing**: Upload and process multiple PDF files
- **🧠 RAG Implementation**: Advanced retrieval-augmented generation for accurate responses
- **💬 Interactive Chat Interface**: Modern, responsive chat UI with message history
- **🔍 Source Citation**: View source documents for each response
- **⚙️ Configurable Settings**: Adjustable chunk size, overlap, and retrieval parameters
- **🚀 Real-time Processing**: Stream responses and processing status
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: Google Generative AI Embeddings
- **Vector Database**: Pinecone
- **Framework**: LangChain
- **Document Processing**: PyPDF
- **Language**: Python 3.8+

## 📋 Prerequisites

- Python 3.8 or higher
- Node.js (for document indexing)
- Google AI API key
- Pinecone API key and index

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Jivan-Jala/GenAI-Powered-Chatbot-.git
cd GenAI-Powered-Chatbot-
```

### 2. Install Dependencies

#### Python Dependencies
```bash
pip install -r requirements.txt
```

#### Node.js Dependencies (for document indexing)
```bash
npm install
```

### 3. Environment Setup

Create a `.env` file in the root directory:

```env
# Google AI API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
PINECONE_ENVIRONMENT=your_environment_here
```

### 4. Set Up Pinecone Index

1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Create a new index with:
   - **Dimensions**: 768 (for text-embedding-004)
   - **Metric**: cosine
   - **Cloud**: Choose your preferred cloud provider

### 5. Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📖 Usage Guide

### Document Upload and Processing

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
2. **Configure Settings**: Adjust chunk size, overlap, and top-k results
3. **Process Documents**: Click "Process Documents" to build your knowledge base
4. **Connect to Existing Index**: Use "Connect to Existing Index" to use pre-processed documents

### Chatting with Documents

1. **Ask Questions**: Type your questions in the chat input
2. **View Responses**: Get AI-powered answers based on your documents
3. **Check Sources**: Expand "Source Documents" to see which parts of your documents were used
4. **Clear History**: Use the sidebar button to clear chat history

### Document Indexing (Optional)

For bulk document processing, you can use the Node.js script:

```bash
npm run index
```

This will process all PDFs listed in `index.js` and store them in Pinecone.

## 🔧 Configuration

### Chunk Settings

- **Chunk Size**: Size of text chunks (500-2000 characters)
- **Chunk Overlap**: Overlap between chunks (50-500 characters)
- **Top K**: Number of relevant chunks to retrieve (3-10)

### Model Settings

- **LLM Model**: Gemini 2.0 Flash
- **Embedding Model**: text-embedding-004
- **Temperature**: 0.1 (for consistent responses)

## 📁 Project Structure

```
GenAI-Powered-Chatbot-/
├── app.py                 # Main Streamlit application
├── index_documents.py     # Document indexing script (Python)
├── query_documents.py     # Query processing script (Python)
├── requirements.txt       # Python dependencies
├── package.json           # Node.js dependencies
├── .env                   # Environment variables
├── README.md             # This file
└── node_modules/         # Node.js dependencies
```

## 🎯 Use Cases

- **📚 Educational Content**: Chat with textbooks, research papers, and educational materials
- **📋 Business Documents**: Query company policies, procedures, and documentation
- **🔬 Research**: Analyze scientific papers and technical documentation
- **📖 Knowledge Base**: Create interactive knowledge bases for organizations
- **🎓 Learning Assistant**: Build personalized learning assistants

## 🔒 Security & Privacy

- API keys are stored in environment variables
- Documents are processed locally before being sent to vector database
- No data is stored permanently in the application
- All communications use secure HTTPS

## 🐛 Troubleshooting

### Common Issues

1. **"Please upload and process documents first"**
   - Make sure you've uploaded PDF files and clicked "Process Documents"
   - Check that your Pinecone index is properly configured

2. **"Error connecting to Pinecone"**
   - Verify your Pinecone API key and index name
   - Ensure your Pinecone index exists and is active

3. **"Error processing PDF"**
   - Check that the PDF file is not corrupted
   - Ensure the file is a valid PDF format

4. **Slow responses**
   - Reduce the number of documents being processed
   - Adjust chunk size and top-k parameters
   - Check your internet connection

### Getting Help

- Check the [LangChain Documentation](https://python.langchain.com/)
- Review [Pinecone Documentation](https://docs.pinecone.io/)
- Consult [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the amazing RAG framework
- [Google AI](https://ai.google.dev/) for the Gemini API
- [Pinecone](https://www.pinecone.io/) for vector database services
- [Streamlit](https://streamlit.io/) for the beautiful web interface

## 📞 Support

If you have any questions or need help, please:

1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information

---

**Built with ❤️ for the GenAI community**