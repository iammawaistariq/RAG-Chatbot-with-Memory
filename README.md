# Iqrar RAG System

A robust Retrieval-Augmented Generation (RAG) web application built using **LangChain**, **Gradio 6**, **Pinecone**, and **OpenAI**. The system allows users to upload PDF documents, index them, and interactively query the knowledge base with full conversational memory.

## 🌟 Features

- **Document Management:** Easily upload, index, and delete PDF files directly from the UI.
- **Conversational AI:** Powered by OpenAI's `gpt-3.5-turbo`, enabling intelligent and context-aware responses.
- **Persistent Memory:** Chat histories and indexed file states are tracked and stored using **Redis**.
- **Vector Search:** Uses HuggingFace Embeddings (`all-MiniLM-L6-v2`) and **Pinecone** Serverless for highly efficient vector retrieval.
- **Modern Web UI:** A vibrant, side-by-side interface built natively with Gradio 6.

---

## 📋 Prerequisites

Before running the application, ensure you have the following installed and configured:

1. **Python 3.8+**
2. **OpenAI API Key** (for language generation)
3. **Pinecone API Key** (for vector storage)
4. **Redis Server** (local or remote, for persistent memory tracking)

---

## 🚀 Setup Instructions

### 1. Set Up the Environment
Navigate to the project directory and create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
Install all required Python packages from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory and add your API keys and connection strings:
```env
OPENAI_API_KEY="your_openai_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
REDIS_URL="redis://your_redis_url_here" 
# Example: redis://default:password@host:port/0
```
*(Note: If `REDIS_URL` is not provided in `.env`, the app will default to the fallback URL specified in `main.py`.)*

### 4. Run the Application
Start the Gradio server:
```bash
python main.py
```

The application will launch and provide a local URL (typically `http://127.0.0.1:7861`) where you can interact with the Iqrar RAG System in your browser.

---

## 🛠️ Usage

1. **Upload Documents:** Use the left panel to upload PDF documents. The system will automatically chunk, embed, and upload the vectors to Pinecone.
2. **Manage Files:** Use the **Refresh List** button to see currently indexed files, or delete files directly by typing their name and clicking **Delete Document**.
3. **Chat:** Use the right panel to ask questions based on your uploaded documents. The AI will remember the context of your conversation!
