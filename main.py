import os
import dotenv
import gradio as gr
import redis

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
INDEX_NAME = "iqrar-rag"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
INDEXED_FILES_KEY = "iqrar_indexed_files"

# Initialize Redis client for tracking indexed files
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
except redis.ConnectionError:
    print(f"[WARNING] Could not connect to Redis at {REDIS_URL}. Ensure Redis is running.")
    redis_client = None

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"[INFO] Created index: {INDEX_NAME}")
    return pc

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def get_vectorstore():
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=get_embeddings()
    )

def get_indexed_files():
    if not redis_client:
        return []
    try:
        files = redis_client.smembers(INDEXED_FILES_KEY)
        return list(files) if files else []
    except redis.ConnectionError:
        return []

def add_document(file_path):
    if not file_path:
        return "No file provided."
    
    filename = os.path.basename(file_path)
    if redis_client and redis_client.sismember(INDEXED_FILES_KEY, filename):
        return f"File '{filename}' is already indexed."
    
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = splitter.split_documents(documents)
        
        for doc in docs:
            doc.metadata["source"] = filename
        
        PineconeVectorStore.from_documents(
            documents=docs,
            embedding=get_embeddings(),
            index_name=INDEX_NAME
        )
        
        if redis_client:
            redis_client.sadd(INDEXED_FILES_KEY, filename)
        return f"Successfully indexed '{filename}' ({len(docs)} chunks)."
    except Exception as e:
        return f"Error indexing '{filename}': {str(e)}"

def delete_document(filename):
    if not filename:
        return "No file selected.", gr.update(choices=get_indexed_files())
    
    try:
        pc = init_pinecone()
        index = pc.Index(INDEX_NAME)
        index.delete(filter={"source": {"$eq": filename}})
        if redis_client:
            redis_client.srem(INDEXED_FILES_KEY, filename)
        return f"Successfully deleted '{filename}' from index.", gr.update(choices=get_indexed_files(), value=None)
    except Exception as e:
        return f"Error deleting '{filename}': {str(e)}", gr.update(choices=get_indexed_files())

def setup_rag_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.25)
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "Answer briefly about the Cheezious brand. "
        "Do not answer unrelated questions. "
        "Use only provided context. "
        "If unknown, say you don't know. "
        "Max 3 sentences.\n\nContext: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str):
        return RedisChatMessageHistory(session_id, url=REDIS_URL)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    return conversational_rag_chain

rag_chain = setup_rag_chain()

def clear_memory(session_id="default"):
    try:
        history = RedisChatMessageHistory(session_id, url=REDIS_URL)
        history.clear()
    except Exception:
        pass
    return []

import traceback

def build_ui():
    with gr.Blocks(title="Iqrar RAG System") as demo:
        gr.Markdown("# Iqrar RAG Interface")
        with gr.Row():
            # Left Column: Document Management
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📄 Document Management")
                
                with gr.Group():
                    file_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                    upload_btn = gr.Button("Index Document", variant="primary")
                    
                gr.Markdown("<br/>")
                
                with gr.Group():
                    gr.Markdown("**Indexed Files List**")
                    indexed_files_display = gr.Markdown("No files indexed yet.")
                    refresh_btn = gr.Button("↻ Refresh List", size="sm")
                    
                gr.Markdown("<br/>")
                
                with gr.Group():
                    delete_filename_input = gr.Textbox(label="File Name to Delete", placeholder="e.g. document.pdf")
                    delete_btn = gr.Button("Delete Document", variant="stop")
                    
                status_box = gr.Textbox(label="Status Logs", interactive=False, lines=3)
                
                def update_display():
                    files = get_indexed_files()
                    if not files:
                        return "No files indexed yet."
                    return "\n".join([f"- {f}" for f in files])

                # Callbacks
                upload_btn.click(
                    fn=add_document, 
                    inputs=file_upload, 
                    outputs=[status_box]
                ).then(fn=update_display, outputs=indexed_files_display)
                
                def handle_delete(filename):
                    print(f"[LOG] Attempting to delete: {filename}")
                    if not filename:
                        return "No filename provided."
                    if not redis_client or not redis_client.sismember(INDEXED_FILES_KEY, filename):
                        return f"File '{filename}' does not exist in the index."
                    
                    try:
                        pc = init_pinecone()
                        index = pc.Index(INDEX_NAME)
                        index.delete(filter={"source": {"$eq": filename}})
                        redis_client.srem(INDEXED_FILES_KEY, filename)
                        return f"Successfully deleted '{filename}' from index."
                    except Exception as e:
                        traceback.print_exc()
                        return f"Error deleting '{filename}': {str(e)}"
                
                delete_btn.click(
                    fn=handle_delete,
                    inputs=delete_filename_input,
                    outputs=[status_box]
                ).then(fn=update_display, outputs=indexed_files_display)
                
                refresh_btn.click(fn=update_display, outputs=indexed_files_display)
                demo.load(fn=update_display, outputs=indexed_files_display)

            # Right Column: Chatbot
            with gr.Column(scale=3):
                gr.Markdown("### 💬 Chat Assistant")
                chatbot = gr.Chatbot(height=600)
                msg = gr.Textbox(label="Your Question", placeholder="Type your question here and press enter...")
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

                def user_message(user_msg, history):
                    print(f"[LOG] User message received: {user_msg}")
                    print(f"[LOG] Current history type: {type(history)}, content: {history}")
                    history = history or []
                    # Gradio 6+ standard format
                    history.append({"role": "user", "content": user_msg})
                    return "", history
                
                def bot_message(history):
                    if not history:
                        return history
                    
                    try:
                        user_msg = history[-1]["content"]
                        # Gradio 6 Multimodal parsing
                        if isinstance(user_msg, list):
                            text_parts = []
                            for item in user_msg:
                                if isinstance(item, dict) and 'text' in item:
                                    text_parts.append(item['text'])
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            user_msg = " ".join(text_parts)
                        elif isinstance(user_msg, tuple):
                            user_msg = str(user_msg[0])
                        elif not isinstance(user_msg, str):
                            user_msg = str(user_msg)

                        print(f"[LOG] Processing parsed message: {user_msg}")
                        response = rag_chain.invoke(
                            {"input": user_msg},
                            config={"configurable": {"session_id": "default"}}
                        )
                        print(f"[LOG] RAG Chain Output: {response}")
                        history.append({"role": "assistant", "content": response["answer"]})
                    except Exception as e:
                        print("[ERROR] Exception in RAG chain:")
                        traceback.print_exc()
                        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    
                    return history

                msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot_message, chatbot, chatbot
                )
                submit_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot_message, chatbot, chatbot
                )
                
                clear_btn.click(fn=clear_memory, inputs=[], outputs=[chatbot], queue=False)

    return demo

if __name__ == "__main__":
    init_pinecone()
    ui = build_ui()
    ui.launch(theme=gr.themes.Base())
