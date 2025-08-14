import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import numpy as np
import pickle

# Load env vars
load_dotenv()

st.set_page_config(page_title="Ollama File QA Chatbot", layout="wide")
st.title("📄 Persistent File QA Chatbot (Ollama + FAISS)")

INDEX_PATH = "faiss_index"

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embedded" not in st.session_state:
    st.session_state.embedded = False

def ocr_pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

uploaded_files = st.file_uploader(
    "Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

# ---- PROCESS FILES ----
if uploaded_files and not st.session_state.embedded:
    try:
        documents = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            if file.name.lower().endswith(".pdf"):
                try:
                    loader = PyMuPDFLoader(tmp_path)
                    docs = loader.load()
                    if not any(d.page_content.strip() for d in docs):
                        raise ValueError("Empty PDF content")
                except Exception:
                    st.warning(f"⚠️ OCR fallback for {file.name}")
                    ocr_text = ocr_pdf_to_text(tmp_path)
                    docs = [Document(page_content=ocr_text, metadata={"source": file.name})]

            elif file.name.lower().endswith(".docx"):
                loader = Docx2txtLoader(tmp_path)
                docs = loader.load()
            else:
                loader = TextLoader(tmp_path)
                docs = loader.load()

            for d in docs:
                d.metadata["source"] = file.name
            documents.extend(docs)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]
        st.write(f"📄 Total chunks after cleaning: {len(chunks)}")
        if not chunks:
            st.error("No valid chunks found.")
            st.stop()

        # Embeddings
        embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        st.write("🔹 Creating FAISS index...")
        vectorstore = FAISS.from_documents(chunks, embeddings_model)

        # Save FAISS index
        vectorstore.save_local(INDEX_PATH)
        st.session_state.vectorstore = vectorstore
        st.session_state.embedded = True

        st.success("✅ Files processed and indexed in FAISS!")

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")

# ---- LOAD EXISTING FAISS INDEX ----
if st.session_state.vectorstore is None and os.path.exists(INDEX_PATH):
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    st.session_state.vectorstore = FAISS.load_local(INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)

# ---- QA ----
# ---- Chat UI ----
if st.session_state.vectorstore:
    llm = OllamaLLM(model="llama3.2")
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something about your uploaded files..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run QA
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"]

            st.markdown(answer)
            # Show sources
            if "source_documents" in result:
                sources = "\n".join(f"- {doc.metadata.get('source', 'Unknown')}" for doc in result["source_documents"])
                st.markdown(f"**Sources:**\n{sources}")

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": answer})
