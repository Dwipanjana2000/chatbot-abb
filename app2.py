import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import google.generativeai as genai
from langchain_core.embeddings import Embeddings
# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Risk & Compliance chatbot", layout="wide")
st.title("Risk & Compliance")
INDEX_PATH = "faiss_index"
class PrecomputedEmbeddings(Embeddings):
    """Use precomputed embeddings inside FAISS."""
    def __init__(self, embedding_dict):
        self.embedding_dict = embedding_dict

    def embed_documents(self, texts):
        return [self.embedding_dict[t] for t in texts]

    def embed_query(self, text):
        return self.embedding_dict.get(text, [0.0] * 768)  # default vector

def ocr_pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def get_gemini_embedding(text):
    """Generate embedding from Gemini."""
    model = "models/embedding-001"
    result = genai.embed_content(model=model, content=text)
    return result["embedding"]

def gemini_chat(prompt):
    """Generate text from Gemini LLM."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# ===============================
# SESSION STATE
# ===============================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embedded" not in st.session_state:
    st.session_state.embedded = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# FILE UPLOAD
# ===============================
uploaded_files = st.file_uploader(
    "Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

# ===============================
# PROCESS FILES
# ===============================
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

        # Precompute embeddings
        st.write("🔹 Creating FAISS index...")
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embedding_map = {t: get_gemini_embedding(t) for t in texts}

        # Build FAISS with precomputed embeddings
        fake_embedder = PrecomputedEmbeddings(embedding_map)
        vectorstore = FAISS.from_texts(texts, embedding=fake_embedder, metadatas=metadatas)

        # Save FAISS index
        vectorstore.save_local(INDEX_PATH)
        st.session_state.vectorstore = vectorstore
        st.session_state.embedded = True

        st.success("✅ Files processed and indexed in FAISS with Gemini embeddings!")

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")

# ===============================
# LOAD EXISTING INDEX
# ===============================
if st.session_state.vectorstore is None and os.path.exists(INDEX_PATH):
    # Placeholder embeddings since we're not recomputing at query time
    from langchain_community.embeddings import FakeEmbeddings
    st.session_state.vectorstore = FAISS.load_local(
        INDEX_PATH, FakeEmbeddings(size=768), allow_dangerous_deserialization=True
    )

# ===============================
# CHAT UI
# ===============================
if st.session_state.vectorstore:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask something about your uploaded files..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant docs
        docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
        context = "\n".join([d.page_content for d in docs])

        # Generate answer from Gemini
        full_prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {prompt}"
        answer = gemini_chat(full_prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)
            sources = "\n".join(f"- {doc.metadata.get('source', 'Unknown')}" for doc in docs)
            st.markdown(f"**Sources:**\n{sources}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
