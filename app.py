"""
Streamlit UI for the RAG pipeline (Groq-powered).

Flow:
  1. Upload PDFs -> FileLoader loads + extracts tables as Markdown
  2. Chunker splits by Markdown headers, then by character length
  3. Embedding builds/persists a Chroma vector store (SBERT embeddings)
  4. Retriever performs similarity search over the vector store
  5. RAGQueryEngine sends retrieved chunks + question to Groq for an answer

Run with:
    streamlit run app.py
"""

import os
import shutil
import tempfile

import streamlit as st

from ingestion_pipeline.read_data import FileLoader
from ingestion_pipeline.chunking import Chunker
from embeddings.embeder import Embedding
from retrieval_pipeline.retriever import Retriever
from generation.llm_query import RAGQueryEngine


PERSIST_DIR = "./vectordb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"question", "answer", "sources"}
if "index_built" not in st.session_state:
    # Detect an already-persisted store from a previous run
    st.session_state.index_built = os.path.isdir(PERSIST_DIR) and bool(
        os.listdir(PERSIST_DIR)
    )


def get_vector_store():
    """Loads the persisted vector store into session state (once)."""
    if st.session_state.vector_store is None:
        with st.spinner("Loading vector store..."):
            st.session_state.vector_store = Embedding.load_vector_store(
                model_name=EMBEDDING_MODEL, persist_directory=PERSIST_DIR
            )
    return st.session_state.vector_store


# ---------------------------------------------------------------------------
# Sidebar: configuration + index building
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Stored only for this session, never written to disk.",
    )
    
# ---------------------------------------------------------------------------
# Main area: search / chat
# ---------------------------------------------------------------------------
st.title("🔎 Document RAG Search")
st.caption("Ask questions grounded in your uploaded documents.")

if not st.session_state.index_built:
    st.warning("No index found yet. Build one from the sidebar first.")
    st.stop()

question = st.chat_input("Ask a question about your documents...")

if question:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    top_k = 4  # Default retrieval chunk count
    vector_store = get_vector_store()
    retriever = Retriever(vector_store, k=top_k)
    
    # Configure the RAGQueryEngine to point to Groq's endpoint
    engine = RAGQueryEngine(
        retriever, 
        api_key=groq_api_key, 
        base_url="https://api.groq.com/openai/v1"
    )

    with st.spinner("Searching and generating answer..."):
        result = engine.ask(question, k=top_k)

    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        }
    )

# Render chat history
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["sources"]:
            with st.expander(f"📚 Sources ({len(turn['sources'])})"):
                for i, (doc, score) in enumerate(turn["sources"], start=1):
                    label = os.path.basename(str(doc.metadata.get("source", "unknown")))
                    page = doc.metadata.get("page")
                    page_str = f", page {page + 1}" if isinstance(page, int) else ""
                    score_str = f" (distance: {score:.3f})" if score is not None else ""
                    st.markdown(f"**[{i}] {label}{page_str}**{score_str}")
                    st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
                    st.divider()