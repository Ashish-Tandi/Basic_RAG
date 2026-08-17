"""
Streamlit UI for the RAG pipeline (Groq-powered).

Flow:
  1. Upload PDFs -> FileLoader loads + extracts tables as Markdown
  2. Chunker splits by Markdown headers, then by character length
  3. Embedding builds/persists a Chroma vector store (SBERT embeddings)
  4. Retriever performs similarity search over the vector store
  5. RAGQueryEngine sends retrieved chunks + question to Groq for an answer

Chats:
  - "New chat" starts a fresh conversation with no memory of other chats.
  - Every chat is saved to disk (CHAT_SESSIONS_FILE) as you use it, and the
    model's own conversational memory is saved to a local sqlite database
    (CHAT_MEMORY_DB) via a LangGraph checkpointer -- so reopening the app
    later shows the same sidebar of past chats, and picking one up again
    still lets the model resolve follow-up questions against that chat's
    earlier turns.

Run with:
    streamlit run app.py
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from embeddings.embeder import Embedding
from retrieval_pipeline.retriever import Retriever
from generation.llm_query import RAGQueryEngine
from api import GROQ_API


PERSIST_DIR = "./vectordb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_SESSIONS_FILE = "./chat_sessions/chat_sessions.json"  # sidebar list + rendered Q/A/sources
CHAT_MEMORY_DB = "./chat_memory/chat_memory.db"  # the model's own LangGraph thread memory


# ---------------------------------------------------------------------------
# Persistence: chat sessions (sidebar list + display history)
# ---------------------------------------------------------------------------
def load_sessions() -> dict:
    """Reads the sidebar chat list + saved turns from disk."""
    if os.path.exists(CHAT_SESSIONS_FILE):
        try:
            with open(CHAT_SESSIONS_FILE, "r") as f:
                data = json.load(f)
            data.setdefault("order", [])
            data.setdefault("sessions", {})
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"order": [], "sessions": {}}


def save_sessions(data: dict) -> None:
    """Writes the chat list atomically so a crash mid-write can't corrupt it."""
    tmp_path = CHAT_SESSIONS_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, CHAT_SESSIONS_FILE)


def make_title(question: str, max_len: int = 50) -> str:
    """Derives a short sidebar label from a chat's first question."""
    title = question.strip().replace("\n", " ")
    return title if len(title) <= max_len else title[: max_len - 1].rstrip() + "…"


def serialize_sources(results) -> list:
    """Converts (Document, score) pairs into JSON-safe dicts for storage."""
    return [
        {"content": doc.page_content, "metadata": doc.metadata, "score": score}
        for doc, score in results
    ]


def touch_session(sessions_data: dict, thread_id: str) -> None:
    """Moves a chat to the front of the sidebar (most recently used first)."""
    if thread_id in sessions_data["order"]:
        sessions_data["order"].remove(thread_id)
    sessions_data["order"].insert(0, thread_id)


# ---------------------------------------------------------------------------
# Cached resources (shared across reruns / browser sessions for this process)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_checkpointer() -> SqliteSaver:
    """The model's conversational memory store -- one sqlite file, persisted
    across app restarts, isolated per thread_id (i.e. per chat)."""
    conn = sqlite3.connect(CHAT_MEMORY_DB, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def get_vector_store():
    """Loads the persisted vector store into session state (once)."""
    if st.session_state.vector_store is None:
        with st.spinner("Loading vector store..."):
            st.session_state.vector_store = Embedding.load_vector_store(
                model_name=EMBEDDING_MODEL, persist_directory=PERSIST_DIR
            )
    return st.session_state.vector_store


def get_engine(top_k: int, api_key: str) -> RAGQueryEngine:
    """Builds a RAGQueryEngine wired to the persisted model-memory store."""
    retriever = Retriever(get_vector_store(), k=top_k)
    return RAGQueryEngine(
        retriever,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        checkpointer=get_checkpointer(),
    )


# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "active_thread_id" not in st.session_state:
    # None = an unsaved "new chat" that doesn't exist on disk yet.
    st.session_state.active_thread_id = None
if "index_built" not in st.session_state:
    # Detect an already-persisted store from a previous run
    st.session_state.index_built = os.path.isdir(PERSIST_DIR) and bool(
        os.listdir(PERSIST_DIR)
    )

sessions_data = load_sessions()

# ---------------------------------------------------------------------------
# Sidebar: configuration + chat list
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    groq_api_key = (
        st.text_input(
            "Groq API key",
            type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Stored only for this session, never written to disk.",
        )
        or GROQ_API
    )

    st.divider()

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.active_thread_id = None
        st.rerun()

    st.subheader("Chats")
    if not sessions_data["order"]:
        st.caption("No chats yet -- ask a question to start one.")
    for thread_id in sessions_data["order"]:
        meta = sessions_data["sessions"][thread_id]
        is_active = thread_id == st.session_state.active_thread_id
        label = ("🟢 " if is_active else "") + (meta["title"] or "New chat")

        col_select, col_delete = st.columns([5, 1])
        if col_select.button(label, key=f"select_{thread_id}", use_container_width=True):
            st.session_state.active_thread_id = thread_id
            st.rerun()
        if col_delete.button("🗑️", key=f"delete_{thread_id}"):
            sessions_data["order"].remove(thread_id)
            del sessions_data["sessions"][thread_id]
            save_sessions(sessions_data)
            # Also wipe the model's memory for this chat, not just the
            # sidebar entry, so a reused thread_id can never resurrect it.
            get_checkpointer().delete_thread(thread_id)
            if st.session_state.active_thread_id == thread_id:
                st.session_state.active_thread_id = None
            st.rerun()

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
    top_k = 4  # Default retrieval chunk count

    # A fresh "New chat" only gets a real thread_id -- and only appears in
    # the sidebar -- once its first question is actually asked.
    thread_id = st.session_state.active_thread_id or str(uuid.uuid4())

    engine = get_engine(api_key=groq_api_key, top_k=top_k)

    with st.spinner("Searching and generating answer..."):
        result = engine.ask(question, thread_id=thread_id, k=top_k)

    if thread_id not in sessions_data["sessions"]:
        sessions_data["sessions"][thread_id] = {
            "title": make_title(question),
            "created_at": datetime.now().isoformat(),
            "turns": [],
        }
    sessions_data["sessions"][thread_id]["turns"].append(
        {
            "question": question,
            "answer": result["answer"],
            "sources": serialize_sources(result["sources"]),
        }
    )
    touch_session(sessions_data, thread_id)
    save_sessions(sessions_data)

    st.session_state.active_thread_id = thread_id
    st.rerun()

# Render the active chat's history (persisted, so it survives reloads/restarts)
active_id = st.session_state.active_thread_id
if active_id is None or active_id not in sessions_data["sessions"]:
    st.info("Start a new conversation by typing a question below.")
else:
    for turn in sessions_data["sessions"][active_id]["turns"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["sources"]:
                with st.expander(f"📚 Sources ({len(turn['sources'])})"):
                    for i, src in enumerate(turn["sources"], start=1):
                        label = RAGQueryEngine._format_source_label(src["metadata"])
                        score = src["score"]
                        score_str = f" (distance: {score:.3f})" if score is not None else ""
                        st.markdown(f"**[{i}] {label}**{score_str}")
                        content = src["content"]
                        st.text(content[:400] + ("..." if len(content) > 400 else ""))
                        st.divider()