import os
import shutil
import hashlib

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class Embedding:

  def __init__(self, data=None):
    """Initializes the Embedding class with the chunked data.

    Args:
        data (list, optional): A list of chunked documents (e.g., output from
          your Chunker.markdown_chunking method). Not required if you only
          intend to call load_vector_store().
    """
    self.data = data

  
  def _make_doc_id(self, doc) -> str:
      """Deterministic ID for a Document: same source + text => same ID.

      This is what lets "append" mode tell new documents apart from ones
      already stored, without needing to compare embeddings.
      """
      metadata = getattr(doc, "metadata", None) or {}
      source = metadata.get("source", "")
      payload = f"{source}::{doc.page_content}".encode("utf-8")
      return hashlib.sha256(payload).hexdigest()


  def _dedupe_documents(self, documents):
      """Drop exact duplicate documents (same computed ID) within a batch."""
      seen = set()
      docs, ids = [], []
      for doc in documents:
          doc_id = self._make_doc_id(doc)
          if doc_id in seen:
              continue
          seen.add(doc_id)
          docs.append(doc)
          ids.append(doc_id)
      return docs, ids


  def sbert_embeder(
      self,
      model_name="all-MiniLM-L6-v2",
      persist_directory="./vectordb",
      mode="append",
      collection_name="langchain",
  ):
      """Generates SBERT embeddings and stores them in a persistent ChromaDB database.

      Args:
          model_name (str): The Hugging Face SBERT model name to use for
            embeddings (default is "all-MiniLM-L6-v2").
          persist_directory (str): The local directory path where ChromaDB will
            save the database files.
          mode (str): "append" to add only documents from self.data that
            aren't already in the store (dedup by content hash). "recreate"
            to wipe any existing database at persist_directory and rebuild
            it from scratch using self.data.
          collection_name (str): Name of the Chroma collection to use.

      Returns:
          Chroma: A LangChain Chroma vector store instance.

      Raises:
          ValueError: If mode isn't "append"/"recreate", or self.data is empty.
      """
      if mode not in ("append", "recreate"):
          raise ValueError(f"mode must be 'append' or 'recreate', got {mode!r}")
      if not self.data:
          raise ValueError("self.data is empty - nothing to embed.")

      embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
      db_exists = os.path.isdir(persist_directory) and bool(os.listdir(persist_directory))

      # ---- RECREATE: wipe whatever is there, then build fresh ----
      if mode == "recreate":
          if db_exists:
              print("Recreate mode: removing existing DB at %s", persist_directory)
              shutil.rmtree(persist_directory, ignore_errors=True)
          os.makedirs(persist_directory, exist_ok=True)

          docs, ids = self._dedupe_documents(self.data)
          vector_store = Chroma.from_documents(
              documents=docs,
              embedding=embeddings_model,
              ids=ids,
              persist_directory=persist_directory,
              collection_name=collection_name,
          )
          return vector_store

      # ---- APPEND, but no DB exists yet: identical to a fresh create ----
      if not db_exists:
          os.makedirs(persist_directory, exist_ok=True)
          docs, ids = self._dedupe_documents(self.data)
          vector_store = Chroma.from_documents(
              documents=docs,
              embedding=embeddings_model,
              ids=ids,
              persist_directory=persist_directory,
              collection_name=collection_name,
          )
          return vector_store

      # ---- APPEND, DB exists: only insert documents not already present ----
      vector_store = Chroma(
          embedding_function=embeddings_model,
          persist_directory=persist_directory,
          collection_name=collection_name,
      )

      docs, ids = self._dedupe_documents(self.data)
      existing = vector_store.get(ids=ids)          # only fetches these specific IDs
      existing_ids = set(existing.get("ids", []))

      new_docs = [d for d, i in zip(docs, ids) if i not in existing_ids]
      new_ids = [i for i in ids if i not in existing_ids]

      if new_docs:
          vector_store.add_documents(documents=new_docs, ids=new_ids)
          
      return vector_store