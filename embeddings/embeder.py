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

  def sbert_embeder(self, model_name="all-MiniLM-L6-v2", persist_directory="./vectordb"):
    """Generates SBERT embeddings and stores them in a persistent ChromaDB database.

    Args:
        model_name (str): The Hugging Face SBERT model name to use for
          embeddings (default is "BAAI/bge-m3").
        persist_directory (str): The local directory path where ChromaDB will
          save the database files.

    Returns:
        Chroma: A LangChain Chroma vector store instance.
    """
    # Initialize the SBERT embedding model via LangChain
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    # Create and persist the ChromaDB vector store automatically from documents
    vector_store = Chroma.from_documents(
        documents=self.data,
        embedding=embeddings_model,
        persist_directory=persist_directory,
    )

    return vector_store

  @staticmethod
  def load_vector_store(model_name="all-MiniLM-L6-v2", persist_directory="./vectordb"):
    """Loads an existing persisted ChromaDB vector store without re-embedding.

    Use this in the search/query app so you don't rebuild embeddings on
    every run -- only call sbert_embeder() once during ingestion.

    Args:
        model_name (str): Must match the model used when the store was built.
        persist_directory (str): Path to the existing persisted ChromaDB.

    Returns:
        Chroma: A LangChain Chroma vector store instance backed by the
          existing on-disk data.
    """
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    vector_store = Chroma(
        embedding_function=embeddings_model,
        persist_directory=persist_directory,
    )

    return vector_store