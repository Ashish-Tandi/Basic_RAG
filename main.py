"""
CLI ingestion script: loads PDFs from ./doc, chunks them, and builds/persists
a Chroma vector store. Run this once (or whenever your source documents
change) before using app.py to search/chat.

Usage:
    python main.py
"""

from ingestion_pipeline.read_data import FileLoader
from ingestion_pipeline.chunking import Chunker
from embeddings.embeder import Embedding


if __name__ == "__main__":
    loader = FileLoader("doc")
    docs = loader.load_pdf()
    print(f"Loaded {len(docs)} page(s) from ./doc")

    chunker = Chunker(loaded_data=docs)
    chunked = chunker.markdown_chunking(chunk_size=384, chunk_overlap=32, number_of_header=3)
    print(f"Created {len(chunked)} chunk(s)")

    embeder = Embedding(chunked)
    vector_store = embeder.sbert_embeder(persist_directory="./vectordb", mode="recreate")
    print("Vector store built and persisted to ./vectordb")

    first_chunk_text = chunked[0].page_content
    print("\nOriginal Text Chunk Preview:")
    print(f"--> {first_chunk_text[:150]}...\n")

    single_vector = vector_store._embedding_function.embed_query(first_chunk_text)

    print("--- Vector Space Properties ---")
    print(f"Dimensionality (Length of vector): {len(single_vector)}")
    print(f"Data type of vector elements: {type(single_vector[0])}")
    print(f"First 10 numbers in the vector space: {single_vector[:10]}")

    print("\nIndex ready. Run 'streamlit run app.py' to search and chat over these documents.")