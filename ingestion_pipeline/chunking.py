from read_data import FileLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class DocumentChunker:
    """
    Handles chunking of loaded document data (with metadata) into smaller,
    manageable segments optimized for LLM ingestion.
    """

    def __init__(self, loaded_data: list):
        """
        Initializes the chunker with loaded data from FileLoader.load_pdf().

        Args:
            loaded_data (list): A list of objects or dictionaries containing
                                'page_content' and 'metadata'.
        """
        self.loaded_data = loaded_data

    def markdown_chunking(self, chunk_size: int = 512, chunk_overlap: int = 64, number_of_header: int = 3) -> list:
        """
        Splits the loaded data containing Markdown text and metadata into
        semantic chunks.

        Args:
            chunk_size (int): Maximum size of each chunk (default is 512).
            chunk_overlap (int): The amount of overlap between consecutive
                                 chunks (default is 64).
            number_of_header (int): The amount of headers to split on (default is 3).

        Returns:
            list: A list of chunked documents with updated metadata.
        """

        headers_to_split_on = [("#"*(no+1), f"Header {no}") for no in range(number_of_header)]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # Secondary splitter to ensure chunks don't exceed the max chunk_size limit
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunked_documents = []

        for doc in self.loaded_data:
            # Extract content and existing metadata depending on object or dict structure
            content = doc.page_content
            metadata = doc.metadata

            # Step 1: Split by markdown headers
            header_splits = markdown_splitter.split_text(content)

            # Step 2: Further split large header sections by character length
            final_splits = text_splitter.split_documents(header_splits)

            # Step 3: Merge back the original file-level metadata into each chunk
            for chunk in final_splits:
                chunk.metadata.update(metadata)
                chunked_documents.append(chunk)

        return chunked_documents

if __name__ == "__main__":
    loader = FileLoader("../doc")
    docs = loader.load_pdf()
    chunker = DocumentChunker(loaded_data=docs)
    chunked = chunker.markdown_chunking()
    print(chunked[100])