import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

class FileLoader:
    """
    Read and shows the contents of files given in file_location.

    Args:
         file_location (str): path to file
    """

    def __init__(self, file_location: str):
        self.file_location = file_location

    def load_pdf(self) -> list[Document]:
        """
        Loads text, extracts tables (in Markdown format) from multiple PDF
        file directly using LangChain's PyMuPDFLoader.

        Returns:
            documents (list): list of content of pages in documents
        """
        documents = []
        for filename in os.listdir(self.file_location):
            file_path = os.path.join(self.file_location, filename)

            if os.path.isfile(file_path):
                print(f"Loading content from: {file_path} using PyMuPDFLoader..")
                loader = PyMuPDFLoader(
                    file_path=file_path,
                    extract_tables="markdown",
                    # extract_images=True
                )

                documents.extend(loader.load())
        return documents

    @staticmethod
    def show_page(documents: list[Document], page_number: int) -> None:
        """
        For given page number, it prints content of page(given page number's)
        from the loaded documents.
        """
        if not documents:
            print("No documents available to display.")
            return

        # Pick a random document/page
        page = documents[page_number - 1]
        page_number = page.metadata.get("page", "Unknown")

        print(f"\n Selected Page ({page_number + 1 }) ===")
        print("-" * 50)
        print(page.page_content)
        print("-" * 50)
        print("Metadata:", page.metadata)
