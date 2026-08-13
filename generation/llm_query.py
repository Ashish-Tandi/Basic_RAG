import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from retrieval_pipeline.retriever import Retriever


SYSTEM_PROMPT = (
    "You are an expert optical engineering assistant that answers questions "
    "using ONLY the provided context excerpts from the user's documents. "
    "Your answers must be detailed and technically rigorous, appropriate for "
    "an audience with a background in optics, photonics, or optical engineering. "
    "Where relevant, include specific technical details such as wavelengths, "
    "refractive indices, numerical aperture, focal length, aberrations, "
    "coatings, tolerances, units, and standard formulas — but only if these "
    "are present in or directly derivable from the provided context. "
    "Do not use outside knowledge and do not guess. "
    "If the answer is not contained in the context, say so clearly rather "
    "than speculating. "
    "When you use information from an excerpt, cite it inline using its "
    "label, e.g. [1], [2]."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)


class RAGQueryEngine:
    """
    Connects a Retriever (vector search) to a chat model via a LangChain
    LCEL chain (prompt | model) to produce grounded answers with cited
    sources.
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str = "Groq",
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
    ):
        """
        Args:
            retriever (Retriever): Vector store retriever instance.
            model (str): Chat model to use for generating answers. Groq
                         model names (e.g. "llama-3.3-70b-versatile"),
                         not "Groq" itself, since this is passed straight
                         through to the OpenAI-compatible /chat/completions
                         endpoint.
            api_key (str, optional): API key. Falls back to the
                                      GROQ_API_KEY environment variable.
            base_url (str, optional): Override the API endpoint to use any
                                       OpenAI-compatible provider (e.g. Gemini's
                                       https://generativelanguage.googleapis.com/v1beta/openai/).
        """
        self.retriever = retriever
        self.model_name = model

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "No API key found. Set the GROQ_API_KEY environment "
                "variable or pass api_key explicitly."
            )

        self.llm = ChatOpenAI(
            model=model,
            api_key=key,
            base_url=base_url,
        )

    @staticmethod
    def _format_source_label(metadata: dict) -> str:
        """Builds a short human-readable label for a chunk's origin."""
        source = metadata.get("source", "unknown source")
        page = metadata.get("page")
        headers = [v for k, v in metadata.items() if k.startswith("Header") and v]
        label = os.path.basename(str(source))
        if page is not None:
            label += f", page {page + 1 if isinstance(page, int) else page}"
        if headers:
            label += f" — {' > '.join(headers)}"
        return label

    def _build_context(self, results: List[Tuple[Document, Optional[float]]]) -> str:
        """Turns (Document, score) pairs into a numbered context block."""
        blocks = []
        for i, (doc, _score) in enumerate(results, start=1):
            label = self._format_source_label(doc.metadata)
            blocks.append(f"[{i}] ({label})\n{doc.page_content}")
        return "\n\n---\n\n".join(blocks)

    def ask(
        self,
        question: str,
        k: int = 4,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Runs the full RAG flow: retrieve relevant chunks, then ask the LLM
        to answer the question grounded in those chunks.

        Args:
            question (str): The user's natural language question.
            k (int): Number of chunks to retrieve.
            temperature (float): LLM sampling temperature.
            max_tokens (int): Max tokens for the generated answer.

        Returns:
            dict: {
                "answer": str,
                "sources": list[tuple[Document, float]]  # what was retrieved
            }
        """
        results = self.retriever.search(question, k=k)

        if not results:
            return {
                "answer": "No relevant content was found in the indexed documents.",
                "sources": [],
            }

        context = self._build_context(results)

        try:
            # .bind() applies per-call sampling params without mutating the
            # shared self.llm, so concurrent calls with different
            # temperature/max_tokens don't interfere with each other.
            bound_chain = _PROMPT | self.llm.bind(
                temperature=temperature, max_tokens=max_tokens
            )
            response = bound_chain.invoke({"context": context, "question": question})
        except Exception as e:
            raise RuntimeError(f"Failed to get answer from model: {e}") from e

        return {"answer": response.content, "sources": results}