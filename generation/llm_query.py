import os
from typing import List, Optional, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from retrieval_pipeline.retriever import Retriever


CONDENSE_SYSTEM_PROMPT = (
    "Given a chat history and a follow-up question, rewrite the follow-up "
    "into a standalone question that can be understood with no other "
    "context — resolve pronouns and implicit references (e.g. 'it', 'that "
    "lens', 'the previous one') using the chat history. "
    "Do NOT answer the question. Output ONLY the rewritten question, "
    "nothing else. If the follow-up is already standalone, return it "
    "unchanged."
)

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
    "label, e.g. [1], [2]. "
    "Use the earlier turns of this conversation to resolve follow-up "
    "questions (e.g. pronouns, 'the previous answer', 'that lens'), but "
    "still ground every factual claim in the provided context excerpts."
    "Only If the user asks for a comparison between multiple elements, systems, "
    "or concepts, provide a detailed technical comparison in the text, "
)


class RAGState(TypedDict):
    """Graph state: running chat history + the latest turn's retrieval."""

    messages: Annotated[List[BaseMessage], add_messages]
    standalone_question: str
    sources: List[Tuple[Document, Optional[float]]]
    k: int


class RAGQueryEngine:
    """
    Connects a Retriever (vector search) to a chat model via a LangGraph
    StateGraph (retrieve -> generate) to produce grounded, cited answers
    that also carry conversation memory across turns.

    Memory works via a LangGraph checkpointer keyed by `thread_id`: each
    call to `ask(question, thread_id=...)` loads that thread's prior
    messages, appends the new question, and persists the updated history
    after the answer is generated — so follow-up questions ("what about
    its NA?", "and the coating?") resolve against earlier turns.
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        checkpointer: Optional[BaseCheckpointSaver] = None,
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
            checkpointer (BaseCheckpointSaver, optional): Where conversation
                         state is persisted. Defaults to an in-process
                         `InMemorySaver` (lost on restart). Pass e.g. a
                         `SqliteSaver`/`PostgresSaver` for durable, multi-
                         process memory.
        """
        self.retriever = retriever
        self.model_name = model

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "No API key found. Set the GROQ_API_KEY environment "
                "variable or pass api_key explicitly."
            )

        self.llm = ChatOpenAI(model=model, api_key=key, base_url=base_url)
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(RAGState)
        builder.add_node("condense", self._condense_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("generate", self._generate_node)
        builder.add_edge(START, "condense")
        builder.add_edge("condense", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _condense_node(self, state: RAGState) -> dict:
        """
        Rewrites the latest question into a standalone query for retrieval.

        Without this, a follow-up like "what about its focal length?" gets
        searched against the vector store literally — with no subject noun
        to match against, retrieval returns whatever loosely matches
        "focal length" in isolation, not chunks about the specific item
        from earlier turns. The LLM then answers grounded in those
        unrelated chunks, which looks like it "forgot" the conversation.
        This step only affects what's searched for; the user's original
        wording is still what's sent to the model in `_generate_node`.
        """
        messages = state["messages"]
        question = messages[-1].content
        history = messages[:-1]

        if not history:
            # First turn of the thread: nothing to resolve against yet.
            return {"standalone_question": question}

        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history
        )
        condense_input = [
            SystemMessage(content=CONDENSE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Chat history:\n{history_text}\n\n"
                    f"Follow-up question: {question}\n\n"
                    f"Standalone question:"
                )
            ),
        ]

        try:
            # temperature=0 regardless of the answer's configured
            # temperature — query rewriting should be deterministic, not
            # creative. Falls back to the raw question if this call fails
            # for any reason, rather than breaking retrieval entirely.
            response = self.llm.bind(temperature=0.0).invoke(condense_input)
            standalone_question = response.content.strip() or question
        except Exception:
            standalone_question = question

        return {"standalone_question": standalone_question}

    def _retrieve_node(self, state: RAGState) -> dict:
        """Retrieves chunks for the (possibly rewritten) standalone question."""
        results = self.retriever.search(
            state["standalone_question"], k=state["k"]
        )
        return {"sources": results}

    def _generate_node(self, state: RAGState) -> dict:
        """Builds grounded context and asks the LLM, given full chat history."""
        results = state["sources"]

        if not results:
            answer = "No relevant content was found in the indexed documents."
            return {"messages": [AIMessage(content=answer)]}

        context = self._build_context(results)
        context_message = HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {state['messages'][-1].content}"
        )
        # Full running history (minus the raw last question, which is
        # replaced by the context-augmented version above) plus a fixed
        # system prompt, so the model sees prior turns for follow-ups.
        history = state["messages"][:-1]
        llm_input = [SystemMessage(content=SYSTEM_PROMPT), *history, context_message]

        try:
            response = self.llm.invoke(llm_input)
        except Exception as e:
            raise RuntimeError(f"Failed to get answer from model: {e}") from e

        return {"messages": [AIMessage(content=response.content)]}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        thread_id: str = "default",
        k: int = 4,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Runs the full RAG flow for one conversation turn: retrieve relevant
        chunks, then ask the LLM to answer grounded in those chunks and in
        the prior turns of this thread.

        Args:
            question (str): The user's natural language question.
            thread_id (str): Conversation identifier. Reuse the same value
                              across calls to keep memory of earlier turns;
                              use a new value to start a fresh conversation.
            k (int): Number of chunks to retrieve.
            temperature (float): LLM sampling temperature.
            max_tokens (int): Max tokens for the generated answer.

        Returns:
            dict: {
                "answer": str,
                "sources": list[tuple[Document, float]],  # this turn's retrieval
                "thread_id": str,
            }
        """
        # .bind() applies per-call sampling params without mutating the
        # shared self.llm, so concurrent calls with different
        # temperature/max_tokens don't interfere with each other.
        bound_llm = self.llm.bind(temperature=temperature, max_tokens=max_tokens)
        original_llm, self.llm = self.llm, bound_llm
        try:
            config = {"configurable": {"thread_id": thread_id}}
            final_state = self.graph.invoke(
                {"messages": [HumanMessage(content=question)], "k": k},
                config=config,
            )
        finally:
            self.llm = original_llm

        return {
            "answer": final_state["messages"][-1].content,
            "sources": final_state["sources"],
            "thread_id": thread_id,
        }