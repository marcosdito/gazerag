"""
RAG pipeline: question → retrieval → LLM → answer with citations.

Architecture:
- ChromaDB stores embeddings and chunk metadata (built by scripts/ingest_papers.py).
- LlamaIndex orchestrates retrieval and prompt assembly.
- Groq serves the LLM (Llama 3.1 8B Instant — fast, free tier).
- sentence-transformers generates query embeddings (same model used at ingest).

The pipeline supports two query modes:
- `ask(question)` — pure literature Q&A
- `ask_with_context(question, context)` — augments the prompt with caller-
  provided text (used by the Streamlit app to inject gaze analysis results).
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.utils.config import (
    CHROMA_COLLECTION,
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    TOP_K_RETRIEVAL,
    validate,
)


# -----------------------------------------------------------------------------
# Prompt template
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a research assistant specialized in eye-tracking and visual attention.

Your job is to answer the user's question using ONLY the information in the provided context. \
If the context does not contain enough information to answer, say so explicitly — do not invent facts.

When you reference information from the context, cite the source using the format \
[source: filename, page N]. Be concise and precise. Prefer concrete numbers and definitions \
over vague summaries."""


HYBRID_SYSTEM_PROMPT = """You are a research assistant specialized in eye-tracking and visual attention.

You have access to two information sources:
1. SCIENTIFIC LITERATURE: passages retrieved from peer-reviewed papers.
2. EXPERIMENTAL DATA: quantitative metrics extracted from the user's own gaze recording.

Your job is to interpret the EXPERIMENTAL DATA in light of the SCIENTIFIC LITERATURE. \
Cite literature passages as [source: filename, page N]. Be specific about which user metric \
maps to which literature finding. If the literature does not cover something the user asked, \
say so explicitly — do not invent."""


# -----------------------------------------------------------------------------
# Response container
# -----------------------------------------------------------------------------
@dataclass
class RAGResponse:
    """Structured response from the pipeline."""

    answer: str
    sources: list[dict]  # [{source, page, score, snippet}]

    def __str__(self) -> str:
        lines = [self.answer, "", "Sources:"]
        for s in self.sources:
            lines.append(
                f"  • {s['source']} (page {s['page']}, relevance {s['score']:.2f})"
            )
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
class RAGPipeline:
    """Encapsulates the full retrieval + generation flow."""

    def __init__(self) -> None:
        validate()

        # Embedding model — must match what ingest_papers.py used
        self.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

        # LLM via Groq
        self.llm = Groq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,  # low temperature for factual Q&A
        )

        # Make LlamaIndex use these globally
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        # ChromaDB client + collection
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(name=CHROMA_COLLECTION)

        # Wrap Chroma as a LlamaIndex vector store
        vector_store = ChromaVectorStore(chroma_collection=collection)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=self.embed_model
        )

        # Retriever
        self.retriever = self.index.as_retriever(similarity_top_k=TOP_K_RETRIEVAL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _retrieve(self, question: str) -> list[NodeWithScore]:
        """Return the top-K most relevant chunks for a question."""
        return self.retriever.retrieve(question)

    def _format_context(self, nodes: list[NodeWithScore]) -> str:
        """Format retrieved chunks into a prompt-friendly block."""
        blocks: list[str] = []
        for i, node in enumerate(nodes, start=1):
            meta = node.metadata or {}
            source = meta.get("source", "unknown")
            page = meta.get("page", "?")
            text = node.get_content().strip()
            blocks.append(f"[Passage {i} | source: {source}, page {page}]\n{text}")
        return "\n\n".join(blocks)

    def _nodes_to_sources(self, nodes: list[NodeWithScore]) -> list[dict]:
        """Convert retrieved nodes into a plain list of source dicts."""
        sources = []
        for node in nodes:
            meta = node.metadata or {}
            snippet = node.get_content().strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            sources.append({
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "?"),
                "score": float(node.score) if node.score is not None else 0.0,
                "snippet": snippet,
            })
        return sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(self, question: str) -> RAGResponse:
        """Answer a question using only retrieved literature."""
        nodes = self._retrieve(question)

        if not nodes:
            return RAGResponse(
                answer="I could not find any relevant passages in the indexed literature.",
                sources=[],
            )

        context_block = self._format_context(nodes)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== Context ===\n{context_block}\n\n"
            f"=== Question ===\n{question}\n\n"
            f"=== Answer ===\n"
        )

        response = self.llm.complete(prompt)
        return RAGResponse(
            answer=str(response).strip(),
            sources=self._nodes_to_sources(nodes),
        )

    def ask_with_context(self, question: str, experimental_context: str) -> RAGResponse:
        """Answer a question augmented with user-provided experimental metrics.

        This is the hybrid mode used by the Streamlit app: the user uploads a
        gaze trajectory, the analyzer extracts metrics, and those metrics are
        injected into the prompt alongside retrieved literature.
        """
        nodes = self._retrieve(question)
        literature_block = self._format_context(nodes) if nodes else "(no relevant literature found)"

        prompt = (
            f"{HYBRID_SYSTEM_PROMPT}\n\n"
            f"=== Scientific literature ===\n{literature_block}\n\n"
            f"=== Experimental data (from user's gaze recording) ===\n{experimental_context}\n\n"
            f"=== Question ===\n{question}\n\n"
            f"=== Answer ===\n"
        )

        response = self.llm.complete(prompt)
        return RAGResponse(
            answer=str(response).strip(),
            sources=self._nodes_to_sources(nodes),
        )
