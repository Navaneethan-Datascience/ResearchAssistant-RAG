from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

try:
    from Utils.embeddings import BaseEmbedder, Vector, create_embedder  # type: ignore
except Exception:  # pragma: no cover
    try:
        from embeddings import BaseEmbedder, Vector, create_embedder  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import embeddings module. Ensure `embeddings.py` is importable."
        ) from e

try:
    from VectorStore.faiss_store import FaissVectorStore, SearchResult  # type: ignore
except Exception:  # pragma: no cover
    try:
        from faiss_store import FaissVectorStore, SearchResult  # type: ignore
    except Exception:
        FaissVectorStore = None  # type: ignore[misc, assignment]
        SearchResult = None  # type: ignore[misc, assignment]


PathLike = Union[str, Path]


@dataclass(frozen=True)
class RetrievalResult:
    """A user question paired with retrieved context chunks."""

    question: str
    query_embedding: Vector
    hits: List["SearchResult"]


def embed_question(question: str, embedder: BaseEmbedder) -> Vector:
    """
    Convert a user question into an embedding vector.

    Uses the same embedder as document chunks so similarity search is meaningful.
    """
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")
    return embedder.embed_texts([text])[0]


class RAGPipeline:
    """
    Retrieval stage of RAG: embed questions and search the FAISS vector store.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: Optional["FaissVectorStore"] = None,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store

    @classmethod
    def from_saved_index(
        cls,
        index_directory: PathLike,
        *,
        embedder: Optional[BaseEmbedder] = None,
        backend: str = "sentence-transformers",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "RAGPipeline":
        """
        Load a persisted FAISS index and create a pipeline ready for queries.

        Use the same embedding model/backend that was used when building the index.
        """
        if FaissVectorStore is None:
            raise ImportError(
                "FaissVectorStore is not available. Ensure `VectorStore/faiss_store.py` is importable."
            )

        store = FaissVectorStore.load(index_directory)
        emb = embedder or create_embedder(backend=backend, model_name=model_name)
        return cls(embedder=emb, vector_store=store)

    @property
    def embedder(self) -> BaseEmbedder:
        return self._embedder

    @property
    def vector_store(self) -> Optional["FaissVectorStore"]:
        return self._store

    def set_vector_store(self, store: "FaissVectorStore") -> None:
        self._store = store

    def embed_question(self, question: str) -> Vector:
        """Convert user question to embedding."""
        return embed_question(question, self._embedder)

    def retrieve(self, question: str, *, k: int = 5) -> RetrievalResult:
        """
        Embed the question and return top-k similar chunks from the vector store.
        """
        if self._store is None:
            raise ValueError("No vector store attached. Load or set a FaissVectorStore first.")

        query_embedding = self.embed_question(question)
        hits = self._store.search(query_embedding, k=k)
        return RetrievalResult(
            question=question.strip(),
            query_embedding=query_embedding,
            hits=hits,
        )

    def get_context(
        self,
        question: str,
        *,
        k: int = 5,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Retrieve relevant chunks and format them as a single context string for an LLM.
        """
        result = self.retrieve(question, k=k)
        if not result.hits:
            return ""
        return separator.join(hit.document.text for hit in result.hits)
