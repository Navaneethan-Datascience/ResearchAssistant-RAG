from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "FAISS is required for the vector store. Install with: pip install faiss-cpu"
    ) from e

try:
    from Utils.document_loader import LoadedDocument  # type: ignore
    from Utils.embeddings import EmbeddedDocument, Vector  # type: ignore
except Exception:  # pragma: no cover
    try:
        from document_loader import LoadedDocument  # type: ignore
        from embeddings import EmbeddedDocument, Vector  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import LoadedDocument/EmbeddedDocument. "
            "Ensure Utils modules are on PYTHONPATH."
        ) from e


PathLike = Union[str, Path]

_INDEX_FILENAME = "index.faiss"
_DOCUMENTS_FILENAME = "documents.json"
_CONFIG_FILENAME = "config.json"


@dataclass(frozen=True)
class SearchResult:
    score: float
    document: LoadedDocument
    index: int


class FaissVectorStore:
    """
    Open-source vector store backed by FAISS (Facebook AI Similarity Search).

    Stores chunk embeddings from `embeddings.py` and supports similarity search,
    persistence, and incremental adds.

    Install:
      pip install faiss-cpu
      # or GPU build: pip install faiss-gpu
    """

    def __init__(
        self,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ) -> None:
        """
        Args:
            dimension: embedding size. Required before first `add` if not inferred.
            metric: "cosine" (normalized inner product) or "l2" (Euclidean).
        """
        m = metric.strip().lower()
        if m not in {"cosine", "l2"}:
            raise ValueError("metric must be 'cosine' or 'l2'")

        self._metric = m
        self._dimension = dimension
        self._index: Optional[faiss.Index] = None
        self._documents: List[LoadedDocument] = []

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    @property
    def metric(self) -> str:
        return self._metric

    def __len__(self) -> int:
        return len(self._documents)

    def add(self, items: Sequence[EmbeddedDocument]) -> int:
        """
        Add embedded documents to the store.

        Returns:
            Number of vectors added.
        """
        if not items:
            return 0

        vectors = [_as_vector(item.embedding) for item in items]
        matrix = np.vstack(vectors).astype(np.float32)

        if self._dimension is None:
            self._dimension = int(matrix.shape[1])
        elif matrix.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, "
                f"got {matrix.shape[1]}"
            )

        if self._index is None:
            self._index = _create_index(self._dimension, self._metric)

        if self._metric == "cosine":
            faiss.normalize_L2(matrix)

        self._index.add(matrix)
        self._documents.extend(item.document for item in items)
        return len(items)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        k: int = 5,
    ) -> List[SearchResult]:
        """
        Find top-k most similar stored chunks.

        Scores:
          - cosine: inner product (higher is more similar) when vectors are L2-normalized
          - l2: squared L2 distance (lower is more similar)
        """
        if self._index is None or not self._documents:
            return []
        if k <= 0:
            return []

        k = min(k, len(self._documents))
        query = np.array([_as_vector(query_embedding)], dtype=np.float32)
        if self._metric == "cosine":
            faiss.normalize_L2(query)

        scores, indices = self._index.search(query, k)
        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                SearchResult(
                    score=float(score),
                    document=self._documents[int(idx)],
                    index=int(idx),
                )
            )
        return results

    def save(self, directory: PathLike) -> None:
        """Persist FAISS index and document metadata to `directory`."""
        if self._index is None:
            raise ValueError("Cannot save an empty vector store")

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(out / _INDEX_FILENAME))

        docs_payload = [
            {"text": doc.text, "metadata": doc.metadata or {}}
            for doc in self._documents
        ]
        (out / _DOCUMENTS_FILENAME).write_text(
            json.dumps(docs_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        config = {
            "dimension": self._dimension,
            "metric": self._metric,
            "count": len(self._documents),
        }
        (out / _CONFIG_FILENAME).write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: PathLike) -> "FaissVectorStore":
        """Load a vector store previously saved with `save()`."""
        root = Path(directory)
        config_path = root / _CONFIG_FILENAME
        index_path = root / _INDEX_FILENAME
        docs_path = root / _DOCUMENTS_FILENAME

        for p in (config_path, index_path, docs_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        store = cls(dimension=config["dimension"], metric=config["metric"])
        store._index = faiss.read_index(str(index_path))

        docs_payload = json.loads(docs_path.read_text(encoding="utf-8"))
        store._documents = [
            LoadedDocument(text=item["text"], metadata=item.get("metadata", {}))
            for item in docs_payload
        ]
        return store


def _create_index(dimension: int, metric: str) -> faiss.Index:
    if metric == "cosine":
        # Use inner product on L2-normalized vectors for cosine similarity.
        return faiss.IndexFlatIP(dimension)
    return faiss.IndexFlatL2(dimension)


def _as_vector(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("Each embedding must be a 1-D vector")
    return arr
