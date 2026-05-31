from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


try:
    # When `Utils` is a package
    from Utils.document_loader import LoadedDocument  # type: ignore
except Exception:  # pragma: no cover
    try:
        # When `Utils` is on PYTHONPATH and modules are imported directly
        from document_loader import LoadedDocument  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import LoadedDocument. Ensure `document_loader.py` is importable."
        ) from e


Vector = List[float]


@dataclass(frozen=True)
class EmbeddedDocument:
    """
    A `LoadedDocument` paired with its vector embedding.
    """

    document: LoadedDocument
    embedding: Vector


class BaseEmbedder:
    """
    Minimal embedding interface.
    """

    def embed_texts(self, texts: Sequence[str]) -> List[Vector]:
        raise NotImplementedError

    def embed_documents(self, docs: Sequence[LoadedDocument]) -> List[EmbeddedDocument]:
        vectors = self.embed_texts([d.text for d in docs])
        return [EmbeddedDocument(document=d, embedding=v) for d, v in zip(docs, vectors)]


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local semantic embeddings via `sentence-transformers`.

    Install:
      pip install sentence-transformers
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        normalize: bool = True,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "SentenceTransformerEmbedder requires `sentence-transformers`. "
                "Install with: pip install sentence-transformers"
            ) from e

        self._normalize = normalize
        self._batch_size = batch_size
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)  # type: ignore[arg-type]

    def embed_texts(self, texts: Sequence[str]) -> List[Vector]:
        # sentence-transformers can return numpy arrays; convert to plain lists for storage/JSON.
        emb = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return _to_vectors(emb)


class TfidfEmbedder(BaseEmbedder):
    """
    Lightweight fallback embedder using TF-IDF (lexical similarity, not true semantics).

    Useful if you can't install embedding models yet, but still want a working similarity search.

    Install:
      pip install scikit-learn
    """

    def __init__(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "TfidfEmbedder requires `scikit-learn`. Install with: pip install scikit-learn"
            ) from e

        self._vectorizer = TfidfVectorizer()
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfEmbedder":
        self._vectorizer.fit(list(texts))
        self._fitted = True
        return self

    def embed_texts(self, texts: Sequence[str]) -> List[Vector]:
        if not self._fitted:
            # Fit on the same texts for convenience; in a real system fit on the corpus,
            # then reuse the same instance for queries.
            self.fit(texts)
        mat = self._vectorizer.transform(list(texts))
        # Convert sparse rows to dense lists
        return [row.toarray().ravel().astype(float).tolist() for row in mat]


def create_embedder(
    *,
    backend: str = "sentence-transformers",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> BaseEmbedder:
    """
    Factory for embedders.

    - backend="sentence-transformers": best semantic embeddings (recommended)
    - backend="tfidf": fallback lexical embeddings
    """

    b = backend.strip().lower()
    if b in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(
            model_name=model_name, normalize=normalize, batch_size=batch_size, device=device
        )
    if b in {"tfidf", "tf-idf"}:
        return TfidfEmbedder()
    raise ValueError(f"Unknown backend: {backend!r}")


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity in [-1, 1]. For normalized embeddings, this is just the dot product.
    """

    if len(a) != len(b):
        raise ValueError("Vectors must have same dimension")
    denom = (_l2_norm(a) * _l2_norm(b))
    if denom == 0.0:
        return 0.0
    return _dot(a, b) / denom


def top_k_similar(
    query_embedding: Sequence[float],
    corpus: Sequence[EmbeddedDocument],
    *,
    k: int = 5,
) -> List[Tuple[float, EmbeddedDocument]]:
    """
    Return top-k most similar embedded documents to a query embedding.
    """

    if k <= 0:
        return []
    scored: List[Tuple[float, EmbeddedDocument]] = [
        (cosine_similarity(query_embedding, ed.embedding), ed) for ed in corpus
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _l2_norm(v: Sequence[float]) -> float:
    return float(sum(x * x for x in v) ** 0.5)


def _to_vectors(embeddings: Any) -> List[Vector]:
    # Handles numpy arrays, lists, list-of-lists, etc.
    try:
        # numpy: (n,d)
        return embeddings.astype(float).tolist()  # type: ignore[attr-defined]
    except Exception:
        pass

    if isinstance(embeddings, list):
        if not embeddings:
            return []
        if isinstance(embeddings[0], list):
            return [[float(x) for x in row] for row in embeddings]
        # single vector
        return [[float(x) for x in embeddings]]

    # last resort: attempt to iterate rows
    return [[float(x) for x in row] for row in embeddings]

