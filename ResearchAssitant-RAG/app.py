"""
Research Assistant RAG — FastAPI application.

Run from project root:
  uvicorn app:app --reload

Endpoints:
  GET  /          API info
  GET  /health    Service + index status
  POST /ingest     Load Data → chunk → embed → FAISS index
  POST /retrieve   Embed question + search (no LLM)
  POST /ask        Full RAG: retrieve + Ollama answer
"""

from __future__ import annotations

import os

# Avoid slow TensorFlow imports when using sentence-transformers (PyTorch backend).
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root is on PYTHONPATH for Utils / VectorStore imports
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils.chunking import chunk_documents  # noqa: E402
from Utils.document_loader import load_documents  # noqa: E402
from Utils.embeddings import create_embedder  # noqa: E402
from Utils.llm_response import RAGAssistant, format_context_from_hits  # noqa: E402
from Utils.rag_pipeline import RAGPipeline  # noqa: E402
from VectorStore.faiss_store import FaissVectorStore  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "Data"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", ROOT / "VectorStore" / "research_index"))
OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "enter local model name")
OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_BASE_URL", "enter ollama url")
EMBEDDER_BACKEND = os.getenv("RAG_EMBEDDER_BACKEND", "sentence-transformers")
EMBEDDER_MODEL = os.getenv(
    "RAG_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    data_path: Optional[str] = Field(
        default=None,
        description="Folder or file to ingest. Defaults to ./Data",
    )
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0)
    rebuild: bool = Field(
        default=True,
        description="Replace existing index (recommended on re-ingest)",
    )


class IngestResponse(BaseModel):
    data_path: str
    documents_loaded: int
    chunks_created: int
    vectors_stored: int
    index_path: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    score: float
    text: str
    metadata: Dict[str, Any]


class RetrieveResponse(BaseModel):
    question: str
    context: str
    sources: List[SourceChunk]


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
    context: str
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    index_path: str
    vector_count: int
    data_dir: str
    ollama_model: str


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState:
    assistant: Optional[RAGAssistant] = None
    pipeline: Optional[RAGPipeline] = None

    def index_exists(self) -> bool:
        return (INDEX_DIR / "index.faiss").exists()

    def vector_count(self) -> int:
        if self.pipeline and self.pipeline.vector_store:
            return len(self.pipeline.vector_store)
        return 0

    def reload_from_index(self) -> None:
        if not self.index_exists():
            self.assistant = None
            self.pipeline = None
            return
        self.assistant = RAGAssistant.from_index(
            str(INDEX_DIR),
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            embedder_backend=EMBEDDER_BACKEND,
            embedder_model=EMBEDDER_MODEL,
        )
        self.pipeline = self.assistant.pipeline


state = AppState()


def ingest_corpus(
    data_path: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResponse:
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    docs = load_documents(str(data_path))
    if not docs:
        raise ValueError(f"No documents found under: {data_path}")

    chunks = chunk_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise ValueError("Chunking produced no text chunks")

    embedder = create_embedder(
        backend=EMBEDDER_BACKEND,
        model_name=EMBEDDER_MODEL,
    )
    embedded = embedder.embed_documents(chunks)

    store = FaissVectorStore(metric="cosine")
    store.add(embedded)
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    store.save(INDEX_DIR)

    state.reload_from_index()

    return IngestResponse(
        data_path=str(data_path),
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        vectors_stored=len(embedded),
        index_path=str(INDEX_DIR),
    )


def _require_pipeline() -> RAGPipeline:
    if state.pipeline is None:
        if state.index_exists():
            state.reload_from_index()
        if state.pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Vector index not ready. POST /ingest first to build the index.",
            )
    return state.pipeline


def _require_assistant() -> RAGAssistant:
    if state.assistant is None:
        if state.index_exists():
            state.reload_from_index()
        if state.assistant is None:
            raise HTTPException(
                status_code=503,
                detail="Assistant not ready. POST /ingest first, then try again.",
            )
    return state.assistant


def _hits_to_sources(hits) -> List[SourceChunk]:
    return [
        SourceChunk(
            score=hit.score,
            text=hit.document.text,
            metadata=dict(hit.document.metadata or {}),
        )
        for hit in hits
    ]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    if state.index_exists():
        try:
            state.reload_from_index()
        except Exception:
            state.assistant = None
            state.pipeline = None
    yield


app = FastAPI(
    title="Research Assistant RAG",
    description="RAG API: ingest documents, retrieve chunks, answer with Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Research Assistant RAG",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "ingest": "POST /ingest",
            "retrieve": "POST /retrieve",
            "ask": "POST /ask",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ready = state.index_exists()
    return HealthResponse(
        status="ok",
        index_ready=ready and state.pipeline is not None,
        index_path=str(INDEX_DIR),
        vector_count=state.vector_count(),
        data_dir=str(DATA_DIR),
        ollama_model=OLLAMA_MODEL,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(body: IngestRequest) -> IngestResponse:
    """
    Load documents from `Data` (or custom path), chunk, embed, and save FAISS index.
    """
    data_path = Path(body.data_path) if body.data_path else DATA_DIR
    data_path = data_path.expanduser().resolve()

    if body.chunk_overlap >= body.chunk_size:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be smaller than chunk_size",
        )

    if body.rebuild and INDEX_DIR.exists():
        import shutil

        shutil.rmtree(INDEX_DIR, ignore_errors=True)

    try:
        return ingest_corpus(
            data_path,
            chunk_size=body.chunk_size,
            chunk_overlap=body.chunk_overlap,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(body: RetrieveRequest) -> RetrieveResponse:
    """Embed the question and return top-k similar chunks (no LLM)."""
    pipeline = _require_pipeline()
    try:
        result = pipeline.retrieve(body.question, k=body.k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    context = format_context_from_hits(result.hits)
    return RetrieveResponse(
        question=result.question,
        context=context,
        sources=_hits_to_sources(result.hits),
    )


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    """Full RAG: retrieve context from FAISS, then generate answer with Ollama."""
    assistant = _require_assistant()
    try:
        llm_result = assistant.ask(body.question, k=body.k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama unavailable: {e}. Run `ollama serve` and pull the model.",
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {e}") from e

    hits = llm_result.retrieval.hits if llm_result.retrieval else []
    return AskResponse(
        question=llm_result.question,
        answer=llm_result.answer,
        model=llm_result.model,
        context=llm_result.context,
        sources=_hits_to_sources(hits),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
