from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from Utils.rag_pipeline import RAGPipeline, RetrievalResult  # type: ignore
except Exception:  # pragma: no cover
    try:
        from rag_pipeline import RAGPipeline, RetrievalResult  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import RAGPipeline. Ensure `rag_pipeline.py` is importable."
        ) from e

try:
    from VectorStore.faiss_store import SearchResult  # type: ignore
except Exception:  # pragma: no cover
    try:
        from faiss_store import SearchResult  # type: ignore
    except Exception:
        SearchResult = Any  # type: ignore[misc, assignment]


DEFAULT_SYSTEM_PROMPT = """You are a research assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say you do not have enough information.
Be concise, accurate, and mention source filenames when they appear in the context."""


@dataclass(frozen=True)
class LLMResponse:
    question: str
    answer: str
    context: str
    model: str
    retrieval: Optional[RetrievalResult] = None


class OllamaLLM:
    """
    Local LLM client for Ollama (open-source, runs on your machine).

    Prerequisites:
      1. Install Ollama: https://ollama.com
      2. Pull a model: ollama pull llama3.2
      3. Ensure Ollama is running (default: http://localhost:11434)
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:latest",
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a chat completion request to Ollama and return the assistant message text.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": stream,
        }
        if options:
            payload["options"] = options

        url = f"{self.base_url}/api/chat"
        raw = _post_json(url, payload, timeout=self.timeout)
        message = raw.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response: {raw!r}")
        return content.strip()

    def generate_with_context(
        self,
        question: str,
        context: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """Answer a question given retrieved context."""
        user_content = _build_user_prompt(question, context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.chat(messages)


class RAGAssistant:
    """
    End-to-end RAG: retrieve chunks via `RAGPipeline`, then generate with Ollama.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        llm: Optional[OllamaLLM] = None,
        *,
        model: str = "qwen2.5-coder:latest",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.pipeline = pipeline
        self.llm = llm or OllamaLLM(model=model, base_url=base_url)

    @classmethod
    def from_index(
        cls,
        index_directory: str,
        *,
        model: str = "qwen2.5-coder:latest",
        base_url: str = "http://localhost:11434",
        embedder_backend: str = "sentence-transformers",
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "RAGAssistant":
        pipeline = RAGPipeline.from_saved_index(
            index_directory,
            backend=embedder_backend,
            model_name=embedder_model,
        )
        return cls(pipeline, OllamaLLM(model=model, base_url=base_url))

    def ask(
        self,
        question: str,
        *,
        k: int = 5,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        include_metadata: bool = True,
    ) -> LLMResponse:
        """
        Retrieve relevant chunks, pass them as context to Ollama, and return the answer.
        """
        retrieval = self.pipeline.retrieve(question, k=k)
        context = format_context_from_hits(
            retrieval.hits,
            include_metadata=include_metadata,
        )
        answer = self.llm.generate_with_context(
            question,
            context,
            system_prompt=system_prompt,
        )
        return LLMResponse(
            question=question.strip(),
            answer=answer,
            context=context,
            model=self.llm.model,
            retrieval=retrieval,
        )


def format_context_from_hits(
    hits: Sequence["SearchResult"],
    *,
    include_metadata: bool = True,
    separator: str = "\n\n---\n\n",
) -> str:
    """Format retrieved chunks (with optional source metadata) for the LLM prompt."""
    if not hits:
        return ""

    parts: List[str] = []
    for i, hit in enumerate(hits, start=1):
        block = f"[Chunk {i}]\n{hit.document.text}"
        if include_metadata and hit.document.metadata:
            md = hit.document.metadata
            source = md.get("source") or md.get("filename") or "unknown"
            extra = []
            for key in ("page", "slide", "sheet", "chunk_index"):
                if key in md:
                    extra.append(f"{key}={md[key]}")
            meta_line = f"Source: {source}"
            if extra:
                meta_line += " (" + ", ".join(extra) + ")"
            block = f"{meta_line}\n{block}"
        parts.append(block)
    return separator.join(parts)


def _build_user_prompt(question: str, context: str) -> str:
    q = (question or "").strip()
    ctx = (context or "").strip()
    if not ctx:
        return (
            f"Question:\n{q}\n\n"
            "Context:\n(No relevant documents were retrieved.)\n\n"
            "Answer based on the context only."
        )
    return (
        f"Question:\n{q}\n\n"
        f"Context:\n{ctx}\n\n"
        "Answer the question using only the context above."
    )


def _post_json(url: str, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise ConnectionError(
            "Could not reach Ollama. Is it running? Start with: ollama serve"
        ) from e

    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from Ollama: {body[:500]!r}") from e
