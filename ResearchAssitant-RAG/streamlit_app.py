"""
Research Assistant RAG — Streamlit UI.

Run from project root:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app import (  # noqa: E402
    DATA_DIR,
    INDEX_DIR,
    OLLAMA_MODEL,
    ingest_corpus,
)
from Utils.llm_response import RAGAssistant  # noqa: E402


def _index_ready() -> bool:
    return (INDEX_DIR / "index.faiss").exists()


def _vector_count() -> int:
    if not _index_ready():
        return 0
    try:
        from VectorStore.faiss_store import FaissVectorStore

        return len(FaissVectorStore.load(INDEX_DIR))
    except Exception:
        return 0


def _load_assistant() -> RAGAssistant:
    return RAGAssistant.from_index(str(INDEX_DIR))


def _save_uploads(files) -> list[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for f in files:
        path = DATA_DIR / f.name
        path.write_bytes(f.getbuffer())
        saved.append(f.name)
    return saved


def main() -> None:
    st.set_page_config(
        page_title="Research Assistant RAG",
        page_icon="📚",
        layout="wide",
    )

    st.title("Research Assistant")
    st.caption("Upload documents, build the index, then ask questions grounded in your files.")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        chunk_size = st.slider("Chunk size", 500, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, step=50)
        top_k = st.slider("Chunks to retrieve (k)", 1, 10, 5)

        st.divider()
        st.subheader("Upload documents")
        uploaded = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "docx", "pptx", "html", "htm", "csv", "xlsx", "json"],
        )

        if st.button("Save uploads", use_container_width=True, disabled=not uploaded):
            names = _save_uploads(uploaded)
            st.success(f"Saved {len(names)} file(s) to Data/")
            for n in names:
                st.write(f"- `{n}`")

        if st.button("Build / rebuild index", use_container_width=True, type="primary"):
            if not any(DATA_DIR.iterdir()):
                st.error("No files in Data/. Upload documents first.")
            elif chunk_overlap >= chunk_size:
                st.error("Chunk overlap must be smaller than chunk size.")
            else:
                with st.spinner("Loading, chunking, embedding… (first run may take a few minutes)"):
                    try:
                        if INDEX_DIR.exists():
                            import shutil

                            shutil.rmtree(INDEX_DIR, ignore_errors=True)
                        result = ingest_corpus(
                            DATA_DIR,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        st.session_state.pop("assistant", None)
                        st.success(
                            f"Indexed {result.chunks_created} chunks "
                            f"from {result.documents_loaded} document(s)."
                        )
                    except Exception as e:
                        st.error(f"Ingest failed: {e}")

        st.divider()
        st.markdown("**Status**")
        st.write(f"Index ready: **{'Yes' if _index_ready() else 'No'}**")
        st.write(f"Vectors: **{_vector_count()}**")
        st.write(f"Ollama model: `{OLLAMA_MODEL}`")
        st.write(f"Data folder: `{DATA_DIR}`")

    # Main area
    if not _index_ready():
        st.info(
            "Upload files in the sidebar, click **Save uploads**, then **Build / rebuild index** "
            "before asking questions."
        )
    else:
        st.success("Vector index is ready. You can ask questions below.")

    question = st.text_area(
        "Your question",
        placeholder="Ask anything about your documents",
        height=100,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_clicked = st.button("Get answer", type="primary", use_container_width=True)
    with col2:
        retrieve_only = st.checkbox("Retrieve only (no LLM)", help="Skip Ollama; show matching chunks only.")

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        elif not _index_ready():
            st.warning("Build the index first using the sidebar.")
        else:
            try:
                if "assistant" not in st.session_state:
                    with st.spinner("Loading assistant…"):
                        st.session_state["assistant"] = _load_assistant()

                assistant: RAGAssistant = st.session_state["assistant"]

                if retrieve_only:
                    with st.spinner("Searching…"):
                        retrieval = assistant.pipeline.retrieve(question.strip(), k=top_k)
                    st.subheader("Retrieved context")
                    if not retrieval.hits:
                        st.write("No relevant chunks found.")
                    else:
                        for i, hit in enumerate(retrieval.hits, 1):
                            src = hit.document.metadata.get("filename", "unknown")
                            with st.expander(f"Chunk {i} · score {hit.score:.3f} · {src}"):
                                st.write(hit.document.text)
                else:
                    with st.spinner("Retrieving context and generating answer with Ollama…"):
                        result = assistant.ask(question.strip(), k=top_k)

                    st.subheader("Answer")
                    st.markdown(result.answer)

                    with st.expander("Sources used", expanded=False):
                        if result.retrieval and result.retrieval.hits:
                            for i, hit in enumerate(result.retrieval.hits, 1):
                                src = hit.document.metadata.get("filename", "unknown")
                                st.markdown(f"**[{i}] {src}** (score: {hit.score:.3f})")
                                st.text(hit.document.text[:800] + ("…" if len(hit.document.text) > 800 else ""))
                        else:
                            st.write("No sources retrieved.")

            except ConnectionError as e:
                st.error(f"Cannot reach Ollama. Run `ollama serve` and ensure the model is pulled.\n\n{e}")
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
