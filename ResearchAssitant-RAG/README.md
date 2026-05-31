# Research Assistant RAG

A **Retrieval-Augmented Generation (RAG)** system that lets you upload research documents, ask questions in natural language, and get answers grounded in your own files—running entirely on your machine with open-source tools.

## Overview

This project ingests documents from multiple formats, splits them into searchable chunks, embeds them with semantic vectors, stores them in a **FAISS** vector database, and uses a local **Ollama** LLM to generate accurate, context-aware answers. No cloud API keys are required for embeddings or generation.

**Typical flow:** Upload → Chunk → Embed → Index → Ask → Retrieve relevant chunks → LLM answer with sources

## Features

- **Multi-format document loading** — PDF, DOCX, PPTX, TXT, MD, HTML, CSV, XLSX, JSON
- **Smart chunking** — Recursive text splitting with overlap for better retrieval
- **Semantic embeddings** — `sentence-transformers` (with TF-IDF fallback)
- **Vector search** — FAISS for fast similarity search
- **Local LLM** — Ollama integration (e.g. Llama, Qwen)
- **REST API** — FastAPI endpoints for ingest, retrieve, and ask
- **Web UI** — Streamlit app with file upload and Q&A

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌────────────┐     ┌─────────────┐
│  Documents  │────▶│ Chunking │────▶│ Embeddings │────▶│ FAISS Store │
│  (Data/)    │     │          │     │            │     │             │
└─────────────┘     └──────────┘     └────────────┘     └──────┬──────┘
                                                               │
┌─────────────┐     ┌──────────┐     ┌────────────┐            │
│   Answer    │◀────│  Ollama  │◀────│  Retrieve  │◀───────────┘
│             │     │   LLM    │     │  (query)   │
└─────────────┘     └──────────┘     └────────────┘
```

## Project Structure

```
ResearchAssitant-RAG/
├── Data/                    # Uploaded documents (ingest source)
├── VectorStore/
│   ├── faiss_store.py       # FAISS index save/load & search
│   └── research_index/      # Persisted index (generated)
├── Utils/
│   ├── document_loader.py   # Load PDF, DOCX, etc.
│   ├── chunking.py          # Split text into chunks
│   ├── embeddings.py        # Text → vectors
│   ├── rag_pipeline.py      # Query embedding + retrieval
│   └── llm_response.py      # Ollama RAG answers
├── app.py                   # FastAPI REST API
├── streamlit_app.py         # Streamlit UI
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A pulled Ollama model (default: `qwen2.5-coder:latest`)

```bash
ollama pull qwen2.5-coder:latest
ollama serve
```

## Installation

```bash
git clone https://github.com/<your-username>/ResearchAssitant-RAG.git
cd ResearchAssitant-RAG
pip install -r requirements.txt
```

## Usage

### Streamlit UI (recommended)

```bash
streamlit run streamlit_app.py
```

1. Upload files in the sidebar → **Save uploads**
2. Click **Build / rebuild index**
3. Enter a question → **Get answer**

### FastAPI

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive API documentation.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Index and service status |
| `/ingest` | POST | Build vector index from `Data/` |
| `/retrieve` | POST | Search chunks (no LLM) |
| `/ask` | POST | Full RAG answer via Ollama |

**Example**

```bash
curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d "{}"
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d "{\"question\": \"What are the key findings?\", \"k\": 5}"
```

## Configuration

Optional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_DATA_DIR` | `./Data` | Document upload folder |
| `RAG_INDEX_DIR` | `./VectorStore/research_index` | FAISS index path |
| `RAG_OLLAMA_MODEL` | `qwen2.5-coder:latest` | Ollama model name |
| `RAG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `RAG_EMBEDDER_BACKEND` | `sentence-transformers` | Embedding backend |
| `RAG_EMBEDDER_MODEL` | `all-MiniLM-L6-v2` | Embedding model |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Document parsing | pypdf, python-docx, python-pptx, BeautifulSoup, openpyxl |
| Chunking | Custom recursive splitter |
| Embeddings | sentence-transformers / scikit-learn (TF-IDF) |
| Vector DB | FAISS (open source) |
| LLM | Ollama (local) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |

## License

MIT (or specify your license)

## Author

Your name / GitHub profile
