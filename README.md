# Custom RAG from Scratch (Python + Ollama llama3.2)

A lightweight, modular, and fully local Retrieval-Augmented Generation (RAG) system built **entirely from scratch** in Python.

- **Zero Frameworks**: No LangChain, LlamaIndex, ChromaDB, or heavy dependencies.
- **100% Local & Private**: Powered by local [Ollama](https://ollama.com/) running `llama3.2:3b` for generation and `embeddinggemma:latest` for vector embeddings.
- **Transparent Math**: Pure Python cosine similarity computation with automatic NumPy matrix acceleration.
- **Clean CLI & Interactive Chat**: Ingest documents, run one-shot queries with similarity scores and source citations, or chat interactively with streaming output.

---

## Architecture Overview

```
                         [ User Query ]
                               |
                               v
                     [ Ollama Client ]
               (embeddinggemma:latest 768-dim)
                               |
                               v
+-------------------------------------------------------------+
|                     Vector Store (Scratch)                  |
|  - Cosine Similarity:  cos(θ) = (A · B) / (||A|| * ||B||)   |
|  - Top-K Nearest Neighbor Ranking & Similarity Thresholding |
|  - JSON Disk Persistence                                    |
+-------------------------------------------------------------+
                               |
                        [ Top Chunks ]
                               |
                               v
                     [ Prompt Augmented ]
        Context + Source File Names + User Question
                               |
                               v
                     [ Ollama llama3.2:3b ]
                     (Streaming Response)
```

---

## Project Structure

```
rag_project/
├── config.py                 # Configuration (Ollama URLs, model names, chunking parameters)
├── main.py                   # Interactive CLI (ingest, query, chat, stats, clear)
├── test_rag.py               # Unit & integration test suite (6 validation checks)
├── core/
│   ├── __init__.py
│   ├── document.py           # Document & Chunk data models with serialization
│   ├── loader.py             # File and directory loader (.txt, .md, .json, .csv)
│   ├── splitter.py           # Recursive text chunker with sliding window overlap
│   ├── ollama_client.py      # Native HTTP client for local Ollama API (embed & generate)
│   ├── vector_store.py       # Custom vector store with cosine similarity & persistence
│   └── rag_engine.py         # End-to-end RAG orchestrator with prompt engineering
├── data/
│   └── sample_docs/          # Built-in sample knowledge base documents
│       ├── artemis_mission.md
│       ├── quantum_computing.txt
│       └── local_rag_architecture.json
└── storage/
    └── vector_index.json     # Saved vector embeddings and chunk metadata
```

---

## Quickstart

### 1. Requirements & Prerequisites
- Python 3.9+ (Python 3.14 verified)
- [Ollama](https://ollama.com/) installed and running locally
- Required models in Ollama:
  ```bash
  ollama run llama3.2:3b
  ollama pull embeddinggemma:latest
  ```

### 2. Ingest Documents
Ingest a directory of markdown, text, csv, or json files:
```bash
python main.py ingest data/sample_docs
```

Or ingest a single file:
```bash
python main.py ingest path/to/your_document.pdf.txt
```

### 3. Query the Knowledge Base
Ask a question and receive a grounded answer with retrieved citations and match percentages:
```bash
python main.py query "What rocket launches the Orion spacecraft?"
```

```bash
python main.py query "What are the hardware modalities of quantum computing?"
```

### 4. Interactive Chat
Start a real-time terminal chat session with token streaming:
```bash
python main.py chat
```
*Tip: Type `!sources` during chat to inspect the full chunk text of the most recent answer.*

### 5. Check Index Stats
Inspect indexed chunk count, source files, and disk storage:
```bash
python main.py stats
```

### 6. Reset / Clear Index
To wipe the current vector store:
```bash
python main.py clear
```

---

## Running the Automated Test Suite

A comprehensive test suite is included to verify all layers from math to inference:
```bash
python test_rag.py
```

Tests include:
1. Pure cosine similarity math verification (orthogonal, parallel, and inverted vectors).
2. Recursive text chunking with sliding overlap.
3. Multi-format document loading.
4. Ollama HTTP connectivity and 768-dim embedding generation.
5. In-memory vector store indexing, persistence, and reloading.
6. End-to-end query generation with `llama3.2`.
