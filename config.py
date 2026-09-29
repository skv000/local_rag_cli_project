import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAGConfig:
    # Ollama settings
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "llama3.2:3b"
    embedding_model: str = "embeddinggemma:latest"

    # Chunking settings
    chunk_size: int = 500  # Target characters per chunk
    chunk_overlap: int = 100  # Overlap between consecutive chunks

    # Retrieval settings
    top_k: int = 3  # Number of top relevant chunks to retrieve
    similarity_threshold: float = 0.2  # Minimum cosine similarity threshold (0.0 to 1.0)

    # Persistence settings
    storage_dir: Path = Path(__file__).parent / "storage"
    vector_store_file: str = "vector_index.json"

    # Generation parameters
    temperature: float = 0.2
    max_tokens: int = 1024


DEFAULT_CONFIG = RAGConfig()
