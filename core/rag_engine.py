from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

from config import RAGConfig, DEFAULT_CONFIG
from .document import Chunk, SearchResult
from .loader import DocumentLoader
from .ollama_client import OllamaClient
from .splitter import RecursiveTextSplitter
from .vector_store import VectorStore


SYSTEM_PROMPT = """You are an accurate, helpful AI assistant powered by a local RAG (Retrieval-Augmented Generation) system.
Your task is to answer the user's question based ONLY on the provided Context sections below.

Guidelines:
1. Ground your answers strictly in the provided Context.
2. If the answer cannot be found in the context, explicitly state: "Based on the provided documents, I could not find information about that."
3. Cite the relevant source files or chunk numbers when answering.
4. Keep answers concise, factual, and well-structured. Do not hallucinate.
"""


def build_rag_prompt(query: str, search_results: List[SearchResult]) -> str:
    """Builds an augmented prompt containing retrieved context snippets and user query."""
    if not search_results:
        context_str = "No relevant context found in the knowledge base."
    else:
        context_blocks = []
        for idx, res in enumerate(search_results, start=1):
            source_name = Path(res.chunk.source).name
            snippet = res.chunk.text.strip()
            score_pct = f"{res.score * 100:.1f}%"
            context_blocks.append(
                f"[Source {idx}: {source_name} (Similarity: {score_pct})]\n{snippet}"
            )
        context_str = "\n\n---\n\n".join(context_blocks)

    prompt = f"""### Context Information:
{context_str}

### User Question:
{query}

### Answer:"""
    return prompt


class RAGEngine:
    """End-to-end RAG orchestrator combining loading, chunking, indexing, and retrieval generation."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.ollama = OllamaClient(base_url=self.config.ollama_base_url)
        self.splitter = RecursiveTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.vector_store = VectorStore()
        self.index_path = self.config.storage_dir / self.config.vector_store_file

        # Auto-load existing index if available
        self.load_index()

    def check_health(self) -> Tuple[bool, str]:
        """Checks if Ollama is connected and required models are available."""
        if not self.ollama.check_connection():
            return False, f"Could not connect to Ollama at {self.config.ollama_base_url}. Ensure Ollama is running ('ollama serve')."

        try:
            models = self.ollama.list_models()
        except Exception as e:
            return False, f"Connected to Ollama, but failed to list models: {e}"

        # Check LLM model
        llm_match = any(self.config.llm_model in m for m in models)
        if not llm_match:
            return False, f"LLM model '{self.config.llm_model}' not found in Ollama. Available: {models}"

        # Check embedding model
        emb_match = any(self.config.embedding_model in m for m in models)
        if not emb_match:
            return False, f"Embedding model '{self.config.embedding_model}' not found in Ollama. Available: {models}"

        return True, "Ollama is ready with required models."

    def ingest_path(self, target_path: Union[str, Path]) -> int:
        """
        Ingests a single file or a directory into the vector store.
        Returns the number of new chunks indexed.
        """
        path = Path(target_path).resolve()
        if path.is_file():
            documents = [DocumentLoader.load_file(path)]
        elif path.is_dir():
            documents = DocumentLoader.load_directory(path)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not documents:
            print("[Warning] No readable documents found at path.")
            return 0

        print(f"Loaded {len(documents)} document(s). Splitting into chunks...")
        chunks: List[Chunk] = self.splitter.split_documents(documents)
        print(f"Generated {len(chunks)} chunk(s). Computing embeddings using '{self.config.embedding_model}'...")

        # Batch embed chunk texts
        texts = [c.text for c in chunks]
        embeddings = self.ollama.get_embeddings_batch(
            texts=texts,
            model=self.config.embedding_model,
            show_progress=True,
        )

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # Add to vector store and persist
        self.vector_store.add_chunks(chunks)
        self.save_index()
        print(f"Successfully indexed and persisted {len(chunks)} chunks.")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Retrieves top relevant chunks for a question."""
        k = top_k if top_k is not None else self.config.top_k
        thresh = threshold if threshold is not None else self.config.similarity_threshold

        query_embedding = self.ollama.get_embedding(
            text=query,
            model=self.config.embedding_model,
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            threshold=thresh,
        )
        return results

    def query(
        self,
        question: str,
        stream: bool = False,
        top_k: Optional[int] = None,
    ) -> Tuple[Union[str, Generator[str, None, None]], List[SearchResult]]:
        """
        Runs RAG on a user question:
        1. Embeds question and retrieves context.
        2. Formulates augmented prompt.
        3. Generates response using local llama3.2.
        """
        search_results = self.retrieve(question, top_k=top_k)
        prompt = build_rag_prompt(question, search_results)

        response = self.ollama.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            model=self.config.llm_model,
            stream=stream,
            temperature=self.config.temperature,
        )
        return response, search_results

    def save_index(self) -> None:
        """Persists vector store index to disk."""
        self.vector_store.save(self.index_path)

    def load_index(self) -> bool:
        """Loads vector store index from disk if it exists."""
        return self.vector_store.load(self.index_path)

    def clear_index(self) -> None:
        """Clears in-memory index and removes persistent index file."""
        self.vector_store.clear()
        if self.index_path.is_file():
            self.index_path.unlink()
