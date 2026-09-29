import json
import math
from pathlib import Path
from typing import List, Optional, Union

from .document import Chunk, SearchResult

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def pure_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity using pure Python math."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same dimension")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class VectorStore:
    """
    In-memory vector store built from scratch with disk persistence,
    exact cosine similarity matching, and top-k nearest neighbor retrieval.
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self._embeddings_matrix = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add_chunks(self, new_chunks: List[Chunk]) -> None:
        """Adds chunks with precomputed embeddings to the store."""
        valid_chunks = [c for c in new_chunks if c.embedding is not None]
        if len(valid_chunks) < len(new_chunks):
            print(f"[Warning] Skipped {len(new_chunks) - len(valid_chunks)} chunks without embeddings")

        self.chunks.extend(valid_chunks)
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        """Builds cached matrix for fast vectorized operations if numpy is available."""
        if HAS_NUMPY and self.chunks:
            vectors = [c.embedding for c in self.chunks]
            mat = np.array(vectors, dtype=np.float32)
            # Normalize vectors for fast dot-product cosine similarity
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self._embeddings_matrix = mat / norms
        else:
            self._embeddings_matrix = None

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Retrieves top_k most similar chunks for a query embedding.
        Filters by similarity threshold (0.0 to 1.0).
        """
        if not self.chunks:
            return []

        scored_results: List[SearchResult] = []

        if HAS_NUMPY and self._embeddings_matrix is not None:
            # Vectorized cosine similarity: query normalized dot product with normalized matrix
            q_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
            similarities = np.dot(self._embeddings_matrix, q_vec)

            for idx, score in enumerate(similarities):
                float_score = float(score)
                if float_score >= threshold:
                    scored_results.append(SearchResult(chunk=self.chunks[idx], score=float_score))
        else:
            # Fallback pure python cosine similarity
            for chunk in self.chunks:
                score = pure_cosine_similarity(query_embedding, chunk.embedding)
                if score >= threshold:
                    scored_results.append(SearchResult(chunk=chunk, score=score))

        # Sort descending by similarity score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    def save(self, filepath: Union[str, Path]) -> None:
        """Persists the vector index and chunks to disk in JSON format."""
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "total_chunks": len(self.chunks),
            "chunks": [c.to_dict() for c in self.chunks],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: Union[str, Path]) -> bool:
        """Loads vector index and chunks from disk."""
        path = Path(filepath).resolve()
        if not path.is_file():
            return False

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_chunks = data.get("chunks", [])
        self.chunks = [Chunk.from_dict(item) for item in raw_chunks]
        self._rebuild_cache()
        return True

    def clear(self) -> None:
        """Clears all stored chunks and embeddings."""
        self.chunks.clear()
        self._embeddings_matrix = None

    def get_stats(self) -> dict:
        """Returns statistics about the current store."""
        unique_sources = set(c.source for c in self.chunks)
        unique_docs = set(c.doc_id for c in self.chunks)
        total_chars = sum(len(c.text) for c in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(unique_docs),
            "unique_sources": list(unique_sources),
            "total_characters": total_chars,
            "has_numpy_acceleration": HAS_NUMPY,
        }
