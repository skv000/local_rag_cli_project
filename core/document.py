from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class Document:
    """Represents a full document ingested from a file."""
    content: str
    source: str
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            doc_id=data.get("doc_id", str(uuid.uuid4())),
            source=data.get("source", "unknown"),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Chunk:
    """Represents a chunk of a document with its embedding and source information."""
    text: str
    doc_id: str
    source: str
    chunk_index: int
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data.get("chunk_id", str(uuid.uuid4())),
            doc_id=data.get("doc_id", ""),
            source=data.get("source", ""),
            chunk_index=data.get("chunk_index", 0),
            text=data.get("text", ""),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """Represents a chunk retrieved from the vector store with similarity score."""
    chunk: Chunk
    score: float

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.4f}, source='{self.chunk.source}', snippet='{self.chunk.text[:60]}...')"
