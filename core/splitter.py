from typing import List, Optional
from .document import Document, Chunk


class RecursiveTextSplitter:
    """
    Splits text recursively using an ordered hierarchy of separators,
    preserving natural semantic boundaries (paragraphs, sentences, words).
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by separators until pieces fit within chunk_size."""
        final_chunks: List[str] = []

        # Find the first separator present in text
        separator = separators[-1]
        new_separators: List[str] = []

        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        # Split text by the selected separator
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits up to chunk_size with overlap
        good_splits: List[str] = []
        for piece in splits:
            if not piece.strip() and separator != "":
                continue

            # If piece alone exceeds chunk_size and more fine-grained separators exist
            if len(piece) > self.chunk_size and new_separators:
                # Recurse on this oversized piece
                sub_splits = self._split_text(piece, new_separators)
                good_splits.extend(sub_splits)
            else:
                good_splits.append(piece)

        # Group good_splits with sliding window overlap
        accumulated: List[str] = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if accumulated else 0)

            if current_len + piece_len > self.chunk_size and accumulated:
                # Flush current accumulated chunk
                chunk_str = separator.join(accumulated).strip()
                if chunk_str:
                    final_chunks.append(chunk_str)

                # Keep overlap pieces from the end
                overlap_accumulated: List[str] = []
                overlap_len = 0
                for prev_piece in reversed(accumulated):
                    needed_len = len(prev_piece) + (len(separator) if overlap_accumulated else 0)
                    if overlap_len + needed_len <= self.chunk_overlap:
                        overlap_accumulated.insert(0, prev_piece)
                        overlap_len += needed_len
                    else:
                        break

                accumulated = overlap_accumulated
                current_len = overlap_len

            accumulated.append(piece)
            current_len += len(piece) + (len(separator) if len(accumulated) > 1 else 0)

        if accumulated:
            chunk_str = separator.join(accumulated).strip()
            if chunk_str:
                final_chunks.append(chunk_str)

        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """Splits raw string into a list of chunk texts."""
        if not text or not text.strip():
            return []
        return self._split_text(text.strip(), self.separators)

    def split_document(self, document: Document) -> List[Chunk]:
        """Splits a Document into a list of Chunk objects."""
        text_chunks = self.split_text(document.content)
        chunks: List[Chunk] = []

        for idx, chunk_text in enumerate(text_chunks):
            chunk = Chunk(
                text=chunk_text,
                doc_id=document.doc_id,
                source=document.source,
                chunk_index=idx,
                metadata={
                    **document.metadata,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                },
            )
            chunks.append(chunk)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Splits multiple documents into chunks."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
