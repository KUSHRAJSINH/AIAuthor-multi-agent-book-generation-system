"""
RAG text chunker — splits documents into overlapping chunks for embedding.
"""
from __future__ import annotations

from typing import List


class TextChunker:
    """Recursive character-level text splitter with overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._separators = ["\n\n", "\n", ". ", " ", ""]

    def split(self, text: str) -> List[str]:
        """Split *text* into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        return self._recursive_split(text, self._separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        chunks: List[str] = []
        sep = separators[0] if separators else ""
        splits = text.split(sep) if sep else list(text)

        current = ""
        for part in splits:
            candidate = (current + sep + part).lstrip(sep) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                # If single part is too long, recurse with next separator
                if len(part) > self.chunk_size and len(separators) > 1:
                    chunks.extend(self._recursive_split(part, separators[1:]))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current.strip())

        # Apply overlap by prepending the tail of the previous chunk
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped: List[str] = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = overlapped[-1][-self.chunk_overlap:]
                overlapped.append(prev_tail + " " + chunks[i])
            return overlapped

        return chunks

    def split_documents(self, documents: List[str]) -> List[str]:
        all_chunks: List[str] = []
        for doc in documents:
            all_chunks.extend(self.split(doc))
        return all_chunks
