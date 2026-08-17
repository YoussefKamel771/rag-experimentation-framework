from __future__ import annotations

import hashlib
from src.plugins.registry import chunker_registry

from src.ingestion.models import Chunk, Document
from .base import Chunker

@chunker_registry.register("recursive")
class RecursiveChunker(Chunker):
    """
    Character-based recursive chunker.

    It tries boundaries in this order:
        paragraph -> newline -> sentence-ish boundary -> space -> characters

    This is intentionally simple and transparent for experimentation.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: list[str] | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split(self, text: str, separator_index: int = 0) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if separator_index >= len(self.separators):
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = self.separators[separator_index]

        if separator == "":
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(separator)

        if len(parts) == 1:
            return self._split(text, separator_index + 1)

        pieces: list[str] = []
        current = ""

        for part in parts:
            candidate = part if not current else current + separator + part

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                pieces.extend(self._split(current, separator_index + 1))
                current = part
            else:
                pieces.extend(self._split(part, separator_index + 1))
                current = ""

        if current:
            pieces.extend(self._split(current, separator_index + 1))

        return [p.strip() for p in pieces if p.strip()]

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if not chunks or self.chunk_overlap == 0:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            previous = chunks[i - 1]
            overlap = previous[-self.chunk_overlap:]

            # Avoid creating a chunk larger than the configured size.
            current = overlap + " " + chunks[i]
            if len(current) > self.chunk_size:
                current = current[-self.chunk_size:]

            result.append(current.strip())

        return result

    def chunk(self, document: Document) -> list[Chunk]:
        raw_chunks = self._split(document.text)
        raw_chunks = self._add_overlap(raw_chunks)

        chunks: list[Chunk] = []

        for index, text in enumerate(raw_chunks):
            stable_key = (
                f"{document.document_id}:{index}:{hashlib.sha1(text.encode()).hexdigest()[:8]}"
            )
            chunk_id = hashlib.sha1(stable_key.encode()).hexdigest()[:16]

            metadata = {
                **document.metadata,
                "document_id": document.document_id,
                "chunk_index": index,
                "chunking_strategy": "recursive",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                )
            )

        return chunks
