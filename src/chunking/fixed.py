from __future__ import annotations

from src.ingestion.models import Document
from src.plugins.registry import chunker_registry

from .base import Chunker
from .utils import make_chunk


@chunker_registry.register("fixed")
class FixedChunker(Chunker):

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must satisfy "
                "0 <= chunk_overlap < chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document):

        chunks = []

        start = 0
        chunk_index = 0

        step = self.chunk_size - self.chunk_overlap

        while start < len(document.text):

            text = document.text[
                start:start + self.chunk_size
            ]

            chunks.append(
                make_chunk(
                    document,
                    text,
                    chunk_index,
                    "fixed",
                    {
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    },
                )
            )

            chunk_index += 1
            start += step

        return chunks