from __future__ import annotations

from src.ingestion.models import Document
from src.plugins.registry import chunker_registry

from .base import Chunker
from .utils import make_chunk, sentence_split


@chunker_registry.register("sentence")
class SentenceChunker(Chunker):

    def __init__(
        self,
        max_chunk_size: int = 1000,
        overlap_sentences: int = 1,
    ):
        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size must be greater than 0"
            )

        if overlap_sentences < 0:
            raise ValueError(
                "overlap_sentences cannot be negative"
            )

        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def chunk(self, document: Document):

        sentences = sentence_split(document.text)

        chunks = []
        current = []

        for sentence in sentences:

            candidate = " ".join(
                current + [sentence]
            )

            if (
                current
                and len(candidate) > self.max_chunk_size
            ):
                chunks.append(
                    " ".join(current)
                )

                current = (
                    current[-self.overlap_sentences:]
                    + [sentence]
                )

            else:
                current.append(sentence)

        if current:
            chunks.append(" ".join(current))

        return [
            make_chunk(
                document,
                text,
                index,
                "sentence",
                {
                    "max_chunk_size": self.max_chunk_size,
                    "overlap_sentences": self.overlap_sentences,
                },
            )
            for index, text in enumerate(chunks)
        ]