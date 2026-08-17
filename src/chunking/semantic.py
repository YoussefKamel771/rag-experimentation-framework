from __future__ import annotations

import numpy as np

from src.embeddings.base import EmbeddingModel
from src.ingestion.models import Document
from src.plugins.registry import chunker_registry

from .base import Chunker
from .utils import make_chunk, sentence_split


@chunker_registry.register("semantic")
class SemanticChunker(Chunker):

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        max_chunk_size: int = 1200,
        min_chunk_size: int = 150,
        similarity_threshold: float = 0.72,
    ):
        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size must be greater than 0"
            )

        if min_chunk_size < 0:
            raise ValueError(
                "min_chunk_size cannot be negative"
            )

        if not 0 <= similarity_threshold <= 1:
            raise ValueError(
                "similarity_threshold must be between 0 and 1"
            )

        self.embedding_model = embedding_model
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold

    def chunk(self, document: Document):

        sentences = sentence_split(
            document.text
        )

        if len(sentences) <= 1:
            groups = [sentences]

        else:
            groups = self._create_groups(
                sentences
            )

        return [
            make_chunk(
                document,
                " ".join(group),
                index,
                "semantic",
                {
                    "max_chunk_size": self.max_chunk_size,
                    "min_chunk_size": self.min_chunk_size,
                    "similarity_threshold": self.similarity_threshold,
                    "semantic_embedding_model":
                        self.embedding_model.model_name,
                },
            )
            for index, group in enumerate(groups)
            if group
        ]

    def _create_groups(
        self,
        sentences: list[str],
    ):

        vectors = self.embedding_model.embed(
            sentences
        )

        groups = []
        current = [sentences[0]]

        for index in range(1, len(sentences)):

            similarity = float(
                np.dot(
                    vectors[index - 1],
                    vectors[index],
                )
            )

            candidate = " ".join(
                current + [sentences[index]]
            )

            current_text = " ".join(current)

            should_split = (
                (
                    similarity
                    < self.similarity_threshold
                )
                and
                len(current_text)
                >= self.min_chunk_size
            ) or (
                len(candidate)
                > self.max_chunk_size
            )

            if should_split:
                groups.append(current)
                current = [sentences[index]]

            else:
                current.append(sentences[index])

        if current:
            groups.append(current)

        return groups