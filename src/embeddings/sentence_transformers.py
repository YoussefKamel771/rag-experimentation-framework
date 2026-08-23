from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.plugins.registry import embedding_registry
from .base import EmbeddingModel


@embedding_registry.register("sentence_transformers")
class SentenceTransformersEmbedding(EmbeddingModel):

    def __init__(
        self,
        model: str,
        batch_size: int = 32,
        device: str | None = None,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
    ):
        self.model = model
        self.batch_size = batch_size

        is_e5_family = "e5" in model.lower()

        self.query_prefix = (
            query_prefix
            if query_prefix is not None
            else ("query: " if is_e5_family else "")
        )
        self.passage_prefix = (
            passage_prefix
            if passage_prefix is not None
            else ("passage: " if is_e5_family else "")
        )

        self._model = SentenceTransformer(
            model,
            device=device,
        )

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def dimension(self) -> int:
        return int(
            self._model.get_embedding_dimension()
        )

    def embed(
        self,
        texts: list[str],
        is_query: bool = False,
    ) -> np.ndarray:
        """
        is_query:
            Must be True when embedding a search query, False (default)
            when embedding a document/chunk to be indexed. This is what
            selects "query: " vs. "passage: " for e5-family models --
            getting this wrong for an e5 model breaks the asymmetric
            query/passage representation the model was trained on, not
            just "makes results slightly worse".
        """

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        prefix = self.query_prefix if is_query else self.passage_prefix
 
        prepared_texts = [
            self._prepare_text(text, prefix)
            for text in texts
        ]

        vectors = self._model.encode(
            prepared_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    @staticmethod
    def _prepare_text(text: str, prefix: str) -> str:
        if not prefix or text.startswith(prefix):
            return text
 
        return f"{prefix}{text}"