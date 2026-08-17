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
    ):
        self.model = model
        self.batch_size = batch_size

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

    def embed(self, texts: list[str]) -> np.ndarray:

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        prepared_texts = [
            self._prepare_text(text)
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
    def _prepare_text(text: str) -> str:
        """
        Prepare text for E5-style embedding models.

        Indexed documents use the 'passage:' prefix.
        """
        if text.startswith("passage:"):
            return text

        return f"passage: {text}"