from __future__ import annotations

import numpy as np
import httpx
from tqdm import tqdm
from .base import EmbeddingModel
from src.plugins.registry import embedding_registry


@embedding_registry.register("ollama")
class OllamaEmbedding(EmbeddingModel):
    """Ollama embeddings through its OpenAI-compatible /v1/embeddings API."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        self.model = model
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self.base_url = url
        self.timeout = timeout
        self.batch_size = batch_size
        self.show_progress = show_progress

    @property
    def model_name(self) -> str:
        return self.model

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        response = httpx.post(
            f"{self.base_url}/v1/embeddings",
            json={
                "model": self.model,
                "input": texts,
            },
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()["data"]
        data = sorted(data, key=lambda item: item["index"])

        return np.asarray(
            [item["embedding"] for item in data],
            dtype=np.float32,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        batches = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        # Wrap batch loop in tqdm for progress tracking
        progress_bar = tqdm(
            range(0, len(texts), self.batch_size),
            total=total_batches,
            desc=f"Embedding chunks ({self.model})",
            disable=not self.show_progress,
        )

        for start in progress_bar:
            batch = texts[start : start + self.batch_size]
            batches.append(self._embed_batch(batch))

        vectors = np.vstack(batches)

        self.embedding_size = vectors.shape[1]

        # Normalize once so FAISS inner product becomes cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)

        return vectors.astype(np.float32)