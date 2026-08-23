from __future__ import annotations

import time
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
        max_retries: int = 3,
    ):
        self.model = model
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self.base_url = url
        self.timeout = timeout
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.max_retries = max_retries
        self.dimension = 0

    @property
    def model_name(self) -> str:
        return self.model

    def _embed_batch_with_retry(self, client: httpx.Client, texts: list[str]) -> np.ndarray:
        """Sends a single batch request with retry logic for connection drops."""
        url = f"{self.base_url}/v1/embeddings"
        payload = {
            "model": self.model,
            "input": texts,
        }
        headers = {"ngrok-skip-browser-warning": "true"}

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                data = response.json()["data"]
                data = sorted(data, key=lambda item: item["index"])

                return np.asarray(
                    [item["embedding"] for item in data],
                    dtype=np.float32,
                )
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.HTTPStatusError) as e:
                if attempt == self.max_retries:
                    raise e
                # Wait briefly before retrying dropped connections
                time.sleep(2 * attempt)

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        batches = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        # Use a persistent HTTP Client session with reasonable keep-alive timeouts
        limits = httpx.Limits(max_keepalive_connections=5, keepalive_expiry=10.0)
        with httpx.Client(timeout=self.timeout, limits=limits) as client:
            progress_bar = tqdm(
                range(0, len(texts), self.batch_size),
                total=total_batches,
                desc=f"Embedding chunks ({self.model})",
                disable=not self.show_progress,
            )

            for start in progress_bar:
                batch = texts[start : start + self.batch_size]
                batches.append(self._embed_batch_with_retry(client, batch))

        vectors = np.vstack(batches)
        self.dimension = vectors.shape[1]

        # Normalize once so FAISS inner product becomes cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)

        return vectors.astype(np.float32)