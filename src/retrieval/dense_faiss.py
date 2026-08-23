from __future__ import annotations

import json
from pathlib import Path

import faiss

from src.ingestion.models import Chunk
from src.plugins.registry import (
    embedding_registry,
    retriever_registry,
)

from .base import RetrievalResult, Retriever


def load_chunks(path: Path) -> list[Chunk]:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    return [
        Chunk(**item)
        for item in data
    ]


@retriever_registry.register("dense_faiss")
class DenseFAISSRetriever(Retriever):

    def __init__(
        self,
        index_dir: str,
        embedding_provider: str,
        embedding_params: dict | None = None,
    ):
        path = Path(index_dir)

        self.index = faiss.read_index(
            str(path / "index.faiss")
        )

        self.chunks = load_chunks(
            path / "chunks.json"
        )

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "FAISS index/chunks size mismatch"
            )

        self.embedder = embedding_registry.create(
            embedding_provider,
            **(embedding_params or {}),
        )

        if self.index.d != self.embedder.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"index={self.index.d}, "
                f"query={self.embedder.dimension}"
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if not query.strip():
            return []

        k = min(
            max(1, top_k),
            len(self.chunks),
        )

        query_vector = self.embedder.embed(
            [query],
            is_query=True,
        )

        scores, ids = self.index.search(
            query_vector,
            k,
        )

        results = []

        for rank, (score, index) in enumerate(
            zip(scores[0], ids[0]),
            start=1,
        ):
            if index < 0:
                continue

            results.append(
                RetrievalResult(
                    chunk=self.chunks[int(index)],
                    score=float(score),
                    rank=rank,
                    retriever="dense_faiss",
                    metadata={
                        "index_position": int(index)
                    },
                )
            )

        return results