from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from src.plugins.registry import vector_store_registry
from .base import VectorStore
from  src.ingestion.models import Chunk

@vector_store_registry.register("faiss")
class FAISSStore(VectorStore):
    """
    Persistent FAISS store using normalized vectors + inner product,
    which is equivalent to cosine similarity.
    """

    def __init__(self, path: str | Path):
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []
        self.path = path

    def build(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("Number of vectors must equal number of chunks")

        if len(vectors) == 0:
            raise ValueError("Cannot build an empty index")


        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)
        self.chunks = chunks

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Index has not been built")

        output = Path(self.path)
        output.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(output / "index.faiss"))

        chunks_data = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in self.chunks
        ]

        (output / "chunks.json").write_text(
            json.dumps(chunks_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("FAISS saved")

    @classmethod
    def load(cls, path: str | Path) -> "FAISSStore":
        output = Path(path)

        store = cls()
        store.index = faiss.read_index(str(output / "index.faiss"))

        chunks_data = json.loads(
            (output / "chunks.json").read_text(encoding="utf-8")
        )

        store.chunks = [
            Chunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                metadata=item["metadata"],
            )
            for item in chunks_data
        ]

        return store
