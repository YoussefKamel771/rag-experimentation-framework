from __future__ import annotations

import json, os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.plugins.registry import vector_store_registry
from .base import VectorStore


@vector_store_registry.register("qdrant")
class QdrantStore(VectorStore):

    def __init__(
        self,
        path: str | None = "artifacts",
        collection_name: str = "rag_chunks",
        distance: str = "cosine",
    ):
        if distance != "cosine":
            raise ValueError(
                "Only cosine distance is supported currently."
            )

        self.path = path
        self.collection_name = collection_name
        self.distance = distance

        self.client: QdrantClient | None = None
        self.chunks = []
        self.dimension: int | None = None

    def build(self, vectors, chunks) -> None:

        if len(vectors) == 0:
            raise ValueError(
                "Cannot build Qdrant index from empty vectors."
            )

        if len(vectors) != len(chunks):
            raise ValueError(
                f"Vectors/chunks count mismatch: "
                f"{len(vectors)} vectors, "
                f"{len(chunks)} chunks."
            )

        if not self.path:
            raise ValueError(
                "A local Qdrant path is required."
            )

        self.dimension = int(vectors.shape[1])
        self.chunks = chunks

        qdrant_path = Path(self.path)
        qdrant_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = QdrantClient(
            path=str(qdrant_path)
        )

        self._create_collection()

        points = self._create_points(
            vectors,
            chunks,
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def _create_collection(self) -> None:

        if self.client is None:
            raise RuntimeError(
                "Qdrant client has not been initialized."
            )

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def _create_points(self, vectors, chunks):

        points = []

        for vector, chunk in zip(vectors, chunks):

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    str(chunk.chunk_id),
                )
            )

            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                **chunk.metadata,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        return points

    def save(self) -> None:

        output_dir = Path(self.path)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in self.chunks
        ]

        (output_dir / "chunks.json").write_text(
            json.dumps(
                chunks,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        config = {
            "path": str(
                Path(self.path).resolve()
            ),
            "collection_name": self.collection_name,
            "distance": self.distance,
            "dimension": self.dimension,
        }

        (output_dir / "qdrant_config.json").write_text(
            json.dumps(
                config,
                indent=2,
            ),
            encoding="utf-8",
        )