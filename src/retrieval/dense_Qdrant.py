from __future__ import annotations

from src.ingestion.models import Chunk
from src.plugins.registry import (
    embedding_registry,
    retriever_registry,
)

from .base import RetrievalResult, Retriever


@retriever_registry.register("dense_qdrant")
class DenseQdrantRetriever(Retriever):

    def __init__(
        self,
        path: str,
        collection_name: str,
        embedding_provider: str,
        embedding_params: dict | None = None,
    ):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(path=path)
        self.collection_name = collection_name

        self.embedder = embedding_registry.create(
            embedding_provider,
            **(embedding_params or {}),
        )

        # self._validate_dimension()

    def _validate_dimension(self):
        collection = self.client.get_collection(
            collection_name=self.collection_name
        )

        collection_dimension = (
            collection.config.params.vectors.size
        )

        if collection_dimension != self.embedder.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"collection={collection_dimension}, "
                f"query={self.embedder.dimension}"
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if not query.strip():
            return []

        top_k = max(1, top_k)

        query_vector = self.embedder.embed(
            [query],
            is_query=True,
        )[0].tolist()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results = []

        for rank, point in enumerate(
            response.points,
            start=1,
        ):
            payload = dict(point.payload or {})

            chunk_id = str(
                payload.pop("chunk_id")
            )

            text = str(
                payload.pop("text")
            )

            chunk = Chunk(
                chunk_id=chunk_id,
                text=text,
                metadata=payload,
            )

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(point.score),
                    rank=rank,
                    retriever="dense_qdrant",
                    metadata={
                        "point_id": str(point.id),
                    },
                )
            )

        return results