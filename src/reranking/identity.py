from __future__ import annotations

from src.plugins.registry import reranker_registry
from src.retrieval.base import RetrievalResult

from .base import Reranker


@reranker_registry.register("identity")
class IdentityReranker(Reranker):

    def __init__(
            self,
            model: str = None,
            batch_size: int = 16,
            device: str | None = None,
        ):
        pass

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if not results:
            return []

        limit = max(1, top_k)

        reranked_results = []

        for rank, result in enumerate(
            results[:limit],
            start=1,
        ):

            metadata = {
                **result.metadata,

                "reranker": "identity",

                "original_rank": result.rank,

                "retrieval_score": result.score,

                "reranker_score": result.score,
            }

            reranked_results.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=result.score,
                    rank=rank,
                    retriever=result.retriever,
                    metadata=metadata,
                )
            )

        return reranked_results