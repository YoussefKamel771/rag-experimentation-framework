from __future__ import annotations

from abc import ABC, abstractmethod

from src.retrieval.base import RetrievalResult


class Reranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Re-rank retrieved candidates according to the query.

        Args:
            query: User query.
            results: Candidate retrieval results.
            top_k: Number of final results to return.

        Returns:
            Re-ranked retrieval results.
        """
        raise NotImplementedError