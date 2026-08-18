from __future__ import annotations

import numpy as np

from src.plugins.registry import reranker_registry
from src.retrieval.base import RetrievalResult

from .base import Reranker


@reranker_registry.register("cross_encoder")
class CrossEncoderReranker(Reranker):

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        device: str | None = None,
    ):
        try:
            from sentence_transformers import CrossEncoder

        except ImportError as exc:
            raise ImportError(
                "CrossEncoder reranking requires "
                "sentence-transformers. "
                "Install it with: "
                "pip install -e '.[reranking]'"
            ) from exc

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        self.model_name = model
        self.batch_size = batch_size

        self.model = CrossEncoder(
            model,
            device=device,
        )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if not results:
            return []

        if not query.strip():
            return []

        limit = max(1, top_k)

        # --------------------------------------------------
        # Prepare query/document pairs
        # --------------------------------------------------

        pairs = [
            (query, result.chunk.text)
            for result in results
        ]

        # --------------------------------------------------
        # Cross-encoder scoring
        # --------------------------------------------------

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        scores = np.asarray(
            scores,
            dtype=np.float32,
        ).reshape(-1)

        if len(scores) != len(results):
            raise RuntimeError(
                "Cross-encoder returned an unexpected "
                "number of scores"
            )

        # --------------------------------------------------
        # Create reranked results
        # --------------------------------------------------

        candidates = []

        for result, score in zip(results, scores):

            reranker_score = float(score)

            metadata = {
                **result.metadata,

                "reranker": "cross_encoder",

                "reranker_model": self.model_name,

                "original_rank": result.rank,

                "retrieval_score": result.score,

                "reranker_score": reranker_score,
            }

            candidates.append(
                RetrievalResult(
                    chunk=result.chunk,

                    score=reranker_score,

                    rank=0,

                    retriever=result.retriever,

                    metadata=metadata,
                )
            )

        # --------------------------------------------------
        # Sort according to cross-encoder score
        # --------------------------------------------------

        candidates.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        # --------------------------------------------------
        # Assign final ranks
        # --------------------------------------------------

        final_results = []

        for rank, result in enumerate(
            candidates[:limit],
            start=1,
        ):

            final_results.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=result.score,
                    rank=rank,
                    retriever=result.retriever,
                    metadata=result.metadata,
                )
            )

        return final_results