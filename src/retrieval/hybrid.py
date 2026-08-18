from __future__ import annotations

from src.plugins.registry import retriever_registry

from .base import RetrievalResult, Retriever


def normalize_scores(
    results: list[RetrievalResult],
) -> dict[str, float]:
    """
    Normalize retrieval scores to the range [0, 1].

    Returns:
        {
            chunk_id: normalized_score
        }
    """

    if not results:
        return {}

    scores = [
        result.score
        for result in results
    ]

    minimum = min(scores)
    maximum = max(scores)

    # All scores are identical
    if maximum - minimum < 1e-12:
        return {
            result.chunk.chunk_id: 1.0
            for result in results
        }

    return {
        result.chunk.chunk_id:
            (
                result.score - minimum
            )
            / (
                maximum - minimum
            )
        for result in results
    }


@retriever_registry.register("hybrid")
class HybridRetriever(Retriever):

    def __init__(
        self,
        dense_retriever: Retriever,
        lexical_retriever: Retriever,
        alpha: float = 0.7,
        candidate_k: int = 20,
    ):
        """
        Args:
            dense_retriever:
                Dense retriever such as FAISS or Qdrant.

            lexical_retriever:
                Lexical retriever such as BM25.

            alpha:
                Weight assigned to dense retrieval.

                1.0 = only dense
                0.0 = only lexical
                0.7 = 70% dense + 30% lexical

            candidate_k:
                Number of candidates retrieved from
                each retriever before fusion.
        """

        if not 0 <= alpha <= 1:
            raise ValueError(
                "alpha must be between 0 and 1"
            )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0"
            )

        self.dense_retriever = (
            dense_retriever
        )

        self.lexical_retriever = (
            lexical_retriever
        )

        self.alpha = alpha
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if not query.strip():
            return []

        # -------------------------------------------------
        # 1. Retrieve candidates
        # -------------------------------------------------

        dense_results = (
            self.dense_retriever.retrieve(
                query,
                self.candidate_k,
            )
        )

        lexical_results = (
            self.lexical_retriever.retrieve(
                query,
                self.candidate_k,
            )
        )

        # -------------------------------------------------
        # 2. Normalize scores
        # -------------------------------------------------

        dense_scores = normalize_scores(
            dense_results
        )

        lexical_scores = normalize_scores(
            lexical_results
        )

        # -------------------------------------------------
        # 3. Collect all unique chunks
        # -------------------------------------------------

        chunks = {}

        for result in dense_results:
            chunks[result.chunk.chunk_id] = (
                result.chunk
            )

        for result in lexical_results:
            chunks[result.chunk.chunk_id] = (
                result.chunk
            )

        # -------------------------------------------------
        # 4. Calculate hybrid score
        # -------------------------------------------------

        combined_results = []

        for chunk_id, chunk in chunks.items():

            dense_score = dense_scores.get(
                chunk_id,
                0.0,
            )

            lexical_score = lexical_scores.get(
                chunk_id,
                0.0,
            )

            hybrid_score = (
                self.alpha * dense_score
                +
                (1 - self.alpha)
                * lexical_score
            )

            combined_results.append(
                {
                    "chunk": chunk,
                    "score": hybrid_score,
                    "dense_score": dense_score,
                    "lexical_score": lexical_score,
                }
            )

        # -------------------------------------------------
        # 5. Sort by hybrid score
        # -------------------------------------------------

        combined_results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # -------------------------------------------------
        # 6. Create final RetrievalResult objects
        # -------------------------------------------------

        top_k = max(
            1,
            min(
                top_k,
                len(combined_results),
            ),
        )

        results = []

        for rank, item in enumerate(
            combined_results[:top_k],
            start=1,
        ):

            results.append(
                RetrievalResult(
                    chunk=item["chunk"],
                    score=float(
                        item["score"]
                    ),
                    rank=rank,
                    retriever="hybrid",
                    metadata={
                        "dense_score":
                            item["dense_score"],

                        "lexical_score":
                            item["lexical_score"],

                        "alpha":
                            self.alpha,

                        "candidate_k":
                            self.candidate_k,
                    },
                )
            )

        return results