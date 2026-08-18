from __future__ import annotations

import argparse
from typing import Any

from configs.config import get_settings, Settings
from src.pipeline.reranker import retrieve_and_rerank


def run_retrieval_and_rerank(
    query: str,
    config: Settings,
) -> list[Any]:
    """Execute the retrieval and reranking pipeline and print detailed results."""
    # --------------------------------------------------
    # Run retrieval + reranking
    # --------------------------------------------------
    results = retrieve_and_rerank(
        config=config,
        query=query,
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------
    print()
    print("=" * 90)
    print("RAG RETRIEVAL + RERANKING")
    print("=" * 90)

    print(f"Query:          {query}")
    print(f"Retriever:      {config.retrieval.type}")
    print(f"Reranker:       {config.reranker.type}")
    print(f"Candidate K:    {config.retrieval.candidate_k}")
    print(f"Final Top K:    {config.reranker.top_k}")
    print(f"Final results:  {len(results)}")

    print("=" * 90)

    for result in results:
        print()
        print("-" * 90)

        print(f"Rank:              {result.rank}")
        print(f"Chunk ID:          {result.chunk.chunk_id}")
        print(f"Retriever:         {result.retriever}")

        print(
            f"Retrieval score:    "
            f"{result.metadata.get('retrieval_score', 'N/A')}"
        )

        print(
            f"Reranker:           "
            f"{result.metadata.get('reranker', 'N/A')}"
        )

        print(
            f"Reranker model:     "
            f"{result.metadata.get('reranker_model', 'N/A')}"
        )

        print(
            f"Original rank:      "
            f"{result.metadata.get('original_rank', 'N/A')}"
        )

        print(
            f"Reranker score:     "
            f"{result.metadata.get('reranker_score', result.score):.6f}"
        )

        print("-" * 90)

        print(result.chunk.text[:2000])

    print()
    print("=" * 90)

    return results


def main() -> None:
    # parser = argparse.ArgumentParser(
    #     description="Test retrieval + reranking pipeline"
    # )

    # parser.add_argument(
    #     "--config",
    #     required=True,
    #     help="Path to YAML configuration file",
    # )

    # parser.add_argument(
    #     "--query",
    #     required=True,
    #     help="Query to retrieve and rerank",
    # )

    # args = parser.parse_args()

    config = get_settings(config_path="configs/config.yaml")
    run_retrieval_and_rerank(
        query="what is a duck?",
        config=config,
    )


if __name__ == "__main__":
    main()