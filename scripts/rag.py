from __future__ import annotations

import argparse
import json
from typing import Any

from configs.config import get_settings, Settings
from src.pipeline.rag_pipeline import run_rag


def execute_rag(
    query: str,
    config: Settings,
) -> dict[str, Any]:
    """Execute the complete RAG pipeline and print answer, sources, and pipeline details.

    Returns the pipeline output dictionary containing 'generation' and
    'context'.
    """

    # --------------------------------------------------
    # Run RAG
    # --------------------------------------------------
    result = run_rag(
        config=config,
        query=query,
    )

    generation = result["generation"]
    context = result["context"]

    # --------------------------------------------------
    # Answer
    # --------------------------------------------------
    print()
    print("=" * 90)
    print("ANSWER")
    print("=" * 90)
    print(generation.answer)

    # --------------------------------------------------
    # Sources
    # --------------------------------------------------
    print()
    print("=" * 90)
    print("SOURCES")
    print("=" * 90)
    print(
        json.dumps(
            generation.sources,
            ensure_ascii=False,
            indent=2,
        )
    )

    # --------------------------------------------------
    # Pipeline information
    # --------------------------------------------------
    print()
    print("=" * 90)
    print("PIPELINE")
    print("=" * 90)
    print(f"Retriever:       {config.retrieval.type}")
    print(f"Reranker:        {config.reranker.type}")
    print(f"Context builder: {config.context.type}")
    print(f"Generator:       {config.generation.provider}")
    print(f"Model:           {generation.model}")
    print(f"Context chunks:  {context.total_chunks}")
    print(f"Context chars:   {context.total_characters}")
    print("=" * 90)

    return result


def main() -> None:
    # parser = argparse.ArgumentParser(description="Run the complete RAG pipeline")

    # parser.add_argument(
    #     "--config",
    #     required=True,
    #     help="Path to YAML configuration file",
    # )

    # parser.add_argument(
    #     "--query",
    #     required=True,
    #     help="Question to ask",
    # )

    # args = parser.parse_args()

    config = get_settings(config_path="configs/config.yaml")
    execute_rag(
        query="what is a duck?",
        config=config,
    )


if __name__ == "__main__":
    main()