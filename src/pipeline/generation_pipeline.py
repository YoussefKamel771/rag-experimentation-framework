from __future__ import annotations

from configs.config import Settings

from src.plugins.registry import (
    context_builder_registry,
    generator_registry,
)

from src.plugins.registry_loader import load_all_plugins
from src.pipeline.reranking_pipeline import  retrieve_and_rerank


def create_context_builder(
    config: Settings,
):
    """
    Create the configured context builder.
    """

    load_all_plugins()

    context_config = config.context

    return context_builder_registry.create(
        context_config.type,
        **context_config.params,
    )


def create_generator(
    config: Settings,
):
    """
    Create the configured generation provider.
    """

    load_all_plugins()

    generation_config = config.generation

    return generator_registry.create(
        generation_config.provider,
        **generation_config.params,
    )


def build_sources(context):
    """
    Convert context items into source information
    that can be returned with the generated answer.
    """

    sources = []

    for source_number, item in enumerate(
        context.items,
        start=1,
    ):
        sources.append(
            {
                "source_number": source_number,
                "chunk_id": item.chunk_id,
                "rank": item.rank,
                "score": item.score,
                "metadata": item.metadata,
            }
        )

    return sources


def run_rag(
    config: Settings,
    query: str,
):
    """
    Run the complete RAG pipeline:

        Query
          ↓
        Retrieval
          ↓
        Reranking
          ↓
        Context Building
          ↓
        Generation
          ↓
        Answer
    """

    if not query.strip():
        raise ValueError(
            "query cannot be empty"
        )

    # --------------------------------------------------
    # 1. Retrieval + reranking
    # --------------------------------------------------

    retrieval_results = retrieve_and_rerank(
        config=config,
        query=query,
    )

    # --------------------------------------------------
    # 2. Build context
    # --------------------------------------------------

    context_builder = create_context_builder(
        config
    )

    context = context_builder.build(
        retrieval_results
    )

    # --------------------------------------------------
    # 3. Build source information
    # --------------------------------------------------

    sources = build_sources(context)

    # --------------------------------------------------
    # 4. Generate answer
    # --------------------------------------------------

    generator = create_generator(config)

    generation = generator.generate(
        query=query,
        context=context.text,
        sources=sources,
    )

    # --------------------------------------------------
    # 5. Return complete RAG result
    # --------------------------------------------------

    return {
        "query": query,
        "retrieval_results": retrieval_results,
        "context": context,
        "generation": generation,
    }
