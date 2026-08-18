from __future__ import annotations

from configs.config import Settings

from src.plugins.registry import reranker_registry

from src.plugins.registry_loader import load_all_plugins
from .retrieval_pipeline import create_retriever


def create_reranker(
    config: Settings,
):
    """
    Create the configured reranker plugin.
    """

    load_all_plugins()

    reranker_config = config.reranker

    return reranker_registry.create(
        reranker_config.type,
        **reranker_config.params,
    )


def retrieve_and_rerank(
    config: Settings,
    query: str,
):
    """
    Retrieve candidate chunks and optionally
    rerank them.
    """

    retriever = create_retriever(config)

    retrieval_config = config.retrieval

    reranker_config = config.reranker

    # --------------------------------------------------
    # Number of candidates sent to reranker
    # --------------------------------------------------

    candidate_k = retrieval_config.candidate_k

    # --------------------------------------------------
    # Final number of results
    # --------------------------------------------------

    top_k = reranker_config.top_k

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    candidates = retriever.retrieve(
        query,
        top_k=candidate_k,
    )

    # --------------------------------------------------
    # Reranking
    # --------------------------------------------------

    reranker = create_reranker(config)

    return reranker.rerank(
        query,
        candidates,
        top_k=top_k,
    )