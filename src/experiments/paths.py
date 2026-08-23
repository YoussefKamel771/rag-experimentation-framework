from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.config import Settings


def derive_variant_paths(
    config: Settings,
    experiment_name: str,
    variant_name: str,
    root: str = "artifacts/experiments",
) -> dict[str, Any]:
    """
    Build the dotted-path overrides that give one experiment variant
    its own isolated artifact directory.

    Without this, sweeping embedding models or chunking strategies
    against the SAME vector_store.params.path / Qdrant collection_name
    would silently overwrite the previous variant's index
    (QdrantStore._create_collection calls recreate_collection), making
    it impossible to re-evaluate an earlier variant without rebuilding
    it. Only call this for variants that actually reindex --
    retrieval-method sweeps (experiment 3) reuse one shared index and
    should NOT have their paths isolated.

    Both FAISSStore and QdrantStore write chunks.json into their own
    `self.path` (== vector_store.params.path) on save(), so BM25's
    chunks_path is derived from the same directory rather than from
    output_dir.
    """
    variant_dir = Path(root) / experiment_name / variant_name
    store_dir = variant_dir / config.vector_store.type

    overrides: dict[str, Any] = {
        "output_dir": str(variant_dir),
        "vector_store.params.path": str(store_dir),
    }

    if config.vector_store.type == "qdrant":
        # Qdrant collections live inside the on-disk path above, but
        # recreate_collection() is scoped by name too -- isolate both
        # so nothing about a previous variant can leak into this one.
        overrides["vector_store.params.collection_name"] = (
            f"{experiment_name}_{variant_name}"
        )

    if config.retrieval.lexical is not None:
        overrides["retrieval.lexical.params.chunks_path"] = str(
            store_dir / "chunks.json"
        )

    return overrides