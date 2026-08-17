from __future__ import annotations


import time, os

from pathlib import Path
from src.ingestion.loader import  load_csv_manifest

from configs.config import get_settings, Settings
from src.pipeline.utils import create_and_save_manifest
from src.plugins.registry import (
    chunker_registry,
    embedding_registry,
    vector_store_registry,
)

from src.plugins.registry_loader import load_all_plugins


def build_index(config: Settings) -> dict:
    load_all_plugins()

    started = time.perf_counter()

    print(f"[1/4] Loading documents from: {config.input_dir}")

    documents = load_csv_manifest(Path(config.input_dir))

    print(f"      Loaded {len(documents)} document units/pages")

    # ---------------------------------------------------------
    # 1. Create embedding plugin
    # ---------------------------------------------------------

    print(
        f"[2/4] Creating embedding: "
        f"{config.embedding.provider}"
    )

    embedder = embedding_registry.create(
        config.embedding.provider,
        **config.embedding.params
    )

    # ---------------------------------------------------------
    # 2. Create chunker plugin
    # ---------------------------------------------------------

    print(
        f"[3/4] Chunking with: "
        f"{config.chunking.strategy}"
    )

    chunker = chunker_registry.create(
        config.chunking.strategy,
        **config.chunking.params,
    )

    chunks = []

    for document in documents:
        chunks.extend(chunker.chunk(document))

    print(f"      Created {len(chunks)} chunks")

    # ---------------------------------------------------------
    # 3. Embed chunks
    # ---------------------------------------------------------

    print("      Embedding chunks...")

    vectors = embedder.embed(
        [chunk.text for chunk in chunks]
    )

    embedding_size = int(vectors.shape[1])

    print(f"      Embedding matrix: {vectors.shape}")
    print(f"      Embedding dimension: {embedding_size}")

    # ---------------------------------------------------------
    # 4. Create vector store plugin
    # ---------------------------------------------------------

    print(
        f"[4/4] Building vector store: "
        f"{config.vector_store.type}"
    )

    store = vector_store_registry.create(
        config.vector_store.type,
        **config.vector_store.params,
    )

    store.build(vectors, chunks) 

    store.save()

    elapsed = time.perf_counter() - started

    # ---------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------

    manifest = create_and_save_manifest(
        cfg=config,
        documents_count=len(documents),
        chunks_count=len(chunks),
        vector_dimension=int(embedding_size),
        elapsed_seconds=elapsed,
    )

    print()
    print("Index created successfully.")
    print(f"  Output: {config.output_dir}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Dimension: {embedding_size}")
    print(f"  Time: {elapsed:.2f}s")

    return manifest


def main() -> None:
    config = get_settings()
    build_index(config)
