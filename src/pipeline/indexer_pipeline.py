# src/pipeline/beir_indexer.py
from __future__ import annotations

import time

from configs.config import Settings
from src.ingestion.beir_loader import BEIRDatasetLoader
from src.ingestion.models import Document
from src.pipeline.utils import create_and_save_manifest
from src.plugins.registry import (
    chunker_registry,
    embedding_registry,
    vector_store_registry,
)
from src.plugins.registry_loader import load_all_plugins


class BEIRIndexer:
    """
    Builds a vector index from a BEIR benchmark dataset, reusing the
    same chunking/embedding/vector-store plugins as the regular
    indexing pipeline.
    """

    def __init__(
        self,
        config: Settings,
        documents: list[Document],
    ):
        load_all_plugins()

        self.config = config
        self.documents = documents


    def build(self) -> dict:
        started = time.perf_counter()

        # --------------------------------------------------
        # 1. Embedding
        # --------------------------------------------------
        print(f"[2/4] Creating embedding: {self.config.embedding.provider}")

        embedder = embedding_registry.create(
            self.config.embedding.provider,
            **self.config.embedding.params,
        )

        # --------------------------------------------------
        # 2. Chunking
        # --------------------------------------------------
        print(f"[3/4] Chunking with: {self.config.chunking.strategy}")

        chunker = chunker_registry.create(
            self.config.chunking.strategy,
            **self.config.chunking.params,
        )

        chunks = []
        for document in self.documents:
            chunks.extend(chunker.chunk(document))

        print(f"      Created {len(chunks)} chunks")

        # --------------------------------------------------
        # 3. Embed chunks
        # --------------------------------------------------
        print("      Embedding chunks...")

        vectors = embedder.embed([chunk.text for chunk in chunks])
        embedding_size = int(vectors.shape[1])

        print(f"      Embedding matrix: {vectors.shape}")

        # --------------------------------------------------
        # 4. Vector store
        # --------------------------------------------------
        print(f"[4/4] Building vector store: {self.config.vector_store.type}")

        store = vector_store_registry.create(
            self.config.vector_store.type,
            **self.config.vector_store.params,
        )

        store.build(vectors, chunks)
        store.save()

        elapsed = time.perf_counter() - started

        manifest = create_and_save_manifest(
            cfg=self.config,
            documents_count=len(self.documents),
            chunks_count=len(chunks),
            vector_dimension=embedding_size,
            elapsed_seconds=elapsed,
        )

        print()
        print("BEIR index created successfully.")
        print(f"  Chunks: {len(chunks)}")
        print(f"  Dimension: {embedding_size}")
        print(f"  Time: {elapsed:.2f}s")

        return manifest


def build_index_from_documents(
    config: Settings,
    documents: list[Document],
) -> dict:
    
    indexer = BEIRIndexer(config, documents)
    results = indexer.build()
    return results
