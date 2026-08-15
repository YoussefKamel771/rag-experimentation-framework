from __future__ import annotations

import argparse, os
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from src.chunking.recursive import RecursiveChunker
from src.embeddings.ollama import OllamaEmbedding
from src.ingestion.loader import load_directory, load_csv_manifest
from src.vectorstores.faiss_store import FAISSStore


def build_index(
    input_dir: str,
    output_dir: str,
    embedding_model: str,
    ollama_base_url: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    started = time.perf_counter()

    print(f"[1/4] Loading documents from: {input_dir}")
    documents = load_csv_manifest(Path(input_dir))
    print(f"      Loaded {len(documents)} document units/pages")

    print("[2/4] Chunking")
    chunker = RecursiveChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []
    for document in documents:
        chunks.extend(chunker.chunk(document))

    print(f"      Created {len(chunks)} chunks")

    print(f"[3/4] Embedding with Ollama: {embedding_model}")
    embedder = OllamaEmbedding(
        model=embedding_model,
        base_url=ollama_base_url,
    )
    vectors = embedder.embed([chunk.text for chunk in chunks])
    print(f"      Embedding matrix: {vectors.shape}")

    print("[4/4] Building FAISS index")
    store = FAISSStore()
    store.build(vectors, chunks)
    store.save(output_dir)

    elapsed = time.perf_counter() - started

    manifest = {
        "framework_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "input_dir": str(Path(input_dir).resolve()),
        "documents": len(documents),
        "chunks": len(chunks),
        "chunking": {
            "strategy": "recursive",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "embedding": {
            "provider": "ollama",
            "model": embedding_model,
            "dimension": int(vectors.shape[1]),
            "normalized": True,
        },
        "vector_store": {
            "type": "faiss",
            "metric": "cosine_via_inner_product",
        },
        "elapsed_seconds": round(elapsed, 3),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Index created successfully.")
    print(f"  Output: {output_path}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Dimension: {vectors.shape[1]}")
    print(f"  Time: {elapsed:.2f}s")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the offline RAG index."
    )

    parser.add_argument("--input-dir",
                         default=os.path.join(os.getcwd(), "data/raw/text_data_toc.csv"))
    parser.add_argument("--output-dir",
                         default=os.path.join(os.getcwd(), "artifacts")) 
    parser.add_argument(
        "--embedding-model",
        default="embeddinggemma:300m",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="https://condone-hankering-cushy.ngrok-free.dev",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)

    args = parser.parse_args()

    build_index(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        ollama_base_url=args.ollama_base_url,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
