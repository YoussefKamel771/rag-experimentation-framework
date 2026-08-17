from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from configs.config import Settings
from src.plugins.registry import (
    chunker_registry,
    embedding_registry,
    vector_store_registry,
)

from src.plugins.registry_loader import load_all_plugins
from typing import Any

def create_and_save_manifest(
    cfg: Settings,
    documents_count: int,
    chunks_count: int,
    vector_dimension: int,
    elapsed_seconds: float,
) -> dict:
    """Generates the index manifest metadata and writes it to disk."""
    manifest = {
        "framework_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "input_dir": str(cfg.input_dir.resolve()),
        "documents": documents_count,
        "chunks": chunks_count,
        "chunking": {
            "strategy": cfg.chunking.strategy,
            **cfg.chunking.params,
        },
        "embedding": {
            "provider": cfg.embedding.provider,
            "dimension": vector_dimension,
            **cfg.embedding.params
        },
        "vector_store": {
            "type": cfg.vector_store.type,
            **cfg.vector_store.params,
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
    }

    # Ensure output directory exists and save file
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return manifest

