# RAG Experimentation Framework

A modular, framework-free RAG experimentation project.

## Current scope: Offline / indexing pipeline

```text
Documents
   ↓
Document Loader
   ↓
Cleaning / Normalization
   ↓
Pluggable Chunker
   ↓
Pluggable Embedding Model
   ↓
Pluggable Vector Index
   ↓
Persisted Index + Chunk Metadata
```

The first implementation deliberately avoids LangChain/LlamaIndex so the RAG mechanics remain explicit.

## Requirements

- Python 3.11+
- Ollama running locally
- An Ollama embedding model, for example:

```bash
ollama pull nomic-embed-text
```

## Install

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e .
```

## Add documents

Put PDF, Markdown, or TXT files under:

```text
data/raw/
```

## Build an index

```bash
python scripts/ingest.py \
  --input-dir data/raw \
  --output-dir artifacts/indexes/baseline \
  --embedding-model nomic-embed-text \
  --chunk-size 1000 \
  --chunk-overlap 150
```

The command creates:

```text
artifacts/indexes/baseline/
├── index.faiss
├── chunks.json
└── manifest.json
```

`chunks.json` contains the chunk text and metadata. `manifest.json` records the exact indexing configuration so experiments are reproducible.

## Current modules

- `src/ingestion/` — document loading and cleaning
- `src/chunking/` — chunker interface + recursive chunker
- `src/embeddings/` — embedding interface + Ollama implementation
- `src/vectorstores/` — FAISS persistence
- `src/pipeline/` — offline indexing orchestration

## Design rule

The evaluation dataset should reference stable document/page/section information, not chunk IDs, because chunking itself is an experimental variable.

## Next milestones

1. Add fixed-size and semantic chunkers.
2. Add more embedding providers.
3. Add Qdrant.
4. Implement online retrieval.
5. Add evaluation dataset + retrieval metrics.
6. Add experiment runner and comparison dashboard.
