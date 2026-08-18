# RAG Experimentation Framework — Offline v0.2

The offline pipeline is now configuration-driven and plugin-based.

```text
Documents → Loader → Chunker Registry → Embedding Registry → Vector Store Registry
```

## Plugins

Chunkers: `fixed`, `recursive`, `sentence`, `markdown`, `semantic`

Embedding providers: `ollama`, `sentence_transformers`

Vector stores: `faiss`, `qdrant`

For Arabic/English experiments, `intfloat/multilingual-e5-small` is included as a local Sentence Transformers option. Ollama remains the simplest local provider.

## Install

```bash
pip install -e .
```

For Sentence Transformers:

```bash
pip install -e ".[hf]"
```

## Ollama

```bash
ollama pull nomic-embed-text
```

## Add data

Put PDF/TXT/Markdown files in `data/raw/`.

## List plugins

```bash
python scripts/list_plugins.py
```

## Run experiments

```bash
python scripts/ingest.py --config configs/baseline.yaml
python scripts/ingest.py --config configs/semantic_qdrant.yaml
python scripts/ingest.py --config configs/multilingual_faiss.yaml
python scripts/ingest.py --config configs/markdown_qdrant.yaml
```

Each experiment produces a self-contained artifact directory with a manifest describing the selected plugins and parameters.

## Architecture

```text
                    OFFLINE INDEXING

Documents
   ↓
Loader
   ↓
Chunker Registry
   ├─ fixed
   ├─ recursive
   ├─ sentence
   ├─ markdown
   └─ semantic ───────┐
                       │ needs embeddings
                       ▼
Embedding Registry
   ├─ Ollama
   └─ SentenceTransformers
   ↓
Vector Store Registry
   ├─ FAISS
   └─ Qdrant
   ↓
Artifacts + manifest
```

The evaluation dataset should refer to stable source information (document/page/section), not chunk IDs, because chunking is itself an experimental variable.
