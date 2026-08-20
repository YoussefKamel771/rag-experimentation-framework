# RAG Experimentation Framework

> A modular, plugin-based framework for systematically building, evaluating, and comparing Retrieval-Augmented Generation (RAG) systems.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Plugin%20%2F%20Registry-green.svg)](#plugin--registry-architecture)
[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)](#project-status)

## Overview

RAG systems are rarely improved by changing only one component. Chunking,
embeddings, retrieval, reranking, context construction, and generation all
affect the final answer.

This project provides a configurable experimentation framework where each
major RAG component can be replaced independently and evaluated against the
same dataset.

```text
Documents
   ↓
Chunking → Embedding → Vector Store
                         ↓
Query → Retrieval → Reranking → Context → Generation
                                      ↓
                                  Evaluation
```

## Why this project?

A basic RAG pipeline can answer questions, but it is difficult to determine
*why* it succeeds or fails.

This framework is designed to answer questions such as:

- Which chunking strategy works best?
- Which embedding model gives the best retrieval?
- Does BM25 outperform dense retrieval?
- Does hybrid retrieval improve recall?
- Does reranking improve MRR or nDCG?
- Did retrieval fail, or did the LLM fail to use good context?
- Which complete RAG configuration performs best?

The goal is to make RAG experimentation **systematic, reproducible, and
configuration-driven**.

## Architecture

```text
                         ┌────────────────────┐
                         │     Documents      │
                         └─────────┬──────────┘
                                   ↓
                         ┌────────────────────┐
                         │ Chunking Registry  │
                         └─────────┬──────────┘
                                   ↓
                         ┌────────────────────┐
                         │ Embedding Registry │
                         └─────────┬──────────┘
                                   ↓
                         ┌────────────────────┐
                         │ Vector Store       │
                         │ Registry           │
                         └─────────┬──────────┘
                                   ↓
                              OFFLINE INDEX
════════════════════════════════════════════════════════════
                              ONLINE RAG
                                   ↓
                                Query
                                   ↓
                         ┌────────────────────┐
                         │ Retriever Registry │
                         └─────────┬──────────┘
                                   ↓
                              Candidate Top-N
                                   ↓
                         ┌────────────────────┐
                         │ Reranker Registry  │
                         └─────────┬──────────┘
                                   ↓
                                  Top-K
                                   ↓
                         ┌────────────────────┐
                         │ Context Builder    │
                         │ Registry            │
                         └─────────┬──────────┘
                                   ↓
                         ┌────────────────────┐
                         │ Generator Registry │
                         └─────────┬──────────┘
                                   ↓
                            Answer + Sources
                                   ↓
                         ┌────────────────────┐
                         │ Evaluation Layer   │
                         └────────────────────┘
```

## Current Features

- Modular offline indexing pipeline
- Configurable chunking strategies
- Embedding model registry
- FAISS and Qdrant vector-store support
- Dense retrieval
- BM25 retrieval
- Hybrid retrieval
- Identity and cross-encoder reranking
- Context builder with source preservation and character budget
- Generator registry
- Ollama generation

## Plugin / Registry Architecture

Each major component is independently registered:

```text
chunker_registry
embedding_registry
vector_store_registry
retriever_registry
reranker_registry
context_builder_registry
generator_registry
```

For example:

```text
retriever_registry
├── dense
├── bm25
└── hybrid
```

and:

```text
reranker_registry
├── identity
└── cross_encoder
```

A new implementation can be added without changing the main RAG pipeline.

Example:

```python
@retriever_registry.register("my_retriever")
class MyRetriever:
    ...
```

Then select it from YAML:

```yaml
retriever:
  type: my_retriever
```

This makes the framework suitable for controlled experiments.

## Repository Structure

```text
rag-experimentation-framework/
│
├── configs/
│   ├── rag_full_ollama.yaml
│   ├── rag_full_ollama_dense.yaml
│   └── evaluation_example.yaml
│
├── data/
│   └── evaluation/
│       └── example_dataset.json
│
├── scripts/
│   ├── rag.py
│   ├── evaluate.py
│   └── compare_experiments.py
│
├── src/
│   ├── chunking/
│   ├── embeddings/
│   ├── vectorstores/
│   ├── retrieval/
│   ├── reranking/
│   ├── context/
│   ├── generation/
│   ├── evaluation/
│   ├── pipeline/
│   └── plugins/
│       └── registry.py
│
├── tests/
├── artifacts/
└── README.md
```


## ⚙️ Setup & Installation

### Prerequisites
Make sure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (Anaconda or Miniconda) installed on your system.


### 1. Create and Activate Conda Environment

Create a new Conda environment:

```bash
# Create the environment
conda create -n rag-exp python=3.10 -y

# Activate the environment
conda activate rag-exp

```

### 3. Install Dependencies


```bash
pip install -r requirements.txt
```





## Ollama Setup

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Pull the generation model:

```bash
ollama pull qwen3:4b
```

The models can be changed through YAML configuration.

## Run the RAG Pipeline

```bash
PYTHONPATH=. python scripts/rag.py \
  --config configs/rag_full_ollama.yaml \
  --query "How does sensor fusion improve obstacle detection?"
```

The complete online pipeline is:

```text
Query
 ↓
Hybrid Retrieval
 ↓
Candidate Top-N
 ↓
Cross Encoder
 ↓
Top-K
 ↓
Context Builder
 ↓
Ollama
 ↓
Answer + Sources
```

## Context Building

Retrieved chunks are converted into source-aware context:

```text
[Source 1 | chunk=abc123]
First relevant chunk...

---

[Source 2 | chunk=def456]
Second relevant chunk...
```

A context budget can be configured:

```yaml
context:
  type: simple
  params:
    max_characters: 12000
```

This keeps context construction independent from retrieval and generation.

## Generation

Generation uses a provider registry:

```text
generator_registry
└── ollama
```

Example:

```yaml
generation:
  provider: ollama
  params:
    model: qwen3:4b
    base_url: http://localhost:11434
    temperature: 0.0
```

The provider-independent interface makes future integrations possible without
rewriting the RAG pipeline.




## Roadmap

### Completed

- [x] Modular offline indexing pipeline
- [x] Chunking strategies
- [x] Embedding plugins
- [x] FAISS backend
- [x] Qdrant backend
- [x] Dense retrieval
- [x] BM25 retrieval
- [x] Hybrid retrieval
- [x] Reranking registry
- [x] Cross-encoder reranking
- [x] Context builder
- [x] Generation registry
- [x] Ollama generation

### Planned

- [ ] Evaluation dataset format
- [ ] Retrieval metrics
- [ ] Experiment runner
- [ ] Experiment comparison
- [ ] Stage-level retrieval vs reranking evaluation
- [ ] Context relevance metrics
- [ ] Faithfulness / groundedness evaluation
- [ ] Answer relevance evaluation
- [ ] LLM-as-a-judge
- [ ] Experiment tracking database
- [ ] Results dashboard
- [ ] Automatic hyperparameter sweeps
- [ ] Statistical significance testing
- [ ] More multilingual evaluation
- [ ] Arabic-specific RAG experiments

## Technology Stack

- Python
- Ollama
- FAISS
- Qdrant
- BM25
- Cross-Encoder reranking
- YAML configuration
- JSON evaluation datasets
- Plugin / Registry architecture

The framework is intentionally model- and backend-agnostic.

## Project Status

This is an experimental RAG engineering framework focused on:

- learning RAG deeply
- testing retrieval architectures
- benchmarking models
- understanding RAG failure modes
- prototyping RAG systems
- building reproducible experiments

It is not intended to be a production-ready RAG platform yet.

## Contributing

Contributions are welcome.

A useful contribution should ideally:

1. Use the registry architecture.
2. Include tests.
3. Keep configuration separate from implementation.
4. Document the new component.
5. Include an example experiment where appropriate.

## License

MIT License.

## Author

**Youssef Kamel**

