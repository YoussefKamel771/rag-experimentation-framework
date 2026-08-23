from __future__ import annotations

from configs.config import get_settings
from src.evaluation import beir_to_eval_examples, print_leaderboard, generate_html_report
from src.experiments.runner import ExperimentRunner, ExperimentVariant, apply_overrides
from src.ingestion.beir_loader import BEIRDatasetLoader

EXPERIMENT_NAME = "experiment_3_retrieval"
DATASET = "scifact"
SPLIT = "test"

# --------------------------------------------------------------------
# Chunking, embedding, and vector_store all stay exactly as
# config.yaml defines them and are indexed ONCE. Every variant below
# only changes retrieval.type / reranker.type and is evaluated
# eval-only (reindex=False) against that single shared index -- this
# is the cheap sweep, since nothing about the corpus or its vectors
# changes between variants.
#
# reranker.type is pinned to "identity" (pass-through, no-op) for the
# three non-final variants so the comparison isolates the retrieval
# method itself; only the last variant adds the cross-encoder.
# --------------------------------------------------------------------

BASELINE_OVERRIDES = {
    "embedding.provider": "sentence_transformers",
    "embedding.params": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
        "device": "cpu",
    },
    "chunking.strategy": "recursive",
    "chunking.params": {"chunk_size": 1000, "chunk_overlap": 150},
}


VARIANTS = [
    ExperimentVariant(
        name="dense_only",
        overrides={"retrieval.type": "dense"},
        reindex=False,
    ),
    ExperimentVariant(
        name="bm25_only",
        overrides={"retrieval.type": "bm25"},
        reindex=False,
    ),
    ExperimentVariant(
        name="hybrid",
        overrides={"retrieval.type": "hybrid"},
        reindex=False,
    ),
    ExperimentVariant(
        name="hybrid_with_reranker",
        overrides={
            "retrieval.type": "hybrid",
            "reranker.type": "cross_encoder",
        },
        reindex=False,
    ),
]


def main() -> None:
    base_config = get_settings("configs/config.yaml")
    base_config = apply_overrides(base_config, BASELINE_OVERRIDES)


    loader = BEIRDatasetLoader(dataset=DATASET, split=SPLIT).load()
    documents = loader.to_documents()
    examples = beir_to_eval_examples(loader.get_queries(), loader.get_qrels())

    print(f"Loaded {len(documents)} documents, {len(examples)} scorable queries")

    runner = ExperimentRunner(
        base_config=base_config,
        experiment_name=EXPERIMENT_NAME,
        documents=documents,
    )

    # Build the one shared index up front, at config.yaml's own paths
    # (artifacts/qdrant, chunks.json alongside it) -- not isolated per
    # variant, since every variant below reuses it unchanged.
    runner.build_shared_index()

    results = runner.run_all(
        VARIANTS,
        examples,
        dataset=DATASET,
        split=SPLIT,
        save_dir=f"artifacts/eval/{EXPERIMENT_NAME}",
    )

    print()
    print("=" * 90)
    print("EXPERIMENT 3 -- RETRIEVAL METHOD COMPARISON (retrieval stage)")
    print("=" * 90)
    print_leaderboard(results, stage="retrieval", sort_by="ndcg_cut_10")

    print()
    print("=" * 90)
    print("EXPERIMENT 3 -- RETRIEVAL METHOD COMPARISON (reranking stage)")
    print("=" * 90)
    print("(reranking == retrieval for the three identity-reranker variants;")
    print(" only hybrid_plus_reranker actually reranks)")
    print_leaderboard(results, stage="reranking", sort_by="ndcg_cut_10")

    report_path = generate_html_report(
        results,
        experiment_name=EXPERIMENT_NAME,
        output_dir=f"artifacts/eval/{EXPERIMENT_NAME}",
    )
    print(f"\nOpen the visual report: {report_path}")


if __name__ == "__main__":
    main()