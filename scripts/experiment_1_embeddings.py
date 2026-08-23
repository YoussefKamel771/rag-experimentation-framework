from __future__ import annotations

from configs.config import get_settings
from src.evaluation import pairs_against_baseline, print_variant_pairs
from src.evaluation import beir_to_eval_examples, print_leaderboard, generate_html_report
from src.experiments.runner import ExperimentRunner, ExperimentVariant, apply_overrides
from src.ingestion.beir_loader import BEIRDatasetLoader

EXPERIMENT_NAME = "experiment_1_embedding"
DATASET = "scifact"
SPLIT = "test"

# --------------------------------------------------------------------
# Baseline chunking is fixed for every variant, per the instruction to
# isolate the embedding-model axis. All other config.yaml sections
# (vector_store, retrieval, reranker, context, generation) also stay
# fixed via base_overrides below -- only embedding.* varies per
# variant.
# --------------------------------------------------------------------

BASELINE_CHUNKING_OVERRIDES = {
    "chunking.strategy": "recursive",
    "chunking.params": {"chunk_size": 1000, "chunk_overlap": 100},
}

VARIANTS = [
    ExperimentVariant(
        name="embeddinggemma:300m",
        overrides={
            "embedding.provider": "ollama",
            "embedding.params": {
                "model": "embeddinggemma:300m",
                "base_url": "https://condone-hankering-cushy.ngrok-free.dev",
            },
        },
    ),
    ExperimentVariant(
        name="e5-base-v2",
        overrides={
            "embedding.provider": "sentence_transformers",
            "embedding.params": {
                "model": "intfloat/e5-base-v2",
                "batch_size": 32,
                "device": "cpu",
            },
        },
    ),
    ExperimentVariant(
        name="all_minilm_l6_v2",
        overrides={
            "embedding.provider": "sentence_transformers",
            "embedding.params": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 32,
                "device": "cpu",
            },
        },
    ),
]


def main() -> None:
    base_config = get_settings("configs/config.yaml")

    # Apply the fixed chunking baseline once, up front. Per-variant
    # overrides in VARIANTS only touch embedding.*, so every variant
    # inherits this baseline chunking plus whatever else config.yaml
    # already sets for vector_store/retrieval/reranker.
    base_config = apply_overrides(base_config, BASELINE_CHUNKING_OVERRIDES)

    loader = BEIRDatasetLoader(dataset=DATASET, split=SPLIT).load()
    documents = loader.to_documents()
    examples = beir_to_eval_examples(loader.get_queries(), loader.get_qrels())

    print(f"Loaded {len(documents)} documents, {len(examples)} scorable queries")

    runner = ExperimentRunner(
        base_config=base_config,
        experiment_name=EXPERIMENT_NAME,
        documents=documents,
    )

    results = runner.run_all(
        VARIANTS,
        examples,
        dataset=DATASET,
        split=SPLIT,
        save_dir=f"artifacts/eval/{EXPERIMENT_NAME}",
    )

    pairs = pairs_against_baseline(
        [v.name for v in VARIANTS], baseline_name="all_minilm_l6_v2"
    )
    print_variant_pairs(results, pairs, stage="retrieval")

    print()
    print("=" * 90)
    print("EXPERIMENT 1 -- EMBEDDING MODEL COMPARISON (retrieval stage)")
    print("=" * 90)
    print_leaderboard(results, stage="retrieval", sort_by="ndcg_cut_10")

    print()
    print("=" * 90)
    print("EXPERIMENT 1 -- EMBEDDING MODEL COMPARISON (reranking stage)")
    print("=" * 90)
    print_leaderboard(results, stage="reranking", sort_by="ndcg_cut_10")

    report_path = generate_html_report(
        results,
        experiment_name=EXPERIMENT_NAME,
        output_dir=f"artifacts/eval/{EXPERIMENT_NAME}",
    )
    print(f"\nOpen the visual report: {report_path}")


if __name__ == "__main__":
    main()