from __future__ import annotations

from configs.config import get_settings
from src.evaluation import beir_to_eval_examples, generate_html_report, print_leaderboard
from src.experiments.runner import ExperimentRunner, ExperimentVariant, apply_overrides
from src.ingestion.beir_loader import BEIRDatasetLoader

EXPERIMENT_NAME = "experiment_2_chunking"
DATASET = "scifact"
SPLIT = "test"

BASELINE_EMBEDDING_OVERRIDES = {
    "embedding.provider": "sentence_transformers",
    "embedding.params": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
        "device": "cpu",
    },
}

VARIANTS = [
    ExperimentVariant(
        name="fixed_500_50",
        overrides={
            "chunking.strategy": "fixed",
            "chunking.params": {"chunk_size": 500, "chunk_overlap": 50},
        },
    ),
    ExperimentVariant(
        name="fixed_1000_150",
        overrides={
            "chunking.strategy": "fixed",
            "chunking.params": {"chunk_size": 1000, "chunk_overlap": 150},
        },
    ),
    ExperimentVariant(
        name="recursive_1000_150",
        overrides={
            "chunking.strategy": "recursive",
            "chunking.params": {"chunk_size": 1000, "chunk_overlap": 150},
        },
    ),
    ExperimentVariant(
        name="sentence_1000_1overlap",
        overrides={
            "chunking.strategy": "sentence",
            "chunking.params": {
                "max_chunk_size": 1000,
                "overlap_sentences": 1,
            },
        },
    ),
]


def main() -> None:
    base_config = get_settings("configs/config.yaml")
    base_config = apply_overrides(base_config, BASELINE_EMBEDDING_OVERRIDES)

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

    print()
    print("=" * 90)
    print("EXPERIMENT 2 -- CHUNKING STRATEGY COMPARISON (retrieval stage)")
    print("=" * 90)
    print_leaderboard(results, stage="retrieval", sort_by="ndcg_cut_10")

    print()
    print("=" * 90)
    print("EXPERIMENT 2 -- CHUNKING STRATEGY COMPARISON (reranking stage)")
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