from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from configs.config import Settings
from src.evaluation import EvalExample, RetrievalEvalRun, RetrievalEvaluator, save_eval_run
from src.ingestion.models import Document
from src.pipeline.indexer_pipeline import build_index_from_documents  

from .overrides import apply_overrides
from .paths import derive_variant_paths


@dataclass
class ExperimentVariant:
    """
    One point in a sweep: a name (used for artifact paths + result
    labeling) and the dotted-path config overrides that define it.

    reindex=True  -> chunking/embedding/vector_store changed; this
                      variant gets its own isolated index built before
                      evaluation (experiments 1 and 2).
    reindex=False -> only retrieval/reranker config changed; this
                      variant is evaluated against whatever index is
                      already on disk at the base config's paths
                      (experiment 3 -- build the shared index once,
                      outside the sweep).
    """

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    reindex: bool = True


class ExperimentRunner:
    """
    Runs a list of ExperimentVariants against a base Settings and a
    fixed evaluation example set, producing one RetrievalEvalRun per
    variant.
    """

    def __init__(
        self,
        base_config: Settings,
        experiment_name: str,
        documents: list[Document] | None = None,
        k_values: list[int] | None = None,
        doc_id_field: str = "document_id",
    ):
        self.base_config = base_config
        self.experiment_name = experiment_name
        self.documents = documents
        self.k_values = k_values
        self.doc_id_field = doc_id_field

    def build_shared_index(self) -> None:
        """
        Build a single index at the base config's own paths (no
        isolation). Use this once, up front, for sweeps where no
        variant reindexes -- e.g. comparing retrieval methods against
        one fixed corpus (experiment 3).
        """
        if self.documents is None:
            raise ValueError(
                "documents must be provided to build the shared index"
            )

        print(
            f"[{self.experiment_name}] Building shared index "
            f"(chunking={self.base_config.chunking.strategy}, "
            f"embedding={self.base_config.embedding.provider})"
        )
        build_index_from_documents(self.base_config, self.documents)

    def build_variant_config(self, variant: ExperimentVariant) -> Settings:
        config = apply_overrides(self.base_config, variant.overrides)
        config = apply_overrides(config, {"experiment_name": self.experiment_name})

        if variant.reindex:
            path_overrides = derive_variant_paths(
                config, self.experiment_name, variant.name
            )
            config = apply_overrides(config, path_overrides)

        return config

    def run_variant(
        self,
        variant: ExperimentVariant,
        examples: list[EvalExample],
        dataset: str = "unknown",
        split: str = "unknown",
        save_dir: str | None = None,
    ) -> RetrievalEvalRun:

        config = self.build_variant_config(variant)

        if variant.reindex:
            if self.documents is None:
                raise ValueError(
                    f"Variant '{variant.name}' has reindex=True but no "
                    "documents were provided to ExperimentRunner"
                )
            print(
                f"[{self.experiment_name}/{variant.name}] Building index "
                f"(chunking={config.chunking.strategy}/"
                f"{config.chunking.params}, "
                f"embedding={config.embedding.provider}/"
                f"{config.embedding.params.get('model')})"
            )
            build_index_from_documents(config, self.documents)
        else:
            print(
                f"[{self.experiment_name}/{variant.name}] Reusing shared "
                f"index (retrieval={config.retrieval.type}, "
                f"reranker={config.reranker.type})"
            )

        evaluator = RetrievalEvaluator(
            config,
            k_values=self.k_values,
            doc_id_field=self.doc_id_field,
        )

        run = evaluator.evaluate(examples, dataset=dataset, split=split)
        run.metadata["experiment_name"] = self.experiment_name
        run.metadata["variant_name"] = variant.name
        run.metadata["variant_overrides"] = variant.overrides

        if save_dir:
            path = f"{save_dir.rstrip('/')}/{variant.name}.json"
            saved_path = save_eval_run(run, path)
            print(f"  Saved: {saved_path}")

        return run

    def run_all(
        self,
        variants: list[ExperimentVariant],
        examples: list[EvalExample],
        dataset: str = "unknown",
        split: str = "unknown",
        save_dir: str | None = None,
    ) -> dict[str, RetrievalEvalRun]:

        results: dict[str, RetrievalEvalRun] = {}

        for variant in variants:
            results[variant.name] = self.run_variant(
                variant,
                examples,
                dataset=dataset,
                split=split,
                save_dir=save_dir,
            )

        return results