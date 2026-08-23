from __future__ import annotations

from tqdm import tqdm

from configs.config import Settings
from src.pipeline.reranking_pipeline import create_reranker
from src.pipeline.retrieval_pipeline import create_retriever
from src.plugins.registry_loader import load_all_plugins

from .dedup import dedup_to_document_ranking, doc_ranking_to_run
from .metrics import (
    compute_gold_coverage,
    compute_ir_metrics,
    compute_rank_movement,
)
from .models import EvalExample, QueryStageResult, RetrievalEvalRun, StageEvalResult

DEFAULT_K_VALUES = [1, 3, 5, 10, 20]


class RetrievalEvaluator:
    """
    Evaluates the retrieval and reranking stages independently against
    document-level relevance judgments (e.g. BEIR qrels).

    The retriever and reranker are instantiated ONCE in __init__ and
    reused for every example -- evaluation sets can run into the
    hundreds of queries, and re-instantiating an embedding model /
    Qdrant client / cross-encoder per query (as the current
    request-scoped pipeline functions do) would make evaluation
    loops untenable. This class is a warm, eval-scoped stand-in for
    that until the full RAGPipeline refactor lands.
    """

    def __init__(
        self,
        config: Settings,
        k_values: list[int] | None = None,
        doc_id_field: str = "document_id",
    ):
        load_all_plugins()

        self.config = config
        self.k_values = k_values or DEFAULT_K_VALUES
        self.doc_id_field = doc_id_field

        self.retriever = create_retriever(config)
        self.reranker = create_reranker(config)

    def evaluate(
        self,
        examples: list[EvalExample],
        dataset: str = "unknown",
        split: str = "unknown",
    ) -> RetrievalEvalRun:

        if not examples:
            raise ValueError("examples cannot be empty")

        qrels = {
            example.query_id: example.relevant_doc_ids
            for example in examples
        }

        candidate_k = self.config.retrieval.candidate_k
        top_k = self.config.reranker.top_k

        retrieval_run: dict[str, dict[str, float]] = {}
        reranking_run: dict[str, dict[str, float]] = {}

        retrieval_docs: dict[str, list[str]] = {}
        reranking_docs: dict[str, list[str]] = {}

        for example in tqdm(examples, desc="Evaluating retrieval/reranking"):

            candidates = self.retriever.retrieve(
                example.query_text,
                top_k=candidate_k,
            )

            retrieval_doc_ids = dedup_to_document_ranking(
                candidates,
                self.doc_id_field,
            )
            retrieval_docs[example.query_id] = retrieval_doc_ids
            retrieval_run[example.query_id] = doc_ranking_to_run(
                retrieval_doc_ids
            )

            reranked = self.reranker.rerank(
                example.query_text,
                candidates,
                top_k=top_k,
            )

            reranking_doc_ids = dedup_to_document_ranking(
                reranked,
                self.doc_id_field,
            )
            reranking_docs[example.query_id] = reranking_doc_ids
            reranking_run[example.query_id] = doc_ranking_to_run(
                reranking_doc_ids
            )

        retrieval_stage = self._build_stage_result(
            stage="retrieval",
            qrels=qrels,
            run=retrieval_run,
            doc_rankings=retrieval_docs,
        )

        # Recall ceiling: best any downstream stage could possibly
        # achieve, given what actually made it into the candidate pool.
        retrieval_stage.metadata["gold_coverage"] = compute_gold_coverage(
            qrels, retrieval_docs
        )

        reranking_stage = self._build_stage_result(
            stage="reranking",
            qrels=qrels,
            run=reranking_run,
            doc_rankings=reranking_docs,
        )

        # Did reranking actually help, relative to its own input?
        reranking_stage.metadata["rank_movement"] = compute_rank_movement(
            qrels, retrieval_docs, reranking_docs
        )

        return RetrievalEvalRun(
            dataset=dataset,
            split=split,
            num_examples=len(examples),
            stages={
                "retrieval": retrieval_stage,
                "reranking": reranking_stage,
            },
            config_snapshot={
                "retrieval_type": self.config.retrieval.type,
                "candidate_k": candidate_k,
                "reranker_type": self.config.reranker.type,
                "reranker_top_k": top_k,
                "embedding_provider": self.config.embedding.provider,
                "embedding_model": self.config.embedding.params.get("model"),
            },
        )

    def _build_stage_result(
        self,
        stage: str,
        qrels: dict[str, dict[str, int]],
        run: dict[str, dict[str, float]],
        doc_rankings: dict[str, list[str]],
    ) -> StageEvalResult:

        aggregate_metrics, per_query_metrics = compute_ir_metrics(
            qrels, run, self.k_values
        )

        per_query_results: list[QueryStageResult] = []

        for query_id, doc_ids in doc_rankings.items():

            relevant = qrels.get(query_id, {})
            gold_ids = {
                doc_id for doc_id, grade in relevant.items() if grade > 0
            }

            relevant_found = len(gold_ids.intersection(doc_ids))

            per_query_results.append(
                QueryStageResult(
                    query_id=query_id,
                    metrics=per_query_metrics.get(query_id, {}),
                    ranked_doc_ids=doc_ids,
                    retrieved_count=len(doc_ids),
                    relevant_found=relevant_found,
                )
            )

        return StageEvalResult(
            stage=stage,
            k_values=self.k_values,
            aggregate_metrics=aggregate_metrics,
            per_query=per_query_results,
        )