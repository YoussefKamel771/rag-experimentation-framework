from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalExample:
    """
    A single query with its ground-truth relevance judgments.

    relevant_doc_ids maps document_id -> relevance grade (as in BEIR
    qrels: 0/1, or graded 0-2/0-3 depending on the dataset). Judgments
    are at the *document* level, since that's what qrels give us --
    the pipeline's chunk-level results get deduplicated up to document
    ids before scoring (see dedup.py).
    """

    query_id: str
    query_text: str
    relevant_doc_ids: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryStageResult:
    """
    Per-query outcome for a single pipeline stage (e.g. retrieval,
    reranking). Kept separately from the aggregate so individual
    queries can be inspected when a run-level number moves.
    """

    query_id: str
    metrics: dict[str, float]
    ranked_doc_ids: list[str]
    retrieved_count: int
    relevant_found: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageEvalResult:
    """
    Aggregate + per-query evaluation results for one pipeline stage.
    """

    stage: str
    k_values: list[int]
    aggregate_metrics: dict[str, float]
    per_query: list[QueryStageResult]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalEvalRun:
    """
    Full result of evaluating retrieval (+ reranking) over a set of
    EvalExamples against a specific pipeline configuration.
    """

    dataset: str
    split: str
    num_examples: int
    stages: dict[str, StageEvalResult]
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)