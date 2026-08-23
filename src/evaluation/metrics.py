from __future__ import annotations

import re
from typing import Any

import pytrec_eval

# ---------------------------------------------------------------------
# Types (kept as plain dict aliases -- these mirror pytrec_eval's own
# expected shapes, so no wrapper classes are introduced here):
#
#   qrels: dict[query_id, dict[doc_id, relevance_grade]]
#   run:   dict[query_id, dict[doc_id, score]]
#   doc_ranking: dict[query_id, list[doc_id]]   (ordered, best first)
# ---------------------------------------------------------------------

_METRIC_PREFIX_LABELS = {
    "ndcg_cut": "nDCG",
    "recall": "Recall",
    "P": "Precision",
    "map_cut": "MAP",
}

_METRIC_EXACT_LABELS = {
    "recip_rank": "MRR",
}


def friendly_metric_label(name: str) -> str:
    """
    Turn a raw pytrec_eval metric key (e.g. "ndcg_cut_10", "P_10",
    "recip_rank") into a human-readable label ("nDCG@10",
    "Precision@10", "MRR"). Unrecognized names pass through unchanged.
    """
    if name in _METRIC_EXACT_LABELS:
        return _METRIC_EXACT_LABELS[name]

    match = re.match(r"(ndcg_cut|recall|map_cut|P)_(\d+)$", name)
    if not match:
        return name

    prefix, k = match.groups()
    return f"{_METRIC_PREFIX_LABELS.get(prefix, prefix)}@{k}"


def compute_ir_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k_values: list[int],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Compute standard IR metrics (nDCG@k, Recall@k, Precision@k, MAP@k,
    MRR) via pytrec_eval.

    Returns:
        (aggregate_metrics, per_query_metrics)
    """
    if not qrels or not run:
        return {}, {}

    k_str = ",".join(str(k) for k in sorted(set(k_values)))

    measures = {
        f"ndcg_cut.{k_str}",
        f"recall.{k_str}",
        f"P.{k_str}",
        f"map_cut.{k_str}",
        "recip_rank",
    }

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    raw_per_query = evaluator.evaluate(run)

    per_query_metrics = {
        query_id: dict(scores)
        for query_id, scores in raw_per_query.items()
    }

    aggregate_metrics = _average_metrics(per_query_metrics)

    return aggregate_metrics, per_query_metrics


def _average_metrics(
    per_query_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:

    if not per_query_metrics:
        return {}

    all_metric_names: set[str] = set()
    for scores in per_query_metrics.values():
        all_metric_names.update(scores.keys())

    aggregate: dict[str, float] = {}

    for name in sorted(all_metric_names):
        values = [
            scores[name]
            for scores in per_query_metrics.values()
            if name in scores
        ]
        aggregate[name] = sum(values) / len(values) if values else 0.0

    return aggregate


def compute_gold_coverage(
    qrels: dict[str, dict[str, int]],
    doc_ranking: dict[str, list[str]],
    k: int | None = None,
) -> float:
    """
    Fraction of queries for which at least one relevant document
    appears anywhere in the candidate pool (or within the top-k of it
    if `k` is given).

    This is the "recall ceiling": the best possible score any
    downstream stage (reranker, generator) could achieve for that
    query, since a document that never made it into the candidate
    pool can never be surfaced later. Useful context when comparing
    reranker/generator experiments against each other.
    """
    found = 0
    total = 0

    for query_id, relevant in qrels.items():
        gold_ids = {
            doc_id
            for doc_id, grade in relevant.items()
            if grade > 0
        }

        if not gold_ids:
            continue

        total += 1

        candidates = doc_ranking.get(query_id, [])
        if k is not None:
            candidates = candidates[:k]

        if gold_ids.intersection(candidates):
            found += 1

    return found / total if total else 0.0


def compute_rank_movement(
    qrels: dict[str, dict[str, int]],
    before_ranking: dict[str, list[str]],
    after_ranking: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Track how reranking moved gold documents relative to their
    pre-rerank position. Only considers gold docs present in *both*
    rankings (a doc that fell out of the candidate pool entirely is a
    coverage problem, not a rank-movement one -- see
    compute_gold_coverage).

    Positive delta = document moved to a better (lower-index) rank
    after reranking. Negative = reranking demoted it.
    """
    deltas: list[int] = []
    promoted = 0
    demoted = 0
    unchanged = 0

    for query_id, relevant in qrels.items():
        gold_ids = {
            doc_id
            for doc_id, grade in relevant.items()
            if grade > 0
        }

        before = before_ranking.get(query_id, [])
        after = after_ranking.get(query_id, [])

        before_rank = {doc_id: idx for idx, doc_id in enumerate(before)}
        after_rank = {doc_id: idx for idx, doc_id in enumerate(after)}

        for doc_id in gold_ids:
            if doc_id not in before_rank or doc_id not in after_rank:
                continue

            delta = before_rank[doc_id] - after_rank[doc_id]
            deltas.append(delta)

            if delta > 0:
                promoted += 1
            elif delta < 0:
                demoted += 1
            else:
                unchanged += 1

    return {
        "avg_rank_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "promoted": promoted,
        "demoted": demoted,
        "unchanged": unchanged,
        "n_tracked": len(deltas),
    }