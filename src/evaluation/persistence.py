from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    QueryStageResult,
    RetrievalEvalRun,
    StageEvalResult,
)


def save_eval_run(run: RetrievalEvalRun, path: str | Path) -> Path:
    """
    Serialize a RetrievalEvalRun to JSON, mirroring the pattern used
    for index manifests (src/pipeline/utils.py) so eval results are
    similarly diffable/traceable across runs.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(run)
    payload["saved_at"] = datetime.now(timezone.utc).isoformat()

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


def load_eval_run(path: str | Path) -> RetrievalEvalRun:
    """Reconstruct a RetrievalEvalRun from a JSON file written by save_eval_run."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    stages = {}
    for stage_name, stage_data in data["stages"].items():
        per_query = [
            QueryStageResult(**item) for item in stage_data["per_query"]
        ]
        stages[stage_name] = StageEvalResult(
            stage=stage_data["stage"],
            k_values=stage_data["k_values"],
            aggregate_metrics=stage_data["aggregate_metrics"],
            per_query=per_query,
            metadata=stage_data.get("metadata", {}),
        )

    return RetrievalEvalRun(
        dataset=data["dataset"],
        split=data["split"],
        num_examples=data["num_examples"],
        stages=stages,
        config_snapshot=data.get("config_snapshot", {}),
        metadata=data.get("metadata", {}),
    )


def compare_runs(
    baseline: RetrievalEvalRun,
    candidate: RetrievalEvalRun,
) -> dict[str, dict[str, Any]]:
    """
    Report per-stage, per-metric deltas between two eval runs (e.g.
    before/after swapping a reranker plugin). This is the actual
    payoff of stage-level eval: attributing a change to a specific
    stage instead of only comparing end-to-end numbers.
    """
    comparison: dict[str, dict[str, Any]] = {}

    stage_names = set(baseline.stages) | set(candidate.stages)

    for stage_name in sorted(stage_names):

        base_stage = baseline.stages.get(stage_name)
        cand_stage = candidate.stages.get(stage_name)

        if base_stage is None or cand_stage is None:
            comparison[stage_name] = {"error": "stage missing in one run"}
            continue

        metric_names = (
            set(base_stage.aggregate_metrics)
            | set(cand_stage.aggregate_metrics)
        )

        stage_deltas = {}
        for metric_name in sorted(metric_names):
            base_value = base_stage.aggregate_metrics.get(metric_name, 0.0)
            cand_value = cand_stage.aggregate_metrics.get(metric_name, 0.0)
            stage_deltas[metric_name] = {
                "baseline": base_value,
                "candidate": cand_value,
                "delta": cand_value - base_value,
            }

        comparison[stage_name] = stage_deltas

    return comparison

def pairs_against_baseline(
    variant_names: list[str],
    baseline_name: str,
) -> list[tuple[str, str]]:
    """
    Build (baseline, candidate) pairs comparing every other variant
    against a single designated baseline. Generic across any
    experiment: works whether the baseline is a chunking strategy
    (experiment 2), an embedding model (experiment 1), or a
    retriever/reranker combo (experiment 3) -- it only needs the
    baseline's name to exist among the variants passed in.
    """
    if baseline_name not in variant_names:
        raise ValueError(
            f"baseline_name '{baseline_name}' not found in variant_names: "
            f"{variant_names}"
        )

    return [
        (baseline_name, name)
        for name in variant_names
        if name != baseline_name
    ]


def compare_variant_pairs(
    runs: dict[str, RetrievalEvalRun],
    pairs: list[tuple[str, str]],
    stage: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Run compare_runs() over a list of explicit (baseline, candidate)
    variant-name pairs already present in `runs`, for a single stage.

    Generic across experiments -- `runs` and `pairs` can reference any
    variant names from any experiment; this just wraps compare_runs()
    so the pairing/looping logic isn't duplicated per experiment
    script. Use pairs_against_baseline() to build `pairs` automatically,
    or pass explicit tuples (e.g. experiment 3's retriever/reranker
    crossings) when the comparison isn't "everything vs one baseline".

    Returns:
        {"baseline_name -> candidate_name": {metric: {baseline, candidate, delta}}}
    """
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}

    for baseline_name, candidate_name in pairs:
        for name in (baseline_name, candidate_name):
            if name not in runs:
                raise ValueError(
                    f"Variant '{name}' not found in runs: {list(runs)}"
                )

        deltas = compare_runs(runs[baseline_name], runs[candidate_name])
        comparisons[f"{baseline_name} -> {candidate_name}"] = deltas.get(stage, {})

    return comparisons


def print_variant_pairs(
    runs: dict[str, RetrievalEvalRun],
    pairs: list[tuple[str, str]],
    stage: str,
    metric_columns: list[str] | None = None,
) -> None:
    """
    Pretty-print compare_variant_pairs() output. Generic across
    experiments -- pass whatever pairs and stage make sense for the
    sweep being run.
    """
    from .metrics import friendly_metric_label

    comparisons = compare_variant_pairs(runs, pairs, stage=stage)

    if not comparisons:
        print(f"No variant pairs to compare for stage '{stage}'")
        return

    for label, stage_deltas in comparisons.items():
        print(f"\n{label} ({stage} stage deltas):")

        if not stage_deltas:
            print(f"  No '{stage}' stage data available for this pair")
            continue

        columns = metric_columns or sorted(stage_deltas)

        for metric in columns:
            if metric not in stage_deltas:
                continue
            values = stage_deltas[metric]
            print(
                f"  {friendly_metric_label(metric):<14} "
                f"{values['baseline']:.4f} -> {values['candidate']:.4f} "
                f"({values['delta']:+.4f})"
            )

def build_leaderboard(
    runs: dict[str, RetrievalEvalRun],
    stage: str,
    sort_by: str = "ndcg_cut_10",
) -> list[dict[str, Any]]:
    """
    Flatten N runs (e.g. one per embedding model / chunking strategy /
    retrieval method) into a sorted list of rows for a given stage, so
    a whole sweep can be compared in one table instead of doing
    pairwise compare_runs() calls by hand.

    Each row is {"variant": <name>, **aggregate_metrics, [gold_coverage],
    [rank_movement summary]}. `sort_by` must be a metric name present
    in that stage's aggregate_metrics (pytrec_eval names, e.g.
    "ndcg_cut_10", "recall_10", "recip_rank") -- rows missing it sort last.
    """
    rows: list[dict[str, Any]] = []

    for variant_name, run in runs.items():
        stage_result = run.stages.get(stage)
        if stage_result is None:
            continue

        row: dict[str, Any] = {
            "variant": variant_name,
            **stage_result.aggregate_metrics,
        }

        if "gold_coverage" in stage_result.metadata:
            row["gold_coverage"] = stage_result.metadata["gold_coverage"]

        if "rank_movement" in stage_result.metadata:
            row["avg_rank_delta"] = (
                stage_result.metadata["rank_movement"]["avg_rank_delta"]
            )

        rows.append(row)

    rows.sort(key=lambda row: row.get(sort_by, float("-inf")), reverse=True)

    return rows


def print_leaderboard(
    runs: dict[str, RetrievalEvalRun],
    stage: str,
    sort_by: str = "ndcg_cut_10",
    metric_columns: list[str] | None = None,
) -> None:
    """
    Pretty-print build_leaderboard()'s output as a table with
    human-readable metric labels.

    Columns are the UNION of keys across all rows, not just rows[0] --
    e.g. only the reranking stage carries rank_movement/avg_rank_delta,
    and gold_coverage may be present for some variants but not others
    if a run partially failed. Using rows[0] alone silently drops or
    misaligns those columns; this keeps every row's data visible.
    """
    from .metrics import friendly_metric_label

    _extra_labels = {
        "gold_coverage": "Coverage",
        "avg_rank_delta": "Rank Δ",
    }

    rows = build_leaderboard(runs, stage, sort_by=sort_by)

    if not rows:
        print(f"No results for stage '{stage}'")
        return

    if metric_columns is not None:
        columns = metric_columns
    else:
        seen: dict[str, None] = {}
        for row in rows:
            for key in row:
                if key != "variant":
                    seen[key] = None
        columns = list(seen)

    labels = {
        col: _extra_labels.get(col, friendly_metric_label(col))
        for col in columns
    }

    col_width = 16
    header = f"{'variant':<28}" + "".join(
        f"{labels[col]:>{col_width}}" for col in columns
    )
    print(header)
    print("-" * len(header))

    for rank, row in enumerate(rows, start=1):
        marker = "*" if rank == 1 else " "
        line = f"{marker}{row['variant']:<27}"
        for col in columns:
            value = row.get(col)
            line += (
                f"{value:>{col_width}.4f}"
                if isinstance(value, float)
                else f"{'--':>{col_width}}"
            )
        print(line)

    print(f"\n(* = best by {labels.get(sort_by, sort_by)})")