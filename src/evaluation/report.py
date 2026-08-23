from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display backend needed -- rendering to file only

import matplotlib.pyplot as plt
import numpy as np

from .metrics import friendly_metric_label
from .models import RetrievalEvalRun
from .persistence import build_leaderboard

DEFAULT_CHART_METRICS_TEMPLATE = ["ndcg_cut_{k}", "recall_{k}", "map_cut_{k}", "recip_rank"]


def _chart_metrics(primary_k: int) -> list[str]:
    return [
        metric.format(k=primary_k) if "{k}" in metric else metric
        for metric in DEFAULT_CHART_METRICS_TEMPLATE
    ]


def plot_metric_comparison(
    runs: dict[str, RetrievalEvalRun],
    stage: str,
    metrics: list[str],
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """
    Grouped bar chart: one x-axis group per variant, one bar per
    metric within the group. This is the core "better way to
    visualize" the leaderboard -- differences that are easy to miss
    scanning a column of 4-decimal numbers (e.g. 0.6123 vs 0.6089) are
    immediately visible as bar-height differences.
    """
    variant_names = list(runs.keys())

    series = []
    for metric in metrics:
        values = []
        for name in variant_names:
            stage_result = runs[name].stages.get(stage)
            value = (
                stage_result.aggregate_metrics.get(metric, 0.0)
                if stage_result
                else 0.0
            )
            values.append(value)
        series.append(values)

    n_variants = len(variant_names)
    n_metrics = len(metrics)
    x = np.arange(n_variants)
    width = 0.8 / max(n_metrics, 1)

    fig, ax = plt.subplots(figsize=(max(6.5, n_variants * 1.7), 5))

    for i, (metric, values) in enumerate(zip(metrics, series)):
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=friendly_metric_label(metric))
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(title or f"{stage.capitalize()} stage — metric comparison")
    ax.legend(loc="upper right", fontsize=8, ncol=len(metrics))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def _img_to_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _rows_to_html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No results.</p>"

    def label(col: str) -> str:
        if col == "gold_coverage":
            return "Coverage"
        if col == "avg_rank_delta":
            return "Rank Δ"
        return friendly_metric_label(col)

    header_cells = "".join(f"<th>{label(col)}</th>" for col in columns)

    body_rows = []
    for rank, row in enumerate(rows, start=1):
        cells = []
        for col in columns:
            value = row.get(col)
            cells.append(
                f"<td>{value:.4f}</td>" if isinstance(value, float) else "<td>--</td>"
            )
        row_class = ' class="best"' if rank == 1 else ""
        body_rows.append(
            f'<tr{row_class}><td>{row["variant"]}</td>{"".join(cells)}</tr>'
        )

    return (
        "<table><thead><tr><th>Variant</th>"
        f"{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )


def generate_html_report(
    runs: dict[str, RetrievalEvalRun],
    experiment_name: str,
    output_dir: str | Path,
    primary_k: int = 10,
) -> Path:
    """
    Render one self-contained HTML report for an experiment sweep:
    a grouped bar chart per stage (retrieval, reranking) at the
    primary_k metrics, embedded directly as base64 PNGs (no separate
    image files to keep track of), plus the exact numbers underneath
    in a table with the best variant row highlighted.

    Opens directly in a browser -- no server, no notebook needed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_metrics = _chart_metrics(primary_k)
    sort_metric = f"ndcg_cut_{primary_k}"

    sections = []

    for stage in ("retrieval", "reranking"):

        chart_path = output_dir / f"{stage}_chart.png"
        plot_metric_comparison(
            runs,
            stage,
            chart_metrics,
            chart_path,
            title=f"{experiment_name} — {stage} stage",
        )
        img_uri = _img_to_data_uri(chart_path)
        chart_path.unlink()  # embedded in the HTML; no need to keep the loose file

        rows = build_leaderboard(runs, stage=stage, sort_by=sort_metric)

        seen: dict[str, None] = {}
        for row in rows:
            for key in row:
                if key != "variant":
                    seen[key] = None
        columns = list(seen)

        table_html = _rows_to_html_table(rows, columns)

        sections.append(
            f"""
            <section>
              <h2>{stage.capitalize()} stage</h2>
              <img src="{img_uri}" alt="{stage} chart" />
              {table_html}
            </section>
            """
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>{experiment_name} — evaluation report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 40px auto;
    max-width: 960px;
    color: #1a1a1a;
  }}
  h1 {{ margin-bottom: 2px; }}
  .subtitle {{ color: #666; margin-top: 0; }}
  h2 {{ margin-top: 44px; border-bottom: 1px solid #e0e0e0; padding-bottom: 6px; }}
  img {{ max-width: 100%; margin-top: 12px; }}
  table {{ border-collapse: collapse; margin-top: 18px; width: 100%; font-size: 14px; }}
  th, td {{
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #f5f5f5; }}
  tr.best td {{ background: #e8f5e9; font-weight: 600; }}
  .legend {{ color: #666; font-size: 13px; margin-top: 6px; }}
</style>
</head>
<body>
  <h1>{experiment_name}</h1>
  <p class="subtitle">{len(runs)} variants — best row per stage highlighted (sorted by nDCG@{primary_k})</p>
  {''.join(sections)}
</body>
</html>
"""

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    return report_path