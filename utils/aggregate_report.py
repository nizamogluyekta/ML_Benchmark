"""Benchmark-level reporting utilities."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

LOWER_IS_BETTER = {"mse", "rmse", "mae"}
DEFAULT_METRICS_ORDER = ["rmse", "mae", "mse", "r2"]


def _fmt(val: Optional[float]) -> str:
  if pd.isna(val):
    return "-"
  return f"{val:.4f}"


def _best_record(metric: str, df: pd.DataFrame) -> Optional[pd.Series]:
  metric_df = df[df["metric"] == metric]
  if metric_df.empty:
    return None
  ascending = metric in LOWER_IS_BETTER
  return metric_df.sort_values("value", ascending=ascending).iloc[0]


def _plot_metric(dataset: str, metric: str, df: pd.DataFrame, reports_dir: Path) -> Optional[Path]:
  metric_df = df[df["metric"] == metric]
  if metric_df.empty:
    return None
  metric_df = metric_df.sort_values("value", ascending=metric in LOWER_IS_BETTER)
  plt.figure(figsize=(10, 4))
  plt.bar(metric_df["model"], metric_df["value"], color="#4c72b0")
  plt.title(f"{dataset} – {metric.upper()} comparison")
  plt.ylabel(metric.upper())
  plt.xticks(rotation=45, ha="right")
  plt.tight_layout()
  output = reports_dir / f"{dataset}_{metric}_comparison.png"
  plt.savefig(output, bbox_inches="tight")
  plt.close()
  return output


def _rel(path: Path, base: Path) -> str:
  try:
    return path.relative_to(base).as_posix()
  except ValueError:
    return path.as_posix()


def _model_assets(dataset: str, model: str, reports_dir: Path) -> Dict[str, Path]:
  safe = model.replace(".", "_")
  model_dir = reports_dir / dataset / safe
  assets: Dict[str, Path] = {}
  if not model_dir.exists():
    return assets
  for label, filename in {
    "Report": f"{safe}_performance_report.md",
    "Metrics": "metrics.csv",
    "Metrics JSON": "metrics.json",
    "Predictions": "predictions.csv",
    "Actual vs Predicted": f"{safe}_actual_vs_predicted.png",
    "Scatter": f"{safe}_scatter_plot.png",
  }.items():
    path = model_dir / filename
    if path.exists():
      assets[label] = path
  return assets


def generate_benchmark_report(summary_csv: Path, reports_dir: Path, output_path: Path) -> Optional[Path]:
  if not summary_csv.exists():
    return None
  df = pd.read_csv(summary_csv)
  if df.empty:
    return None

  sections: List[str] = []
  sections.append("# Benchmark Summary")
  sections.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
  sections.append("")

  for dataset in df["dataset"].unique():
    dataset_df = df[df["dataset"] == dataset]
    sections.append(f"## Dataset: {dataset}")

    pivot = dataset_df.pivot_table(index="model", columns="metric", values="value")
    pivot = pivot.reindex(columns=DEFAULT_METRICS_ORDER)
    sections.append("### Metric Table")
    headers = "| Model | " + " | ".join(col.upper() if col is not None else "" for col in pivot.columns) + " |"
    separator = "|" + "---|" * (len(pivot.columns) + 1)
    sections.append(headers)
    sections.append(separator)
    for model, row in pivot.iterrows():
      values = " | ".join(_fmt(row.get(metric)) for metric in pivot.columns)
      sections.append(f"| {model} | {values} |")
    sections.append("")

    sections.append("### Top Models")
    for metric in DEFAULT_METRICS_ORDER:
      rec = _best_record(metric, dataset_df)
      if rec is None:
        continue
      direction = "lowest" if metric in LOWER_IS_BETTER else "highest"
      sections.append(f"- **{metric.upper()}** ({direction}): `{rec['model']}` = {_fmt(rec['value'])}")
    sections.append("")

    for metric in DEFAULT_METRICS_ORDER:
      chart = _plot_metric(dataset, metric, dataset_df, reports_dir)
      if chart:
        rel = _rel(chart, reports_dir)
        sections.append(f"![{dataset} {metric} comparison]({rel})")
    sections.append("")

    sections.append("### Model Artifacts")
    for model in sorted(dataset_df["model"].unique()):
      assets = _model_assets(dataset, model, reports_dir)
      if assets:
        links = " | ".join(f"[{label}]({_rel(path, reports_dir)})" for label, path in assets.items())
        sections.append(f"- `{model}`: {links}")
      else:
        sections.append(f"- `{model}`: (no artifacts found)")
    sections.append("")

  output_path.write_text("\n".join(sections), encoding="utf-8")
  return output_path
