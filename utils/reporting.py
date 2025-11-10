"""Utility helpers for persisting metrics, plots, and textual summaries."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List
import csv
import json


@dataclass
class MetricResult:
  """Structured representation of a single model evaluation."""
  dataset: str
  model: str
  metric: str
  value: float


def ensure_dir(path: Path) -> Path:
  path.mkdir(parents=True, exist_ok=True)
  return path


def save_metrics_csv(metrics: Iterable[MetricResult], output_file: Path) -> None:
  """Write metrics to disk in a normalized long format."""
  ensure_dir(output_file.parent)
  rows = [asdict(metric) for metric in metrics]
  if not rows:
    return
  with output_file.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


def save_metrics_json(metrics: Iterable[MetricResult], output_file: Path) -> None:
  """Persist metrics to JSON for downstream aggregation."""
  ensure_dir(output_file.parent)
  rows = [asdict(metric) for metric in metrics]
  with output_file.open("w") as handle:
    json.dump(rows, handle, indent=2)


def write_run_summary(summary: Dict[str, str], output_file: Path) -> None:
  """Persist a human-readable summary for the run."""
  ensure_dir(output_file.parent)
  with output_file.open("w") as handle:
    for key, value in summary.items():
      handle.write(f"{key}: {value}\n")


def report_placeholder(output_dir: Path, dataset: str, model: str) -> None:
  """Drop a minimal marker file so pipelines can verify execution order."""
  ensure_dir(output_dir)
  marker = output_dir / "_SUCCESS"
  marker.write_text(f"dataset={dataset}\nmodel={model}\n")
