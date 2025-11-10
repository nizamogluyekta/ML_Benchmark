"""Benchmark orchestrator coordinating preprocessing and training."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

import yaml

from scripts.preprocess import run as preprocess_run
from scripts.train_model import run_training
from utils.reporting import MetricResult, save_metrics_csv

BENCHMARK_CONFIG = ROOT_DIR / "configs/benchmark.yaml"
DEFAULT_REPORTS_DIR = ROOT_DIR / "reports"


logger = logging.getLogger(__name__)


def load_benchmark_config(config_path: Path = BENCHMARK_CONFIG) -> Dict[str, Any]:
  with config_path.open() as handle:
    return yaml.safe_load(handle) or {}


def maybe_preprocess(datasets: List[str], force: bool) -> None:
  for dataset in datasets:
    preprocess_run(dataset=dataset, force=force)


def run_benchmark(config_path: Path, reports_dir: Path) -> Path:
  cfg = load_benchmark_config(config_path)
  datasets = cfg.get("datasets", [])
  models = cfg.get("models", [])
  if not datasets or not models:
    raise RuntimeError("Benchmark configuration must specify datasets and models")
  preprocess_force = cfg.get("preprocess", {}).get("force", False)
  maybe_preprocess(datasets, force=preprocess_force)

  all_metrics: List[MetricResult] = []
  for dataset in datasets:
    for model_id in models:
      logger.info("Running model %s on dataset %s", model_id, dataset)
      metrics = run_training(dataset=dataset, model_id=model_id, reports_dir=reports_dir)
      all_metrics.extend(metrics)

  summary_file = reports_dir / cfg.get("reporting", {}).get("summary_file", "benchmark_summary.csv")
  save_metrics_csv(all_metrics, summary_file)
  return summary_file


def main() -> None:
  parser = argparse.ArgumentParser(description="Run the configured ML benchmark suite")
  parser.add_argument("--config", default=str(BENCHMARK_CONFIG), help="Benchmark configuration file path")
  parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="Directory to store reports")
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
  summary_file = run_benchmark(config_path=Path(args.config), reports_dir=Path(args.reports_dir))
  logger.info("Benchmark summary written to %s", summary_file)


if __name__ == "__main__":
  main()
