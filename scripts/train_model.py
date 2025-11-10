"""Training dispatcher for the benchmark framework."""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from models.base import TrainingResult
from utils.reporting import (
  DEFAULT_ERROR_THRESHOLD,
  MetricResult,
  generate_detailed_report,
  report_placeholder,
  save_metrics_csv,
  save_metrics_json,
)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs/datasets.yaml"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


logger = logging.getLogger(__name__)


@dataclass
class DatasetSplits:
  train: pd.DataFrame
  test: pd.DataFrame


def load_dataset_configs(config_path: Path = CONFIG_PATH) -> Dict[str, Dict[str, Any]]:
  with config_path.open() as handle:
    return yaml.safe_load(handle) or {}


def load_splits(dataset_name: str) -> DatasetSplits:
  dataset_dir = PROCESSED_DIR / dataset_name
  train_path = dataset_dir / "training.csv"
  test_path = dataset_dir / "testing.csv"
  if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError(
      f"Processed splits not found for {dataset_name}. Run scripts/preprocess.py first."
    )
  return DatasetSplits(
    train=pd.read_csv(train_path),
    test=pd.read_csv(test_path),
  )


def import_model(model_id: str):
  module_path = f"models.{model_id}"
  module = importlib.import_module(module_path)
  if not hasattr(module, "train"):
    raise AttributeError(f"Model module {module_path} is missing a train function")
  return module.train


def run_training(dataset: str, model_id: str, reports_dir: Path = DEFAULT_REPORTS_DIR) -> List[MetricResult]:
  configs = load_dataset_configs()
  if dataset not in configs:
    raise KeyError(f"Dataset {dataset} is not defined in configs/datasets.yaml")
  dataset_cfg = configs[dataset]
  splits = load_splits(dataset)
  train_fn = import_model(model_id)
  model_reports_dir = reports_dir / dataset / model_id.replace(".", "_")
  model_reports_dir.mkdir(parents=True, exist_ok=True)
  logger.info("Training %s on dataset %s", model_id, dataset)
  try:
    result = train_fn(dataset_name=dataset, dataset_cfg=dataset_cfg, splits=splits, output_dir=model_reports_dir)
  except (RuntimeError, ImportError) as exc:
    error_file = model_reports_dir / "ERROR.txt"
    error_file.write_text(str(exc))
    logger.error("Model %s failed: %s", model_id, exc)
    return []
  if isinstance(result, TrainingResult):
    metrics = result.metrics
    predictions_df = result.predictions
    extra_sections = result.extra_sections
  else:
    metrics = result
    predictions_df = None
    extra_sections = None

  if not metrics:
    logger.warning("Model %s returned no metrics", model_id)
    return []
  metrics_csv = model_reports_dir / "metrics.csv"
  metrics_json = model_reports_dir / "metrics.json"
  save_metrics_csv(metrics, metrics_csv)
  save_metrics_json(metrics, metrics_json)
  if predictions_df is not None:
    predictions_path = model_reports_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    report_cfg = dataset_cfg.get("reporting", {})
    error_threshold = report_cfg.get("error_band", DEFAULT_ERROR_THRESHOLD)
    file_prefix = model_id.replace(".", "_")
    generate_detailed_report(
      predictions_df=predictions_df,
      metrics=metrics,
      output_dir=model_reports_dir,
      model_label=f"{dataset}:{model_id}",
      file_prefix=file_prefix,
      error_threshold=error_threshold,
      extra_sections=extra_sections,
    )

  report_placeholder(model_reports_dir, dataset=dataset, model=model_id)
  return metrics


def main() -> None:
  parser = argparse.ArgumentParser(description="Run a single model on a dataset")
  parser.add_argument("--dataset", required=True, help="Dataset name as defined in configs/datasets.yaml")
  parser.add_argument("--model", required=True, help="Model identifier (e.g., baseline.svm)")
  parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="Directory where reports will be written")
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
  run_training(dataset=args.dataset, model_id=args.model, reports_dir=Path(args.reports_dir))


if __name__ == "__main__":
  main()
