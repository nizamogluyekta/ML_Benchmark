"""Placeholder wavelet-enhanced XGBoost model."""
from __future__ import annotations

from typing import Any, Dict, List

from models.baseline import xgboost as baseline_xgb
from utils.reporting import MetricResult

MODEL_ID = "wavelet.xgboost"


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> List[MetricResult]:
  base_metrics = baseline_xgb.train(dataset_name, dataset_cfg, splits, output_dir)
  return [MetricResult(dataset=m.dataset, model=MODEL_ID, metric=m.metric, value=m.value) for m in base_metrics]
