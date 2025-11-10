"""Placeholder wavelet-enhanced Random Forest model."""
from __future__ import annotations

from typing import Any, Dict, List

from models.baseline import random_forest as baseline_rf
from utils.reporting import MetricResult

MODEL_ID = "wavelet.random_forest"


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> List[MetricResult]:
  base_metrics = baseline_rf.train(dataset_name, dataset_cfg, splits, output_dir)
  return [MetricResult(dataset=m.dataset, model=MODEL_ID, metric=m.metric, value=m.value) for m in base_metrics]
