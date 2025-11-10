"""Helpers shared across model implementations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.reporting import MetricResult, predictions_to_frame


@dataclass
class TrainingResult:
  metrics: List[MetricResult]
  predictions: Optional[pd.DataFrame] = None
  extra_sections: Optional[List[Tuple[str, str]]] = None


def extract_features_and_target(splits, features: Sequence[str], target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  missing = [feat for feat in features if feat not in splits.train.columns]
  if missing:
    raise KeyError(f"Missing features in training split: {missing}")
  if target not in splits.train.columns:
    raise KeyError(f"Target column {target} missing from training split")
  X_train = splits.train[features].to_numpy()
  y_train = splits.train[target].to_numpy()
  X_test = splits.test[features].to_numpy()
  y_test = splits.test[target].to_numpy()
  return X_train, y_train, X_test, y_test


def regression_metrics(dataset: str, model_name: str, y_true: Iterable[float], y_pred: Iterable[float]) -> List[MetricResult]:
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  if not y_true.size:
    raise ValueError("Cannot compute metrics on empty targets")
  mse = mean_squared_error(y_true, y_pred)
  rmse = math.sqrt(mse)
  return [
    MetricResult(dataset=dataset, model=model_name, metric="mse", value=float(mse)),
    MetricResult(dataset=dataset, model=model_name, metric="rmse", value=float(rmse)),
    MetricResult(dataset=dataset, model=model_name, metric="mae", value=float(mean_absolute_error(y_true, y_pred))),
    MetricResult(dataset=dataset, model=model_name, metric="r2", value=float(r2_score(y_true, y_pred))),
  ]


def build_predictions_frame(splits, dataset_cfg, predictions) -> pd.DataFrame:
  test_df = splits.test
  return predictions_to_frame(test_df, predictions, dataset_cfg)
