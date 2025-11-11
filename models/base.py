"""Helpers shared across model implementations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.reporting import MetricResult, predictions_to_frame

ERROR_EPS = 1e-8


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
  mae = mean_absolute_error(y_true, y_pred)
  abs_error = np.abs(y_pred - y_true)
  percent_error = abs_error / np.maximum(np.abs(y_true), ERROR_EPS)
  percent_error_pct = percent_error * 100
  mape = float(percent_error_pct.mean())
  smape = float((200.0 * abs_error / (np.abs(y_true) + np.abs(y_pred) + ERROR_EPS)).mean())
  median_ae = float(np.median(abs_error))
  p90_ae = float(np.percentile(abs_error, 90))
  bias = float((y_pred - y_true).mean())
  coverage_5 = float((percent_error_pct <= 5).mean() * 100)
  coverage_10 = float((percent_error_pct <= 10).mean() * 100)
  metrics = [
    MetricResult(dataset=dataset, model=model_name, metric="mse", value=float(mse)),
    MetricResult(dataset=dataset, model=model_name, metric="rmse", value=float(rmse)),
    MetricResult(dataset=dataset, model=model_name, metric="mae", value=float(mae)),
    MetricResult(dataset=dataset, model=model_name, metric="r2", value=float(r2_score(y_true, y_pred))),
    MetricResult(dataset=dataset, model=model_name, metric="mape", value=mape),
    MetricResult(dataset=dataset, model=model_name, metric="smape", value=smape),
    MetricResult(dataset=dataset, model=model_name, metric="median_ae", value=median_ae),
    MetricResult(dataset=dataset, model=model_name, metric="p90_ae", value=p90_ae),
    MetricResult(dataset=dataset, model=model_name, metric="bias", value=bias),
    MetricResult(dataset=dataset, model=model_name, metric="coverage_within_5pct", value=coverage_5),
    MetricResult(dataset=dataset, model=model_name, metric="coverage_within_10pct", value=coverage_10),
  ]
  if np.all(y_true >= 0) and np.all(y_pred >= 0):
    rmsle = float(math.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))
    metrics.append(MetricResult(dataset=dataset, model=model_name, metric="rmsle", value=rmsle))
  return metrics


def build_predictions_frame(splits, dataset_cfg, predictions) -> pd.DataFrame:
  test_df = splits.test
  return predictions_to_frame(test_df, predictions, dataset_cfg)


def timed_fit_predict(model, X_train, y_train, X_test, *, fit_params: Optional[dict] = None):
  start = perf_counter()
  if fit_params is None:
    fit_params = {}
  model.fit(X_train, y_train, **fit_params)
  train_time = perf_counter() - start
  start = perf_counter()
  predictions = model.predict(X_test)
  predict_time = perf_counter() - start
  return predictions, train_time, predict_time


def timing_metrics(dataset: str, model_name: str, train_time: float, predict_time: float) -> List[MetricResult]:
  return [
    MetricResult(dataset=dataset, model=model_name, metric="train_time_sec", value=float(train_time)),
    MetricResult(dataset=dataset, model=model_name, metric="predict_time_sec", value=float(predict_time)),
  ]
