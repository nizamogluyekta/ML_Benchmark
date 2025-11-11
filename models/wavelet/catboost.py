"""Wavelet-enhanced CatBoost regressor."""
from __future__ import annotations

from typing import Any, Dict

try:
  from catboost import CatBoostRegressor
except ImportError as exc:  # pragma: no cover
  CatBoostRegressor = None
  CATBOOST_IMPORT_ERROR = exc
else:
  CATBOOST_IMPORT_ERROR = None

from models.base import (
  TrainingResult,
  build_predictions_frame,
  regression_metrics,
  timed_fit_predict,
  timing_metrics,
)
from models.wavelet.common import format_wavelet_stats_markdown, prepare_wavelet_data

MODEL_ID = "wavelet.catboost"


def build_model(hyperparams: Dict[str, Any] | None = None):
  if CatBoostRegressor is None:
    raise RuntimeError("catboost is not installed") from CATBOOST_IMPORT_ERROR
  params = {
    "iterations": 900,
    "depth": 8,
    "learning_rate": 0.04,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": False,
  }
  if hyperparams:
    params.update(hyperparams)
  return CatBoostRegressor(**params)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  categorical = [col for col in (prepared.categorical_features or []) if col in prepared.X_train.columns]
  for col in categorical:
    prepared.X_train[col] = prepared.X_train[col].astype(str)
    prepared.X_test[col] = prepared.X_test[col].astype(str)
  model = build_model(dataset_cfg.get("wavelet_catboost_params"))
  predictions, train_time, predict_time = timed_fit_predict(
    model,
    prepared.X_train,
    prepared.y_train,
    prepared.X_test,
    fit_params={"cat_features": categorical or None},
  )
  metrics = regression_metrics(dataset_name, MODEL_ID, prepared.y_test, predictions)
  metrics.extend(timing_metrics(dataset_name, MODEL_ID, train_time, predict_time))
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  extra = format_wavelet_stats_markdown(prepared.wavelet_stats)
  sections = [("Wavelet Detail Coefficients (mean absolute value)", extra)] if extra else None
  return TrainingResult(metrics=metrics, predictions=predictions_df, extra_sections=sections)
