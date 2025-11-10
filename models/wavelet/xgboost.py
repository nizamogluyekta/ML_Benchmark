"""Wavelet-enhanced XGBoost model."""
from __future__ import annotations

from typing import Any, Dict

try:
  from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - optional dependency
  XGBRegressor = None
  XGB_IMPORT_ERROR = exc
else:
  XGB_IMPORT_ERROR = None

from sklearn.pipeline import Pipeline

from models.base import TrainingResult, build_predictions_frame, regression_metrics
from models.wavelet.common import format_wavelet_stats_markdown, prepare_wavelet_data

MODEL_ID = "wavelet.xgboost"


def build_estimator(hyperparams: Dict[str, Any] | None = None):
  if XGBRegressor is None:
    raise RuntimeError("xgboost is not installed") from XGB_IMPORT_ERROR
  params = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
  }
  if hyperparams:
    params.update(hyperparams)
  return XGBRegressor(**params)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  model = build_estimator(dataset_cfg.get("wavelet_xgb_params"))
  pipeline = Pipeline([
    ("pre", prepared.preprocessor),
    ("xgb", model),
  ])
  pipeline.fit(prepared.X_train, prepared.y_train)
  predictions = pipeline.predict(prepared.X_test)
  metrics = regression_metrics(dataset_name, MODEL_ID, prepared.y_test, predictions)
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  extra = format_wavelet_stats_markdown(prepared.wavelet_stats)
  sections = [("Wavelet Detail Coefficients (mean absolute value)", extra)] if extra else None
  return TrainingResult(metrics=metrics, predictions=predictions_df, extra_sections=sections)
