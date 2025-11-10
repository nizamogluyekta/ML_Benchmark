"""Wavelet-enhanced LightGBM model."""
from __future__ import annotations

from typing import Any, Dict

try:
  import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - optional dependency
  lgb = None
  LGB_IMPORT_ERROR = exc
else:
  LGB_IMPORT_ERROR = None

from models.base import TrainingResult, build_predictions_frame, regression_metrics
from models.wavelet.common import format_wavelet_stats_markdown, prepare_wavelet_data

MODEL_ID = "wavelet.lightgbm"


def build_model(hyperparams: Dict[str, Any] | None = None):
  if lgb is None:
    raise RuntimeError("lightgbm is not installed") from LGB_IMPORT_ERROR
  params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 96,
    "learning_rate": 0.04,
    "n_estimators": 900,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "n_jobs": -1,
  }
  if hyperparams:
    params.update(hyperparams)
  return lgb.LGBMRegressor(**params)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  model = build_model(dataset_cfg.get("wavelet_lgbm_params"))
  model.fit(prepared.X_train, prepared.y_train)
  predictions = model.predict(prepared.X_test)
  metrics = regression_metrics(dataset_name, MODEL_ID, prepared.y_test, predictions)
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  extra = format_wavelet_stats_markdown(prepared.wavelet_stats)
  sections = [("Wavelet Detail Coefficients (mean absolute value)", extra)] if extra else None
  return TrainingResult(metrics=metrics, predictions=predictions_df, extra_sections=sections)
