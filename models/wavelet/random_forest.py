"""Wavelet-enhanced Random Forest model."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from models.base import TrainingResult, build_predictions_frame, regression_metrics
from models.wavelet.common import format_wavelet_stats_markdown, prepare_wavelet_data

MODEL_ID = "wavelet.random_forest"


def build_estimator(hyperparams: Dict[str, Any] | None = None) -> RandomForestRegressor:
  params = {
    "n_estimators": 400,
    "random_state": 42,
    "n_jobs": -1,
  }
  if hyperparams:
    params.update(hyperparams)
  return RandomForestRegressor(**params)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  model = build_estimator(dataset_cfg.get("wavelet_rf_params"))
  pipeline = Pipeline([
    ("pre", prepared.preprocessor),
    ("rf", model),
  ])
  pipeline.fit(prepared.X_train, prepared.y_train)
  predictions = pipeline.predict(prepared.X_test)
  metrics = regression_metrics(dataset_name, MODEL_ID, prepared.y_test, predictions)
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  extra = format_wavelet_stats_markdown(prepared.wavelet_stats)
  sections = [("Wavelet Detail Coefficients (mean absolute value)", extra)] if extra else None
  return TrainingResult(metrics=metrics, predictions=predictions_df, extra_sections=sections)
