"""Wavelet-enhanced neural network model."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from models.base import TrainingResult, build_predictions_frame, regression_metrics, timed_fit_predict, timing_metrics
from models.wavelet.common import format_wavelet_stats_markdown, prepare_wavelet_data

MODEL_ID = "wavelet.neural"


def build_estimator(hyperparams: Dict[str, Any] | None = None) -> MLPRegressor:
  params = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation": "relu",
    "learning_rate_init": 5e-4,
    "max_iter": 600,
    "random_state": 42,
  }
  if hyperparams:
    params.update(hyperparams)
  return MLPRegressor(**params)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  estimator = build_estimator(dataset_cfg.get("wavelet_mlp_params"))
  pipeline = Pipeline([
    ("pre", prepared.preprocessor),
    ("mlp", estimator),
  ])
  predictions, train_time, predict_time = timed_fit_predict(pipeline, prepared.X_train, prepared.y_train, prepared.X_test)
  metrics = regression_metrics(dataset_name, MODEL_ID, prepared.y_test, predictions)
  metrics.extend(timing_metrics(dataset_name, MODEL_ID, train_time, predict_time))
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  extra = format_wavelet_stats_markdown(prepared.wavelet_stats)
  sections = [("Wavelet Detail Coefficients (mean absolute value)", extra)] if extra else None
  return TrainingResult(metrics=metrics, predictions=predictions_df, extra_sections=sections)
