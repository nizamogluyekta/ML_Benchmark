"""Wavelet-enhanced deeper scikit-learn MLP."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from models.base import TrainingResult, build_predictions_frame, regression_metrics
from models.wavelet.common import format_wavelet_stats_markdown, prepare_wavelet_data

MODEL_ID = "wavelet.deep_sklearn"


def build_model(preprocessor, hyperparams: Dict[str, Any] | None = None) -> Pipeline:
  params = {
    "hidden_layer_sizes": (512, 256, 128, 64),
    "activation": "relu",
    "learning_rate_init": 4e-4,
    "max_iter": 800,
    "alpha": 5e-5,
    "random_state": 42,
    "early_stopping": True,
    "n_iter_no_change": 30,
  }
  if hyperparams:
    params.update(hyperparams)
  return Pipeline([
    ("pre", preprocessor),
    ("mlp", MLPRegressor(**params)),
  ])


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  pipeline = build_model(prepared.preprocessor, dataset_cfg.get("wavelet_deep_sklearn_params"))
  pipeline.fit(prepared.X_train, prepared.y_train)
  predictions = pipeline.predict(prepared.X_test)
  metrics = regression_metrics(dataset_name, MODEL_ID, prepared.y_test, predictions)
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  extra = format_wavelet_stats_markdown(prepared.wavelet_stats)
  sections = [("Wavelet Detail Coefficients (mean absolute value)", extra)] if extra else None
  return TrainingResult(metrics=metrics, predictions=predictions_df, extra_sections=sections)
