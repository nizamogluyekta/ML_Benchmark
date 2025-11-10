"""Wavelet-enhanced SVM model leveraging the shared preprocessing pipeline."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from models.base import TrainingResult, build_predictions_frame, regression_metrics
from models.wavelet.common import format_wavelet_stats_markdown, prepare_wavelet_data

MODEL_ID = "wavelet.svm"


def build_estimator(hyperparams: Dict[str, Any] | None = None) -> SVR:
  params = {
    "kernel": "rbf",
    "C": 5.0,
    "epsilon": 0.05,
  }
  if hyperparams:
    params.update(hyperparams)
  return SVR(**params)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  model = build_estimator(dataset_cfg.get("wavelet_svm_params"))
  pipeline = Pipeline([
    ("pre", prepared.preprocessor),
    ("svm", model),
  ])
  pipeline.fit(prepared.X_train, prepared.y_train)
  predictions = pipeline.predict(prepared.X_test)
  metrics = regression_metrics(dataset_name, MODEL_ID, prepared.y_test, predictions)
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  extra = format_wavelet_stats_markdown(prepared.wavelet_stats)
  sections = [("Wavelet Detail Coefficients (mean absolute value)", extra)] if extra else None
  return TrainingResult(metrics=metrics, predictions=predictions_df, extra_sections=sections)
