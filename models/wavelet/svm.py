"""Wavelet-enhanced SVM model leveraging the shared preprocessing pipeline."""
from __future__ import annotations

from typing import Any, Dict, List

from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from models.base import regression_metrics
from models.wavelet.common import prepare_wavelet_data
from utils.reporting import MetricResult

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


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> List[MetricResult]:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  model = build_estimator(dataset_cfg.get("wavelet_svm_params"))
  pipeline = Pipeline([
    ("pre", prepared.preprocessor),
    ("svm", model),
  ])
  pipeline.fit(prepared.X_train, prepared.y_train)
  predictions = pipeline.predict(prepared.X_test)
  return regression_metrics(dataset_name, MODEL_ID, prepared.y_test, predictions)
