"""Wavelet-enhanced Random Forest model."""
from __future__ import annotations

from typing import Any, Dict, List

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from models.base import regression_metrics
from models.wavelet.common import prepare_wavelet_data
from utils.reporting import MetricResult

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


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> List[MetricResult]:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  model = build_estimator(dataset_cfg.get("wavelet_rf_params"))
  pipeline = Pipeline([
    ("pre", prepared.preprocessor),
    ("rf", model),
  ])
  pipeline.fit(prepared.X_train, prepared.y_train)
  predictions = pipeline.predict(prepared.X_test)
  return regression_metrics(dataset_name, MODEL_ID, prepared.y_test, predictions)
