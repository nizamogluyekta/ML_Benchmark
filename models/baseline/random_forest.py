"""Random Forest baseline model."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestRegressor

from models.base import (
  TrainingResult,
  build_predictions_frame,
  extract_features_and_target,
  regression_metrics,
  timed_fit_predict,
  timing_metrics,
)


MODEL_ID = "baseline.random_forest"


def build_model(hyperparams: Dict[str, Any] | None = None) -> RandomForestRegressor:
  params = {
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,
  }
  if hyperparams:
    params.update(hyperparams)
  return RandomForestRegressor(**params)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  features = dataset_cfg.get("features")
  target = dataset_cfg.get("target")
  if not features or not target:
    raise KeyError("Dataset configuration must include features and target")
  X_train, y_train, X_test, y_test = extract_features_and_target(splits, features, target)
  model = build_model(dataset_cfg.get("rf_params"))
  predictions, train_time, predict_time = timed_fit_predict(model, X_train, y_train, X_test)
  metrics = regression_metrics(dataset_name, MODEL_ID, y_test, predictions)
  metrics.extend(timing_metrics(dataset_name, MODEL_ID, train_time, predict_time))
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  return TrainingResult(metrics=metrics, predictions=predictions_df)
