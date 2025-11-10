"""Random Forest baseline model."""
from __future__ import annotations

from typing import Any, Dict, List

from sklearn.ensemble import RandomForestRegressor

from models.base import extract_features_and_target, regression_metrics
from utils.reporting import MetricResult


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


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> List[MetricResult]:
  features = dataset_cfg.get("features")
  target = dataset_cfg.get("target")
  if not features or not target:
    raise KeyError("Dataset configuration must include features and target")
  X_train, y_train, X_test, y_test = extract_features_and_target(splits, features, target)
  model = build_model(dataset_cfg.get("rf_params"))
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  return regression_metrics(dataset_name, MODEL_ID, y_test, predictions)
