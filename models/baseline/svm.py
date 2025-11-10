"""Support Vector Machine baseline model."""
from __future__ import annotations

from typing import Any, Dict, List

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from models.base import extract_features_and_target, regression_metrics
from utils.reporting import MetricResult


MODEL_ID = "baseline.svm"


def build_model(hyperparams: Dict[str, Any] | None = None) -> Pipeline:
  params = {
    "kernel": "rbf",
    "C": 1.0,
    "epsilon": 0.1,
  }
  if hyperparams:
    params.update(hyperparams)
  return Pipeline([
    ("scaler", StandardScaler()),
    ("estimator", SVR(**params)),
  ])


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> List[MetricResult]:
  features = dataset_cfg.get("features")
  target = dataset_cfg.get("target")
  if not features or not target:
    raise KeyError("Dataset configuration must include features and target")
  X_train, y_train, X_test, y_test = extract_features_and_target(splits, features, target)
  model = build_model(dataset_cfg.get("svm_params"))
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  return regression_metrics(dataset_name, MODEL_ID, y_test, predictions)
