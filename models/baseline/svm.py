"""Support Vector Machine baseline model."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from models.base import TrainingResult, build_predictions_frame, extract_features_and_target, regression_metrics


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


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  features = dataset_cfg.get("features")
  target = dataset_cfg.get("target")
  if not features or not target:
    raise KeyError("Dataset configuration must include features and target")
  X_train, y_train, X_test, y_test = extract_features_and_target(splits, features, target)
  model = build_model(dataset_cfg.get("svm_params"))
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  metrics = regression_metrics(dataset_name, MODEL_ID, y_test, predictions)
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  return TrainingResult(metrics=metrics, predictions=predictions_df)
