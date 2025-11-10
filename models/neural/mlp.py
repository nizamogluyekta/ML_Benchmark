"""Simple MLP regressor baseline relying on scikit-learn."""
from __future__ import annotations

from typing import Any, Dict, List

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from models.base import extract_features_and_target, regression_metrics
from utils.reporting import MetricResult

MODEL_ID = "neural.mlp"


def build_model(hyperparams: Dict[str, Any] | None = None) -> Pipeline:
  params = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "learning_rate_init": 1e-3,
    "max_iter": 500,
    "random_state": 42,
  }
  if hyperparams:
    params.update(hyperparams)
  return Pipeline([
    ("scaler", StandardScaler()),
    ("estimator", MLPRegressor(**params)),
  ])


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> List[MetricResult]:
  features = dataset_cfg.get("features")
  target = dataset_cfg.get("target")
  if not features or not target:
    raise KeyError("Dataset configuration must include features and target")
  X_train, y_train, X_test, y_test = extract_features_and_target(splits, features, target)
  model = build_model(dataset_cfg.get("mlp_params"))
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  return regression_metrics(dataset_name, MODEL_ID, y_test, predictions)
