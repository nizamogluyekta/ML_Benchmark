"""Deeper scikit-learn MLP regressor baseline."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import TrainingResult, build_predictions_frame, extract_features_and_target, regression_metrics

MODEL_ID = "deep.sklearn_mlp"


def build_model(hyperparams: Dict[str, Any] | None = None) -> Pipeline:
  params = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation": "relu",
    "learning_rate_init": 5e-4,
    "max_iter": 600,
    "alpha": 1e-4,
    "random_state": 42,
    "early_stopping": True,
    "n_iter_no_change": 20,
  }
  if hyperparams:
    params.update(hyperparams)
  return Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(**params)),
  ])


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  features = dataset_cfg.get("features")
  target = dataset_cfg.get("target")
  if not features or not target:
    raise KeyError("Dataset configuration must include features and target")
  X_train, y_train, X_test, y_test = extract_features_and_target(splits, features, target)
  pipeline = build_model(dataset_cfg.get("deep_sklearn_params"))
  pipeline.fit(X_train, y_train)
  predictions = pipeline.predict(X_test)
  metrics = regression_metrics(dataset_name, MODEL_ID, y_test, predictions)
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  return TrainingResult(metrics=metrics, predictions=predictions_df)
