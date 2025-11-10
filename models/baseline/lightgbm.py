"""LightGBM baseline regressor."""
from __future__ import annotations

from typing import Any, Dict

try:
  import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - optional dependency
  lgb = None
  LGB_IMPORT_ERROR = exc
else:
  LGB_IMPORT_ERROR = None

from models.base import TrainingResult, build_predictions_frame, extract_features_and_target, regression_metrics

MODEL_ID = "baseline.lightgbm"


def build_model(hyperparams: Dict[str, Any] | None = None):
  if lgb is None:
    raise RuntimeError("lightgbm is not installed") from LGB_IMPORT_ERROR
  params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "n_estimators": 800,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
  }
  if hyperparams:
    params.update(hyperparams)
  return lgb.LGBMRegressor(**params)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  features = dataset_cfg.get("features")
  target = dataset_cfg.get("target")
  if not features or not target:
    raise KeyError("Dataset configuration must include features and target")
  X_train, y_train, X_test, y_test = extract_features_and_target(splits, features, target)
  model = build_model(dataset_cfg.get("lgbm_params"))
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  metrics = regression_metrics(dataset_name, MODEL_ID, y_test, predictions)
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  return TrainingResult(metrics=metrics, predictions=predictions_df)
