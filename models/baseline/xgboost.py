"""XGBoost baseline model."""
from __future__ import annotations

from typing import Any, Dict, List

try:
  from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - optional dependency
  XGBRegressor = None
  XGB_IMPORT_ERROR = exc
else:
  XGB_IMPORT_ERROR = None

from models.base import extract_features_and_target, regression_metrics
from utils.reporting import MetricResult


MODEL_ID = "baseline.xgboost"


def build_model(hyperparams: Dict[str, Any] | None = None):
  if XGBRegressor is None:
    raise RuntimeError("xgboost is not installed") from XGB_IMPORT_ERROR
  params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
  }
  if hyperparams:
    params.update(hyperparams)
  return XGBRegressor(**params)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> List[MetricResult]:
  features = dataset_cfg.get("features")
  target = dataset_cfg.get("target")
  if not features or not target:
    raise KeyError("Dataset configuration must include features and target")
  X_train, y_train, X_test, y_test = extract_features_and_target(splits, features, target)
  model = build_model(dataset_cfg.get("xgb_params"))
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  return regression_metrics(dataset_name, MODEL_ID, y_test, predictions)
