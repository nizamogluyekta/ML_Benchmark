"""Wavelet-enhanced XGBoost model."""
from __future__ import annotations

from typing import Any, Dict, List

try:
  from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - optional dependency
  XGBRegressor = None
  XGB_IMPORT_ERROR = exc
else:
  XGB_IMPORT_ERROR = None

from sklearn.pipeline import Pipeline

from models.base import regression_metrics
from models.wavelet.common import prepare_wavelet_data
from utils.reporting import MetricResult

MODEL_ID = "wavelet.xgboost"


def build_estimator(hyperparams: Dict[str, Any] | None = None):
  if XGBRegressor is None:
    raise RuntimeError("xgboost is not installed") from XGB_IMPORT_ERROR
  params = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
  }
  if hyperparams:
    params.update(hyperparams)
  return XGBRegressor(**params)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> List[MetricResult]:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  model = build_estimator(dataset_cfg.get("wavelet_xgb_params"))
  pipeline = Pipeline([
    ("pre", prepared.preprocessor),
    ("xgb", model),
  ])
  pipeline.fit(prepared.X_train, prepared.y_train)
  predictions = pipeline.predict(prepared.X_test)
  return regression_metrics(dataset_name, MODEL_ID, prepared.y_test, predictions)
