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

from models.base import TrainingResult, build_predictions_frame, regression_metrics, timed_fit_predict, timing_metrics

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
  train_df = splits.train[features].copy()
  test_df = splits.test[features].copy()
  categorical = dataset_cfg.get("categorical_features", [])
  for col in categorical or []:
    if col in train_df.columns:
      train_df[col] = train_df[col].astype("category")
    if col in test_df.columns:
      test_df[col] = test_df[col].astype("category")
  y_train = splits.train[target]
  y_test = splits.test[target]
  model = build_model(dataset_cfg.get("lgbm_params"))
  predictions, train_time, predict_time = timed_fit_predict(model, train_df, y_train, test_df)
  metrics = regression_metrics(dataset_name, MODEL_ID, y_test, predictions)
  metrics.extend(timing_metrics(dataset_name, MODEL_ID, train_time, predict_time))
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  return TrainingResult(metrics=metrics, predictions=predictions_df)
