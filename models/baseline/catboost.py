"""CatBoost baseline regressor."""
from __future__ import annotations

from typing import Any, Dict

try:
  from catboost import CatBoostRegressor
except ImportError as exc:  # pragma: no cover
  CatBoostRegressor = None
  CATBOOST_IMPORT_ERROR = exc
else:
  CATBOOST_IMPORT_ERROR = None

from models.base import TrainingResult, build_predictions_frame, regression_metrics

MODEL_ID = "baseline.catboost"


def build_model(hyperparams: Dict[str, Any] | None = None):
  if CatBoostRegressor is None:
    raise RuntimeError("catboost is not installed") from CATBOOST_IMPORT_ERROR
  params = {
    "iterations": 800,
    "depth": 8,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": False,
  }
  if hyperparams:
    params.update(hyperparams)
  return CatBoostRegressor(**params)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  features = dataset_cfg.get("features")
  target = dataset_cfg.get("target")
  if not features or not target:
    raise KeyError("Dataset configuration must include features and target")
  train_df = splits.train[features].copy()
  test_df = splits.test[features].copy()
  y_train = splits.train[target]
  y_test = splits.test[target]

  categorical = [col for col in (dataset_cfg.get("categorical_features") or []) if col in train_df.columns]
  for col in categorical:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

  model = build_model(dataset_cfg.get("catboost_params"))
  model.fit(train_df, y_train, cat_features=categorical or None)
  predictions = model.predict(test_df)
  metrics = regression_metrics(dataset_name, MODEL_ID, y_test, predictions)
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  return TrainingResult(metrics=metrics, predictions=predictions_df)
