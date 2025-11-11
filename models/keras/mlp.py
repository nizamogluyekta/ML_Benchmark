"""Keras (TensorFlow) MLP baseline."""
from __future__ import annotations

from typing import Any, Dict

try:
  import tensorflow as tf
except ImportError as exc:  # pragma: no cover
  tf = None
  TF_IMPORT_ERROR = exc
else:  # pragma: no cover
  TF_IMPORT_ERROR = None

import numpy as np
from sklearn.preprocessing import StandardScaler

from models.base import TrainingResult, build_predictions_frame, extract_features_and_target, regression_metrics

MODEL_ID = "keras.mlp"


def build_model(input_dim: int, hyperparams: Dict[str, Any] | None = None):
  if tf is None:
    raise RuntimeError("tensorflow is not installed") from TF_IMPORT_ERROR
  params = {
    "hidden_units": [256, 128, 64],
    "dropout": 0.1,
    "lr": 1e-3,
    "epochs": 60,
    "batch_size": 64,
    "validation_split": 0.15,
    "patience": 8,
  }
  if hyperparams:
    params.update(hyperparams)
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
  for units in params["hidden_units"]:
    model.add(tf.keras.layers.Dense(units, activation="relu"))
    if params["dropout"]:
      model.add(tf.keras.layers.Dropout(params["dropout"]))
  model.add(tf.keras.layers.Dense(1))
  optimizer = tf.keras.optimizers.Adam(learning_rate=params["lr"])
  model.compile(optimizer=optimizer, loss="mse")
  return model, params


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  features = dataset_cfg.get("features")
  target = dataset_cfg.get("target")
  if not features or not target:
    raise KeyError("Dataset configuration must include features and target")
  X_train, y_train, X_test, y_test = extract_features_and_target(splits, features, target)
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  model, params = build_model(X_train.shape[1], dataset_cfg.get("keras_params"))
  callbacks = []
  if params["patience"]:
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"], restore_best_weights=True))
  model.fit(
    X_train,
    y_train,
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    validation_split=params["validation_split"],
    callbacks=callbacks,
    verbose=0,
  )
  predictions = model.predict(X_test, batch_size=params["batch_size"], verbose=0).ravel()
  metrics = regression_metrics(dataset_name, MODEL_ID, y_test, predictions)
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  return TrainingResult(metrics=metrics, predictions=predictions_df)
