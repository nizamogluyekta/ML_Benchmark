"""Wavelet-enhanced Keras MLP."""
from __future__ import annotations

from typing import Any, Dict
from time import perf_counter

try:
  import tensorflow as tf
except ImportError as exc:  # pragma: no cover
  tf = None
  TF_IMPORT_ERROR = exc
else:  # pragma: no cover
  TF_IMPORT_ERROR = None

from models.base import TrainingResult, build_predictions_frame, regression_metrics, timing_metrics
from models.wavelet.common import format_wavelet_stats_markdown, prepare_wavelet_data

MODEL_ID = "wavelet.keras_mlp"


def build_model(input_dim: int, hyperparams: Dict[str, Any] | None = None):
  if tf is None:
    raise RuntimeError("tensorflow is not installed") from TF_IMPORT_ERROR
  params = {
    "hidden_units": [512, 256, 128, 64],
    "dropout": 0.15,
    "lr": 8e-4,
    "epochs": 70,
    "batch_size": 64,
    "validation_split": 0.2,
    "patience": 10,
  }
  if hyperparams:
    params.update(hyperparams)
  model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(input_dim,))])
  for units in params["hidden_units"]:
    model.add(tf.keras.layers.Dense(units, activation="relu"))
    if params["dropout"]:
      model.add(tf.keras.layers.Dropout(params["dropout"]))
  model.add(tf.keras.layers.Dense(1))
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]), loss="mse")
  return model, params


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  preprocessor = prepared.preprocessor
  X_train = preprocessor.fit_transform(prepared.X_train, prepared.y_train)
  X_test = preprocessor.transform(prepared.X_test)

  model, params = build_model(X_train.shape[1], dataset_cfg.get("wavelet_keras_params"))
  callbacks = []
  if params["patience"]:
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"], restore_best_weights=True))
  start = perf_counter()
  model.fit(
    X_train,
    prepared.y_train,
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    validation_split=params["validation_split"],
    callbacks=callbacks,
    verbose=0,
  )
  train_time = perf_counter() - start
  start = perf_counter()
  predictions = model.predict(X_test, batch_size=params["batch_size"], verbose=0).ravel()
  predict_time = perf_counter() - start
  metrics = regression_metrics(dataset_name, MODEL_ID, prepared.y_test, predictions)
  metrics.extend(timing_metrics(dataset_name, MODEL_ID, train_time, predict_time))
  predictions_df = build_predictions_frame(splits, dataset_cfg, predictions)
  extra = format_wavelet_stats_markdown(prepared.wavelet_stats)
  sections = [("Wavelet Detail Coefficients (mean absolute value)", extra)] if extra else None
  return TrainingResult(metrics=metrics, predictions=predictions_df, extra_sections=sections)
