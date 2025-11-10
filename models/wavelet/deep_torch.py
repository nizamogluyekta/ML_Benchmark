"""Wavelet-enhanced PyTorch MLP."""
from __future__ import annotations

from typing import Any, Dict

try:
  import torch
  from torch import nn
  from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - optional dependency
  torch = None
  nn = None
  DataLoader = None
  TensorDataset = None
  TORCH_IMPORT_ERROR = exc
else:
  TORCH_IMPORT_ERROR = None

import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

from models.base import TrainingResult, build_predictions_frame, regression_metrics
from models.wavelet.common import format_wavelet_stats_markdown, prepare_wavelet_data

MODEL_ID = "wavelet.deep_torch"
logger = logging.getLogger(__name__)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
  return torch.as_tensor(array, dtype=torch.float32)


def build_model(input_dim: int, hyperparams: Dict[str, Any] | None = None):
  if torch is None:
    raise RuntimeError("PyTorch is not installed") from TORCH_IMPORT_ERROR
  params = {
    "hidden_dims": (256, 128, 64),
    "dropout": 0.1,
    "lr": 8e-4,
    "epochs": 150,
    "batch_size": 64,
    "weight_decay": 5e-5,
    "patience": 15,
    "log_every": 20,
  }
  if hyperparams:
    params.update(hyperparams)

  layers = []
  prev = input_dim
  for dim in params["hidden_dims"]:
    layers.append(nn.Linear(prev, dim))
    layers.append(nn.ReLU())
    if params["dropout"]:
      layers.append(nn.Dropout(params["dropout"]))
    prev = dim
  layers.append(nn.Linear(prev, 1))
  model = nn.Sequential(*layers)
  return model, params


def _train_loop(model: nn.Module, train_loader: DataLoader, val_data, params: Dict[str, Any]) -> None:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
  X_val, y_val = (tensor.to(device) for tensor in val_data)
  best_val = float("inf")
  patience = params.get("patience", 20)
  wait = 0

  for epoch in range(1, params["epochs"] + 1):
    model.train()
    for batch_X, batch_y in train_loader:
      batch_X = batch_X.to(device)
      batch_y = batch_y.to(device)
      optimizer.zero_grad(set_to_none=True)
      preds = model(batch_X).squeeze(-1)
      loss = criterion(preds, batch_y)
      loss.backward()
      optimizer.step()
    model.eval()
    with torch.no_grad():
      val_loss = criterion(model(X_val).squeeze(-1), y_val).item()
    if epoch % params["log_every"] == 0:
      logger.info("[wavelet.deep_torch] epoch %d/%d val_loss=%.6f", epoch, params["epochs"], val_loss)
    if val_loss + 1e-6 < best_val:
      best_val = val_loss
      best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
      wait = 0
    else:
      wait += 1
      if wait >= patience:
        break
  if 'best_state' in locals():
    model.load_state_dict(best_state)


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  prepared = prepare_wavelet_data(splits, dataset_cfg)
  scaler = StandardScaler()
  X_train = scaler.fit_transform(prepared.X_train)
  X_test = scaler.transform(prepared.X_test)

  model, params = build_model(X_train.shape[1], dataset_cfg.get("wavelet_deep_params"))
  train_dataset = TensorDataset(_to_tensor(X_train), _to_tensor(prepared.y_train.to_numpy()))
  train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
  val_data = (_to_tensor(X_test), _to_tensor(prepared.y_test.to_numpy()))
  _train_loop(model, train_loader, val_data, params)

  model.eval()
  with torch.no_grad():
    preds = model(_to_tensor(X_test)).squeeze(-1).cpu().numpy()

  metrics = regression_metrics(dataset_name, MODEL_ID, prepared.y_test, preds)
  predictions_df = build_predictions_frame(splits, dataset_cfg, preds)
  extra = format_wavelet_stats_markdown(prepared.wavelet_stats)
  sections = [("Wavelet Detail Coefficients (mean absolute value)", extra)] if extra else None
  return TrainingResult(metrics=metrics, predictions=predictions_df, extra_sections=sections)
