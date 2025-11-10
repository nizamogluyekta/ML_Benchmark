"""PyTorch-based multilayer perceptron for regression."""
from __future__ import annotations

from typing import Any, Dict, Tuple

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

from models.base import TrainingResult, build_predictions_frame, extract_features_and_target, regression_metrics

MODEL_ID = "deep.torch_mlp"
logger = logging.getLogger(__name__)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
  return torch.as_tensor(array, dtype=torch.float32)


def build_model(input_dim: int, hyperparams: Dict[str, Any] | None = None) -> Tuple[nn.Module, Dict[str, Any]]:
  if torch is None:
    raise RuntimeError("PyTorch is not installed") from TORCH_IMPORT_ERROR
  params = {
    "hidden_dims": (128, 64),
    "dropout": 0.05,
    "lr": 1e-3,
    "epochs": 120,
    "batch_size": 64,
    "weight_decay": 1e-4,
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


def _train_loop(model: nn.Module, train_loader: DataLoader, val_data: Tuple[torch.Tensor, torch.Tensor], params: Dict[str, Any]) -> None:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
  X_val, y_val = (tensor.to(device) for tensor in val_data)

  model.train()
  for epoch in range(1, params["epochs"] + 1):
    for batch_X, batch_y in train_loader:
      batch_X = batch_X.to(device)
      batch_y = batch_y.to(device)
      optimizer.zero_grad(set_to_none=True)
      preds = model(batch_X).squeeze(-1)
      loss = criterion(preds, batch_y)
      loss.backward()
      optimizer.step()
    # simple early stop hook (not implemented yet, placeholder for extension)
    with torch.no_grad():
      model.eval()
      val_loss = criterion(model(X_val).squeeze(-1), y_val).item()
      if epoch % params["log_every"] == 0:
        logger.info("[deep.torch_mlp] epoch %d/%d val_loss=%.6f", epoch, params["epochs"], val_loss)
      model.train()


def train(dataset_name: str, dataset_cfg: Dict[str, Any], splits, output_dir) -> TrainingResult:
  features = dataset_cfg.get("features")
  target = dataset_cfg.get("target")
  if not features or not target:
    raise KeyError("Dataset configuration must include features and target")
  X_train, y_train, X_test, y_test = extract_features_and_target(splits, features, target)

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  model, params = build_model(X_train_scaled.shape[1], dataset_cfg.get("deep_params"))
  batch_size = params["batch_size"]

  train_dataset = TensorDataset(_to_tensor(X_train_scaled), _to_tensor(y_train))
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_tensors = (_to_tensor(X_test_scaled), _to_tensor(y_test))
  _train_loop(model, train_loader, val_tensors, params)

  model.eval()
  with torch.no_grad():
    preds = model(_to_tensor(X_test_scaled)).squeeze(-1).cpu().numpy()

  metrics = regression_metrics(dataset_name, MODEL_ID, y_test, preds)
  predictions_df = build_predictions_frame(splits, dataset_cfg, preds)
  return TrainingResult(metrics=metrics, predictions=predictions_df)
