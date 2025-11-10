"""Shared preprocessing pipeline for wavelet-based models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from utils.wavelet import augment_with_wavelet_context, summarize_wavelet_details


@dataclass
class WaveletPreparedData:
  X_train: pd.DataFrame
  y_train: pd.Series
  X_test: pd.DataFrame
  y_test: pd.Series
  preprocessor: ColumnTransformer
  numeric_features: List[str]
  categorical_features: List[str]
  wavelet_stats: Dict[str, float]


def engineer_wavelet_frame(df: pd.DataFrame, dataset_cfg: Dict[str, Any]) -> pd.DataFrame:
  df = df.copy()
  target_column = dataset_cfg.get("target")
  if not target_column:
    raise KeyError("Dataset configuration must define a target column")
  date_column = dataset_cfg.get("datetime", "DATE_KEY")
  wavelet_cfg = dataset_cfg.get("wavelet", {})
  augment_with_wavelet_context(
    df,
    target_column=target_column,
    date_column=date_column,
    wavelet=wavelet_cfg.get("wavelet", "db4"),
    level=wavelet_cfg.get("level", 3),
  )
  return df


def _compile_feature_lists(df: pd.DataFrame, dataset_cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
  base_numeric = [col for col in dataset_cfg.get("features", []) if col in df.columns]
  engineered = [
    "sin_day",
    "cos_day",
    "month",
    "temp_sunshine",
    "humidity_sunshine",
    "sunshine_squared",
    "temp_squared",
    "has_rain",
  ]
  target = dataset_cfg.get("target")
  wavelet_cols: List[str] = []
  if target:
    wavelet_cols = [col for col in df.columns if col.startswith(f"{target}_wavelet") or col.startswith(f"{target}_d")]
  numeric = [col for col in base_numeric + engineered + wavelet_cols if col in df.columns]
  categorical = [col for col in dataset_cfg.get("categorical_features", []) if col in df.columns]
  return numeric, categorical


def _build_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
  transformers = []
  if numeric:
    transformers.append(("num", Pipeline([("scaler", StandardScaler()), ("power", PowerTransformer())]), numeric))
  if categorical:
    transformers.append(("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
  if not transformers:
    raise ValueError("No features available for wavelet preprocessing")
  return ColumnTransformer(transformers)


def prepare_wavelet_data(splits, dataset_cfg: Dict[str, Any]) -> WaveletPreparedData:
  train_df = engineer_wavelet_frame(splits.train, dataset_cfg)
  test_df = engineer_wavelet_frame(splits.test, dataset_cfg)
  numeric, categorical = _compile_feature_lists(train_df, dataset_cfg)
  feature_cols = numeric + categorical
  if not feature_cols:
    raise ValueError("Wavelet pipeline requires at least one feature column")

  X_train = train_df[feature_cols]
  y_train = train_df[dataset_cfg["target"]]
  X_test = test_df[feature_cols]
  y_test = test_df[dataset_cfg["target"]]

  preprocessor = _build_preprocessor(numeric, categorical)
  stats = summarize_wavelet_details(train_df, dataset_cfg["target"])

  return WaveletPreparedData(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    preprocessor=preprocessor,
    numeric_features=numeric,
    categorical_features=categorical,
    wavelet_stats=stats,
  )
