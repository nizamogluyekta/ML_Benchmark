"""Wavelet feature construction utilities ported from the VeriBilimi project."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

try:
  import pywt
except ImportError as exc:  # pragma: no cover - optional dependency
  pywt = None
  PYWT_IMPORT_ERROR = exc
else:
  PYWT_IMPORT_ERROR = None


def _require_pywt() -> None:
  if pywt is None:
    raise RuntimeError("pywt is required for wavelet transforms") from PYWT_IMPORT_ERROR


def add_wavelet_features(df: pd.DataFrame, column: str, wavelet: str = "db4", level: int = 3) -> List[str]:
  """Append approximation + detail coefficients as new columns."""
  if column not in df.columns:
    raise KeyError(f"Column {column} not found in dataframe")
  _require_pywt()

  series = df[column].astype(float).to_numpy()
  coeffs = pywt.wavedec(series, wavelet, level=level)

  approx_coeffs = [c.copy() for c in coeffs]
  for idx in range(1, len(approx_coeffs)):
    approx_coeffs[idx] = np.zeros_like(approx_coeffs[idx])
  approx_series = pywt.waverec(approx_coeffs, wavelet)[: len(series)]

  added_cols = []
  approx_col = f"{column}_wavelet_approx"
  df[approx_col] = approx_series
  added_cols.append(approx_col)

  for idx in range(1, len(coeffs)):
    coeff_tmp = [np.zeros_like(c) for c in coeffs]
    coeff_tmp[idx] = coeffs[idx]
    detail_series = pywt.waverec(coeff_tmp, wavelet)[: len(series)]
    detail_col = f"{column}_d{len(coeffs) - idx}"
    df[detail_col] = detail_series
    added_cols.append(detail_col)

  return added_cols


def add_temporal_features(df: pd.DataFrame, date_column: str = "DATE_KEY") -> List[str]:
  """Add seasonal signals derived from a datetime column."""
  if date_column not in df.columns:
    return []
  df[date_column] = pd.to_datetime(df[date_column])
  df["day_of_year"] = df[date_column].dt.dayofyear
  df["month"] = df[date_column].dt.month
  df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
  df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
  return ["day_of_year", "month", "sin_day", "cos_day"]


def add_interaction_features(df: pd.DataFrame) -> List[str]:
  """Add interaction and nonlinear terms used by the legacy wavelet scripts."""
  added = []
  if {"ORTALAMA_GUNLUK_SICAKLIK_C", "GUNLUK_GUNESLENME_SURESI_saat"}.issubset(df.columns):
    df["temp_sunshine"] = df["ORTALAMA_GUNLUK_SICAKLIK_C"] * df["GUNLUK_GUNESLENME_SURESI_saat"]
    added.append("temp_sunshine")
    df["sunshine_squared"] = df["GUNLUK_GUNESLENME_SURESI_saat"] ** 2
    added.append("sunshine_squared")
    df["temp_squared"] = df["ORTALAMA_GUNLUK_SICAKLIK_C"] ** 2
    added.append("temp_squared")
  if {"ORTALAMA_GUNLUK_NEM_%", "GUNLUK_GUNESLENME_SURESI_saat"}.issubset(df.columns):
    df["humidity_sunshine"] = df["ORTALAMA_GUNLUK_NEM_%"] * df["GUNLUK_GUNESLENME_SURESI_saat"]
    added.append("humidity_sunshine")
  if "TOPLAM_GUNLUK_YAGIS_mm" in df.columns:
    df["has_rain"] = (df["TOPLAM_GUNLUK_YAGIS_mm"] > 0).astype(int)
    added.append("has_rain")
  return added


def augment_with_wavelet_context(
  df: pd.DataFrame,
  target_column: str,
  date_column: str = "DATE_KEY",
  wavelet: str = "db4",
  level: int = 3,
) -> List[str]:
  """Apply temporal, interaction, and wavelet augmentations."""
  added_cols: List[str] = []
  added_cols += add_temporal_features(df, date_column=date_column)
  added_cols += add_interaction_features(df)
  added_cols += add_wavelet_features(df, column=target_column, wavelet=wavelet, level=level)
  return added_cols


def summarize_wavelet_details(df: pd.DataFrame, target_column: str) -> Dict[str, float]:
  """Compute summary stats for detail coefficients to include in reports."""
  stats: Dict[str, float] = {}
  prefix = f"{target_column}_d"
  for col in df.columns:
    if not col.startswith(prefix):
      continue
    suffix = col[len(prefix):]
    label = suffix if suffix.startswith("d") else f"d{suffix}"
    stats[label] = float(df[col].abs().mean())
  return stats
