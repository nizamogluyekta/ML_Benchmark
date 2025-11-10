"""Wavelet feature construction utilities."""
from __future__ import annotations

from typing import Tuple
import numpy as np

try:
  import pywt
except ImportError as exc:  # pragma: no cover - optional dependency
  pywt = None
  MISSING_WAVELET_ERROR = exc
else:
  MISSING_WAVELET_ERROR = None


def apply_wavelet_transform(series: np.ndarray, wavelet: str = "db4", level: int = 1) -> Tuple[np.ndarray, np.ndarray]:
  """Return approximation and detail coefficients for a given series."""
  if pywt is None:
    raise RuntimeError("pywt is required for wavelet transforms") from MISSING_WAVELET_ERROR
  coeffs = pywt.wavedec(series, wavelet, level=level)
  approx, *details = coeffs
  return approx, np.concatenate(details) if details else np.array([])


def augment_features(series: np.ndarray, wavelet: str = "db4", level: int = 1) -> np.ndarray:
  """Concatenate original values with approximation coefficients."""
  approx, details = apply_wavelet_transform(series, wavelet=wavelet, level=level)
  if details.size:
    return np.vstack([series, approx[: series.shape[0]], details[: series.shape[0]]]).T
  return np.vstack([series, approx[: series.shape[0]]]).T
