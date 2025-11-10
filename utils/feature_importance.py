"""Shared helpers for computing feature importances across models."""
from __future__ import annotations

from typing import Dict, Sequence
import numpy as np
import pandas as pd


def compute_importance_from_estimator(estimator, feature_names: Sequence[str]) -> pd.DataFrame:
  """Return a dataframe with normalized importance scores.

  Works with estimators that expose either ``feature_importances_`` or
  ``coef_`` attributes. Values are normalized so they sum to 1.
  """
  if hasattr(estimator, "feature_importances_"):
    raw = np.asarray(estimator.feature_importances_, dtype=float)
  elif hasattr(estimator, "coef_"):
    raw = np.abs(np.asarray(estimator.coef_).ravel())
  else:
    raise ValueError("Estimator does not expose feature importances")

  total = raw.sum()
  normalized = raw / total if total else raw
  return pd.DataFrame({"feature": feature_names, "importance": normalized})


def top_k_features(importance_df: pd.DataFrame, k: int = 10) -> Dict[str, float]:
  """Return the top-k features as a dict for quick reporting."""
  subset = importance_df.sort_values("importance", ascending=False).head(k)
  return dict(zip(subset.feature, subset.importance))
