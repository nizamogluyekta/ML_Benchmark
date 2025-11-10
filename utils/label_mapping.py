"""Label encoding utilities reused across datasets."""
from __future__ import annotations

from typing import Dict, Iterable


def create_mapping(labels: Iterable[str]) -> Dict[str, int]:
  """Create a deterministic label-to-index mapping."""
  return {label: idx for idx, label in enumerate(sorted(set(labels)))}


def inverse_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
  """Invert a mapping dictionary."""
  return {idx: label for label, idx in mapping.items()}
