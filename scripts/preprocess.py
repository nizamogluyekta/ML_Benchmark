"""Dataset preprocessing CLI."""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs/datasets.yaml"


logger = logging.getLogger(__name__)


def load_dataset_configs(config_path: Path = CONFIG_PATH) -> Dict[str, Dict[str, Any]]:
  with config_path.open() as handle:
    return yaml.safe_load(handle) or {}


def _copy_split_file(src: Path, dest: Path, force: bool) -> None:
  if dest.exists() and not force:
    return
  if not src.exists():
    raise FileNotFoundError(f"Source split not found: {src}")
  dest.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dest)


def copy_source_splits(dataset_name: str, dataset_cfg: Dict[str, Any], force: bool = False) -> bool:
  source_files = dataset_cfg.get("source_files") or {}
  if not source_files:
    return False

  processed_dir = PROCESSED_DIR / dataset_name
  processed_dir.mkdir(parents=True, exist_ok=True)

  copied = False
  for key, filename in ("train", "training.csv"), ("test", "testing.csv"):
    src_value = source_files.get(key)
    if not src_value:
      continue
    src_path = Path(src_value)
    if not src_path.is_absolute():
      src_path = PROJECT_ROOT / src_path
    dest_path = processed_dir / filename
    _copy_split_file(src_path, dest_path, force)
    copied = True
  return copied


def write_placeholder_split(dataset_name: str, feature_list: Any, target: str, force: bool = False) -> None:
  processed_dir = PROCESSED_DIR / dataset_name
  processed_dir.mkdir(parents=True, exist_ok=True)
  headers = list(feature_list or []) + [target]
  for split in ("training.csv", "testing.csv"):
    split_path = processed_dir / split
    if split_path.exists() and not force:
      continue
    with split_path.open("w", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(headers)
  (processed_dir / "README.txt").write_text(
    "This dataset was generated as a placeholder. Replace with actual processed data.\n"
  )


def persist_metadata(dataset_name: str, dataset_cfg: Dict[str, Any]) -> None:
  processed_dir = PROCESSED_DIR / dataset_name
  processed_dir.mkdir(parents=True, exist_ok=True)
  metadata_file = processed_dir / "metadata.yaml"
  metadata_file.write_text(yaml.safe_dump(dataset_cfg, sort_keys=False))


def preprocess_dataset(dataset_name: str, dataset_cfg: Dict[str, Any], force: bool = False) -> Path:
  logger.info("Preparing dataset %s", dataset_name)
  persist_metadata(dataset_name, dataset_cfg)
  copied = copy_source_splits(dataset_name, dataset_cfg, force=force)
  if not copied:
    write_placeholder_split(dataset_name, dataset_cfg.get("features"), dataset_cfg.get("target", "target"), force=force)
  return PROCESSED_DIR / dataset_name


def run(dataset: str | None, force: bool = False) -> None:
  configs = load_dataset_configs()
  if not configs:
    raise RuntimeError("No dataset configurations found")
  datasets = [dataset] if dataset else list(configs.keys())
  for name in datasets:
    if name not in configs:
      raise KeyError(f"Dataset {name} is not defined in configs/datasets.yaml")
    preprocess_dataset(name, configs[name], force=force)


def main() -> None:
  parser = argparse.ArgumentParser(description="Preprocess datasets for the ML benchmark")
  parser.add_argument("--dataset", help="Name of the dataset to preprocess. Defaults to all defined datasets.")
  parser.add_argument("--force", action="store_true", help="Overwrite existing processed splits.")
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
  run(dataset=args.dataset, force=args.force)


if __name__ == "__main__":
  main()
