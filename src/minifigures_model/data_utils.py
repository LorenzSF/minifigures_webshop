"""Shared dataset and path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minifigures_model.constants import get_data_folder

DATA_DIR = get_data_folder()
IMAGES_DIR = DATA_DIR / "minifigures"
DATASETS_DIR = DATA_DIR / "datasets"
PREDICTIONS_DIR = DATA_DIR / "predictions"
BASE_DATASET_PATH = DATA_DIR / "dataset.json"
LABELED_DATASET_PATH = DATA_DIR / "dataset_labeled.json"
MERGED_DATASET_PATH = DATA_DIR / "dataset_merged_labeled.json"
ALL_PREDICTIONS_PATH = PREDICTIONS_DIR / "all_predictions.json"
FIXED_LR_PREDICTIONS_PATH = PREDICTIONS_DIR / "all_predictions_fixed_lr.json"


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, payload: Any, *, sort_keys: bool = False) -> None:
    """Write a JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(temp_path, "w") as f:
        json.dump(payload, f, indent=4, sort_keys=sort_keys)
    temp_path.replace(path)


def load_dataset(path: Path) -> dict[str, list[str]]:
    """Load a dataset JSON file."""
    return load_json(path)


def save_dataset(dataset: dict[str, list[str]], path: Path) -> None:
    """Save a dataset JSON file."""
    write_json(path, dataset, sort_keys=True)


def get_latest_dataset_path(pattern: str, data_dir: Path = DATASETS_DIR) -> Path:
    """Return the newest dataset split matching the provided glob pattern."""
    candidates = sorted(data_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        msg = f"No dataset files found for pattern: {pattern}"
        raise FileNotFoundError(msg)
    return candidates[0]


def get_classes(dataset: dict[str, list[str]]) -> list[str]:
    """Infer the sorted class list from a dataset mapping."""
    return sorted({label for labels in dataset.values() for label in labels})


def list_image_paths(images_dir: Path = IMAGES_DIR) -> list[Path]:
    """Return the available PNG images."""
    return [path for path in sorted(images_dir.glob("*.png")) if not path.name.startswith("._")]
