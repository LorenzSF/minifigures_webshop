"""Merge the base dataset with the latest exported Label Studio dataset."""

import json
from pathlib import Path

DATA_DIR = Path("/workspaces/updated-minifigures-webshop-2026-LorenzSF/data/data")
OUTPUT_PATH = DATA_DIR / "dataset_merged_labeled.json"


def get_latest_dataset_labeled_path(data_dir: Path = DATA_DIR) -> Path:
    """Return the latest exported dataset_labeled json."""
    candidates = sorted(data_dir.glob("dataset_labeled*.json"))
    if not candidates:
        raise FileNotFoundError("No dataset_labeled*.json files found.")
    return candidates[-1]


def merge_datasets(reference_path: Path, candidate_path: Path, output_path: Path) -> None:
    """Merge datasets without duplicating keys.

    Existing labels in the reference dataset are preserved.
    Only new tags from the candidate dataset are added.
    """
    with open(reference_path) as f:
        reference = json.load(f)

    with open(candidate_path) as f:
        candidate = json.load(f)

    merged = dict(reference)
    added_count = 0

    for tag, labels in candidate.items():
        if tag not in merged:
            merged[tag] = labels
            added_count += 1

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=4)

    print(f"reference_path: {reference_path}")
    print(f"candidate_path: {candidate_path}")
    print(f"output_path: {output_path}")
    print(f"reference_count: {len(reference)}")
    print(f"candidate_count: {len(candidate)}")
    print(f"added_count: {added_count}")
    print(f"merged_count: {len(merged)}")


if __name__ == "__main__":
    latest_candidate_path = get_latest_dataset_labeled_path()
    merge_datasets(
        reference_path=DATA_DIR / "dataset.json",
        candidate_path=latest_candidate_path,
        output_path=OUTPUT_PATH,
    )
