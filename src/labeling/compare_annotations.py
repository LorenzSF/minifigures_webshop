"""Compare existing dataset labels with the latest exported Label Studio dataset."""

import json
from pathlib import Path

DATA_DIR = Path("/workspaces/updated-minifigures-webshop-2026-LorenzSF/data/data")
IMAGES_DIR = DATA_DIR / "minifigures"


def get_latest_dataset_labeled_path(data_dir: Path = DATA_DIR) -> Path:
    """Return the latest exported dataset_labeled json."""
    candidates = sorted(data_dir.glob("dataset_labeled*.json"))
    if not candidates:
        raise FileNotFoundError("No dataset_labeled*.json files found.")
    return candidates[-1]


def compare_datasets(reference_path: Path, candidate_path: Path) -> None:
    """Print overlap and label differences between two datasets."""
    with open(reference_path) as f:
        reference = json.load(f)

    with open(candidate_path) as f:
        candidate = json.load(f)

    reference_keys = set(reference)
    candidate_keys = set(candidate)
    overlap = sorted(reference_keys & candidate_keys)
    only_candidate = sorted(candidate_keys - reference_keys)

    print(f"reference_count: {len(reference)}")
    print(f"candidate_count: {len(candidate)}")
    print(f"overlap_count: {len(overlap)}")
    print(f"only_candidate_count: {len(only_candidate)}")

    mismatches = []
    for tag in overlap:
        reference_labels = sorted(reference[tag])
        candidate_labels = sorted(candidate[tag])
        if reference_labels != candidate_labels:
            mismatches.append((tag, reference_labels, candidate_labels))

    print(f"mismatch_count: {len(mismatches)}")
    print()

    if overlap:
        print("overlapping_tags:")
        for tag in overlap:
            print(f"  {tag}")
            print(f"    image: {IMAGES_DIR / f'{tag}.png'}")
        print()

    if mismatches:
        print("mismatches:")
        for tag, reference_labels, candidate_labels in mismatches:
            print(f"  {tag}")
            print(f"    image: {IMAGES_DIR / f'{tag}.png'}")
            print(f"    reference: {reference_labels}")
            print(f"    candidate: {candidate_labels}")


if __name__ == "__main__":
    latest_candidate_path = get_latest_dataset_labeled_path()
    reference_path = DATA_DIR / "dataset.json"
    print(f"reference_path: {reference_path}")
    print(f"latest_candidate: {latest_candidate_path}")
    print()
    compare_datasets(reference_path=reference_path, candidate_path=latest_candidate_path)
