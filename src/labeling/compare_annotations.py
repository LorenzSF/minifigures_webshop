"""Compare existing dataset labels with the exported Label Studio dataset."""

from pathlib import Path

from minifigures_model.data_utils import (
    BASE_DATASET_PATH,
    IMAGES_DIR,
    LABELED_DATASET_PATH,
    load_dataset,
)


def compare_datasets(reference_path: Path, candidate_path: Path) -> None:
    """Print overlap and label differences between two datasets."""
    reference = load_dataset(reference_path)
    candidate = load_dataset(candidate_path)

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
    reference_path = BASE_DATASET_PATH
    latest_candidate_path = LABELED_DATASET_PATH
    print(f"reference_path: {reference_path}")
    print(f"latest_candidate: {latest_candidate_path}")
    print()
    compare_datasets(reference_path=reference_path, candidate_path=latest_candidate_path)
