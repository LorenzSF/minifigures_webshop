"""Merge the base dataset with the exported Label Studio dataset."""

from pathlib import Path

from minifigures_model.data_utils import (
    BASE_DATASET_PATH,
    LABELED_DATASET_PATH,
    MERGED_DATASET_PATH,
    load_dataset,
    save_dataset,
)


def merge_datasets(reference_path: Path, candidate_path: Path, output_path: Path) -> None:
    """Merge datasets without duplicating keys.

    Existing labels in the reference dataset are preserved.
    Only new tags from the candidate dataset are added.
    """
    reference = load_dataset(reference_path)
    candidate = load_dataset(candidate_path)

    merged = dict(reference)
    added_count = 0

    # Add only new tags from the exported dataset.
    for tag, labels in candidate.items():
        if tag not in merged:
            merged[tag] = labels
            added_count += 1

    save_dataset(merged, output_path)

    print(f"reference_path: {reference_path}")
    print(f"candidate_path: {candidate_path}")
    print(f"output_path: {output_path}")
    print(f"reference_count: {len(reference)}")
    print(f"candidate_count: {len(candidate)}")
    print(f"added_count: {added_count}")
    print(f"merged_count: {len(merged)}")


if __name__ == "__main__":
    merge_datasets(
        reference_path=BASE_DATASET_PATH,
        candidate_path=LABELED_DATASET_PATH,
        output_path=MERGED_DATASET_PATH,
    )
