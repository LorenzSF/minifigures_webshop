"""Export Label Studio annotations to a simple dataset JSON file."""

import json
from pathlib import Path

from minifigures_model.data_utils import DATA_DIR, LABELED_DATASET_PATH, write_json


def export_label_studio_annotations_to_simple_json(
    source_dir: Path, destination_path: Path = LABELED_DATASET_PATH
) -> None:
    """Convert label-studio annotations to json."""
    dataset = {}

    # Iterate over exported Label Studio files.
    for path in source_dir.iterdir():
        with open(path) as f:
            metadata = json.load(f)

        if not metadata["was_cancelled"]:
            labels = metadata["result"][0]["value"]["choices"]
            image_path: str = metadata["task"]["data"]["image"]
            image_tag = image_path.split("/")[-1][:-4]
            dataset[image_tag] = labels

    write_json(destination_path, dataset, sort_keys=True)
    print(f"output_path: {destination_path}")
    print(f"image_count: {len(dataset)}")


if __name__ == "__main__":
    label_studio_annotations_dir = DATA_DIR / "target_annotations"
    export_label_studio_annotations_to_simple_json(label_studio_annotations_dir)
