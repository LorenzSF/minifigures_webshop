"""Export catalog face embeddings for similarity search."""
# ruff: noqa: INP001

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from minifigures_model.data_utils import IMAGES_DIR, list_image_paths
from minifigures_model.face_cropper import DEFAULT_FACE_CROPPER_TAG, FaceCropper, crop_face
from minifigures_model.face_index import FACE_EMBEDDINGS_PATH, IndexedFace, save_face_index
from minifigures_model.face_similarity import DEFAULT_FACE_EMBEDDING_MODEL_TAG, embed_face
from minifigures_model.model import EncoderDecoder


def export_face_embeddings(
    cropper_tag: str = DEFAULT_FACE_CROPPER_TAG,
    embedding_model_tag: str = DEFAULT_FACE_EMBEDDING_MODEL_TAG,
    image_dir: Path = IMAGES_DIR,
    output_path: Path = FACE_EMBEDDINGS_PATH,
) -> Path:
    """Export face embeddings for all catalog images."""
    cropper = FaceCropper.load(cropper_tag)
    embedding_model = EncoderDecoder.load(embedding_model_tag)
    image_paths = list_image_paths(image_dir)
    if not image_paths:
        msg = f"No PNG catalog images found in {image_dir}."
        raise FileNotFoundError(msg)

    print(f"images_total: {len(image_paths)}")
    entries = {}
    for idx, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            face_box = cropper.predict_box(image_rgb)
            face_image = crop_face(image_rgb, face_box)
            embedding = embed_face(face_image, embedding_model)
        entries[image_path.stem] = IndexedFace(
            tag=image_path.stem, box=face_box, embedding=embedding
        )
        if idx % 250 == 0 or idx == len(image_paths):
            print(f"processed_count: {idx}")

    path = save_face_index(
        entries, output_path, embedding_model_tag=embedding_model_tag, cropper_tag=cropper_tag
    )
    print(f"output_path: {path}")
    print(f"image_count: {len(entries)}")
    return path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cropper-tag", default=DEFAULT_FACE_CROPPER_TAG)
    parser.add_argument("--embedding-model-tag", default=DEFAULT_FACE_EMBEDDING_MODEL_TAG)
    parser.add_argument("--image-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--output-path", type=Path, default=FACE_EMBEDDINGS_PATH)
    return parser.parse_args()


def main() -> None:
    """Run the export command."""
    args = parse_args()
    export_face_embeddings(
        cropper_tag=args.cropper_tag,
        embedding_model_tag=args.embedding_model_tag,
        image_dir=args.image_dir,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
