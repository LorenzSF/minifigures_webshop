"""Marketplace page."""
# ruff: noqa: INP001

from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from minifigures_app.utils import get_image, list_im_tags

ITEMS_PER_PAGE = 9
GRID_COLUMNS = 3
NOT_LABELED = "Not labeled"
LABELS_DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "data" / "dataset_merged_labeled.json"
)


def main():
    """Run the Streamlit Market page."""
    st.set_page_config(page_title="Market", page_icon="💰")

    # page description
    st.title("💰 Market")
    st.markdown("---")

    # Ensure clean session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = 0

    # get images
    all_image_tags = list_im_tags()
    labels_by_image = _load_labels_by_image()
    selected_labels = st.multiselect("Filter by label", options=_get_label_options(labels_by_image))
    image_tags = _filter_image_tags(
        image_tags=all_image_tags, labels_by_image=labels_by_image, selected_labels=selected_labels
    )
    _clamp_current_page(total_items=len(image_tags))
    selected_image_tag = _get_selected_market_image(image_tags)

    if not image_tags:
        st.info("No products match the selected labels.")
        return

    if selected_image_tag:
        _render_selected_image_detail(selected_image_tag, labels_by_image)
        if st.button("Back to market"):
            _clear_market_selection()
            st.rerun()
        return

    # paginate results
    image_slice = image_tags[
        st.session_state.current_page * ITEMS_PER_PAGE : min(
            st.session_state.current_page * ITEMS_PER_PAGE + ITEMS_PER_PAGE, len(image_tags)
        )
    ]
    n_rows = ceil(len(image_slice) / GRID_COLUMNS)
    for row in range(n_rows):
        for i, col in enumerate(st.columns(GRID_COLUMNS)):
            image_index = row * GRID_COLUMNS + i
            if image_index >= len(image_slice):
                continue
            image_tag = image_slice[image_index]
            image = get_image(image_tag)

            col.write(image_tag)
            col.caption(", ".join(_get_labels_for_image(image_tag, labels_by_image)))
            col.image(_rescale_image(image, resolution=200))
            col.button(
                "View", key=f"view_{image_tag}", on_click=_select_market_image, args=(image_tag,)
            )

    # paging buttons
    st.markdown("""---""")
    cols = st.columns(5)
    cols[2].text(
        f"Page: {st.session_state.current_page + 1} / {ceil(len(image_tags) / ITEMS_PER_PAGE)}"
    )

    # If I am not at the beginning
    if (
        image_tags
        and (st.session_state.current_page != 0)
        and (_ := cols[0].button("Previous Page"))
    ):
        st.session_state.current_page -= 1
        st.rerun()

    # If I am not in the end
    if image_tags and (len(image_slice) == ITEMS_PER_PAGE) and (_ := cols[-1].button("Next Page")):
        st.session_state.current_page += 1
        st.rerun()


def _load_labels_by_image() -> dict[str, list[str]]:
    """Load image labels from the merged labeled dataset."""
    with LABELS_DATASET_PATH.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        message = f"Expected a JSON object in {LABELS_DATASET_PATH}."
        raise TypeError(message)

    labels_by_image: dict[str, list[str]] = {}
    for image_tag, labels in payload.items():
        if not isinstance(image_tag, str):
            continue
        if not isinstance(labels, list):
            labels_by_image[image_tag] = []
            continue
        labels_by_image[image_tag] = [label for label in labels if isinstance(label, str)]
    return labels_by_image


def _get_label_options(labels_by_image: dict[str, list[str]]) -> list[str]:
    """Get filter options from existing dataset labels."""
    labels = sorted({label for labels in labels_by_image.values() for label in labels})
    return [*labels, NOT_LABELED]


def _filter_image_tags(
    image_tags: list[str], labels_by_image: dict[str, list[str]], selected_labels: list[str]
) -> list[str]:
    """Filter image tags by any selected label."""
    if not selected_labels:
        return image_tags

    selected_label_set = set(selected_labels)
    return [
        image_tag
        for image_tag in image_tags
        if selected_label_set.intersection(_get_labels_for_image(image_tag, labels_by_image))
    ]


def _get_labels_for_image(image_tag: str, labels_by_image: dict[str, list[str]]) -> list[str]:
    """Return image labels, grouping unlabeled images under a fallback label."""
    labels = labels_by_image.get(image_tag, [])
    if labels:
        return labels
    return [NOT_LABELED]


def _clamp_current_page(total_items: int) -> None:
    """Keep the selected page inside the current filtered result range."""
    last_page = max(ceil(total_items / ITEMS_PER_PAGE) - 1, 0)
    st.session_state.current_page = min(st.session_state.current_page, last_page)


def _get_selected_market_image(image_tags: list[str]) -> str | None:
    """Return the selected marketplace image when it is part of the current result set."""
    selected_image_tag = st.session_state.get("market_selected_image_tag")
    if selected_image_tag in image_tags:
        return selected_image_tag

    st.session_state.market_selected_image_tag = None
    return None


def _render_selected_image_detail(image_tag: str, labels_by_image: dict[str, list[str]]) -> None:
    """Render a larger marketplace image with its associated labels."""
    image = get_image(image_tag)
    labels = _get_labels_for_image(image_tag, labels_by_image)

    image_column, details_column = st.columns([1.35, 1], gap="large")
    with image_column:
        st.image(_rescale_image(image, resolution=520), width="stretch")
    with details_column:
        st.subheader("Name")
        st.write(image_tag)
        st.subheader("Category")
        st.markdown("\n".join(f"- {label}" for label in labels))


def _select_market_image(image_tag: str) -> None:
    """Store the marketplace image selected by a View button."""
    st.session_state.market_selected_image_tag = image_tag


def _clear_market_selection() -> None:
    """Clear the selected marketplace image."""
    st.session_state.market_selected_image_tag = None


def _rescale_image(im: Image.Image, resolution: int) -> Image.Image:
    """Rescale image to specified resolution."""
    size = (resolution, resolution)
    return ImageOps.pad(ImageOps.contain(im, size=size), size=size, color="white")


if __name__ == "__main__":
    main()
