"""Test minifigures-app."""

import minifigures_app


def test_import() -> None:
    """Test that the app can be imported."""
    assert isinstance(minifigures_app.__name__, str)
