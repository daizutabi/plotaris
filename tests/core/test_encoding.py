from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from plotaris.core.encoding import SIZES, Encoding, create_palette

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    data = {
        "a": ["A", "B", "C", "A", "B", "C"],
        "b": [1, 1, 1, 2, 2, 2],
        "value": [10, 20, 30, 15, 25, 35],
    }

    return pl.DataFrame(data)


def test_create_palette_list(df: pl.DataFrame) -> None:
    result = create_palette(
        df,
        ("a",),
        [1, 2, 3],
        [1, 2, 3],
    )
    expected = {("A",): 1, ("B",): 2, ("C",): 3}
    assert result == expected


def test_create_palette_dict(df: pl.DataFrame) -> None:
    result = create_palette(
        df,
        ("a",),
        {("A",): 10, ("B",): 20},
        [1, 2],
    )
    expected = {("A",): 10, ("B",): 20, ("C",): 1}
    assert result == expected


def test_encoding_get() -> None:
    enc = Encoding(x="a", y="b", color=("c",), size=("d",))
    assert enc.get("color") == ("c",)
    assert enc.get("size") == ("d",)
    assert enc.get("shape") == ()


def test_encoding_get_error() -> None:
    with pytest.raises(KeyError):
        enc = Encoding()
        enc.get("invalid")


def test_encoding_items() -> None:
    enc = Encoding(x="a", y="b", color=("c",), size=("d",))
    items = list(enc.items())
    assert items == [("color", ("c",)), ("size", ("d",))]


def test_encoding_palettes(mocker: MockerFixture) -> None:
    mock_create_palette = mocker.patch(
        "plotaris.core.encoding.create_palette",
        return_value="a",
    )

    enc = Encoding(x="a", y="b", color=("a",), size=("b",))
    palettes = enc.palettes("a", size=[10, 20])  # pyright: ignore[reportArgumentType]
    assert palettes == {"color": "a", "size": "a"}
    mock_create_palette.assert_called_with("a", ("b",), [10, 20], SIZES)
