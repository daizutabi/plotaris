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


def test_encoding_items() -> None:
    enc = Encoding(color=("c",), size=("d",))
    items = list(enc.items())
    assert items == [("color", ("c",)), ("size", ("d",))]


def test_encoding_build_palettes_mock(mocker: MockerFixture) -> None:
    mock_create_palette = mocker.patch(
        "plotaris.core.encoding.create_palette",
        return_value="a",
    )

    enc = Encoding(color=("a",), size=("b",))
    palettes = enc.build_palettes(pl.DataFrame(), size=[10, 20])
    assert palettes == {"color": "a", "size": "a"}
    mock_create_palette.assert_called_with(mocker.ANY, ("b",), [10, 20], SIZES)


def test_encoding_build_palettes(df: pl.DataFrame) -> None:
    enc = Encoding(size=("a",))
    palettes = enc.build_palettes(df)
    assert palettes == {"size": {("A",): 50, ("B",): 100, ("C",): 150}}


@pytest.mark.parametrize(
    ("index", "size", "shape"),
    [
        (0, 50, "o"),
        (1, 100, "o"),
        (2, 150, "o"),
        (3, 200, "s"),
        (4, 250, "s"),
        (5, 50, "s"),
    ],
)
def test_encoding_get_properties(
    df: pl.DataFrame,
    index: int,
    size: int,
    shape: str,
) -> None:
    enc = Encoding(size=("a", "b"), shape=("b",))
    palettes = enc.build_palettes(df)
    x = df.row(index, named=True)
    assert enc.get_properties(x, palettes) == {"size": size, "shape": shape}
