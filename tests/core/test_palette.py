from __future__ import annotations

import polars as pl
import pytest

from plotaris.core.palette import Base, Palette, create_palette


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    data = {
        "a": ["A", "B", "C", "A", "B", "C"],
        "b": [1, 1, 1, 2, 2, 2],
        "value": [10, 20, 10, 20, 10, 30],
    }

    return pl.DataFrame(data)


def test_create_palette(data: pl.DataFrame) -> None:
    mapping = {("A",): 1, ("B",): 2}
    result = create_palette(data, ("a",), [10, 20, 30], mapping)
    assert result == {("A",): 1, ("B",): 2, ("C",): 30}


def test_create_palette_empty(data: pl.DataFrame) -> None:
    result = create_palette(data, ("a",))
    assert result == {("A",): None, ("B",): None, ("C",): None}


def test_create_palette_mapping_empty(data: pl.DataFrame) -> None:
    result = create_palette(data, ("a",), [1, 2])
    assert result == {("A",): 1, ("B",): 2, ("C",): 1}


def test_create_palette_default_empty(data: pl.DataFrame) -> None:
    mapping = {("A",): 10, ("B",): 20}
    result = create_palette(data, ("a",), mapping=mapping)
    assert result == {("A",): 10, ("B",): 20, ("C",): None}


@pytest.fixture
def base() -> Base:
    return Base(color="a", size=("a", "b"), shape="value", invalid=None)


@pytest.mark.parametrize(
    ("index", "value"),
    [(0, 100), (1, 2), (2, 3), (3, 200), (4, 2), (5, 3)],
)
def test_base_mapping_default(
    base: Base,
    data: pl.DataFrame,
    index: int,
    value: int,
) -> None:
    result = (
        base.mapping(size={("A", 1): 100, ("A", 2): 200})
        .default(size=[1, 2, 3])
        .set(data)
        .get(data.row(index, named=True))
    )
    assert result == {"color": None, "size": value, "shape": None}


@pytest.mark.parametrize(
    ("index", "color", "shape"),
    [(0, 1, "o"), (1, 2, "s"), (2, 1, "o"), (3, 1, "s"), (4, 2, "o"), (5, 1, "^")],
)
def test_base_default(
    base: Base,
    data: pl.DataFrame,
    index: int,
    color: int,
    shape: str,
) -> None:
    result = (
        base.default(color=[1, 2], shape=["o", "s", "^"])
        .set(data)
        .get(data.row(index, named=True))
    )
    assert result == {"color": color, "size": None, "shape": shape}


@pytest.mark.parametrize(
    ("index", "color", "shape"),
    [
        (0, 1, "o"),
        (1, 2, "s"),
        (2, None, "o"),
        (3, 1, "s"),
        (4, 2, "o"),
        (5, None, None),
    ],
)
def test_base_mapping(
    base: Base,
    data: pl.DataFrame,
    index: int,
    color: int | None,
    shape: str | None,
) -> None:
    result = (
        base.mapping(color={"A": 1, "B": 2}, shape={(10,): "o", (20,): "s"})
        .set(data)
        .get(data.row(index, named=True))
    )
    assert result == {"color": color, "size": None, "shape": shape}


@pytest.mark.parametrize(
    ("start", "stop", "color"),
    [(0, 0, None), (0, 1, 10), (0, 3, 10), (0, 4, None), (3, 4, 20)],
)
def test_base_get_by_dataframe(
    base: Base,
    data: pl.DataFrame,
    start: int,
    stop: int,
    color: int | None,
) -> None:
    base = Base(color="b").default(color=[10, 20]).set(data)
    assert base.get(data[start:stop]) == {"color": color}


def test_palette(data: pl.DataFrame) -> None:
    palette = Palette("a", ("a", "b"))
    assert palette.columns == {"color": ("a",), "size": ("a", "b")}
    result = palette.set(data).get(data.row(0, named=True))
    assert result == {"color": "#1f77b4", "size": 50}
