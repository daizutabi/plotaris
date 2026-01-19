from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import polars as pl
import pytest

from plotaris.core.axisgrid import FacetGrid

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pytest_mock import MockerFixture

    from plotaris.core.axisgrid import FacetAxes


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [3, 3, 4, 4, 5, 5],
            "x": range(6),
        },
    )


@pytest.fixture(scope="module")
def grid(data: pl.DataFrame) -> FacetGrid:
    return FacetGrid(data, row="a", col="b")


def test_nrows_ncols(grid: FacetGrid) -> None:
    assert grid.nrows == 2
    assert grid.ncols == 3


def test_axes(grid: FacetGrid) -> None:
    assert len(grid.facet_axes) == 6
    assert (0, 0) in grid.facet_axes
    assert (0, 1) in grid.facet_axes
    assert (0, 2) in grid.facet_axes
    assert (1, 0) in grid.facet_axes
    assert (1, 1) in grid.facet_axes
    assert (1, 2) in grid.facet_axes


def test_get_axes(grid: FacetGrid) -> None:
    assert grid.facet_axes.get_axes(0, 0) is grid._axes[0, 0]  # pyright: ignore[reportPrivateUsage]


def test_get_axes_none(grid: FacetGrid) -> None:
    assert grid.facet_axes.get_axes(10, 10) is None


@pytest.mark.parametrize(
    ("name", "rcs"),
    [
        ("has_data", [(0, 0), (0, 1), (1, 1), (1, 2)]),
        ("is_left", [(0, 0), (1, 0)]),
        ("is_top", [(0, 0), (0, 1), (0, 2)]),
        ("is_right", [(0, 2), (1, 2)]),
        ("is_bottom", [(1, 0), (1, 1), (1, 2)]),
        ("is_leftmost", [(0, 0), (1, 1)]),
        ("is_topmost", [(0, 0), (0, 1), (1, 2)]),
        ("is_rightmost", [(0, 1), (1, 2)]),
        ("is_bottommost", [(0, 0), (1, 1), (1, 2)]),
    ],
)
def test_facet_axes_filter(
    grid: FacetGrid,
    name: str,
    rcs: list[tuple[int, int]],
) -> None:
    result = grid.facet_axes.filter(**{name: True}).axes  # pyright: ignore[reportArgumentType]
    expected = [grid.facet_axes[rc] for rc in rcs]
    assert result == expected


@pytest.fixture(scope="module")
def grid_delaxes(data: pl.DataFrame) -> FacetGrid:
    return FacetGrid(data, row="a", col="b").delaxes()


@pytest.mark.parametrize(
    ("name", "rcs"),
    [
        ("has_data", [(0, 0), (0, 1), (1, 1), (1, 2)]),
        ("is_left", [(0, 0)]),
        ("is_top", [(0, 0), (0, 1)]),
        ("is_right", [(1, 2)]),
        ("is_bottom", [(1, 1), (1, 2)]),
        ("is_leftmost", [(0, 0), (1, 1)]),
        ("is_topmost", [(0, 0), (0, 1), (1, 2)]),
        ("is_rightmost", [(0, 1), (1, 2)]),
        ("is_bottommost", [(0, 0), (1, 1), (1, 2)]),
    ],
)
def test_axes_property_after_delaxes(
    grid_delaxes: FacetGrid,
    name: str,
    rcs: list[tuple[int, int]],
) -> None:
    result = grid_delaxes.facet_axes.filter(**{name: True}).axes  # pyright: ignore[reportArgumentType]
    expected = [grid_delaxes.facet_axes[rc] for rc in rcs]
    assert result == expected


@pytest.mark.parametrize(
    ("kwargs", "n"),
    [
        ({"has_data": True}, 4),
        ({"is_topmost": True}, 3),
        ({"is_topmost": True, "row": 0}, 2),
        ({"is_topmost": True, "row": 1}, 1),
    ],
)
def test_select(data: pl.DataFrame, kwargs: dict[str, Any], n: int) -> None:
    grid = FacetGrid(data, row="a", col="b")
    assert len(grid.select(**kwargs).axes) == n


def test_map(grid: FacetGrid) -> None:
    axes: list[Axes] = []

    def func(facet_axes: FacetAxes) -> None:  # pyright: ignore[reportUnusedParameter]
        axes.append(plt.gca())

    grid.map(func)

    assert axes == grid.axes


def test_map_axes(grid: FacetGrid) -> None:
    axes: list[Axes] = []

    def func(ax: Axes, x: int, *, y: int) -> None:
        axes.append(ax)
        assert x == 1
        assert y == 2

    grid.map_axes(func, 1, y=2)

    assert axes[0] == grid.facet_axes[0, 0]
    assert axes[-1] == grid.facet_axes[1, 2]


def test_map_dataframe(mocker: MockerFixture) -> None:
    df = pl.DataFrame(
        {
            "category": ["A", "A", "B", "B"],
            "x": [1, 2, 3, 4],
            "y": [5, 6, 7, 8],
        },
    )

    grid = FacetGrid(df, row="category")
    assert grid.nrows == 2
    assert grid.ncols == 1
    assert (0, 0) in grid.facet_axes
    assert (1, 0) in grid.facet_axes

    def plot(data: pl.DataFrame, ms: int, *, color: str) -> None:
        ax = plt.gca()
        ax.scatter(data["x"], data["y"], ms=ms, color=color)  # pyright: ignore[reportUnknownMemberType]

    mock_scatter = mocker.patch("matplotlib.axes.Axes.scatter")
    grid.map_dataframe(plot, 5, color="red")

    assert mock_scatter.call_count == 2
    calls = mock_scatter.call_args_list

    (x, y), kwargs = calls[0]
    assert isinstance(x, pl.Series)
    assert isinstance(y, pl.Series)
    assert x.to_list() == [1, 2]
    assert y.to_list() == [5, 6]
    assert kwargs["ms"] == 5
    assert kwargs["color"] == "red"

    (x, y), kwargs = calls[1]
    assert isinstance(x, pl.Series)
    assert isinstance(y, pl.Series)
    assert x.to_list() == [3, 4]
    assert y.to_list() == [7, 8]
    assert kwargs["ms"] == 5
    assert kwargs["color"] == "red"


def test_set(grid: FacetGrid, mocker: MockerFixture) -> None:
    mock_set = mocker.patch("matplotlib.axes.Axes.set")
    grid.set(x=1, y=2)
    assert mock_set.call_count == 6
    calls = mock_set.call_args_list
    _, kwargs = calls[0]
    assert kwargs["x"] == 1
    assert kwargs["y"] == 2


def test_display(grid: FacetGrid) -> None:
    assert grid._display_() is grid.figure  # pyright: ignore[reportPrivateUsage]
    assert grid.facet_axes._display_() is grid.figure  # pyright: ignore[reportPrivateUsage]
    assert next(iter(grid.facet_axes))._display_() is grid._axes[0, 0]  # pyright: ignore[reportPrivateUsage]
    assert grid.facet_axes.filter(row=10)._display_() is None  # pyright: ignore[reportPrivateUsage]
