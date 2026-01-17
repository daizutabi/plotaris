from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from plotaris.core.axisgrid import FacetGrid

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pytest_mock import MockerFixture

    from plotaris.core.data import Facet


@pytest.fixture(scope="module")
def grid() -> FacetGrid:
    data = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [3, 3, 4, 4, 5, 5],
            "x": range(6),
        },
    )
    return FacetGrid(data, row="a", col="b")


def test_nrows_ncols(grid: FacetGrid) -> None:
    assert grid.nrows == 2
    assert grid.ncols == 3


def test_axes(grid: FacetGrid) -> None:
    assert len(grid.axes) == 4
    assert (0, 0) in grid.axes
    assert (0, 1) in grid.axes
    assert (1, 1) in grid.axes
    assert (1, 2) in grid.axes


def test_display(grid: FacetGrid) -> None:
    assert grid._display_() is grid.figure  # pyright: ignore[reportPrivateUsage]


def test_iter(grid: FacetGrid) -> None:
    assert list(grid) == list(grid.axes.values())


def test_items(grid: FacetGrid) -> None:
    axes = [a for a, _ in grid.items()]
    assert axes == list(grid.axes.values())


def test_map_facet(grid: FacetGrid) -> None:
    axes: list[Axes] = []

    def func(facet: Facet, *, ax: Axes) -> None:  # pyright: ignore[reportUnusedParameter]
        axes.append(ax)

    grid.map_facet(func)

    assert axes == list(grid.axes.values())


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
    assert (0, 0) in grid.axes
    assert (1, 0) in grid.axes

    def plot(data: pl.DataFrame, ms: int, *, ax: Axes, color: str) -> None:
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
