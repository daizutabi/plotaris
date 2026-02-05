from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import polars as pl
import pytest

from plotaris.core.axisgrid import FacetAxes, FacetGrid

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from pytest_mock import MockerFixture


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


def test_axes(grid: FacetGrid) -> None:
    assert len(grid.facet_axes) == 6
    assert (0, 0) in grid.facet_axes
    assert (0, 1) in grid.facet_axes
    assert (0, 2) in grid.facet_axes
    assert (1, 0) in grid.facet_axes
    assert (1, 1) in grid.facet_axes
    assert (1, 2) in grid.facet_axes


def test_get_axes(grid: FacetGrid) -> None:
    assert grid.facet_axes.get_axes(0, 0) is grid.figure.axes[0]


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
    result = grid.facet_axes.filter(predicate=None, **{name: True}).axes
    expected = list(starmap(grid.facet_axes.get_axes, rcs))
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
    result = grid_delaxes.facet_axes.filter(predicate=None, **{name: True}).axes
    expected = list(starmap(grid_delaxes.facet_axes.get_axes, rcs))
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


def test_all(data: pl.DataFrame) -> None:
    grid = FacetGrid(data, row="a", col="b")
    assert len(grid.axes) == 6
    grid.select(has_data=True)
    assert len(grid.axes) == 4
    grid.all()
    assert len(grid.axes) == 6


def test_map(grid: FacetGrid) -> None:
    axes: list[Axes] = []

    def func(facet_axes: FacetAxes) -> None:
        assert isinstance(facet_axes, FacetAxes)
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

    assert axes[0] == grid.facet_axes.get_axes(0, 0)
    assert axes[-1] == grid.facet_axes.get_axes(1, 2)


def test_map_dataframe(mocker: MockerFixture) -> None:
    df = pl.DataFrame(
        {
            "category": ["A", "A", "B", "B"],
            "x": [1, 2, 3, 4],
            "y": [5, 6, 7, 8],
        },
    )

    grid = FacetGrid(df, row="category")
    assert grid.facet_data.nrows == 2
    assert grid.facet_data.ncols == 1
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


def test_set_titles_margin_titles_true(data: pl.DataFrame) -> None:
    grid = FacetGrid(data, row="a", col="b")
    grid.set_titles({"b": "{:.1f}"}, a="A(m)", margin_titles=True)
    assert grid.facet_axes[0, 0].axes.get_title() == "b=3.0"
    assert grid.facet_axes[0, 1].axes.get_title() == "b=4.0"
    assert grid.facet_axes[0, 2].axes.get_title() == "b=5.0"
    assert grid.facet_axes[1, 0].axes.get_title() == ""
    assert grid.facet_axes[1, 1].axes.get_title() == ""
    assert grid.facet_axes[1, 2].axes.get_title() == ""
    assert len(grid.figure.axes) == 8
    assert grid.figure.axes[-2].get_ylabel() == "A=1m"
    assert grid.figure.axes[-1].get_ylabel() == "A=2m"


def test_set_titles_margin_titles_false(data: pl.DataFrame) -> None:
    grid = FacetGrid(data, row="a", col="b")
    grid.set_titles({"b": "{:.1f}"}, a="A(m)", margin_titles=False)
    assert grid.facet_axes[0, 0].axes.get_title() == "A=1m, b=3.0"
    assert grid.facet_axes[0, 1].axes.get_title() == "A=1m, b=4.0"
    assert grid.facet_axes[0, 2].axes.get_title() == ""
    assert grid.facet_axes[1, 0].axes.get_title() == ""
    assert grid.facet_axes[1, 1].axes.get_title() == "A=2m, b=4.0"
    assert grid.facet_axes[1, 2].axes.get_title() == "A=2m, b=5.0"


def test_set_titles_margin_titles_true_delaxes(data: pl.DataFrame) -> None:
    grid = FacetGrid(data, row="a", col="b").delaxes()
    grid.set_titles({"b": "{:.1f}"}, a="A(m)", margin_titles=True)
    assert grid.facet_axes[0, 0].axes.get_title() == "b=3.0"
    assert grid.facet_axes[0, 1].axes.get_title() == "b=4.0"
    assert grid.facet_axes[1, 2].axes.get_title() == "b=5.0"
    assert len(grid.figure.axes) == 6
    assert grid.figure.axes[-2].get_ylabel() == "A=1m"
    assert grid.figure.axes[-1].get_ylabel() == "A=2m"


def test_set_titles_margin_titles_false_delaxes(data: pl.DataFrame) -> None:
    grid = FacetGrid(data, row="a", col="b").delaxes()
    grid.set_titles({"b": "{:.1f}"}, a="A(m)", margin_titles=False)
    assert grid.facet_axes[0, 0].axes.get_title() == "A=1m, b=3.0"
    assert grid.facet_axes[0, 1].axes.get_title() == "A=1m, b=4.0"
    assert grid.facet_axes[1, 1].axes.get_title() == "A=2m, b=4.0"
    assert grid.facet_axes[1, 2].axes.get_title() == "A=2m, b=5.0"


def test_display(grid: FacetGrid) -> None:
    assert grid._display_() is grid.figure  # pyright: ignore[reportPrivateUsage]
