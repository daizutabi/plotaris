from __future__ import annotations

from typing import Literal

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from plotaris.matplotlib.facet import Facet, FacetData, FacetLabel
from plotaris.matplotlib.label import Label


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        (None, "a=1, b=x, c=10, d=y"),
        ("row", "a=1, b=x"),
        ("col", "c=10, d=y"),
    ],
)
def test_facet_label_format_dim(
    dim: Literal["row", "col"] | None,
    expected: str,
) -> None:
    label = FacetLabel(Label({"a": 1, "b": "x"}), Label({"c": 10, "d": "y"}), dim=dim)
    assert label.format() == expected


def test_facet_label_format_format() -> None:
    label = FacetLabel(Label({"a": 1e5}), Label({"b": 2.1234e-5}), sep=";", eq="~")
    assert label.format({"b": ("B", "_{:.2g}_")}, a="A(V)") == "A~100kV;B~_2.1e-05_"


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [3, 3, 4, 4, 5, 5],
            "x": range(6),
        },
    )


def test_row_col_wrap_row_col(data: pl.DataFrame) -> None:
    facet_data = FacetData(data, row="a", col="b")
    assert facet_data.row == ("a",)
    assert facet_data.col == ("b",)
    assert facet_data.wrap is None


def test_row_col_wrap_row(data: pl.DataFrame) -> None:
    facet_data = FacetData(data, row="a", wrap=2)
    assert facet_data.row == ("a",)
    assert facet_data.col == ()
    assert facet_data.wrap == 2


def test_row_col_wrap_col(data: pl.DataFrame) -> None:
    facet_data = FacetData(data, col=("a", "b"), wrap=2)
    assert facet_data.row == ()
    assert facet_data.col == ("a", "b")
    assert facet_data.wrap == 2


def test_row_col_wrap_none(data: pl.DataFrame) -> None:
    facet_data = FacetData(data)
    assert facet_data.row == ()
    assert facet_data.col == ()
    assert facet_data.wrap is None


def test_index_row_col(data: pl.DataFrame) -> None:
    facet_data = FacetData(data, row="a", col="b")
    assert facet_data.index(0, 0) == 0
    assert facet_data.index(0, 1) == 1
    assert facet_data.index(0, 2) is None
    assert facet_data.index(1, 0) is None
    assert facet_data.index(1, 1) == 2
    assert facet_data.index(1, 2) == 3


def test_index_row(data: pl.DataFrame) -> None:
    facet_data = FacetData(data, row="a")
    assert facet_data.index(0, 0) == 0
    assert facet_data.index(1, 0) == 1
    assert facet_data.index(2, 0) is None


def test_index_col(data: pl.DataFrame) -> None:
    facet_data = FacetData(data, col="b")
    assert facet_data.index(0, 0) == 0
    assert facet_data.index(0, 1) == 1
    assert facet_data.index(0, 2) == 2
    assert facet_data.index(0, 3) is None


def test_index_none(data: pl.DataFrame) -> None:
    facet_data = FacetData(data)
    assert facet_data.index(0, 0) == 0
    assert facet_data.index(0, 1) is None


def test_index_row_wrap(data: pl.DataFrame) -> None:
    facet_data = FacetData(data, row="x", wrap=4)
    assert facet_data.index(0, 0) == 0
    assert facet_data.index(1, 0) == 1
    assert facet_data.index(2, 0) == 2
    assert facet_data.index(3, 0) == 3
    assert facet_data.index(0, 1) == 4
    assert facet_data.index(1, 1) == 5


def test_index_col_wrap(data: pl.DataFrame) -> None:
    facet_data = FacetData(data, col="x", wrap=3)
    assert facet_data.index(0, 0) == 0
    assert facet_data.index(0, 1) == 1
    assert facet_data.index(0, 2) == 2
    assert facet_data.index(1, 0) == 3
    assert facet_data.index(1, 1) == 4
    assert facet_data.index(1, 2) == 5


@pytest.mark.parametrize(
    ("row", "col", "wrap", "nrows", "ncols"),
    [
        ("a", "b", None, 2, 3),
        ("b", "a", None, 3, 2),
        ("x", None, None, 6, 1),
        (None, "x", None, 1, 6),
        ("x", None, 3, 3, 2),
        (None, "x", 3, 2, 3),
        (["a", "b"], None, None, 4, 1),
        (None, ["a", "b"], 3, 2, 3),
    ],
)
def test_nrows_ncols(
    data: pl.DataFrame,
    row: str | list[str] | None,
    col: str | list[str] | None,
    wrap: int | None,
    nrows: int,
    ncols: int,
) -> None:
    facet_data = FacetData(data, row=row, col=col, wrap=wrap)
    assert facet_data.nrows == nrows
    assert facet_data.ncols == ncols


@pytest.fixture(scope="module")
def facet_data_2x3(data: pl.DataFrame) -> FacetData:
    return FacetData(data, row="a", col="b")


@pytest.fixture(scope="module")
def facet_data_wrapped(data: pl.DataFrame) -> FacetData:
    return FacetData(data, row="x", wrap=4)


def test_coordinates(facet_data_2x3: FacetData) -> None:
    expected_cells = {(0, 0), (0, 1), (1, 1), (1, 2)}
    assert set(facet_data_2x3.coordinates()) == expected_cells


def test_data_access(facet_data_2x3: FacetData, data: pl.DataFrame) -> None:
    df = facet_data_2x3.data(0, 0)
    assert df is not None
    assert_frame_equal(df, data[0:2])
    assert facet_data_2x3.data(0, 2) is None
    assert facet_data_2x3[0, 1].has_data
    assert facet_data_2x3[0, 2].has_data is False


def test_iteration(facet_data_2x3: FacetData) -> None:
    all_facets = list(facet_data_2x3)
    assert len(all_facets) == 6
    assert isinstance(all_facets[0], Facet)
    row, col = all_facets[0]
    assert row == 0
    assert col == 0


@pytest.mark.parametrize(
    ("r", "c", "is_left", "is_top", "is_right", "is_bottom"),
    [
        (0, 0, True, True, False, False),
        (1, 0, True, False, False, False),
        (2, 0, True, False, False, False),
        (3, 0, True, False, False, True),
        (0, 1, False, True, True, False),
        (1, 1, False, False, True, False),
        (2, 1, False, False, True, False),
        (3, 1, False, False, True, True),
    ],
)
def test_facet_grid_boundary_attrs(
    facet_data_wrapped: FacetData,
    r: int,
    c: int,
    is_left: bool,
    is_top: bool,
    is_right: bool,
    is_bottom: bool,
) -> None:
    facet = facet_data_wrapped[r, c]
    assert facet.is_left is is_left
    assert facet.is_top is is_top
    assert facet.is_right is is_right
    assert facet.is_bottom is is_bottom


@pytest.mark.parametrize(
    ("r", "c", "is_leftmost", "is_topmost", "is_rightmost", "is_bottommost"),
    [
        (0, 0, True, True, False, True),
        (0, 1, False, True, True, False),
        (0, 2, False, False, False, False),
        (1, 0, False, False, False, False),
        (1, 1, True, False, False, True),
        (1, 2, False, True, True, True),
    ],
)
def test_facet_data_boundary_attrs(
    facet_data_2x3: FacetData,
    r: int,
    c: int,
    is_leftmost: bool,
    is_topmost: bool,
    is_rightmost: bool,
    is_bottommost: bool,
) -> None:
    facet = facet_data_2x3[r, c]
    assert facet.is_leftmost is is_leftmost
    assert facet.is_topmost is is_topmost
    assert facet.is_rightmost is is_rightmost
    assert facet.is_bottommost is is_bottommost


def test_facet_label(facet_data_2x3: FacetData) -> None:
    facet_0_0 = facet_data_2x3[0, 0]
    assert facet_0_0.label.row.data == {"a": 1}
    assert facet_0_0.label.col.data == {"b": 3}

    facet_0_2 = facet_data_2x3[0, 2]
    assert facet_0_2.label.row.data == {}
    assert facet_0_2.label.col.data == {}

    facet_data_row_only = FacetData(facet_data_2x3.group.data[0], row="a")
    facet = facet_data_row_only[0, 0]
    assert facet.label.row.data == {"a": 1}
    assert facet.label.col.data == {}


def test_facet_collection_filter_has_data(facet_data_2x3: FacetData) -> None:
    facets = facet_data_2x3.facets
    assert len(facets.filter(has_data=True)) == 4
    assert len(facets.filter(has_data=False)) == 2


def test_facet_collection_filter_by_row_col(facet_data_2x3: FacetData) -> None:
    facets = facet_data_2x3.facets
    assert len(facets.filter(row=0)) == 3
    assert len(facets.filter(col=1)) == 2
    assert len(facets.filter(row=1, col=2)) == 1


def test_facet_collection_filter_by_boundary(facet_data_2x3: FacetData) -> None:
    facets = facet_data_2x3.facets
    assert len(facets.filter(is_left=True)) == 2
    assert len(facets.filter(is_top=True)) == 3
    assert len(facets.filter(is_right=True)) == 2
    assert len(facets.filter(is_bottom=True)) == 3
    assert len(facets.filter(is_leftmost=True)) == 2
    assert len(facets.filter(is_topmost=True)) == 3
    assert len(facets.filter(is_rightmost=True)) == 2
    assert len(facets.filter(is_bottommost=True)) == 3


def test_facet_collection_filter_by_predicate(facet_data_2x3: FacetData) -> None:
    facets = facet_data_2x3.facets

    def predicate(facet: Facet) -> bool:
        return facet.row == 1 and facet.has_data

    assert len(facets.filter(predicate)) == 2


def test_facet_collection_filter_chaining(facet_data_2x3: FacetData) -> None:
    facets = facet_data_2x3.facets
    result = facets.filter(is_bottom=True).filter(has_data=True)
    assert len(result) == 2
    for facet in result:
        assert facet.row == 1
        assert facet.has_data


def test_facet_collection_access(facet_data_2x3: FacetData) -> None:
    facets = facet_data_2x3.facets
    assert (0, 1) in facets
    assert (5, 5) not in facets
    assert facets[0, 1].row == 0
    assert facets[0, 1].col == 1
    assert facets.get(1, 1) is not None
    assert facets.get(9, 9) is None
