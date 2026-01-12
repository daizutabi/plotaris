from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from plotaris.core.data import FacetData, GroupedData, group_by


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [3, 3, 4, 4, 5, 5],
            "x": range(6),
        },
    )


def test_mapping_str_str(data: pl.DataFrame) -> None:
    result = GroupedData(data, {"row": "a", "col": "b"})

    expected = pl.DataFrame({"row": [0, 0, 1, 1], "col": [0, 1, 1, 2]})
    assert_frame_equal(result.index, expected, check_dtypes=False)

    assert len(result) == 4
    assert result.item(0, "row") == (1,)
    assert result.item(1, "row") == (1,)
    assert result.item(2, "row") == (2,)
    assert result.item(3, "row") == (2,)
    assert result.item(3, "row", named=True) == {"a": 2}
    assert result.item(0, "col") == (3,)
    assert result.item(1, "col") == (4,)
    assert result.item(2, "col") == (4,)
    assert result.item(3, "col") == (5,)
    assert result.item(3, "col", named=True) == {"b": 5}
    assert result.get_label(0) == {"row": (1,), "col": (3,)}
    assert result.get_label(1) == {"row": (1,), "col": (4,)}
    assert result.get_label(2) == {"row": (2,), "col": (4,)}
    assert result.get_label(3) == {"row": (2,), "col": (5,)}
    assert result.get_label(0, named=True) == {"row": {"a": 1}, "col": {"b": 3}}
    assert result.get_label(1, named=True) == {"row": {"a": 1}, "col": {"b": 4}}
    assert result.get_label(2, named=True) == {"row": {"a": 2}, "col": {"b": 4}}
    assert result.get_label(3, named=True) == {"row": {"a": 2}, "col": {"b": 5}}
    assert result.get_labels() == [
        {"row": (1,), "col": (3,)},
        {"row": (1,), "col": (4,)},
        {"row": (2,), "col": (4,)},
        {"row": (2,), "col": (5,)},
    ]
    assert result.get_labels(named=True) == [
        {"row": {"a": 1}, "col": {"b": 3}},
        {"row": {"a": 1}, "col": {"b": 4}},
        {"row": {"a": 2}, "col": {"b": 4}},
        {"row": {"a": 2}, "col": {"b": 5}},
    ]
    assert result.n_unique("row") == 2
    assert result.n_unique("col") == 3


def test_mapping_str_str_duplicated(data: pl.DataFrame) -> None:
    result = GroupedData(data, {"row": "b", "col": "b"})

    expected = pl.DataFrame({"row": [0, 1, 2], "col": [0, 1, 2]})
    assert_frame_equal(result.index, expected, check_dtypes=False)

    assert len(result) == 3
    assert result.item(0, "row") == (3,)
    assert result.item(1, "row") == (4,)
    assert result.item(2, "row") == (5,)
    assert result.item(2, "row", named=True) == {"b": 5}
    assert result.item(0, "col") == (3,)
    assert result.item(1, "col") == (4,)
    assert result.item(2, "col") == (5,)
    assert result.item(2, "col", named=True) == {"b": 5}
    assert result.n_unique("row") == 3
    assert result.n_unique("col") == 3


@pytest.mark.parametrize(("name", "values"), [("a", [1, 2]), ("b", [3, 4, 5])])
def test_mapping_str(data: pl.DataFrame, name: str, values: list[int]) -> None:
    result = GroupedData(data, {"row": name})

    n = len(values)
    expected = pl.DataFrame({"row": range(n)})
    assert_frame_equal(result.index, expected, check_dtypes=False)

    assert len(result) == n
    assert result.n_unique("row") == n
    assert result.n_unique("col") == 0


def test_mapping_iterable(data: pl.DataFrame) -> None:
    result = GroupedData(data, {"row": ("a", "b")})

    expected = pl.DataFrame({"row": [0, 1, 2, 3]})
    assert_frame_equal(result.index, expected, check_dtypes=False)

    assert len(result) == 4
    assert result.item(0, "row") == (1, 3)
    assert result.item(1, "row") == (1, 4)
    assert result.item(2, "row") == (2, 4)
    assert result.item(3, "row") == (2, 5)
    assert result.item(3, "row", named=True) == {"a": 2, "b": 5}
    assert result.n_unique("row") == 4
    assert result.n_unique("col") == 0


def test_mapping_iterable_str(data: pl.DataFrame) -> None:
    result = GroupedData(data, {"row": ("b", "a"), "col": "a"})

    expected = pl.DataFrame({"row": [0, 1, 2, 3], "col": [0, 0, 1, 1]})
    assert_frame_equal(result.index, expected, check_dtypes=False)

    assert len(result) == 4
    assert result.item(0, "row") == (3, 1)
    assert result.item(1, "row") == (4, 1)
    assert result.item(2, "row") == (4, 2)
    assert result.item(3, "row") == (5, 2)
    assert result.item(3, "row", named=True) == {"b": 5, "a": 2}
    assert result.item(0, "col") == (1,)
    assert result.item(1, "col") == (1,)
    assert result.item(2, "col") == (2,)
    assert result.item(3, "col") == (2,)
    assert result.item(3, "col", named=True) == {"a": 2}
    assert result.n_unique("row") == 4
    assert result.n_unique("col") == 2


@pytest.mark.parametrize("mapping", [{}, {"row": ()}])
def test_mapping_empty(
    data: pl.DataFrame,
    mapping: dict[str, str | tuple[str, ...]],
) -> None:
    result = GroupedData(data, mapping)

    expected = pl.DataFrame([{}])

    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert len(result) == 1
    assert_frame_equal(result.data[0], data)


def test_mapping_str_empty(data: pl.DataFrame) -> None:
    result = GroupedData(data, {"row": "a", "col": ()})

    expected = pl.DataFrame({"row": [0, 1], "col": [0, 0]})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert len(result) == 2
    assert result.item(0, "row") == (1,)
    assert result.item(1, "row") == (2,)
    assert result.item(0, "col") == ()
    assert result.item(1, "col") == ()
    assert result.item(1, "col", named=True) == {}


def test_data_empty() -> None:
    result = GroupedData(pl.DataFrame(), {"row": ["a"]})
    expected = pl.DataFrame({"row": []})
    assert_frame_equal(result.index, expected, check_dtypes=False)


def test_group_by_no_data() -> None:
    group, dfs = group_by(pl.DataFrame({"x": []}), "x")
    assert_frame_equal(group, pl.DataFrame({"x": []}))
    assert len(dfs) == 0


def test_facet_data_empty() -> None:
    result = FacetData(pl.DataFrame(), ("a",), ("b",))
    expected = pl.DataFrame({"row": [], "col": []})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.nrows == 0
    assert result.ncols == 0
    assert result.get_labels() == []


def test_facet_row_col(data: pl.DataFrame) -> None:
    result = FacetData(data, row="a", col="b")
    expected = pl.DataFrame({"row": [0, 0, 1, 1], "col": [0, 1, 1, 2]})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.nrows == 2
    assert result.ncols == 3
    assert result.get_labels(named=True) == [
        {"row": {"a": 1}, "col": {"b": 3}},
        {"row": {"a": 1}, "col": {"b": 4}},
        {"row": {"a": 2}, "col": {"b": 4}},
        {"row": {"a": 2}, "col": {"b": 5}},
    ]
    assert result.cells() == [(0, 0), (0, 1), (1, 1), (1, 2)]
    assert result.cells(empty=True) == [(0, 2), (1, 0)]


def test_facet_row_empty(data: pl.DataFrame) -> None:
    result = FacetData(data, col="a")
    expected = pl.DataFrame({"row": [0, 0], "col": [0, 1]})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.nrows == 1
    assert result.ncols == 2
    assert result.get_labels() == [
        {"row": (), "col": (1,)},
        {"row": (), "col": (2,)},
    ]
    assert result.cells() == [(0, 0), (0, 1)]
    assert result.cells(empty=True) == []


def test_facet_col_empty(data: pl.DataFrame) -> None:
    result = FacetData(data, row="b")
    expected = pl.DataFrame({"row": [0, 1, 2], "col": [0, 0, 0]})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.nrows == 3
    assert result.ncols == 1
    assert result.get_labels(named=True) == [
        {"row": {"b": 3}, "col": {}},
        {"row": {"b": 4}, "col": {}},
        {"row": {"b": 5}, "col": {}},
    ]
    assert result.cells() == [(0, 0), (1, 0), (2, 0)]
    assert result.cells(empty=True) == []


def test_facet_row_col_empty(data: pl.DataFrame) -> None:
    result = FacetData(data)
    expected = pl.DataFrame({"row": [0], "col": [0]})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.nrows == 1
    assert result.ncols == 1
    assert result.get_labels() == [{"row": (), "col": ()}]
    assert result.cells() == [(0, 0)]
    assert result.cells(empty=True) == []


def test_facet_row_wrap(data: pl.DataFrame) -> None:
    result = FacetData(data, row="x", wrap=2)
    expected = pl.DataFrame({"row": [0, 1, 0, 1, 0, 1], "col": [0, 0, 1, 1, 2, 2]})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.nrows == 2
    assert result.ncols == 3
    assert result.get_labels(named=True) == [
        {"row": {"x": 0}, "col": {}},
        {"row": {"x": 1}, "col": {}},
        {"row": {"x": 2}, "col": {}},
        {"row": {"x": 3}, "col": {}},
        {"row": {"x": 4}, "col": {}},
        {"row": {"x": 5}, "col": {}},
    ]
    assert result.cells() == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    assert result.cells(empty=True) == []


def test_facet_col_wrap(data: pl.DataFrame) -> None:
    result = FacetData(data, col="x", wrap=4)
    expected = pl.DataFrame({"row": [0, 0, 0, 0, 1, 1], "col": [0, 1, 2, 3, 0, 1]})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.nrows == 2
    assert result.ncols == 4
    assert result.get_labels(named=True) == [
        {"row": {}, "col": {"x": 0}},
        {"row": {}, "col": {"x": 1}},
        {"row": {}, "col": {"x": 2}},
        {"row": {}, "col": {"x": 3}},
        {"row": {}, "col": {"x": 4}},
        {"row": {}, "col": {"x": 5}},
    ]
    assert result.cells() == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
    assert result.cells(empty=True) == [(1, 2), (1, 3)]


def test_facet_get(data: pl.DataFrame) -> None:
    result = FacetData(data, row="a", col="b")

    expected = pl.DataFrame({"a": [1, 1], "b": [3, 3], "x": [0, 1]})
    df = result.get(0, 0)
    assert isinstance(df, pl.DataFrame)
    assert_frame_equal(df, expected, check_dtypes=False)

    expected = pl.DataFrame({"a": [2, 2], "b": [5, 5], "x": [4, 5]})
    df = result.get(1, 2)
    assert isinstance(df, pl.DataFrame)
    assert_frame_equal(df, expected, check_dtypes=False)

    assert result.get(0, 2) is None


@pytest.mark.parametrize(
    ("row", "col", "expected"),
    [
        (0, 0, True),
        (0, 1, False),
        (0, 2, False),
        (1, 0, False),
        (1, 1, True),
        (1, 2, False),
    ],
)
def test_facet_is_leftmost(
    data: pl.DataFrame,
    row: int,
    col: int,
    expected: bool,
) -> None:
    result = FacetData(data, row="a", col="b")
    assert result.is_leftmost(row, col) == expected


@pytest.mark.parametrize(
    ("row", "col", "expected"),
    [
        (0, 0, False),
        (0, 1, True),
        (0, 2, False),
        (1, 0, False),
        (1, 1, False),
        (1, 2, True),
    ],
)
def test_facet_is_rightmost(
    data: pl.DataFrame,
    row: int,
    col: int,
    expected: bool,
) -> None:
    result = FacetData(data, row="a", col="b")
    assert result.is_rightmost(row, col) == expected


@pytest.mark.parametrize(
    ("row", "col", "expected"),
    [
        (0, 0, True),
        (0, 1, True),
        (0, 2, False),
        (1, 0, False),
        (1, 1, False),
        (1, 2, True),
    ],
)
def test_facet_is_topmost(
    data: pl.DataFrame,
    row: int,
    col: int,
    expected: bool,
) -> None:
    result = FacetData(data, row="a", col="b")
    assert result.is_topmost(row, col) == expected


@pytest.mark.parametrize(
    ("row", "col", "expected"),
    [
        (0, 0, True),
        (0, 1, False),
        (0, 2, False),
        (1, 0, False),
        (1, 1, True),
        (1, 2, True),
    ],
)
def test_facet_is_bottommost(
    data: pl.DataFrame,
    row: int,
    col: int,
    expected: bool,
) -> None:
    result = FacetData(data, row="a", col="b")
    assert result.is_bottommost(row, col) == expected


def test_facet_iterate(data: pl.DataFrame) -> None:
    result = FacetData(data, row="a", col="b")
    facets = list(result.iter_facets())

    assert len(facets) == 4

    facet = facets[0]
    assert facet.row == 0
    assert facet.col == 0
    assert facet.row_label == {"a": 1}
    assert facet.col_label == {"b": 3}
    assert facet.is_leftmost
    assert facet.is_topmost
    assert not facet.is_rightmost
    assert facet.is_bottommost

    facet = facets[-1]
    assert facet.row == 1
    assert facet.col == 2
    assert facet.row_label == {"a": 2}
    assert facet.col_label == {"b": 5}
    assert not facet.is_leftmost
    assert facet.is_topmost
    assert facet.is_rightmost
    assert facet.is_bottommost
