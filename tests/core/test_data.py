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
    assert result.group(0) == {"row": {"a": 1}, "col": {"b": 3}}
    assert result.group(1) == {"row": {"a": 1}, "col": {"b": 4}}
    assert result.group(2) == {"row": {"a": 2}, "col": {"b": 4}}
    assert result.group(3) == {"row": {"a": 2}, "col": {"b": 5}}
    assert result.groups() == [
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
    assert result.groups() == []


def test_facet_row_col(data: pl.DataFrame) -> None:
    result = FacetData(data, row="a", col="b")
    expected = pl.DataFrame({"row": [0, 0, 1, 1], "col": [0, 1, 1, 2]})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.nrows == 2
    assert result.ncols == 3
    assert result.groups() == [
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
    assert result.groups() == [
        {"row": {}, "col": {"a": 1}},
        {"row": {}, "col": {"a": 2}},
    ]
    assert result.cells() == [(0, 0), (0, 1)]
    assert result.cells(empty=True) == []


def test_facet_col_empty(data: pl.DataFrame) -> None:
    result = FacetData(data, row="b")
    expected = pl.DataFrame({"row": [0, 1, 2], "col": [0, 0, 0]})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.nrows == 3
    assert result.ncols == 1
    assert result.groups() == [
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
    assert result.groups() == [{"row": {}, "col": {}}]
    assert result.cells() == [(0, 0)]
    assert result.cells(empty=True) == []


def test_facet_row_wrap(data: pl.DataFrame) -> None:
    result = FacetData(data, row="x", wrap=2)
    expected = pl.DataFrame({"row": [0, 1, 0, 1, 0, 1], "col": [0, 0, 1, 1, 2, 2]})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.nrows == 2
    assert result.ncols == 3
    assert result.groups() == [
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
    assert result.groups() == [
        {"row": {}, "col": {"x": 0}},
        {"row": {}, "col": {"x": 1}},
        {"row": {}, "col": {"x": 2}},
        {"row": {}, "col": {"x": 3}},
        {"row": {}, "col": {"x": 4}},
        {"row": {}, "col": {"x": 5}},
    ]
    assert result.cells() == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
    assert result.cells(empty=True) == [(1, 2), (1, 3)]
