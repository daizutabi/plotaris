from __future__ import annotations

import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from plotaris.core.data import FacetData, group_by


@pytest.fixture(scope="module")
def data() -> DataFrame:
    return DataFrame(
        {"a": [1, 1, 1, 2, 2, 2], "b": [3, 3, 4, 4, 5, 5], "x": range(6)},
    )


def test_facet_frame_row_col(data: DataFrame) -> None:
    result = FacetData(data, row="a", col="b")

    expected = DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [3, 4, 4, 5],
            "_row_index": [0, 0, 1, 1],
            "_col_index": [0, 1, 1, 2],
        },
    )

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == 4
    assert result.n_rows == 2
    assert result.n_cols == 3


def test_facet_frame_row(data: DataFrame) -> None:
    result = FacetData(data, row="a")

    expected = DataFrame(
        {
            "a": [1, 2],
            "_row_index": [0, 1],
            "_col_index": [0, 0],
        },
    )

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == 2
    assert result.n_rows == 2
    assert result.n_cols == 1


def test_facet_frame_col(data: DataFrame) -> None:
    result = FacetData(data, col="b")

    expected = DataFrame(
        {
            "b": [3, 4, 5],
            "_row_index": [0, 0, 0],
            "_col_index": [0, 1, 2],
        },
    )

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == 3
    assert result.n_rows == 1
    assert result.n_cols == 3


def test_facet_frame_row_list(data: DataFrame) -> None:
    result = FacetData(data, row=["a", "b"])

    expected = DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [3, 4, 4, 5],
            "_row_index": [0, 1, 2, 3],
            "_col_index": [0, 0, 0, 0],
        },
    )

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == 4
    assert result.n_rows == 4
    assert result.n_cols == 1


def test_facet_frame_no_facet(data: DataFrame) -> None:
    result = FacetData(data)

    expected = DataFrame(
        {
            "_row_index": [0],
            "_col_index": [0],
        },
    )

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == 1
    assert result.n_rows == 1
    assert result.n_cols == 1


def test_group_by_no_by() -> None:
    data = DataFrame({"x": [1, 2, 3]})
    group, dfs = group_by(data)
    assert_frame_equal(group, DataFrame([{}]))
    assert len(dfs) == 1
    assert_frame_equal(dfs[0], data)


def test_group_by_empty() -> None:
    group, dfs = group_by(DataFrame(), "x")
    assert_frame_equal(group, DataFrame({"x": []}))
    assert len(dfs) == 0


def test_group_by_no_data() -> None:
    group, dfs = group_by(DataFrame({"x": []}), "x")
    assert_frame_equal(group, DataFrame({"x": []}))
    assert len(dfs) == 0
