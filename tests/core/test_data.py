from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from plotaris.core.data import GroupedData, group_by


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [3, 3, 4, 4, 5, 5],
            "x": range(6),
        },
    )


def test_dimensions_str_str(data: pl.DataFrame) -> None:
    result = GroupedData(data, {"row": "a", "col": "b"})

    expected = pl.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [3, 4, 4, 5],
            "_row_index": [0, 0, 1, 1],
            "_col_index": [0, 1, 1, 2],
        },
    )

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == 4
    assert result.n_unique("row") == 2
    assert result.n_unique("col") == 3


@pytest.mark.parametrize(("name", "values"), [("a", [1, 2]), ("b", [3, 4, 5])])
def test_dimensions_str(data: pl.DataFrame, name: str, values: list[int]) -> None:
    result = GroupedData(data, {"row": name})

    n = len(values)
    expected = pl.DataFrame({name: values, "_row_index": range(n)})

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == n
    assert result.n_unique("row") == n
    assert result.n_unique("col") == 0


def test_dimensions_list(data: pl.DataFrame) -> None:
    result = GroupedData(data, {"row": ["a", "b"]})

    expected = pl.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [3, 4, 4, 5],
            "_row_index": [0, 1, 2, 3],
        },
    )

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == 4
    assert result.n_unique("row") == 4
    assert result.n_unique("col") == 0


def test_dimensions_none(data: pl.DataFrame) -> None:
    result = GroupedData(data, {})

    expected = pl.DataFrame([{}])

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == 1


def test_group_by_no_by() -> None:
    data = pl.DataFrame({"x": [1, 2, 3]})
    group, dfs = group_by(data)
    assert_frame_equal(group, pl.DataFrame([{}]))
    assert len(dfs) == 1
    assert_frame_equal(dfs[0], data)


def test_group_by_empty() -> None:
    group, dfs = group_by(pl.DataFrame(), "x")
    assert_frame_equal(group, pl.DataFrame({"x": []}))
    assert len(dfs) == 0


def test_group_by_no_data() -> None:
    group, dfs = group_by(pl.DataFrame({"x": []}), "x")
    assert_frame_equal(group, pl.DataFrame({"x": []}))
    assert len(dfs) == 0
