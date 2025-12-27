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


def test_mapping_str_str(data: pl.DataFrame) -> None:
    result = GroupedData(data, {"row": "a", "col": "b"})

    expected = pl.DataFrame({"row": [0, 0, 1, 1], "col": [0, 1, 1, 2]})

    assert_frame_equal(result.group, expected, check_dtypes=False)
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
    assert result.n_unique("row") == 2
    assert result.n_unique("col") == 3


def test_mapping_str_str_duplicated(data: pl.DataFrame) -> None:
    result = GroupedData(data, {"row": "b", "col": "b"})

    expected = pl.DataFrame({"row": [0, 1, 2], "col": [0, 1, 2]})

    assert_frame_equal(result.group, expected, check_dtypes=False)
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

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == n
    assert result.n_unique("row") == n
    assert result.n_unique("col") == 0


def test_mapping_list(data: pl.DataFrame) -> None:
    result = GroupedData(data, {"row": ["a", "b"]})

    expected = pl.DataFrame({"row": [0, 1, 2, 3]})

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == 4
    assert result.item(0, "row") == (1, 3)
    assert result.item(1, "row") == (1, 4)
    assert result.item(2, "row") == (2, 4)
    assert result.item(3, "row") == (2, 5)
    assert result.item(3, "row", named=True) == {"a": 2, "b": 5}
    assert result.n_unique("row") == 4
    assert result.n_unique("col") == 0


def test_mapping_list_str(data: pl.DataFrame) -> None:
    result = GroupedData(data, {"row": ["b", "a"], "col": "a"})

    expected = pl.DataFrame({"row": [0, 1, 2, 3], "col": [0, 0, 1, 1]})

    assert_frame_equal(result.group, expected, check_dtypes=False)
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


def test_mapping_none(data: pl.DataFrame) -> None:
    result = GroupedData(data, {})

    expected = pl.DataFrame([{}])

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == 1


def test_data_empty() -> None:
    result = GroupedData(pl.DataFrame(), {"row": ["a"]})

    expected = pl.DataFrame({"row": []})

    assert_frame_equal(result.group, expected, check_dtypes=False)
    assert len(result) == 0


def test_group_by_no_data() -> None:
    group, dfs = group_by(pl.DataFrame({"x": []}), "x")
    assert_frame_equal(group, pl.DataFrame({"x": []}))
    assert len(dfs) == 0
