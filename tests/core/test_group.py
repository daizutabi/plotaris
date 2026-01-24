from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from plotaris.core.group import Group, group_by, to_tuple, with_index


@pytest.mark.parametrize(
    ("values", "expected"),
    [(None, ()), ("abc", ("abc",)), (["abc"], ("abc",))],
)
def test_to_tuple(values: str | list[str] | None, expected: tuple[str, ...]) -> None:
    assert to_tuple(values) == expected


@pytest.mark.parametrize("data", [pl.DataFrame(), pl.DataFrame({"a": [1, 2]})])
def test_group_by_without_columns(data: pl.DataFrame) -> None:
    index, dfs = group_by(data)
    assert_frame_equal(index, pl.DataFrame([{}]))
    assert len(dfs) == 1
    assert_frame_equal(dfs[0], data)


def test_group_by_empty() -> None:
    data = pl.DataFrame({"a": [], "b": []})
    index, dfs = group_by(data, "a")
    assert_frame_equal(index, pl.DataFrame({"a": []}))
    assert len(index) == 0
    assert dfs == []


@pytest.mark.parametrize("columns", [("a",), ("a", "a")])
def test_group_by_one(columns: tuple[str, ...]) -> None:
    data = pl.DataFrame({"a": [1, 2, 1, 2], "b": [9, 8, 7, 6]})
    index, dfs = group_by(data, *columns)
    assert_frame_equal(index, pl.DataFrame({"a": [1, 2]}))
    assert len(dfs) == 2
    assert_frame_equal(dfs[0], data[[0, 2]])
    assert_frame_equal(dfs[1], data[[1, 3]])


@pytest.mark.parametrize(
    "columns",
    [("a", "b"), ("b", "a"), (["a"], "b"), (["a"], ["b"]), (["b", "a"], "a")],
)
def test_group_by_two(columns: Any) -> None:
    data = pl.DataFrame({"a": [1, 2, 1, 2], "b": [9, 8, 7, 6]})
    index, dfs = group_by(data, *columns)
    assert_frame_equal(index, data)
    assert len(dfs) == 4
    assert_frame_equal(dfs[0], data[0])
    assert_frame_equal(dfs[1], data[1])
    assert_frame_equal(dfs[2], data[2])
    assert_frame_equal(dfs[3], data[3])


@pytest.mark.parametrize(
    ("columns", "values"),
    [
        ([], [0, 0, 0, 0]),
        (["a"], [0, 1, 0, 1]),
        (["b"], [0, 0, 1, 1]),
        (["a", "b"], [0, 1, 2, 3]),
        (["b", "a"], [0, 1, 2, 3]),
    ],
)
def test_with_index(columns: list[str], values: list[int]) -> None:
    data = pl.DataFrame({"a": [1, 2, 1, 2], "b": [9, 9, 8, 8], "c": [10, 11, 12, 13]})
    result = with_index(data, columns, "x")
    expected = data.with_columns(x=pl.Series(values))
    assert_frame_equal(result, expected, check_dtypes=False)


def test_group_empty() -> None:
    gr = Group(pl.DataFrame())
    assert_frame_equal(gr.index, pl.DataFrame())
    assert gr.data == []


def test_group_empty_with_columns() -> None:
    data = pl.DataFrame({"A": [], "B": [], "C": []})
    gr = Group(data, a="A", b=("B", "C"))
    assert_frame_equal(gr.index, pl.DataFrame({"a": [], "b": []}))
    assert gr.data == []


def test_group_one() -> None:
    data = pl.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]})
    gr = Group(data, a="A", b=["A"])

    expected = pl.DataFrame({"a": [0, 1], "b": [0, 1]})
    assert_frame_equal(gr.index, expected, check_dtypes=False)
    assert len(gr) == 2
    assert_frame_equal(gr[0], data[0:2])
    assert_frame_equal(gr[1], data[2:])


def test_group_two() -> None:
    data = pl.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]})
    gr = Group(data, a="A", b=["B"])

    expected = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    assert_frame_equal(gr.index, expected, check_dtypes=False)
    assert len(gr) == 4
    it = iter(gr)
    assert_frame_equal(gr[0], next(it))
    assert_frame_equal(gr[1], next(it))
    assert_frame_equal(gr[2], next(it))
    assert_frame_equal(gr[3], next(it))


def test_group_columns_empty() -> None:
    data = pl.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]})
    gr = Group(data, a=[], b=[])
    expected = pl.DataFrame({"a": [0], "b": [0]})
    assert_frame_equal(gr.index, expected, check_dtypes=False)
    df = pl.DataFrame()
    assert_frame_equal(gr.mapping["a"], df, check_dtypes=False)
    assert_frame_equal(gr.mapping["b"], df, check_dtypes=False)
    assert len(gr) == 1
    assert_frame_equal(gr[0], data)


def test_group_without_columns() -> None:
    data = pl.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]})
    gr = Group(data)
    assert_frame_equal(gr.index, pl.DataFrame([{}]))
    assert gr.mapping == {}
    assert len(gr) == 1
    assert_frame_equal(gr[0], data)


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [3, 3, 4, 4, 5, 5],
            "x": range(6),
        },
    )


@pytest.mark.parametrize(("name", "values"), [("a", [1, 2]), ("b", [3, 4, 5])])
def test_group_columns_str(data: pl.DataFrame, name: str, values: list[int]) -> None:
    gr = Group(data, row=name)

    n = len(values)
    expected = pl.DataFrame({"row": range(n)})
    assert_frame_equal(gr.index, expected, check_dtypes=False)

    assert len(gr) == n
    assert gr.n_unique("row") == n


def test_group_columns_str_str(data: pl.DataFrame) -> None:
    gr = Group(data, row="a", col="b")

    expected = pl.DataFrame({"row": [0, 0, 1, 1], "col": [0, 1, 1, 2]})
    assert_frame_equal(gr.index, expected, check_dtypes=False)

    expected = pl.DataFrame({"a": [1, 2]})
    assert_frame_equal(gr.mapping["row"], expected, check_dtypes=False)
    expected = pl.DataFrame({"b": [3, 4, 5]})
    assert_frame_equal(gr.mapping["col"], expected, check_dtypes=False)

    assert len(gr) == 4
    assert gr.item(0, "row") == (1,)
    assert gr.item(1, "row") == (1,)
    assert gr.item(2, "row") == (2,)
    assert gr.item(3, "row") == (2,)
    assert gr.item(3, "row", named=True) == {"a": 2}
    assert gr.items("row") == [(1,), (1,), (2,), (2,)]
    assert gr.items("row", named=True) == [{"a": 1}, {"a": 1}, {"a": 2}, {"a": 2}]
    assert gr.item(0, "col") == (3,)
    assert gr.item(1, "col") == (4,)
    assert gr.item(2, "col") == (4,)
    assert gr.item(3, "col") == (5,)
    assert gr.item(3, "col", named=True) == {"b": 5}
    assert gr.dimension(0) == {"row": (1,), "col": (3,)}
    assert gr.dimension(1) == {"row": (1,), "col": (4,)}
    assert gr.dimension(2) == {"row": (2,), "col": (4,)}
    assert gr.dimension(3) == {"row": (2,), "col": (5,)}
    assert gr.dimension(0, named=True) == {"row": {"a": 1}, "col": {"b": 3}}
    assert gr.dimension(1, named=True) == {"row": {"a": 1}, "col": {"b": 4}}
    assert gr.dimension(2, named=True) == {"row": {"a": 2}, "col": {"b": 4}}
    assert gr.dimension(3, named=True) == {"row": {"a": 2}, "col": {"b": 5}}
    assert gr.dimensions() == [
        {"row": (1,), "col": (3,)},
        {"row": (1,), "col": (4,)},
        {"row": (2,), "col": (4,)},
        {"row": (2,), "col": (5,)},
    ]
    assert gr.dimensions(named=True) == [
        {"row": {"a": 1}, "col": {"b": 3}},
        {"row": {"a": 1}, "col": {"b": 4}},
        {"row": {"a": 2}, "col": {"b": 4}},
        {"row": {"a": 2}, "col": {"b": 5}},
    ]
    assert gr.n_unique("row") == 2
    assert gr.n_unique("col") == 3


def test_group_columns_str_str_duplicated(data: pl.DataFrame) -> None:
    gr = Group(data, row="b", col="b")

    expected = pl.DataFrame({"row": [0, 1, 2], "col": [0, 1, 2]})
    assert_frame_equal(gr.index, expected, check_dtypes=False)

    expected = pl.DataFrame({"b": [3, 4, 5]})
    assert_frame_equal(gr.mapping["row"], expected, check_dtypes=False)
    assert_frame_equal(gr.mapping["col"], expected, check_dtypes=False)

    assert len(gr) == 3
    assert gr.item(0, "row") == (3,)
    assert gr.item(1, "row") == (4,)
    assert gr.item(2, "row") == (5,)
    assert gr.item(2, "row", named=True) == {"b": 5}
    assert gr.item(0, "col") == (3,)
    assert gr.item(1, "col") == (4,)
    assert gr.item(2, "col") == (5,)
    assert gr.item(2, "col", named=True) == {"b": 5}
    assert gr.n_unique("row") == 3
    assert gr.n_unique("col") == 3


def test_group_columns_tuple(data: pl.DataFrame) -> None:
    gr = Group(data, row=("a", "b"))

    expected = pl.DataFrame({"row": [0, 1, 2, 3]})
    assert_frame_equal(gr.index, expected, check_dtypes=False)

    expected = pl.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 4, 5]})
    assert_frame_equal(gr.mapping["row"], expected, check_dtypes=False)

    assert len(gr) == 4
    assert gr.item(0, "row") == (1, 3)
    assert gr.item(1, "row") == (1, 4)
    assert gr.item(2, "row") == (2, 4)
    assert gr.item(3, "row") == (2, 5)
    assert gr.item(3, "row", named=True) == {"a": 2, "b": 5}
    assert gr.n_unique("row") == 4


def test_group_columns_tuple_str(data: pl.DataFrame) -> None:
    gr = Group(data, row=("b", "a"), col="a")

    expected = pl.DataFrame({"row": [0, 1, 2, 3], "col": [0, 0, 1, 1]})
    assert_frame_equal(gr.index, expected, check_dtypes=False)

    expected = pl.DataFrame({"b": [3, 4, 4, 5], "a": [1, 1, 2, 2]})
    assert_frame_equal(gr.mapping["row"], expected, check_dtypes=False)
    expected = pl.DataFrame({"a": [1, 2]})
    assert_frame_equal(gr.mapping["col"], expected, check_dtypes=False)

    assert len(gr) == 4
    assert gr.item(0, "row") == (3, 1)
    assert gr.item(1, "row") == (4, 1)
    assert gr.item(2, "row") == (4, 2)
    assert gr.item(3, "row") == (5, 2)
    assert gr.item(3, "row", named=True) == {"b": 5, "a": 2}
    assert gr.item(0, "col") == (1,)
    assert gr.item(1, "col") == (1,)
    assert gr.item(2, "col") == (2,)
    assert gr.item(3, "col") == (2,)
    assert gr.item(3, "col", named=True) == {"a": 2}
    assert gr.n_unique("row") == 4
    assert gr.n_unique("col") == 2


def test_group_columns_str_empty(data: pl.DataFrame) -> None:
    gr = Group(data, row="a", col=())

    expected = pl.DataFrame({"row": [0, 1], "col": [0, 0]})
    assert_frame_equal(gr.index, expected, check_dtypes=False)

    expected = pl.DataFrame({"a": [1, 2]})
    assert_frame_equal(gr.mapping["row"], expected, check_dtypes=False)

    assert len(gr) == 2
    assert gr.item(0, "row") == (1,)
    assert gr.item(1, "row") == (2,)
    assert gr.item(0, "col") == ()
    assert gr.item(1, "col") == ()
    assert gr.item(1, "col", named=True) == {}


def test_group_data_empty() -> None:
    result = Group(pl.DataFrame(), row=["a"])
    expected = pl.DataFrame({"row": []})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.mapping == {}
