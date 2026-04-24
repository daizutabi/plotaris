from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal as _assert_frame_equal

from plotaris.matplotlib.group import Group, group_by, with_index


def assert_frame_equal(left: pl.DataFrame, right: pl.DataFrame, /) -> None:
    _assert_frame_equal(left, right, check_dtypes=False)


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
        ([], [None, None, None, None]),
        (["a"], [0, 1, 0, 1]),
        (["b"], [0, 0, 1, 1]),
        (["a", "b"], [0, 1, 2, 3]),
        (["b", "a"], [0, 1, 2, 3]),
    ],
)
def test_with_index(columns: list[str], values: list[int | None]) -> None:
    data = pl.DataFrame({"a": [1, 2, 1, 2], "b": [9, 9, 8, 8], "c": [10, 11, 12, 13]})
    result = with_index(data, columns, "x")
    expected = data.with_columns(x=pl.Series(values))
    assert_frame_equal(result, expected)


def test_group_empty() -> None:
    gr = Group(pl.DataFrame())
    assert_frame_equal(gr.indices(), pl.DataFrame())
    assert gr.data == []


def test_group_empty_with_columns() -> None:
    data = pl.DataFrame({"A": [], "B": [], "C": []})
    gr = Group(data, a="A", b=("B", "C"))
    assert_frame_equal(gr.indices(), pl.DataFrame({"a": [], "b": []}))
    assert gr.mapping == {"a": ("A",), "b": ("B", "C")}
    assert gr.data == []
    assert "a" in gr
    assert "c" not in gr


def test_group_one() -> None:
    data = pl.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]})
    gr = Group(data, a="A", b=["A"])

    expected = pl.DataFrame({"a": [0, 1], "b": [0, 1]})
    assert_frame_equal(gr.indices(), expected)
    assert_frame_equal(gr.indices(["a", "b"]), expected)
    assert_frame_equal(gr.indices("b"), expected.select("b"))

    expected = pl.DataFrame({"A": [1, 2]})
    assert_frame_equal(gr.keys(), expected)
    assert_frame_equal(gr.keys("a"), expected)
    assert_frame_equal(gr.keys(["a", "b"]), expected)

    assert len(gr) == 2
    assert_frame_equal(gr[0], data[0:2])
    assert_frame_equal(gr[1], data[2:])


def test_group_two() -> None:
    data = pl.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]})
    gr = Group(data, a="A", b=["B"])

    expected = pl.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]})
    assert_frame_equal(gr.indices(), expected)
    assert_frame_equal(gr.indices(["a", "b"]), expected)
    assert_frame_equal(gr.indices("a", "b"), expected)
    assert_frame_equal(gr.indices("b"), expected.select("b"))

    expected = pl.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]})
    assert_frame_equal(gr.keys(), expected)
    assert_frame_equal(gr.keys("a"), expected.select("A"))
    assert_frame_equal(gr.keys("a", "b"), expected)

    assert len(gr) == 4
    it = iter(gr)
    assert_frame_equal(gr[0], next(it))
    assert_frame_equal(gr[1], next(it))
    assert_frame_equal(gr[2], next(it))
    assert_frame_equal(gr[3], next(it))


def test_group_columns_empty() -> None:
    data = pl.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]})
    gr = Group(data, a=[], b=[])

    expected = pl.DataFrame({"a": [None], "b": [None]})
    assert_frame_equal(gr.indices(), expected)

    assert_frame_equal(gr.keys(), pl.DataFrame())

    assert len(gr) == 1
    assert_frame_equal(gr[0], data)


def test_group_without_columns() -> None:
    data = pl.DataFrame({"A": [1, 1, 2, 2], "B": [3, 4, 3, 4]})
    gr = Group(data)

    assert_frame_equal(gr.indices(), pl.DataFrame([]))

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
    assert gr.indices("row")["row"].to_list() == list(range(n))

    assert len(gr) == n


def test_group_columns_str_str(data: pl.DataFrame) -> None:
    gr = Group(data, row="a", col="b")

    expected = pl.DataFrame({"row": [0, 0, 1, 1], "col": [0, 1, 1, 2]})
    assert_frame_equal(gr.indices(), expected)

    expected = pl.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 4, 5]})
    assert_frame_equal(gr.keys(), expected)
    assert_frame_equal(gr.keys("row"), expected.select("a"))
    assert_frame_equal(gr.keys("col"), expected.select("b"))

    assert len(gr) == 4

    dim = gr.dimension_keys()
    assert_frame_equal(dim["row"], expected.select("a"))
    assert_frame_equal(dim["col"], expected.select("b"))

    assert gr.labels(1) == {"row": {"a": 1}, "col": {"b": 4}}
    assert gr.labels(2, merge=True) == {"a": 2, "b": 4}
    assert gr.labels() == [
        {"row": {"a": 1}, "col": {"b": 3}},
        {"row": {"a": 1}, "col": {"b": 4}},
        {"row": {"a": 2}, "col": {"b": 4}},
        {"row": {"a": 2}, "col": {"b": 5}},
    ]
    assert gr.labels(merge=True) == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 4},
        {"a": 2, "b": 5},
    ]


def test_group_columns_str_str_duplicated(data: pl.DataFrame) -> None:
    gr = Group(data, row="b", col="b")

    expected = pl.DataFrame({"row": [0, 1, 2], "col": [0, 1, 2]})
    assert_frame_equal(gr.indices(), expected)

    expected = pl.DataFrame({"b": [3, 4, 5]})
    assert_frame_equal(gr.keys(), expected)

    assert len(gr) == 3

    dim = gr.dimension_keys()
    assert_frame_equal(dim["row"], expected)
    assert_frame_equal(dim["col"], expected)

    assert gr.labels(1) == {"row": {"b": 4}, "col": {"b": 4}}
    assert gr.labels(2, merge=True) == {"b": 5}
    assert gr.labels() == [
        {"row": {"b": 3}, "col": {"b": 3}},
        {"row": {"b": 4}, "col": {"b": 4}},
        {"row": {"b": 5}, "col": {"b": 5}},
    ]
    assert gr.labels(merge=True) == [{"b": 3}, {"b": 4}, {"b": 5}]


def test_group_columns_tuple(data: pl.DataFrame) -> None:
    gr = Group(data, row=("a", "b"))

    expected = pl.DataFrame({"row": [0, 1, 2, 3]})
    assert_frame_equal(gr.indices(), expected)

    expected = pl.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 4, 5]})
    assert_frame_equal(gr.keys(), expected)

    assert len(gr) == 4

    dim = gr.dimension_keys()
    assert_frame_equal(dim["row"], expected)

    assert gr.labels(1) == {"row": {"a": 1, "b": 4}}
    assert gr.labels(2, merge=True) == {"a": 2, "b": 4}
    assert gr.labels() == [
        {"row": {"a": 1, "b": 3}},
        {"row": {"a": 1, "b": 4}},
        {"row": {"a": 2, "b": 4}},
        {"row": {"a": 2, "b": 5}},
    ]
    assert gr.labels(merge=True) == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 4},
        {"a": 2, "b": 5},
    ]


def test_group_columns_tuple_str(data: pl.DataFrame) -> None:
    gr = Group(data, row=("b", "a"), col="a")

    expected = pl.DataFrame({"row": [0, 1, 2, 3], "col": [0, 0, 1, 1]})
    assert_frame_equal(gr.indices(), expected)

    expected = pl.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 4, 5]})
    assert_frame_equal(gr.keys(), expected)
    assert_frame_equal(gr.keys("row"), expected)
    assert_frame_equal(gr.keys("col"), expected.select("a"))

    dim = gr.dimension_keys()
    assert_frame_equal(dim["row"], expected)
    assert_frame_equal(dim["col"], expected.select("a"))

    assert gr.labels(1) == {"row": {"b": 4, "a": 1}, "col": {"a": 1}}
    assert gr.labels(2, merge=True) == {"b": 4, "a": 2}
    assert gr.labels() == [
        {"row": {"b": 3, "a": 1}, "col": {"a": 1}},
        {"row": {"b": 4, "a": 1}, "col": {"a": 1}},
        {"row": {"b": 4, "a": 2}, "col": {"a": 2}},
        {"row": {"b": 5, "a": 2}, "col": {"a": 2}},
    ]
    assert gr.labels(merge=True) == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 4},
        {"a": 2, "b": 5},
    ]


def test_group_columns_str_empty(data: pl.DataFrame) -> None:
    gr = Group(data, row="a", col=())

    expected = pl.DataFrame({"row": [0, 1], "col": [None, None]})
    assert_frame_equal(gr.indices(), expected)

    expected = pl.DataFrame({"a": [1, 2]})
    assert_frame_equal(gr.keys(), expected)

    assert len(gr) == 2

    dim = gr.dimension_keys()
    assert_frame_equal(dim["row"], expected)
    assert_frame_equal(dim["col"], pl.DataFrame())


def test_group_data_empty() -> None:
    result = Group(pl.DataFrame(), row=["a"])
    expected = pl.DataFrame({"row": []})
    assert_frame_equal(result.indices(), expected)
