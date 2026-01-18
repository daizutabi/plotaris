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

    expected = pl.DataFrame({"row": [0]} if mapping else [{}])

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


@pytest.fixture(scope="module")
def facet_data(data: pl.DataFrame) -> FacetData:
    return FacetData(data, row="a", col="b")


def test_facet_row_col(facet_data: FacetData) -> None:
    expected = pl.DataFrame({"row": [0, 0, 1, 1], "col": [0, 1, 1, 2]})
    assert_frame_equal(facet_data.index, expected, check_dtypes=False)
    assert facet_data.nrows == 2
    assert facet_data.ncols == 3
    assert facet_data.get_labels(named=True) == [
        {"row": {"a": 1}, "col": {"b": 3}},
        {"row": {"a": 1}, "col": {"b": 4}},
        {"row": {"a": 2}, "col": {"b": 4}},
        {"row": {"a": 2}, "col": {"b": 5}},
    ]
    facets = facet_data.facets()
    assert len(facets) == 6
    assert len(facets.filter(has_data=True)) == 4
    assert len(facets.filter(has_data=False)) == 2


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
    facets = result.facets()
    assert len(facets) == 2
    assert len(facets.filter(has_data=True)) == 2
    assert len(facets.filter(has_data=False)) == 0


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
    facets = result.facets()
    assert len(facets) == 3
    assert len(facets.filter(has_data=True)) == 3
    assert len(facets.filter(has_data=False)) == 0


def test_facet_row_col_empty(data: pl.DataFrame) -> None:
    result = FacetData(data)
    expected = pl.DataFrame({"row": [0], "col": [0]})
    assert_frame_equal(result.index, expected, check_dtypes=False)
    assert result.nrows == 1
    assert result.ncols == 1
    assert result.get_labels() == [{"row": (), "col": ()}]
    facets = result.facets()
    assert len(facets) == 1
    assert len(facets.filter(has_data=True)) == 1
    assert len(facets.filter(has_data=False)) == 0


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
    facets = result.facets()
    assert len(facets) == 6
    assert len(facets.filter(has_data=True)) == 6
    assert len(facets.filter(has_data=False)) == 0


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
    facets = result.facets()
    assert len(facets) == 8
    assert len(facets.filter(has_data=True)) == 6
    assert len(facets.filter(has_data=False)) == 2


def test_facet_get(facet_data: FacetData) -> None:
    expected = pl.DataFrame({"a": [1, 1], "b": [3, 3], "x": [0, 1]})
    df = facet_data.get(0, 0)
    assert isinstance(df, pl.DataFrame)
    assert_frame_equal(df, expected, check_dtypes=False)

    expected = pl.DataFrame({"a": [2, 2], "b": [5, 5], "x": [4, 5]})
    df = facet_data.get(1, 2)
    assert isinstance(df, pl.DataFrame)
    assert_frame_equal(df, expected, check_dtypes=False)

    assert facet_data.get(0, 2) is None


def test_cell_itere(facet_data: FacetData) -> None:
    r, c = facet_data.facet(0, 0)
    assert r == 0
    assert c == 0


@pytest.mark.parametrize(
    ("name", "row", "col", "expected"),
    [
        ("has_data", 0, 0, True),
        ("has_data", 0, 1, True),
        ("has_data", 0, 2, False),
        ("has_data", 1, 0, False),
        ("has_data", 1, 1, True),
        ("has_data", 1, 2, True),
        ("is_left", 0, 0, True),
        ("is_left", 0, 1, False),
        ("is_left", 0, 2, False),
        ("is_left", 1, 0, True),
        ("is_left", 1, 1, False),
        ("is_left", 1, 2, False),
        ("is_top", 0, 0, True),
        ("is_top", 0, 1, True),
        ("is_top", 0, 2, True),
        ("is_top", 1, 0, False),
        ("is_top", 1, 1, False),
        ("is_top", 1, 2, False),
        ("is_right", 0, 0, False),
        ("is_right", 0, 1, False),
        ("is_right", 0, 2, True),
        ("is_right", 1, 0, False),
        ("is_right", 1, 1, False),
        ("is_right", 1, 2, True),
        ("is_bottom", 0, 0, False),
        ("is_bottom", 0, 1, False),
        ("is_bottom", 0, 2, False),
        ("is_bottom", 1, 0, True),
        ("is_bottom", 1, 1, True),
        ("is_bottom", 1, 2, True),
        ("is_leftmost", 0, 0, True),
        ("is_leftmost", 0, 1, False),
        ("is_leftmost", 0, 2, False),
        ("is_leftmost", 1, 0, False),
        ("is_leftmost", 1, 1, True),
        ("is_leftmost", 1, 2, False),
        ("is_topmost", 0, 0, True),
        ("is_topmost", 0, 1, True),
        ("is_topmost", 0, 2, False),
        ("is_topmost", 1, 0, False),
        ("is_topmost", 1, 1, False),
        ("is_topmost", 1, 2, True),
        ("is_rightmost", 0, 0, False),
        ("is_rightmost", 0, 1, True),
        ("is_rightmost", 0, 2, False),
        ("is_rightmost", 1, 0, False),
        ("is_rightmost", 1, 1, False),
        ("is_rightmost", 1, 2, True),
        ("is_bottommost", 0, 0, True),
        ("is_bottommost", 0, 1, False),
        ("is_bottommost", 0, 2, False),
        ("is_bottommost", 1, 0, False),
        ("is_bottommost", 1, 1, True),
        ("is_bottommost", 1, 2, True),
    ],
)
def test_facet_attribute(
    facet_data: FacetData,
    name: str,
    row: int,
    col: int,
    expected: bool,
) -> None:
    assert getattr(facet_data.facet(row, col), name) is expected


def test_facets(facet_data: FacetData) -> None:
    facets = list(facet_data.facets())

    assert len(facets) == 6

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


def test_cells_filter_predicate(facet_data: FacetData) -> None:
    facets = facet_data.facets().filter(lambda c: c.row == 0)
    assert len(facets) == 3
    assert all(f.row == 0 for f in facets)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("row", 0, [(0, 0), (0, 1), (0, 2)]),
        ("row", 1, [(1, 0), (1, 1), (1, 2)]),
        ("row", 2, []),
        ("col", 0, [(0, 0), (1, 0)]),
        ("col", 1, [(0, 1), (1, 1)]),
        ("col", 2, [(0, 2), (1, 2)]),
    ],
)
def test_facets_filter_row_col(
    facet_data: FacetData,
    name: str,
    value: int,
    expected: list[tuple[int, int]],
) -> None:
    facets = facet_data.facets().filter(**{name: value})  # pyright: ignore[reportArgumentType]
    result = [(f.row, f.col) for f in facets]
    assert result == expected


@pytest.mark.parametrize(
    ("name", "expected"),
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
def test_facets_filter_true(
    facet_data: FacetData,
    name: str,
    expected: list[tuple[int, int]],
) -> None:
    facets = facet_data.facets().filter(**{name: True})  # pyright: ignore[reportArgumentType]
    result = [(f.row, f.col) for f in facets]
    assert result == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("has_data", [(0, 2), (1, 0)]),
        ("is_left", [(0, 1), (0, 2), (1, 1), (1, 2)]),
        ("is_top", [(1, 0), (1, 1), (1, 2)]),
        ("is_right", [(0, 0), (0, 1), (1, 0), (1, 1)]),
        ("is_bottom", [(0, 0), (0, 1), (0, 2)]),
        ("is_leftmost", [(0, 1), (0, 2), (1, 0), (1, 2)]),
        ("is_topmost", [(0, 2), (1, 0), (1, 1)]),
        ("is_rightmost", [(0, 0), (0, 2), (1, 0), (1, 1)]),
        ("is_bottommost", [(0, 1), (0, 2), (1, 0)]),
    ],
)
def test_facets_filter_false(
    facet_data: FacetData,
    name: str,
    expected: list[tuple[int, int]],
) -> None:
    facets = facet_data.facets().filter(**{name: False})  # pyright: ignore[reportArgumentType]
    result = [(f.row, f.col) for f in facets]
    assert result == expected


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("row", 0, [(0, 0), (0, 1)]),
        ("row", 1, [(1, 1), (1, 2)]),
        ("row", 2, []),
        ("col", 0, [(0, 0)]),
        ("col", 1, [(0, 1), (1, 1)]),
        ("col", 2, [(1, 2)]),
    ],
)
def test_facets_filter_row_col_has_data(
    facet_data: FacetData,
    name: str,
    value: int,
    expected: list[tuple[int, int]],
) -> None:
    facets = facet_data.facets().filter(has_data=True, **{name: value})  # pyright: ignore[reportArgumentType]
    result = [(f.row, f.col) for f in facets]
    assert result == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
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
def test_facets_filter_true_has_data(
    facet_data: FacetData,
    name: str,
    expected: list[tuple[int, int]],
) -> None:
    facets = facet_data.facets().filter(has_data=True, **{name: True})  # pyright: ignore[reportArgumentType]
    result = [(f.row, f.col) for f in facets]
    assert result == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("is_left", [(0, 1), (1, 1), (1, 2)]),
        ("is_top", [(1, 1), (1, 2)]),
        ("is_right", [(0, 0), (0, 1), (1, 1)]),
        ("is_bottom", [(0, 0), (0, 1)]),
        ("is_leftmost", [(0, 1), (1, 2)]),
        ("is_topmost", [(1, 1)]),
        ("is_rightmost", [(0, 0), (1, 1)]),
        ("is_bottommost", [(0, 1)]),
    ],
)
def test_facets_filter_false_has_data(
    facet_data: FacetData,
    name: str,
    expected: list[tuple[int, int]],
) -> None:
    facets = facet_data.facets().filter(has_data=True, **{name: False})  # pyright: ignore[reportArgumentType]
    result = [(f.row, f.col) for f in facets]
    assert result == expected


def test_iter(facet_data: FacetData) -> None:
    assert len(list(facet_data)) == len(facet_data.facets()._items)  # pyright: ignore[reportPrivateUsage]
