from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from plotaris.core.data import FacetFrame


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.DataFrame(
        {"a": [1, 1, 1, 2, 2, 2], "b": [3, 3, 4, 4, 5, 5], "x": range(6)},
    )


def test_facet_frame_row_col(data: pl.DataFrame) -> None:
    ff = FacetFrame(data, row="a", col="b")

    expected = pl.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [3, 4, 4, 5],
            "_row_index": [0, 0, 1, 1],
            "_col_index": [0, 1, 1, 2],
        },
    )

    assert_frame_equal(ff.group, expected, check_dtypes=False)
    assert len(ff) == 4
    assert ff.n_rows == 2
    assert ff.n_cols == 3


def test_facet_frame_row(data: pl.DataFrame) -> None:
    ff = FacetFrame(data, row="a")

    expected = pl.DataFrame(
        {
            "a": [1, 2],
            "_row_index": [0, 1],
            "_col_index": [0, 0],
        },
    )

    assert_frame_equal(ff.group, expected, check_dtypes=False)
    assert len(ff) == 2
    assert ff.n_rows == 2
    assert ff.n_cols == 1


def test_facet_frame_col(data: pl.DataFrame) -> None:
    ff = FacetFrame(data, col="b")

    expected = pl.DataFrame(
        {
            "b": [3, 4, 5],
            "_row_index": [0, 0, 0],
            "_col_index": [0, 1, 2],
        },
    )

    assert_frame_equal(ff.group, expected, check_dtypes=False)
    assert len(ff) == 3
    assert ff.n_rows == 1
    assert ff.n_cols == 3


def test_facet_frame_row_list(data: pl.DataFrame) -> None:
    ff = FacetFrame(data, row=["a", "b"])

    expected = pl.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [3, 4, 4, 5],
            "_row_index": [0, 1, 2, 3],
            "_col_index": [0, 0, 0, 0],
        },
    )

    assert_frame_equal(ff.group, expected, check_dtypes=False)
    assert len(ff) == 4
    assert ff.n_rows == 4
    assert ff.n_cols == 1
