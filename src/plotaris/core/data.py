from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable


def _to_list(columns: Iterable[str] | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def group_by(data: pl.DataFrame, *by: str) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    names: list[tuple[Any, ...]] = []
    dfs: list[pl.DataFrame] = []

    for name, df in data.group_by(*by, maintain_order=True):
        names.append(name)
        dfs.append(df)

    group = pl.DataFrame(names, schema=by, orient="row")

    return group, dfs


def with_index(data: pl.DataFrame, names: list[str], name: str) -> pl.DataFrame:
    if not names:
        return data.with_columns(pl.lit(0).alias(name))

    return data.join(
        data.select(names).unique(maintain_order=True).with_row_index(name),
        on=names,
        maintain_order="left",
    )


ROW_INDEX = "_row_index"
COL_INDEX = "_col_index"


class FacetFrame:
    group: pl.DataFrame
    data: list[pl.DataFrame]
    row: list[str]
    col: list[str]
    wrap: int | None

    def __init__(
        self,
        data: pl.DataFrame,
        row: Iterable[str] | None = None,
        col: Iterable[str] | None = None,
        wrap: int | None = None,
    ) -> None:
        self.row = _to_list(row)
        self.col = _to_list(col)
        self.group, self.data = group_by(data, *self.row, *self.col)

        self.group = with_index(self.group, self.row, ROW_INDEX)
        self.group = with_index(self.group, self.col, COL_INDEX)

        self.wrap = wrap

    def __len__(self) -> int:
        return len(self.group)

    @property
    def n_rows(self) -> int:
        return cast("int", self.group[ROW_INDEX].max()) + 1

    @property
    def n_cols(self) -> int:
        return cast("int", self.group[COL_INDEX].max()) + 1

    def keys(self) -> Iterable[tuple[int, int]]:
        yield from self.group.select(ROW_INDEX, COL_INDEX).iter_rows()
