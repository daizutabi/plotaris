from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable


ROW_INDEX = "_row_index"
COL_INDEX = "_col_index"


class FacetData:
    group: pl.DataFrame
    data: list[pl.DataFrame]
    row: list[str]
    col: list[str]

    def __init__(
        self,
        data: pl.DataFrame,
        row: Iterable[str] | None = None,
        col: Iterable[str] | None = None,
    ) -> None:
        self.row = to_list(row)
        self.col = to_list(col)
        self.group, self.data = group_by(data, *self.row, *self.col)

        self.group = with_index(self.group, self.row, ROW_INDEX)
        self.group = with_index(self.group, self.col, COL_INDEX)

    def __len__(self) -> int:
        return len(self.group)

    @property
    def n_rows(self) -> int:
        max_val = self.group[ROW_INDEX].max()
        return 0 if max_val is None else cast("int", max_val) + 1

    @property
    def n_cols(self) -> int:
        max_val = self.group[COL_INDEX].max()
        return 0 if max_val is None else cast("int", max_val) + 1

    def keys(self) -> Iterable[tuple[int, int]]:
        yield from self.group.select(ROW_INDEX, COL_INDEX).iter_rows()


def to_list(columns: Iterable[str] | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def group_by(data: pl.DataFrame, *by: str) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    if not by:
        return pl.DataFrame([{}]), [data]

    if data.is_empty():
        return pl.DataFrame(schema=by), []

    groups = list(data.group_by(*by, maintain_order=True))

    if not groups:
        return pl.DataFrame(schema=by), []

    names, dfs = zip(*groups, strict=True)
    group = pl.DataFrame(list(names), schema=by, orient="row")

    return group, list(dfs)


def with_index(data: pl.DataFrame, columns: list[str], name: str) -> pl.DataFrame:
    if not columns:
        return data.with_columns(pl.lit(0).alias(name))

    return data.join(
        data.select(columns).unique(maintain_order=True).with_row_index(name),
        on=columns,
        maintain_order="left",
    )
