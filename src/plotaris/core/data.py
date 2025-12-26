from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable


class GroupedData:
    dimensions: dict[str, list[str]]
    group: pl.DataFrame
    data: list[pl.DataFrame]

    def __init__(
        self,
        data: pl.DataFrame,
        dimensions: dict[str, str | Iterable[str]],
    ) -> None:
        self.dimensions = {name: to_list(cs) for name, cs in dimensions.items()}

        by = (c for cs in self.dimensions.values() for c in cs)
        self.group, self.data = group_by(data, *by)

        for name, cs in self.dimensions.items():
            self.group = with_index(self.group, cs, index_name(name))

    def __len__(self) -> int:
        return len(self.group)

    def n_unique(self, name: str) -> int:
        """Returns the number of unique values for a given dimension."""
        name = index_name(name)

        if name not in self.group.columns:
            return 0

        max_val = self.group[name].max()
        return 0 if max_val is None else cast("int", max_val) + 1


def index_name(name: str) -> str:
    return f"_{name}_index"


def to_list(columns: str | Iterable[str] | None) -> list[str]:
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
