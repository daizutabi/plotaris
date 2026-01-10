from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class GroupedData:
    mapping: dict[str, tuple[str, ...]]
    group: pl.DataFrame
    data: list[pl.DataFrame]

    def __init__(
        self,
        data: pl.DataFrame,
        mapping: Mapping[str, str | Iterable[str] | None],
    ) -> None:
        self.mapping = {name: to_tuple(cs) for name, cs in mapping.items()}

        if data.is_empty():
            self.group = pl.DataFrame({n: [] for n in self.mapping})
            self.data = []
            return

        by = sorted({c for cs in self.mapping.values() for c in cs})

        if not by:
            self.group = pl.DataFrame([{}])
            self.data = [data]
            return

        group, self.data = group_by(data, *by)

        for name, cs in self.mapping.items():
            group = with_index(group, cs, f"_{name}_index")

        named_exprs = {name: f"_{name}_index" for name in self.mapping}
        self.group = group.select(**named_exprs)

    def __len__(self) -> int:
        return len(self.group)

    @overload
    def item(
        self,
        index: int,
        name: str,
        *,
        named: Literal[False] = ...,
    ) -> tuple[Any, ...]: ...

    @overload
    def item(
        self,
        index: int,
        name: str,
        *,
        named: Literal[True],
    ) -> dict[str, Any]: ...

    def item(
        self,
        index: int,
        name: str,
        *,
        named: bool = False,
    ) -> tuple[Any, ...] | dict[str, Any]:
        columns = self.mapping[name]
        df = self.data[index].select(columns)

        if len(df) == 0:
            return {} if named else ()

        return df.row(0, named=named)

    def n_unique(self, name: str) -> int:
        """Returns the number of unique values for a given dimension."""
        if name not in self.group.columns:
            return 0

        max_val = self.group[name].max()
        return 0 if max_val is None else cast("int", max_val) + 1


def to_tuple(values: str | Iterable[str] | None, /) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(values)


def group_by(data: pl.DataFrame, *by: str) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    groups = list(data.group_by(*by, maintain_order=True))

    if not groups:
        return pl.DataFrame(schema=by), []

    names, dataframes = zip(*groups, strict=True)
    group = pl.DataFrame(list(names), schema=by, orient="row")

    return group, list(dataframes)


def with_index(data: pl.DataFrame, columns: Sequence[str], name: str) -> pl.DataFrame:
    if not columns:
        return data.with_columns(pl.lit(0).alias(name))

    return data.join(
        data.select(columns).unique(maintain_order=True).with_row_index(name),
        on=columns,
        maintain_order="left",
    )


class FacetData(GroupedData):
    nrows: int
    ncols: int

    def __init__(
        self,
        data: pl.DataFrame,
        row: str | Iterable[str] | None = None,
        col: str | Iterable[str] | None = None,
        wrap: int | None = None,
    ) -> None:
        super().__init__(data, {"row": row, "col": col})

        if not row and not col:
            self.group: pl.DataFrame = pl.DataFrame({"row": [0], "col": [0]})

        elif row and wrap:
            self.group = self.group.with_columns(
                (pl.col("row") % wrap).alias("row"),
                (pl.col("row") // wrap).alias("col"),
            )

        elif col and wrap:
            self.group = self.group.with_columns(
                (pl.col("col") // wrap).alias("row"),
                (pl.col("col") % wrap).alias("col"),
            )

        self.nrows = self.n_unique("row")
        self.ncols = self.n_unique("col")
