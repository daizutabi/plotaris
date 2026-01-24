from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


def to_tuple(values: str | Iterable[str] | None, /) -> tuple[str, ...]:
    """Convert a value to a tuple of strings.

    This utility function handles None, a single string, or an iterable of
    strings and ensures the output is always a tuple of strings.

    Args:
        values: The input value to convert.

    Returns:
        A tuple of strings.
    """
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(values)


def group_by(
    data: pl.DataFrame,
    *columns: str | Iterable[str],
) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """Group a DataFrame and return index and a list of data.

    Args:
        data: The DataFrame to group.
        by: The column names to group by.

    Returns:
        A tuple containing:
            - A DataFrame of unique group keys.
            - A list of DataFrames, each corresponding to a group.
    """
    cs = [[c] if isinstance(c, str) else c for c in columns]
    by = sorted(set(chain.from_iterable(cs)))

    if not by:
        return pl.DataFrame([{}]), [data]

    if data.is_empty():
        return pl.DataFrame(schema=by), []

    groups = data.group_by(*by, maintain_order=True)
    keys, dataframes = zip(*groups, strict=True)
    index = pl.DataFrame(keys, schema=by, orient="row")

    return index, list(dataframes)


def with_index(data: pl.DataFrame, columns: Sequence[str], name: str) -> pl.DataFrame:
    """Add a column with a unique integer index for a set of columns.

    This is equivalent to a multi-column "factorize" operation. It finds the
    unique combinations of values in `columns`, assigns an integer index to
    each unique combination, and joins this index back to the original
    DataFrame.

    Args:
        data: The DataFrame to add the index column to.
        columns: A sequence of column names to create the index from.
        name: The name for the new index column.

    Returns:
        The DataFrame with the new index column.
    """
    if not columns:
        return data.with_columns(pl.lit(None).alias(name))

    return data.join(
        data.select(columns).unique(maintain_order=True).with_row_index(name),
        on=columns,
        maintain_order="left",
    )


class Group:
    index: pl.DataFrame
    """A DataFrame where each row corresponds to a data group.

    It contains the original group keys (actual values from the DataFrame columns)
    and their corresponding integer indices for each dimension (e.g., "_row_index",
    "_col_index").
    """
    mapping: dict[str, tuple[str, ...]]
    """A mapping from dimension names (e.g., "row", "col") to a tuple of column names."""  # noqa: E501
    data: list[pl.DataFrame]
    """A list of DataFrames, where each DataFrame is a subgroup of the original data."""

    def __init__(self, data: pl.DataFrame, **columns: str | Iterable[str]) -> None:
        self.mapping = {dim: to_tuple(cols) for dim, cols in columns.items()}

        if data.is_empty():
            self.index = pl.DataFrame({n: [] for n in self.mapping})
            self.data = []
            return

        index, self.data = group_by(data, *self.mapping.values())

        for dim, cols in self.mapping.items():
            index = with_index(index, cols, self._get_index_column(dim))

        self.index = index

    def _get_index_column(self, dimension: str, /) -> str:
        return f"_{dimension}_index"

    def __getitem__(self, index: int) -> pl.DataFrame:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[pl.DataFrame]:
        return iter(self.data)

    def n_unique(self, dimension: str, /) -> int:
        """Get the number of unique values for a given dimension.

        Args:
            dimension: The name of the dimension (e.g., "row", "col").

        Returns:
            The number of unique values in the dimension.
        """
        return self.index[self._get_index_column(dimension)].n_unique()

    @overload
    def item(
        self,
        index: int,
        dimension: str,
        *,
        named: Literal[False] = ...,
    ) -> tuple[Any, ...]: ...

    @overload
    def item(
        self,
        index: int,
        dimension: str,
        *,
        named: Literal[True],
    ) -> dict[str, Any]: ...

    def item(
        self,
        index: int,
        dimension: str,
        *,
        named: bool = False,
    ) -> tuple[Any, ...] | dict[str, Any]:
        if columns := self.mapping[dimension]:
            return self.index.select(columns).row(index, named=named)
        return {} if named else ()

    @overload
    def items(
        self,
        dimension: str,
        *,
        named: Literal[False] = ...,
    ) -> list[tuple[Any, ...]]: ...

    @overload
    def items(
        self,
        dimension: str,
        *,
        named: Literal[True],
    ) -> list[dict[str, Any]]: ...

    def items(
        self,
        dimension: str,
        *,
        named: bool = False,
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        if named:
            return [self.item(i, dimension, named=True) for i in range(len(self))]

        return [self.item(i, dimension, named=False) for i in range(len(self))]

    @overload
    def dimension(
        self,
        index: int,
        *,
        named: Literal[False] = ...,
    ) -> dict[str, tuple[Any, ...]]: ...

    @overload
    def dimension(
        self,
        index: int,
        *,
        named: Literal[True],
    ) -> dict[str, dict[str, Any]]: ...

    def dimension(
        self,
        index: int,
        *,
        named: bool = False,
    ) -> dict[str, tuple[Any, ...]] | dict[str, dict[str, Any]]:
        if named:
            return {n: self.item(index, n, named=True) for n in self.mapping}

        return {n: self.item(index, n, named=False) for n in self.mapping}

    @overload
    def dimensions(
        self,
        *,
        named: Literal[False] = ...,
    ) -> list[dict[str, tuple[Any, ...]]]: ...

    @overload
    def dimensions(
        self,
        *,
        named: Literal[True],
    ) -> list[dict[str, dict[str, Any]]]: ...

    def dimensions(
        self,
        *,
        named: bool = False,
    ) -> list[dict[str, tuple[Any, ...]]] | list[dict[str, dict[str, Any]]]:
        if named:
            return [self.dimension(i, named=True) for i in range(len(self))]

        return [self.dimension(i, named=False) for i in range(len(self))]
