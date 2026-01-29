from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

import polars as pl

from plotaris.utils import to_tuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


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


def _index_name(dimension: str | None = None, /) -> str:
    """Get the internal column name for a dimension's integer index."""
    if dimension:
        return f"_{dimension}_index"
    return "^_.*_index$"


def _flatten(*values: str | Iterable[str]) -> Iterator[str]:
    for value in values:
        if isinstance(value, str):
            yield value
        else:
            yield from value


class Group:
    mapping: dict[str, tuple[str, ...]]
    """A mapping from dimension names (e.g., "row", "col") to a tuple of column names."""  # noqa: E501
    data: list[pl.DataFrame]
    """A list of DataFrames, where each DataFrame is a subgroup of the original data."""

    _index: pl.DataFrame
    """A DataFrame where each row corresponds to a data group.

    It contains the original group keys (actual values from the DataFrame columns)
    and their corresponding integer indices for each dimension (e.g., "_row_index",
    "_col_index").
    """

    def __init__(self, data: pl.DataFrame, **columns: str | Iterable[str]) -> None:
        self.mapping = {dim: to_tuple(cols) for dim, cols in columns.items()}

        if data.is_empty():
            schema = [_index_name(d) for d in self.mapping]
            self._index = pl.DataFrame(schema=schema)
            self.data = []
            return

        index, self.data = group_by(data, *self.mapping.values())

        for dim, cols in self.mapping.items():
            index = with_index(index, cols, _index_name(dim))

        self._index = index

    def __getitem__(self, index: int) -> pl.DataFrame:
        """Get the data subgroup at a given index."""
        return self.data[index]

    def __len__(self) -> int:
        """Get the total number of subgroups."""
        return len(self.data)

    def __iter__(self) -> Iterator[pl.DataFrame]:
        """Iterate over all data subgroups."""
        return iter(self.data)

    def __contains__(self, dimension: str) -> bool:
        return dimension in self.mapping

    def indices(self, *dimension: str | Iterable[str]) -> pl.DataFrame:
        if not dimension:
            dimension = tuple(self.mapping)

        named = {d: _index_name(d) for d in _flatten(*dimension)}
        return self._index.select(**named)

    def keys(self, *dimension: str | Iterable[str]) -> pl.DataFrame:
        if not dimension:
            return self._index.select(pl.exclude(_index_name()))

        cs = [self.mapping[d] for d in _flatten(*dimension)]
        columns = sorted(set(chain.from_iterable(cs)))
        return self._index.select(columns)

    def dimension_keys(self) -> dict[str, pl.DataFrame]:
        return {dim: self.keys(dim) for dim in self.mapping}
