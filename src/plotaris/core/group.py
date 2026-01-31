from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, overload

import polars as pl

from plotaris.utils import to_tuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


def group_by(
    data: pl.DataFrame,
    *columns: str | Iterable[str],
) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """Groups a DataFrame and returns an index and a list of dataframes.

    Args:
        data: The DataFrame to group.
        *columns: Column names to group by. Can be specified as individual
            strings or iterables of strings.

    Returns:
        A tuple containing:
            - A DataFrame of unique group keys.
            - A list of DataFrames, where each corresponds to a group.
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
    """Adds a column with a unique integer index for a set of columns.

    This is equivalent to a multi-column "factorize" operation. It finds the
    unique combinations of values in `columns`, assigns an integer index to
    each unique combination, and joins this index back to the original
    DataFrame.

    Args:
        data: The DataFrame to add the index column to.
        columns: A sequence of column names to create the index from.
        name: The name for the new index column.

    Returns:
        A new DataFrame with the added index column.
    """
    if not columns:
        return data.with_columns(pl.lit(None).alias(name))

    return data.join(
        data.select(columns).unique(maintain_order=True).with_row_index(name),
        on=columns,
        maintain_order="left",
    )


def _index_name(dimension: str | None = None, /) -> str:
    """Gets the internal column name for a dimension's integer index.

    Args:
        dimension: The name of the dimension (e.g., "row", "col").

    Returns:
        The internal column name for the index, or a regex pattern to match all
        index columns if no dimension is provided.
    """
    if dimension:
        return f"_{dimension}_index"
    return "^_.*_index$"


def _flatten(*values: str | Iterable[str]) -> Iterator[str]:
    """Flattens an iterable of strings and other iterables of strings.

    Args:
        *values: One or more strings or iterables of strings.

    Yields:
        Individual strings.
    """
    for value in values:
        if isinstance(value, str):
            yield value
        else:
            yield from value


class Group:
    """Manages data grouped by specified dimensions.

    This class splits a DataFrame into subgroups based on the unique values in
    columns associated with dimensions (e.g., "row", "col"). It assigns integer
    indices to each group for easy access and manipulation.

    Attributes:
        mapping: A dictionary mapping dimension names (e.g., "row", "col") to
            tuples of column names.
        data: A list of DataFrames, where each DataFrame is a subgroup of the
            original data.
    """

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
        """Initializes the Group object.

        Args:
            data: The DataFrame to group.
            **columns: Keyword arguments where keys are dimension names and values
                are the column names to group by for that dimension. Column
                names can be a single string or an iterable of strings.
        """
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
        """Gets the data subgroup at a given index."""
        return self.data[index]

    def __len__(self) -> int:
        """Gets the total number of subgroups."""
        return len(self.data)

    def __iter__(self) -> Iterator[pl.DataFrame]:
        """Iterates over all data subgroups."""
        return iter(self.data)

    def __contains__(self, dimension: str) -> bool:
        """Checks if the given dimension exists in the group mapping.

        Args:
            dimension: The name of the dimension to check.

        Returns:
            True if the dimension exists, False otherwise.
        """
        return dimension in self.mapping

    def indices(self, *dimension: str | Iterable[str]) -> pl.DataFrame:
        """Gets the integer indices for the specified dimensions.

        If no dimensions are provided, it returns indices for all dimensions.

        Args:
            *dimension: The names of the dimensions to get indices for.

        Returns:
            A DataFrame containing the integer index columns for the requested
            dimensions.
        """
        if not dimension:
            dimension = tuple(self.mapping)

        named = {d: _index_name(d) for d in _flatten(*dimension)}
        return self._index.select(**named)

    def keys(self, *dimension: str | Iterable[str]) -> pl.DataFrame:
        """Gets the group key values for the specified dimensions.

        Group keys are the unique combinations of values in the original columns
        used for grouping. If no dimensions are provided, it returns keys for all
        dimensions.

        Args:
            *dimension: The names of the dimensions to get keys for.

        Returns:
            A DataFrame containing the key columns for the requested dimensions.
        """
        if not dimension:
            return self._index.select(pl.exclude(_index_name()))

        cs = [self.mapping[d] for d in _flatten(*dimension)]
        columns = sorted(set(chain.from_iterable(cs)))
        return self._index.select(columns)

    def dimension_keys(self) -> dict[str, pl.DataFrame]:
        """Gets a dictionary mapping each dimension to its keys.

        Returns:
            A dictionary where keys are dimension names and values are DataFrames
            containing the keys for that dimension.
        """
        return {dim: self.keys(dim) for dim in self.mapping}

    @overload
    def labels(self, index: int) -> dict[str, dict[str, Any]]: ...

    @overload
    def labels(self, index: None = None) -> list[dict[str, dict[str, Any]]]: ...

    def labels(
        self,
        index: int | None = None,
    ) -> dict[str, dict[str, Any]] | list[dict[str, dict[str, Any]]]:
        """Gets the labels for one or all data groups.

        Each group is defined by a unique combination of values from the columns
        specified in the dimensions. This method retrieves these values.

        Args:
            index: The integer index of a specific group. If None (default),
                labels for all groups are returned.

        Returns:
            If `index` is an integer, returns a dictionary mapping dimension names
            to the key-value pairs for that group.
            Example: `{"row": {"col_a": 1}, "col": {"col_b": "x"}}`

            If `index` is None, returns a list of these dictionaries, with one
            entry for each group.
        """
        if index is not None:
            return {dim: self.keys(dim).row(index, named=True) for dim in self.mapping}

        dim_keys = self.dimension_keys()
        return [
            {dim: keys.row(i, named=True) for dim, keys in dim_keys.items()}
            for i in range(len(self))
        ]
