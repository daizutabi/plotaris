from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class GroupedData:
    """Group a DataFrame and provide integer indices for accessing groups.

    This class takes a DataFrame and a mapping of dimension names (e.g., "row",
    "col") to column names in the DataFrame. It groups the data by these
    columns and generates a unique integer index for each combination of
    values in each dimension.
    """

    mapping: dict[str, tuple[str, ...]]
    """A dictionary mapping dimension names to a tuple of column names."""
    index: pl.DataFrame
    """A DataFrame where each row corresponds to a data group.

    Columns are dimension names (e.g., "row", "col") and values are
    the generated integer indices for that dimension.
    """
    data: list[pl.DataFrame]
    """A list of DataFrames, where each DataFrame is a chunk of the original data."""

    def __init__(
        self,
        data: pl.DataFrame,
        mapping: Mapping[str, str | Iterable[str] | None],
    ) -> None:
        """Initialize the GroupedData.

        This method groups the data and creates an integer-based index for
        each specified dimension.

        Args:
            data: The input DataFrame to be grouped.
            mapping: A dictionary that defines the grouping. Keys are
                dimension names (e.g., "row", "col"), and values are the
                column names from the DataFrame to group by for that
                dimension.
        """
        self.mapping = {name: to_tuple(cs) for name, cs in mapping.items()}

        if data.is_empty():
            self.index = pl.DataFrame({n: [] for n in self.mapping})
            self.data = []
            return

        by = sorted({c for cs in self.mapping.values() for c in cs})

        if not by:
            self.index = pl.DataFrame({n: [0] for n in mapping} if mapping else [{}])
            self.data = [data]
            return

        index, self.data = group_by(data, *by)

        for name, cs in self.mapping.items():
            index = with_index(index, cs, f"_{name}_index")

        named_exprs = {name: f"_{name}_index" for name in self.mapping}
        self.index = index.select(**named_exprs)

    def __len__(self) -> int:
        """Get the total number of data groups.

        Returns:
            The number of groups.
        """
        return len(self.index)

    def n_unique(self, name: str) -> int:
        """Get the number of unique values for a given dimension.

        Args:
            name: The name of the dimension (e.g., "row", "col").

        Returns:
            The number of unique values in the dimension.
        """
        if name not in self.index.columns:
            return 0

        max_val = self.index[name].max()
        return 0 if max_val is None else cast("int", max_val) + 1

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
        """Retrieve the grouping values for a specific group and dimension.

        Args:
            index: The integer index of the data group.
            name: The name of the dimension (e.g., "row", "col").
            named: If True, return a dictionary mapping column names to
                values. If False, return a tuple of values.

        Returns:
            A tuple or dictionary of the grouping values.
        """
        columns = self.mapping[name]
        df = self.data[index].select(columns)

        if len(df) == 0:
            return {} if named else ()

        return df.row(0, named=named)

    @overload
    def get_label(
        self,
        index: int,
        *,
        named: Literal[False] = ...,
    ) -> dict[str, tuple[Any, ...]]: ...

    @overload
    def get_label(
        self,
        index: int,
        *,
        named: Literal[True],
    ) -> dict[str, dict[str, Any]]: ...

    def get_label(
        self,
        index: int,
        *,
        named: bool = False,
    ) -> dict[str, tuple[Any, ...]] | dict[str, dict[str, Any]]:
        """Retrieve all grouping values for a single data group.

        Args:
            index: The integer index of the data group.
            named: If True, the values for each dimension will be dictionaries.
                If False, they will be tuples.

        Returns:
            A dictionary mapping dimension names to their grouping values.
        """
        if named:
            return {n: self.item(index, n, named=True) for n in self.mapping}

        return {n: self.item(index, n, named=False) for n in self.mapping}

    @overload
    def get_labels(
        self,
        *,
        named: Literal[False] = ...,
    ) -> list[dict[str, tuple[Any, ...]]]: ...

    @overload
    def get_labels(
        self,
        *,
        named: Literal[True],
    ) -> list[dict[str, dict[str, Any]]]: ...

    def get_labels(
        self,
        *,
        named: bool = False,
    ) -> list[dict[str, tuple[Any, ...]]] | list[dict[str, dict[str, Any]]]:
        """Retrieve the labels for all data groups.

        Args:
            named: If True, the values for each dimension will be dictionaries.
                If False, they will be tuples.

        Returns:
            A list of dictionaries, where each dictionary is a group's label.
        """
        if named:
            return [self.get_label(i, named=True) for i in range(len(self))]

        return [self.get_label(i, named=False) for i in range(len(self))]


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


def group_by(data: pl.DataFrame, *by: str) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """Group a DataFrame and return keys and data chunks.

    This is a wrapper around `polars.DataFrame.group_by` that formats the
    output into a DataFrame of group keys and a list of DataFrames, which is
    a more convenient structure for subsequent processing.

    Args:
        data: The DataFrame to group.
        by: The column names to group by.

    Returns:
        A tuple containing:
            - A DataFrame of unique group keys.
            - A list of DataFrames, each corresponding to a group.
    """
    groups = list(data.group_by(*by, maintain_order=True))

    if not groups:
        return pl.DataFrame(schema=by), []

    names, dataframes = zip(*groups, strict=True)
    index = pl.DataFrame(list(names), schema=by, orient="row")

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
        return data.with_columns(pl.lit(0).alias(name))

    return data.join(
        data.select(columns).unique(maintain_order=True).with_row_index(name),
        on=columns,
        maintain_order="left",
    )
