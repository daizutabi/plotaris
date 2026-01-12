"""Provide data structures for handling grouped and faceted data.

The main classes, `GroupedData` and `FacetData`, are used to partition a
DataFrame into smaller chunks based on grouping variables, which is a core
operation for creating faceted plots (small multiples).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence


class GroupedData:
    """Group a DataFrame and provides integer indices for accessing groups.

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
            self.index = pl.DataFrame([{}])
            self.data = [data]
            return

        index, self.data = group_by(data, *by)

        for name, cs in self.mapping.items():
            index = with_index(index, cs, f"_{name}_index")

        named_exprs = {name: f"_{name}_index" for name in self.mapping}
        self.index = index.select(**named_exprs)

    def __len__(self) -> int:
        """Return the total number of data groups."""
        return len(self.index)

    def n_unique(self, name: str) -> int:
        """Return the number of unique values for a given dimension.

        Args:
            name: The name of the dimension (e.g., "row", "col").

        Returns:
            The number of unique values.
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
            name: The name of the dimension.
            named: If True, returns a dictionary mapping column names to
                values. Otherwise, returns a tuple of values.

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
                Otherwise, they will be tuples.

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
                Otherwise, they will be tuples.

        Returns:
            A list of dictionaries, where each dictionary is a group's label.
        """
        if named:
            return [self.get_label(i, named=True) for i in range(len(self))]

        return [self.get_label(i, named=False) for i in range(len(self))]


def to_tuple(values: str | Iterable[str] | None, /) -> tuple[str, ...]:
    """Convert a value to a tuple of strings.

    Handles None, a single string, or an iterable of strings.

    Args:
        values: The input value.

    Returns:
        A tuple of strings.
    """
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(values)


def group_by(data: pl.DataFrame, *by: str) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """Group a DataFrame and returns keys and data chunks.

    This is a wrapper around `polars.DataFrame.group_by` that formats the
    output into a DataFrame of group keys and a list of DataFrames.

    Args:
        data: The DataFrame to group.
        *by: The column names to group by.

    Returns:
        A tuple containing a DataFrame of group keys and a list of the
        group DataFrames.
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


@dataclass(frozen=True)
class Facet:
    row: int
    """The row index of the facet cell."""
    col: int
    """The column index of the facet cell."""
    data: pl.DataFrame
    """The DataFrame associated with this cell."""
    row_label: dict[str, Any]
    """The label for the row dimension."""
    col_label: dict[str, Any]
    """The label for the column dimension."""
    is_leftmost: bool
    """True if the cell is the leftmost occupied cell in its row."""
    is_topmost: bool
    """True if the cell is the topmost occupied cell in its column."""
    is_rightmost: bool
    """True if the cell is the rightmost occupied cell in its row."""
    is_bottommost: bool
    """True if the cell is the bottommost occupied cell in its column."""


class FacetData(GroupedData):
    """A specialized `GroupedData` for creating 2D facet grids.

    This class manages the mapping of data to a grid of subplots defined by
    row and column variables. It also handles wrapping a 1D facet layout
    into a 2D grid.
    """

    nrows: int
    """The number of rows in the facet grid."""
    ncols: int
    """The number of columns in the facet grid."""
    _lookup: dict[tuple[int, int], int]
    """A lookup from (row, col) coordinates to an index in self.data."""
    _min_col_for_row: dict[int, int]
    """A mapping from row index to the minimum occupied column index."""
    _max_col_for_row: dict[int, int]
    """A mapping from row index to the maximum occupied column index."""
    _min_row_for_col: dict[int, int]
    """A mapping from column index to the minimum occupied row index."""
    _max_row_for_col: dict[int, int]
    """A mapping from column index to the maximum occupied row index."""

    def __init__(
        self,
        data: pl.DataFrame,
        row: str | Iterable[str] | None = None,
        col: str | Iterable[str] | None = None,
        wrap: int | None = None,
    ) -> None:
        """Initialize the FacetData.

        Args:
            data: The input DataFrame.
            row: Column(s) to define the rows of the facet grid.
            col: Column(s) to define the columns of the facet grid.
            wrap: If provided, wraps a 1D facet grid (defined by `row` or
                `col`) into a 2D grid with this many columns (if `col` is
                set) or rows (if `row` is set).
        """
        super().__init__(data, {"row": row, "col": col})

        if not row and not col:
            # For a non-faceted plot, create a single 1x1 grid.
            self.index: pl.DataFrame = pl.DataFrame({"row": [0], "col": [0]})

        elif row and wrap:
            self.index = self.index.with_columns(
                (pl.col("row") % wrap).alias("row"),
                (pl.col("row") // wrap).alias("col"),
            )

        elif col and wrap:
            self.index = self.index.with_columns(
                (pl.col("col") // wrap).alias("row"),
                (pl.col("col") % wrap).alias("col"),
            )

        self.nrows = self.n_unique("row")
        self.ncols = self.n_unique("col")

        self._prepare()

    def _prepare(self) -> None:
        it = enumerate(self.index.rows())
        self._lookup = {(cast("int", r), cast("int", c)): i for i, (r, c) in it}

        self._min_col_for_row = {}
        self._max_col_for_row = {}
        self._min_row_for_col = {}
        self._max_row_for_col = {}

        for r, c in self._lookup:
            self._min_col_for_row[r] = min(c, self._min_col_for_row.get(r, c))
            self._max_col_for_row[r] = max(c, self._max_col_for_row.get(r, c))
            self._min_row_for_col[c] = min(r, self._min_row_for_col.get(c, r))
            self._max_row_for_col[c] = max(r, self._max_row_for_col.get(c, r))

    def cells(self, *, empty: bool = False) -> list[tuple[int, int]]:
        """Return the coordinates of cells in the facet grid.

        Args:
            empty: If True, includes grid cells that have no data. If False
                (default), returns only cells that are occupied by data.

        Returns:
            A list of (row, col) integer tuples.
        """
        occupied = list(self._lookup)

        if not empty:
            return occupied

        all_ = [(r, c) for r in range(self.nrows) for c in range(self.ncols)]
        return sorted(set(all_) - set(occupied))

    def get(self, row: int, col: int) -> pl.DataFrame | None:
        """Return the DataFrame for a specific cell.

        Args:
            row: The row index of the cell.
            col: The column index of the cell.

        Returns:
            The DataFrame corresponding to the cell at (row, col), or None
            if the cell is empty.
        """
        if (index := self._lookup.get((row, col))) is not None:
            return self.data[index]
        return None

    def is_leftmost(self, row: int, col: int) -> bool:
        """Check if a cell is the leftmost occupied cell in its row."""
        return self._min_col_for_row.get(row) == col

    def is_topmost(self, row: int, col: int) -> bool:
        """Check if a cell is the topmost occupied cell in its column."""
        return self._min_row_for_col.get(col) == row

    def is_rightmost(self, row: int, col: int) -> bool:
        """Check if a cell is the rightmost occupied cell in its row."""
        return self._max_col_for_row.get(row) == col

    def is_bottommost(self, row: int, col: int) -> bool:
        """Check if a cell is the bottommost occupied cell in its column."""
        return self._max_row_for_col.get(col) == row

    def iter_facets(
        self,
        *,
        leftmost: bool = False,
        topmost: bool = False,
        rightmost: bool = False,
        bottommost: bool = False,
    ) -> Iterator[Facet]:
        """Iterate over occupied facets, yielding coordinates, data, and labels.

        This iterator provides all necessary information for drawing each
        individual facet that contains data, optionally filtering by edge
        conditions.

        Args:
            leftmost: If True, only yields cells that are the leftmost
                occupied cell in their row.
            topmost: If True, only yields cells that are the topmost
                occupied cell in their column.
            rightmost: If True, only yields cells that are the rightmost
                occupied cell in their row.
            bottommost: If True, only yields cells that are the bottommost
                occupied cell in their column.

        Yields:
            Facet: A Facet dataclass instance.
        """
        # _lookup has (row, col) -> data_index
        # Iterate in the order the lookup was built (which is from self.index.rows())
        for (row, col), index in self._lookup.items():
            is_leftmost = self.is_leftmost(row, col)
            is_topmost = self.is_topmost(row, col)
            is_rightmost = self.is_rightmost(row, col)
            is_bottommost = self.is_bottommost(row, col)

            if leftmost and not is_leftmost:
                continue
            if topmost and not is_topmost:
                continue
            if rightmost and not is_rightmost:
                continue
            if bottommost and not is_bottommost:
                continue

            labels = self.get_label(index, named=True)
            yield Facet(
                row=row,
                col=col,
                data=self.data[index],
                row_label=labels["row"],
                col_label=labels["col"],
                is_leftmost=is_leftmost,
                is_topmost=is_topmost,
                is_rightmost=is_rightmost,
                is_bottommost=is_bottommost,
            )

    def __iter__(self) -> Iterator[Facet]:
        """Iterate over all occupied facets."""
        return self.iter_facets()
