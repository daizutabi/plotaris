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
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence


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
            self.index = pl.DataFrame({n: [0] for n in mapping} if mapping else [{}])
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
class Cell:
    row: int
    """The row index of the facet cell."""
    col: int
    """The column index of the facet cell."""
    has_data: bool
    """True if the cell has associated data."""
    is_leftmost: bool
    """True if the cell is the leftmost occupied cell in its row."""
    is_topmost: bool
    """True if the cell is the topmost occupied cell in its column."""
    is_rightmost: bool
    """True if the cell is the rightmost occupied cell in its row."""
    is_bottommost: bool
    """True if the cell is the bottommost occupied cell in its column."""

    def __iter__(self) -> Iterator[int]:
        yield self.row
        yield self.col


@dataclass(frozen=True)
class Facet(Cell):
    data: pl.DataFrame
    """The DataFrame associated with this cell."""
    row_label: dict[str, Any]
    """The label for the row dimension."""
    col_label: dict[str, Any]
    """The label for the column dimension."""


class Collection[T: Cell]:
    items: list[T]

    def __init__(self, items: Iterable[T]) -> None:
        self.items = list(items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def filter(
        self,
        predicate: Callable[[T], bool] | None = None,
        *,
        has_data: bool | None = None,
        is_leftmost: bool | None = None,
        is_topmost: bool | None = None,
        is_rightmost: bool | None = None,
        is_bottommost: bool | None = None,
    ) -> Collection[T]:
        """Filter the collection based on a predicate and optional conditions."""
        items = self.items
        if predicate:
            items = (item for item in items if predicate(item))
        if has_data is not None:
            items = (item for item in items if item.has_data is has_data)
        if is_leftmost is not None:
            items = (item for item in items if item.is_leftmost is is_leftmost)
        if is_topmost is not None:
            items = (item for item in items if item.is_topmost is is_topmost)
        if is_rightmost is not None:
            items = (item for item in items if item.is_rightmost is is_rightmost)
        if is_bottommost is not None:
            items = (item for item in items if item.is_bottommost is is_bottommost)
        return Collection(items)


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

        if row and wrap:
            self.index: pl.DataFrame = self.index.with_columns(
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

    def cell(self, row: int, col: int) -> Cell:
        """Return a Cell for the specified coordinates."""
        return Cell(
            row,
            col,
            has_data=(row, col) in self._lookup,
            is_leftmost=self._min_col_for_row.get(row) == col,
            is_topmost=self._min_row_for_col.get(col) == row,
            is_rightmost=self._max_col_for_row.get(row) == col,
            is_bottommost=self._max_row_for_col.get(col) == row,
        )

    def cells(self) -> Collection[Cell]:
        """Return a collection of all cells in the facet grid."""
        items = [self.cell(r, c) for r in range(self.nrows) for c in range(self.ncols)]
        return Collection(items)

    def facet(self, row: int, col: int) -> Facet | None:
        cell = self.cell(row, col)

        if not cell.has_data:
            return None

        index = self._lookup[row, col]
        labels = self.get_label(index, named=True)

        return Facet(
            row=cell.row,
            col=cell.col,
            has_data=cell.has_data,
            is_leftmost=cell.is_leftmost,
            is_topmost=cell.is_topmost,
            is_rightmost=cell.is_rightmost,
            is_bottommost=cell.is_bottommost,
            data=self.data[index],
            row_label=labels["row"],
            col_label=labels["col"],
        )

    def facets(self) -> Collection[Facet]:
        items = [f for c in self.cells() if (f := self.facet(c.row, c.col))]
        return Collection(items)

    def __iter__(self) -> Iterator[Facet]:
        """Iterate over all occupied facets."""
        yield from self.facets()

    def get(self, row: int, col: int) -> pl.DataFrame | None:
        """Return the DataFrame for a specific cell.

        Args:
            row: The row index of the cell.
            col: The column index of the cell.

        Returns:
            The DataFrame corresponding to the cell at (row, col), or None
            if the cell is empty.
        """
        if facet := self.facet(row, col):
            return facet.data

        return None
