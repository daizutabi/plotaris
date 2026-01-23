from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

import polars as pl

from .group import GroupedData

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


@dataclass(frozen=True)
class Facet:
    """Represent a single cell in the facet grid, which may or may not contain data."""

    row: int
    """The row index of the facet cell."""
    col: int
    """The column index of the facet cell."""
    is_left: bool
    """True if the cell is in the leftmost column (col = 0) of the grid."""
    is_top: bool
    """True if the cell is in the topmost row (row = 0) of the grid."""
    is_right: bool
    """True if the cell is in the rightmost column of the grid."""
    is_bottom: bool
    """True if the cell is in the bottommost row of the grid."""
    is_leftmost: bool
    """True if the cell is the leftmost occupied cell in its row."""
    is_topmost: bool
    """True if the cell is the topmost occupied cell in its column."""
    is_rightmost: bool
    """True if the cell is the rightmost occupied cell in its row."""
    is_bottommost: bool
    """True if the cell is the bottommost occupied cell in its column."""
    data: pl.DataFrame | None
    """The DataFrame subset for this facet, or `None` if the facet is empty."""
    row_label: dict[str, Any]
    """The label for the row dimension."""
    col_label: dict[str, Any]
    """The label for the column dimension."""

    def __iter__(self) -> Iterator[int]:
        """Allow unpacking the cell as `row, col`."""
        yield self.row
        yield self.col

    @property
    def has_data(self) -> bool:
        return self.data is not None


class FacetCollection[T: Facet]:
    """A collection of `Facet` objects, providing methods for filtering and access."""

    _items: list[T]
    _lookup: dict[tuple[int, int], T]

    def __init__(self, items: Iterable[T]) -> None:
        """Initialize the Collection.

        Args:
            items: An iterable of items to be stored in the collection.
        """
        self._items = list(items)
        self._lookup = {(i.row, i.col): i for i in self._items}

    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the items in the collection."""
        return iter(self._items)

    def __len__(self) -> int:
        """Return the number of items in the collection."""
        return len(self._items)

    def __contains__(self, other: Any) -> bool:
        return other in self._lookup

    def get(self, row: int, col: int) -> T | None:
        return self._lookup.get((row, col))

    def filter(
        self,
        predicate: Callable[[T], bool] | None = None,
        *,
        row: int | None = None,
        col: int | None = None,
        has_data: bool | None = None,
        is_left: bool | None = None,
        is_top: bool | None = None,
        is_right: bool | None = None,
        is_bottom: bool | None = None,
        is_leftmost: bool | None = None,
        is_topmost: bool | None = None,
        is_rightmost: bool | None = None,
        is_bottommost: bool | None = None,
    ) -> Self:
        """Filter the collection based on a predicate and/or attributes.

        Args:
            predicate: A callable that returns True for items to be included.
            row: If specified, select only facets in this absolute row index.
            col: If specified, select only facets in this absolute column index.
            has_data: Filter by the `has_data` attribute.
            is_left: Filter by the `is_left` attribute.
            is_top: Filter by the `is_top` attribute.
            is_right: Filter by the `is_right` attribute.
            is_bottom: Filter by the `is_bottom` attribute.
            is_leftmost: Filter by the `is_leftmost` attribute.
            is_topmost: Filter by the `is_topmost` attribute.
            is_rightmost: Filter by the `is_rightmost` attribute.
            is_bottommost: Filter by the `is_bottommost` attribute.

        Returns:
            A new `Collection` containing only the filtered items.
        """
        items = iter(self._items)
        if predicate:
            items = (item for item in items if predicate(item))
        if row is not None:
            items = (item for item in items if item.row == row)
        if col is not None:
            items = (item for item in items if item.col == col)
        if has_data is not None:
            items = (item for item in items if item.has_data is has_data)
        if is_left is not None:
            items = (item for item in items if item.is_left is is_left)
        if is_top is not None:
            items = (item for item in items if item.is_top is is_top)
        if is_right is not None:
            items = (item for item in items if item.is_right is is_right)
        if is_bottom is not None:
            items = (item for item in items if item.is_bottom is is_bottom)
        if is_leftmost is not None:
            items = (item for item in items if item.is_leftmost is is_leftmost)
        if is_topmost is not None:
            items = (item for item in items if item.is_topmost is is_topmost)
        if is_rightmost is not None:
            items = (item for item in items if item.is_rightmost is is_rightmost)
        if is_bottommost is not None:
            items = (item for item in items if item.is_bottommost is is_bottommost)
        return self.__class__(items)


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
    _min_col_for_row: dict[int, int]
    _max_col_for_row: dict[int, int]
    _min_row_for_col: dict[int, int]
    _max_row_for_col: dict[int, int]

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
                `col`) into a 2D grid with this many columns.
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
        """Compute and cache lookup tables for grid metadata."""
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

    def facet(self, row: int, col: int) -> Facet:
        """Get a `Facet` object for the specified grid coordinates.

        Args:
            row: The row index of the cell.
            col: The column index of the cell.

        Returns:
            A `Facet` instance with metadata for the specified location.
        """
        if (row, col) in self._lookup:
            index = self._lookup[row, col]
            data = self.data[index]
            labels = self.get_label(index, named=True)
            row_label = labels["row"]
            col_label = labels["col"]
        else:
            data = None
            row_label = {}
            col_label = {}

        return Facet(
            row,
            col,
            is_left=(col == 0),
            is_top=(row == 0),
            is_right=(col == self.ncols - 1),
            is_bottom=(row == self.nrows - 1),
            is_leftmost=self._min_col_for_row.get(row) == col,
            is_topmost=self._min_row_for_col.get(col) == row,
            is_rightmost=self._max_col_for_row.get(row) == col,
            is_bottommost=self._max_row_for_col.get(col) == row,
            data=data,
            row_label=row_label,
            col_label=col_label,
        )

    def facets(self) -> FacetCollection[Facet]:
        """Get a collection of all facets in the grid, including empty ones.

        Returns:
            A `FacetCollection` containing a `Facet` object for every
            position in the grid.
        """
        items = [self.facet(r, c) for r in range(self.nrows) for c in range(self.ncols)]
        return FacetCollection(items)

    def __iter__(self) -> Iterator[Facet]:
        """Iterate over all facets in the grid."""
        yield from self.facets()

    def get(self, row: int, col: int) -> pl.DataFrame | None:
        """Get the DataFrame for a specific cell.

        Args:
            row: The row index of the cell.
            col: The column index of the cell.

        Returns:
            The DataFrame corresponding to the cell at (row, col), or `None`
            if the cell is empty.
        """
        return self.facet(row, col).data
