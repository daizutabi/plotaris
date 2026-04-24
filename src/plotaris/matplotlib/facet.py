from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

from .group import Group
from .label import Label

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    import polars as pl

    from .label import Format


@dataclass
class FacetLabel:
    row: Label
    """The label for the row dimension."""
    col: Label
    """The label for the column dimension."""
    dim: Literal["row", "col"] | None = None
    eq: str = "="
    sep: str = ", "

    def __post_init__(self) -> None:
        self.row.eq = self.col.eq = self.eq
        self.row.sep = self.col.sep = self.sep

    def format(
        self,
        formats: dict[str, Format | tuple[str, Format]] | None = None,
        /,
        **kwargs: Format | tuple[str, Format],
    ) -> str:
        labels: list[str] = []
        if self.dim != "col" and (label := self.row.format(formats, **kwargs)):
            labels.append(label)
        if self.dim != "row" and (label := self.col.format(formats, **kwargs)):
            labels.append(label)
        return self.sep.join(labels)


@dataclass(frozen=True)
class Facet:
    """Represent a single cell in the facet grid, which may or may not contain data."""

    data: pl.DataFrame | None = field(repr=False)
    """The DataFrame subset for this facet, or `None` if the facet is empty."""
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
    label: FacetLabel
    """The labels for the row and column dimensions."""

    def __iter__(self) -> Iterator[int]:
        """Allow unpacking the cell as `row, col`."""
        yield self.row
        yield self.col

    @property
    def has_data(self) -> bool:
        """True if this facet cell contains data."""
        return self.data is not None


class FacetCollection[T: Facet]:
    """A collection of `Facet` objects, providing methods for filtering and access."""

    _items: list[T]
    """The internal list of Facet objects in the collection."""
    _lookup: dict[tuple[int, int], T]
    """A dictionary mapping (row, col) coordinates to Facet objects for quick access."""

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
        """Return the number of facets in the collection."""
        return len(self._items)

    def __contains__(self, other: Any) -> bool:
        """Check if a facet or (row, col) tuple exists in the collection."""
        return other in self._lookup

    def __getitem__(self, rc: tuple[int, int]) -> T:
        """Get a specific Facet object by its (row, col) coordinates."""
        return self._lookup[rc]

    def get(self, row: int, col: int) -> T | None:
        """Get a specific Facet object by row and column, returning None if not found.

        Args:
            row: The row index of the facet.
            col: The column index of the facet.

        Returns:
            The Facet object at (row, col), or None if no such facet exists.
        """
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


class FacetData:
    group: Group
    """The underlying Group object that manages the data partitioning."""
    row: tuple[str, ...]
    """The column(s) used to define the rows of the facet grid."""
    col: tuple[str, ...]
    """The column(s) used to define the columns of the facet grid."""
    wrap: int | None
    """If provided, the number of columns to wrap a 1D facet grid into 2D."""
    nrows: int
    """The number of rows in the facet grid."""
    ncols: int
    """The number of columns in the facet grid."""
    facets: FacetCollection[Facet]
    """A collection of all facets in the grid, including empty ones."""
    _lookup: dict[tuple[int, int], int]
    """A mapping from (row, col) to the corresponding index in `group.data`."""

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
        if row and col:
            self.group = Group(data, row=row, col=col)
            idx = self.group.indices("row", "col")
            irows = enumerate(idx.rows())
            self._lookup = {(r, c): i for i, (r, c) in irows}

        elif row and not col:
            self.group = Group(data, row=row)
            idx = self.group.indices("row")
            irows = enumerate(idx.rows())
            if wrap:
                self._lookup = {divmod(r, wrap)[::-1]: i for i, (r,) in irows}
            else:
                self._lookup = {(r, 0): i for i, (r,) in irows}

        elif col and not row:
            self.group = Group(data, col=col)
            idx = self.group.indices("col")
            irows = enumerate(idx.rows())
            if wrap:
                self._lookup = {divmod(c, wrap): i for i, (c,) in irows}
            else:
                self._lookup = {(0, c): i for i, (c,) in irows}

        else:
            self.group = Group(data)
            self._lookup = {(0, 0): 0}

        self.row = self.group.mapping.get("row", ())
        self.col = self.group.mapping.get("col", ())
        self.wrap = wrap
        self.nrows = max(r for (r, _) in self._lookup) + 1
        self.ncols = max(c for (_, c) in self._lookup) + 1

        self.facets = _create_facets(self)

    def index(self, row: int, col: int) -> int | None:
        """Get the internal Group index for a specific facet cell.

        Args:
            row: The row index of the cell.
            col: The column index of the cell.

        Returns:
            The integer index within the Group's data list, or None if the cell is empty.
        """  # noqa: E501
        return self._lookup.get((row, col), None)

    def coordinates(self) -> Iterator[tuple[int, int]]:
        """Iterate over the (row, col) coordinates of all occupied cells."""
        yield from self._lookup

    def __getitem__(self, rc: tuple[int, int]) -> Facet:
        """Get a specific Facet object by its (row, col) coordinates."""
        return self.facets[rc]

    def __iter__(self) -> Iterator[Facet]:
        """Iterate over all facets in the grid."""
        yield from self.facets

    def data(self, row: int, col: int) -> pl.DataFrame | None:
        """Get the DataFrame for a specific cell.

        Args:
            row: The row index of the cell.
            col: The column index of the cell.

        Returns:
            The DataFrame corresponding to the cell at (row, col), or `None`
            if the cell is empty.
        """
        return self[row, col].data


def _create_facets(facet_data: FacetData) -> FacetCollection[Facet]:
    """Internal helper to create a FacetCollection from FacetData."""
    min_col_for_row: dict[int, int] = {}
    max_col_for_row: dict[int, int] = {}
    min_row_for_col: dict[int, int] = {}
    max_row_for_col: dict[int, int] = {}

    for r, c in facet_data.coordinates():
        min_col_for_row[r] = min(c, min_col_for_row.get(r, c))
        max_col_for_row[r] = max(c, max_col_for_row.get(r, c))
        min_row_for_col[c] = min(r, min_row_for_col.get(c, r))
        max_row_for_col[c] = max(r, max_row_for_col.get(c, r))

    facets = [
        _create_facet(
            facet_data,
            row,
            col,
            min_col_for_row,
            min_row_for_col,
            max_col_for_row,
            max_row_for_col,
        )
        for row in range(facet_data.nrows)
        for col in range(facet_data.ncols)
    ]

    return FacetCollection(facets)


def _create_facet(
    facet_data: FacetData,
    row: int,
    col: int,
    min_col_for_row: dict[int, int],
    min_row_for_col: dict[int, int],
    max_col_for_row: dict[int, int],
    max_row_for_col: dict[int, int],
) -> Facet:
    """Internal helper to create a single Facet object."""
    index = facet_data.index(row, col)

    if index is None:
        data = None
        label = FacetLabel(Label(), Label())

    else:
        data = facet_data.group[index]
        labels = facet_data.group.labels(index)
        row_label = Label(labels.get("row", {}))
        col_label = Label(labels.get("col", {}))
        label = FacetLabel(row_label, col_label)

    return Facet(
        data,
        row,
        col,
        is_left=(col == 0),
        is_top=(row == 0),
        is_right=(col == facet_data.ncols - 1),
        is_bottom=(row == facet_data.nrows - 1),
        is_leftmost=min_col_for_row.get(row) == col,
        is_topmost=min_row_for_col.get(col) == row,
        is_rightmost=max_col_for_row.get(row) == col,
        is_bottommost=max_row_for_col.get(col) == row,
        label=label,
    )
