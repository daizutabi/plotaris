from __future__ import annotations

from typing import TYPE_CHECKING, Any, Concatenate, Literal, Self, cast

import matplotlib.pyplot as plt

from .data import FacetData

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .data import Facet


class FacetGrid:
    """Manage a grid of subplots for faceted plotting.

    This class creates a matplotlib Figure and a grid of Axes based on the
    faceting structure defined by `FacetData`. It provides methods to map
    plotting functions across the grid.
    """

    data: pl.DataFrame
    """The input DataFrame."""
    facet_data: FacetData
    """An instance of `FacetData` that manages the data partitioning for the grid."""
    figure: Figure
    """The main matplotlib `Figure` object."""
    axes: dict[tuple[int, int], Axes]
    """A dictionary mapping grid coordinates `(row, col)` to `Axes` objects."""

    def __init__(
        self,
        data: pl.DataFrame,
        row: str | Iterable[str] | None = None,
        col: str | Iterable[str] | None = None,
        *,
        wrap: int | None = None,
        sharex: bool | Literal["none", "all", "row", "col"] = True,
        sharey: bool | Literal["none", "all", "row", "col"] = True,
        constrained_layout: bool = True,
        subplot_kw: dict[str, Any] | None = None,
        gridspec_kw: dict[str, Any] | None = None,
        **fig_kw: Any,
    ) -> None:
        """Initialize the FacetGrid.

        Args:
            data: The input DataFrame for plotting.
            row: Column(s) used to create rows of subplots.
            col: Column(s) used to create columns of subplots.
            wrap: If specified, wrap a 1D facet definition into a 2D grid
                with this many columns.
            sharex: Whether to share the x-axis among subplots. See
                `matplotlib.pyplot.subplots` for details.
            sharey: Whether to share the y-axis among subplots. See
                `matplotlib.pyplot.subplots` for details.
            constrained_layout: Whether to use constrained layout for the figure.
            subplot_kw: Keyword arguments passed to `matplotlib.pyplot.subplots`
                for each subplot.
            gridspec_kw: Keyword arguments passed to the `GridSpec` constructor.
            **fig_kw: Additional keyword arguments passed to
                `matplotlib.pyplot.figure`.
        """
        self.data = data
        self.facet_data = FacetData(data, row, col, wrap)

        self.figure, axes = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
            self.nrows,
            self.ncols,
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
            constrained_layout=constrained_layout,
            subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw,
            **fig_kw,
        )

        cells = self.facet_data.cells()
        self.axes = {(c.row, c.col): cast("Axes", axes[c.row, c.col]) for c in cells}

        # rcs = ((r, c) for r in range(self.nrows) for c in range(self.ncols))
        # self.axes = {rc: cast("Axes", axes[*rc]) for rc in rcs}

    @property
    def nrows(self) -> int:
        """Get the number of rows in the facet grid."""
        return self.facet_data.nrows

    @property
    def ncols(self) -> int:
        """Get the number of columns in the facet grid."""
        return self.facet_data.ncols

    def get_axes(self, row: int, col: int) -> Axes | None:
        """Get the `Axes` object at a specific grid location.

        This method safely retrieves an axis, returning `None` if the axis has
        been deleted from the figure.

        Args:
            row: The row index of the axis.
            col: The column index of the axis.

        Returns:
            The `Axes` object at `(row, col)`, or `None` if it does not exist.
        """
        ax = self.axes.get((row, col))
        return ax if ax in self.figure.axes else None

    def select_axes(
        self,
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
    ) -> list[Axes]:
        """Selects a subset of `Axes` objects from the grid based on specified criteria.

        This method allows for flexible filtering of the grid's `Axes` objects
        by their properties, such as whether they contain data, their absolute
        position in the grid, or their relative position (e.g., leftmost, topmost).

        Args:
            row: If specified, select only axes in this absolute row index.
            col: If specified, select only axes in this absolute column index.
            has_data: If True, select only axes with associated data.
                If False, select only axes without associated data (empty cells).
                If None, do not filter by data presence.
            is_left: If True, select only axes that are in the leftmost column.
                If False, select axes not in the leftmost column.
                If None, do not filter by this property.
            is_top: If True, select only axes that are in the topmost row.
                If False, select axes not in the topmost row.
                If None, do not filter by this property.
            is_right: If True, select only axes that are in the rightmost column.
                If False, select axes not in the rightmost column.
                If None, do not filter by this property.
            is_bottom: If True, select only axes that are in the bottommost row.
                If False, select axes not in the bottommost row.
                If None, do not filter by this property.
            is_leftmost: If True, select only axes that are the leftmost
                occupied cell in their row. If False, select axes that are
                not the leftmost occupied cell in their row. If None, do not
                filter by this property.
            is_topmost: Similar to `is_leftmost`, but for the topmost
                occupied cell in its column.
            is_rightmost: Similar to `is_leftmost`, but for the rightmost
                occupied cell in its row.
            is_bottommost: Similar to `is_leftmost`, but for the bottommost
                occupied cell in its column.

        Returns:
            A list of `Axes` objects that match all specified criteria.
        """

        cells = self.facet_data.cells().filter(
            row=row,
            col=col,
            has_data=has_data,
            is_left=is_left,
            is_top=is_top,
            is_right=is_right,
            is_bottom=is_bottom,
            is_leftmost=is_leftmost,
            is_topmost=is_topmost,
            is_rightmost=is_rightmost,
            is_bottommost=is_bottommost,
        )
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    def delaxes(self) -> Self:
        """Delete all empty axes from the figure.

        This is useful for cleaning up the layout when some facets do not
        contain data.

        Returns:
            The `FacetGrid` instance for method chaining.
        """
        for ax in self.select_axes(has_data=False):
            self.figure.delaxes(ax)
        return self

    def __iter__(self) -> Iterator[Axes]:
        """Iterate over the axes that have data."""
        yield from self.select_axes(has_data=True)

    def items(self) -> Iterator[tuple[Axes, Facet]]:
        """Iterate over pairs of (`Axes`, `Facet`) for facets with data.

        This is analogous to `dict.items()`, providing both the plotting
        surface (`Axes`) and the data/metadata container (`Facet`).

        Yields:
            Tuples of (`Axes`, `Facet`).
        """
        for facet in self.facet_data:
            if ax := self.get_axes(facet.row, facet.col):
                yield ax, facet

    def map_facet[**P](
        self,
        func: Callable[Concatenate[Facet, P], Any],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Apply a plotting function to each facet.

        The function is called with the `Facet` object as the first argument.

        Args:
            func: A callable that accepts a `Facet` as the first argument.
            *args: Additional positional arguments to pass to `func`.
            **kwargs: Additional keyword arguments to pass to `func`.

        Returns:
            The `FacetGrid` instance for method chaining.
        """
        for ax, facet in self.items():
            plt.sca(ax)
            func(facet, *args, **kwargs)

        return self

    def map_dataframe[**P](
        self,
        func: Callable[Concatenate[pl.DataFrame, P], Any],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Apply a plotting function to each facet's DataFrame.

        The function is called with the `polars.DataFrame` subset for each
        facet as the first argument.

        Args:
            func: A callable that accepts a `polars.DataFrame` as the
                first argument.
            *args: Additional positional arguments to pass to `func`.
            **kwargs: Additional keyword arguments to pass to `func`.

        Returns:
            The `FacetGrid` instance for method chaining.
        """
        for ax, facet in self.items():
            plt.sca(ax)
            func(facet.data, *args, **kwargs)

        return self

    def _display_(self) -> Figure:
        """Return the figure for display in IPython environments."""
        return self.figure
