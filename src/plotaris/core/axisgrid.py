from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, cast

import matplotlib.pyplot as plt

from .data import FacetData

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .data import Facet


class Plottable[T](Protocol):
    """A protocol for a callable that can be plotted on a matplotlib Axes.

    This defines the expected signature for plotting functions that can be passed
    to methods like `FacetGrid.map_facet`.
    """

    def __call__(
        self,
        arg: T,
        /,
        *args: Any,
        ax: Axes,
        **kwargs: Any,
    ) -> Any:
        """Define the signature for a plottable function.

        Args:
            arg: The primary data argument (e.g., a DataFrame or a Facet).
            *args: Additional positional arguments.
            ax: The matplotlib Axes object to plot on.
            **kwargs: Additional keyword arguments for the plotting function.
        """
        ...


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

        rcs = ((r, c) for r in range(self.nrows) for c in range(self.ncols))
        self.axes = {rc: cast("Axes", axes[*rc]) for rc in rcs}

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

    @property
    def left_axes(self) -> list[Axes]:
        """Get a list of the axes in the leftmost column of the grid."""
        return [a for row in range(self.nrows) if (a := self.get_axes(row, 0))]

    @property
    def top_axes(self) -> list[Axes]:
        """Get a list of the axes in the top row of the grid."""
        return [a for col in range(self.ncols) if (a := self.get_axes(0, col))]

    @property
    def right_axes(self) -> list[Axes]:
        """Get a list of the axes in the rightmost column of the grid."""
        col = self.ncols - 1
        return [a for row in range(self.nrows) if (a := self.get_axes(row, col))]

    @property
    def bottom_axes(self) -> list[Axes]:
        """Get a list of the axes in the bottom row of the grid."""
        row = self.nrows - 1
        return [a for col in range(self.ncols) if (a := self.get_axes(row, col))]

    @property
    def data_axes(self) -> list[Axes]:
        """Get a list of axes that have corresponding data."""
        cells = self.facet_data.cells().filter(has_data=True)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    @property
    def empty_axes(self) -> list[Axes]:
        """Get a list of axes that do not have corresponding data."""
        cells = self.facet_data.cells().filter(has_data=False)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    @property
    def leftmost_axes(self) -> list[Axes]:
        """Get a list of the leftmost axes that contain data in each row."""
        cells = self.facet_data.cells().filter(is_leftmost=True)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    @property
    def topmost_axes(self) -> list[Axes]:
        """Get a list of the topmost axes that contain data in each column."""
        cells = self.facet_data.cells().filter(is_topmost=True)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    @property
    def rightmost_axes(self) -> list[Axes]:
        """Get a list of the rightmost axes that contain data in each row."""
        cells = self.facet_data.cells().filter(is_rightmost=True)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    @property
    def bottommost_axes(self) -> list[Axes]:
        """Get a list of the bottommost axes that contain data in each column."""
        cells = self.facet_data.cells().filter(is_bottommost=True)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    def delaxes(self) -> Self:
        """Delete all empty axes from the figure.

        This is useful for cleaning up the layout when some facets do not
        contain data.

        Returns:
            The `FacetGrid` instance for method chaining.
        """
        for ax in self.empty_axes:
            self.figure.delaxes(ax)
        return self

    def __iter__(self) -> Iterator[Axes]:
        """Iterate over the axes that have data."""
        yield from self.data_axes

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

    def map_facet(
        self,
        func: Plottable[Facet],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Apply a plotting function to each facet.

        The function is called with the `Facet` object as the first argument.

        Args:
            func: A callable that accepts a `Facet` and a matplotlib `Axes`.
            *args: Additional positional arguments to pass to `func`.
            **kwargs: Additional keyword arguments to pass to `func`.

        Returns:
            The `FacetGrid` instance for method chaining.
        """
        for ax, facet in self.items():
            func(facet, *args, ax=ax, **kwargs)

        return self

    def map_dataframe(
        self,
        func: Plottable[pl.DataFrame],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Apply a plotting function to each facet's DataFrame.

        The function is called with the `polars.DataFrame` subset for each
        facet as the first argument.

        Args:
            func: A callable that accepts a `polars.DataFrame` and a
                matplotlib `Axes`.
            *args: Additional positional arguments to pass to `func`.
            **kwargs: Additional keyword arguments to pass to `func`.

        Returns:
            The `FacetGrid` instance for method chaining.
        """
        for ax, facet in self.items():
            func(facet.data, *args, ax=ax, **kwargs)

        return self

    def _display_(self) -> Figure:
        """Return the figure for display in IPython environments."""
        return self.figure
