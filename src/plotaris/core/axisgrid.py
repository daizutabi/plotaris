from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Concatenate, Literal, Self

import matplotlib.pyplot as plt

from .data import Facet, FacetCollection, FacetData

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from numpy.typing import NDArray


@dataclass(frozen=True)
class FacetAxes(Facet):
    """Represents a `Facet` that is paired with its corresponding `Axes` object."""

    axes: Axes
    """The `Axes` object for this facet."""

    @classmethod
    def from_facet(
        cls,
        facet: Facet,
        axes: Axes,
    ) -> Self:
        kwargs = {f.name: getattr(facet, f.name) for f in fields(facet)}
        return cls(**kwargs, axes=axes)

    def _display_(self) -> Axes:
        """Return the axes for display in IPython environments."""
        return self.axes


class FacetAxesCollection(FacetCollection[FacetAxes]):
    def get_axes(self, row: int, col: int) -> Axes | None:
        if facet_axes := self.get(row, col):
            return facet_axes.axes
        return None

    def __getitem__(self, rc: tuple[int, int]) -> Axes | None:
        return self.get_axes(*rc)

    @property
    def axes(self) -> list[Axes]:
        return [facet_axes.axes for facet_axes in self]

    def map[**P](
        self,
        func: Callable[Concatenate[FacetAxes, P], Any],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Apply a plotting function to each facet in the collection.

        The function is called with the `FacetAxes` object as the first argument.

        Args:
            func: A callable that accepts a `FacetAxes` object as the first argument.
            *args: Additional positional arguments to pass to `func`.
            **kwargs: Additional keyword arguments to pass to `func`.

        Returns:
            The collection instance for method chaining.
        """
        for facet_axes in self:
            plt.sca(facet_axes.axes)
            func(facet_axes, *args, **kwargs)

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
        facet as the first argument. This is only called for facets that
        have data.

        Args:
            func: A callable that accepts a `polars.DataFrame` as the
                first argument.
            *args: Additional positional arguments to pass to `func`.
            **kwargs: Additional keyword arguments to pass to `func`.

        Returns:
            The collection instance for method chaining.
        """
        for facet_axes in self:
            if facet_axes.data is not None:
                plt.sca(facet_axes.axes)
                func(facet_axes.data, *args, **kwargs)

        return self

    def map_axes[**P](
        self,
        func: Callable[Concatenate[Axes, P], Any],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Apply a function to each axes in the collection.

        The function is called with the `Axes` object as the first argument.

        Args:
            func: A callable that accepts a `Axes` object as the first argument.
            *args: Additional positional arguments to pass to `func`.
            **kwargs: Additional keyword arguments to pass to `func`.

        Returns:
            The collection instance for method chaining.
        """
        for axes in self.axes:
            func(axes, *args, **kwargs)

        return self

    def set(self, **kwargs: Any) -> Self:
        for axes in self.axes:
            axes.set(**kwargs)

        return self

    def _display_(self) -> Figure | SubFigure | None:
        """Return the figure for display in IPython environments."""
        if axes := self.axes:
            return axes[0].figure
        return None


class FacetGrid:
    """Manage a grid of subplots for faceted plotting.

    This class is the main entry point. It creates the matplotlib Figure and
    manages the collection of `FacetAxes` objects. Plotting is typically done
    by accessing the `.facet_axes` collection and using its `map` or `filter`
    methods.
    """

    data: pl.DataFrame
    """The input DataFrame."""
    facet_data: FacetData
    """An instance of `FacetData` that manages the data partitioning for the grid."""
    facet_axes: FacetAxesCollection
    """A `FacetAxesCollection` containing all facet-axes pairs in the grid.

    This is the main object used for filtering and mapping.
    """
    figure: Figure
    """The main matplotlib `Figure` object."""
    _axes: NDArray[Any]

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

        self.figure, self._axes = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
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

        self.set_facet_axes()

    def set_facet_axes(self) -> Self:
        facets = self.facet_data.facets()
        facet_axes = (FacetAxes.from_facet(f, self._axes[f.row, f.col]) for f in facets)
        self.facet_axes = FacetAxesCollection(facet_axes)
        return self

    @property
    def nrows(self) -> int:
        """Get the number of rows in the facet grid."""
        return self.facet_data.nrows

    @property
    def ncols(self) -> int:
        """Get the number of columns in the facet grid."""
        return self.facet_data.ncols

    @property
    def axes(self) -> list[Axes]:
        """A list of all `Axes` objects in the grid."""
        return self.facet_axes.axes

    def delaxes(self) -> Self:
        """Delete all empty axes from the figure.

        This is useful for cleaning up the layout when some facets do not
        contain data.

        Returns:
            The `FacetGrid` instance for method chaining.
        """
        for ax in self.facet_axes.filter(has_data=False).axes:
            self.figure.delaxes(ax)

        self.facet_axes = FacetAxesCollection(f for f in self.facet_axes if f.has_data)
        return self

    def select(
        self,
        predicate: Callable[[FacetAxes], bool] | None = None,
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
        self.set_facet_axes()
        self.facet_axes = self.facet_axes.filter(
            predicate=predicate,
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
        return self

    def map[**P](
        self,
        func: Callable[Concatenate[FacetAxes, P], Any],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Apply a plotting function to each facet in the collection.

        The function is called with the `FacetAxes` object as the first argument.

        Args:
            func: A callable that accepts a `FacetAxes` object as the first argument.
            *args: Additional positional arguments to pass to `func`.
            **kwargs: Additional keyword arguments to pass to `func`.

        Returns:
            The collection instance for method chaining.
        """
        self.facet_axes.map(func, *args, **kwargs)
        return self

    def map_axes[**P](
        self,
        func: Callable[Concatenate[Axes, P], Any],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Apply a function to each axes in the collection.

        The function is called with the `Axes` object as the first argument.

        Args:
            func: A callable that accepts a `Axes` object as the first argument.
            *args: Additional positional arguments to pass to `func`.
            **kwargs: Additional keyword arguments to pass to `func`.

        Returns:
            The collection instance for method chaining.
        """
        self.facet_axes.map_axes(func, *args, **kwargs)
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
        facet as the first argument. This is only called for facets that
        have data.

        Args:
            func: A callable that accepts a `polars.DataFrame` as the
                first argument.
            *args: Additional positional arguments to pass to `func`.
            **kwargs: Additional keyword arguments to pass to `func`.

        Returns:
            The collection instance for method chaining.
        """
        self.facet_axes.map_dataframe(func, *args, **kwargs)
        return self

    def set(self, **kwargs: Any) -> Self:
        self.facet_axes.set(**kwargs)
        return self

    def _display_(self) -> Figure:
        """Return the figure for display in IPython environments."""
        return self.figure
