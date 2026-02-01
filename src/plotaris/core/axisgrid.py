from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Concatenate, Literal, Self

import matplotlib.pyplot as plt

from .facet import Facet, FacetCollection, FacetData

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .label import Format


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

    def set_title(
        self,
        formats: dict[str, Format | tuple[str, Format]] | None = None,
        *,
        dim: Literal["row", "col"] | None,
        loc: Literal["top", "right"],
        ax: Axes | None = None,
        **kwargs: Format | tuple[str, Format],
    ) -> Self:
        """Set a title on the `Axes` object for this facet.

        The title can be placed at the top of the axes or on the right side.
        It uses the `FacetLabel` associated with the facet to generate the title string.

        Args:
            formats: A dictionary mapping data keys to format specifiers for the
                `plotaris.core.label.Label.format` method.
            dim: The facet dimension ("row" or "col") to use for the title.
                If `None`, all dimensions are used.
            loc: The location of the title, either "top" or "right".
            ax: An optional `Axes` object to set the title on. If `None`,
                uses the `axes` attribute of the `FacetAxes`.
            **kwargs: Additional per-key format specifiers provided as keyword
                arguments, merged with `formats`.

        Returns:
            The `FacetAxes` instance for method chaining.
        """
        self.label.dim = dim
        label = self.label.format(formats, **kwargs)
        if not label:
            return self

        ax = ax or self.axes

        if loc == "top":
            ax.set_title(label)  # pyright: ignore[reportUnknownMemberType]
        else:
            ax = ax.twinx()
            ax.set_yticks([])  # pyright: ignore[reportUnknownMemberType]
            ax.set_ylabel(label)  # pyright: ignore[reportUnknownMemberType]

        return self


class FacetAxesCollection(FacetCollection[FacetAxes]):
    def get_axes(self, row: int, col: int) -> Axes | None:
        if facet_axes := self.get(row, col):
            return facet_axes.axes
        return None

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
    _facet_axes: FacetAxesCollection
    """A `FacetAxesCollection` containing all facet-axes pairs in the grid."""
    _selected_facet_axes: FacetAxesCollection | None
    """A selected `FacetAxesCollection`."""
    figure: Figure
    """The main matplotlib `Figure` object."""

    def __init__(
        self,
        data: pl.DataFrame,
        row: str | Iterable[str] | None = None,
        col: str | Iterable[str] | None = None,
        wrap: int | None = None,
        *,
        sharex: bool | Literal["none", "all", "row", "col"] = True,
        sharey: bool | Literal["none", "all", "row", "col"] = True,
        constrained_layout: bool = True,
        subplot_kw: dict[str, Any] | None = None,
        gridspec_kw: dict[str, Any] | None = None,
        figure: Figure | None = None,
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
            figure: An optional `Figure` object to use for the grid.
            **fig_kw: Additional keyword arguments passed to
                `matplotlib.pyplot.figure`.
        """
        self.data = data
        self.facet_data = FacetData(data, row, col, wrap)

        self.figure = figure or plt.figure(  # pyright: ignore[reportUnknownMemberType]
            constrained_layout=constrained_layout,
            **fig_kw,
        )

        axes = self.figure.subplots(
            self.facet_data.nrows,
            self.facet_data.ncols,
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
            subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw,
        )

        facets = self.facet_data.facets
        facet_axes = (FacetAxes.from_facet(f, axes[f.row, f.col]) for f in facets)
        self._facet_axes = FacetAxesCollection(facet_axes)
        self._selected_facet_axes = None

    @property
    def facet_axes(self) -> FacetAxesCollection:
        return self._selected_facet_axes or self._facet_axes

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
        for ax in self._facet_axes.filter(has_data=False).axes:
            self.figure.delaxes(ax)

        self._facet_axes = FacetAxesCollection(f for f in self.facet_axes if f.has_data)
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
        """Select a subset of facets from the grid.

        Note:
            Calling this method will reset any prior selection.

        Args:
            predicate: A callable that takes a `FacetAxes` object and returns
                `True` if it should be included in the selection.
            row: The row index to select.
            col: The column index to select.
            has_data: If `True`, select only facets that have associated data.
            is_left: If `True`, select facets in the first column.
            is_top: If `True`, select facets in the first row.
            is_right: If `True`, select facets in the last column.
            is_bottom: If `True`, select facets in the last row.
            is_leftmost: If `True`, select the first visible facet in each row.
            is_topmost: If `True`, select the first visible facet in each column.
            is_rightmost: If `True`, select the last visible facet in each row.
            is_bottommost: If `True`, select the last visible facet in each column.

        Returns:
            The `FacetGrid` instance for method chaining.
        """
        self._selected_facet_axes = self._facet_axes.filter(
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

    def all(self) -> Self:
        """Clear any active facet selection, returning to the full set of facets."""
        self._selected_facet_axes = None
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

    def set_titles(
        self,
        formats: dict[str, Format | tuple[str, Format]] | None = None,
        *,
        margin_titles: bool = True,
        **kwargs: Format | tuple[str, Format],
    ) -> Self:
        """Sets titles for the facet grid, typically along the top and right edges.

        This method leverages the advanced formatting capabilities of the
        `FacetLabel` and `plotaris.core.label.Label` classes to create
        descriptive titles for each facet. Top titles typically represent
        row dimensions, while right titles represent column dimensions.

        Args:
            formats: A dictionary mapping data keys to format specifiers for the
                `plotaris.core.label.Label.format` method.
            margin_titles: If `True`, titles are placed in the figure margins
                (top and right). If `False`, titles are placed within each subplot.
            **kwargs: Additional per-key format specifiers provided as keyword
                arguments, merged with `formats`.

        Returns:
            The `FacetGrid` instance for method chaining.
        """

        def set_title(
            facet_axes: FacetAxes,
            dim: Literal["row", "col"] | None,
            loc: Literal["top", "right"],
        ) -> None:
            if dim is None:
                fa = facet_axes
            elif loc == "top":
                row, col = 0, facet_axes.col
                fa = self.facet_axes.get(row, col) or facet_axes
            else:
                row, col = facet_axes.row, self.facet_data.ncols - 1
                fa = self.facet_axes.get(row, col) or facet_axes

            facet_axes.set_title(formats, dim=dim, loc=loc, ax=fa.axes, **kwargs)

        if margin_titles and self.facet_data.wrap is None:
            self.facet_axes.filter(is_topmost=True).map(set_title, "col", "top")
            self.facet_axes.filter(is_rightmost=True).map(set_title, "row", "right")
        else:
            self.facet_axes.map(set_title, None, "top")

        return self

    def _display_(self) -> Figure:
        """Return the figure for display in IPython environments."""
        return self.figure
