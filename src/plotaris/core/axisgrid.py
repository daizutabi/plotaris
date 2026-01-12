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
    def __call__(
        self,
        arg: T,
        /,
        *args: Any,
        ax: Axes,
        **kwargs: Any,
    ) -> Any: ...


class FacetGrid:
    data: pl.DataFrame
    facet_data: FacetData
    figure: Figure
    axes: dict[tuple[int, int], Axes]

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

        for r, c in self.facet_data.cells(empty=True):
            self.figure.delaxes(cast("Axes", axes[r, c]))

        self.axes = {rc: cast("Axes", axes[*rc]) for rc in self.facet_data.cells()}

        for f in self.facet_data.iter_facets(bottommost=True):
            self.axes[f.row, f.col].tick_params(labelbottom=True)  # pyright: ignore[reportUnknownMemberType]

        for f in self.facet_data.iter_facets(leftmost=True):
            self.axes[f.row, f.col].tick_params(labelleft=True)  # pyright: ignore[reportUnknownMemberType]

    @property
    def nrows(self) -> int:
        return self.facet_data.nrows

    @property
    def ncols(self) -> int:
        return self.facet_data.ncols

    def _display_(self) -> Figure:
        return self.figure

    def __iter__(self) -> Iterator[Axes]:
        for facet in self.facet_data:
            yield self.axes[facet.row, facet.col]

    def items(self) -> Iterator[tuple[Axes, Facet]]:
        for facet in self.facet_data:
            ax = self.axes[facet.row, facet.col]
            yield ax, facet

    def map_facet(
        self,
        func: Plottable[Facet],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
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
        for ax, facet in self.items():
            func(facet.data, *args, ax=ax, **kwargs)

        return self
