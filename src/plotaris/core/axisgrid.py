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

        rcs = ((r, c) for r in range(self.nrows) for c in range(self.ncols))
        self.axes = {rc: cast("Axes", axes[*rc]) for rc in rcs}

        # for f in self.facet_data.iter_facets(bottommost=True):
        #     self.axes[f.row, f.col].tick_params(labelbottom=True)

        # for f in self.facet_data.iter_facets(leftmost=True):
        #     self.axes[f.row, f.col].tick_params(labelleft=True)

    @property
    def nrows(self) -> int:
        return self.facet_data.nrows

    @property
    def ncols(self) -> int:
        return self.facet_data.ncols

    def get_axes(self, row: int, col: int) -> Axes | None:
        ax = self.axes.get((row, col))
        return ax if ax in self.figure.axes else None

    @property
    def left_axes(self) -> list[Axes]:
        return [a for row in range(self.nrows) if (a := self.get_axes(row, 0))]

    @property
    def top_axes(self) -> list[Axes]:
        return [a for col in range(self.ncols) if (a := self.get_axes(0, col))]

    @property
    def right_axes(self) -> list[Axes]:
        col = self.ncols - 1
        return [a for row in range(self.nrows) if (a := self.get_axes(row, col))]

    @property
    def bottom_axes(self) -> list[Axes]:
        row = self.nrows - 1
        return [a for col in range(self.ncols) if (a := self.get_axes(row, col))]

    @property
    def data_axes(self) -> list[Axes]:
        cells = self.facet_data.cells().filter(has_data=True)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    @property
    def empty_axes(self) -> list[Axes]:
        cells = self.facet_data.cells().filter(has_data=False)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    @property
    def leftmost_axes(self) -> list[Axes]:
        cells = self.facet_data.cells().filter(is_leftmost=True)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    @property
    def topmost_axes(self) -> list[Axes]:
        cells = self.facet_data.cells().filter(is_topmost=True)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    @property
    def rightmost_axes(self) -> list[Axes]:
        cells = self.facet_data.cells().filter(is_rightmost=True)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    @property
    def bottommost_axes(self) -> list[Axes]:
        cells = self.facet_data.cells().filter(is_bottommost=True)
        return [a for cell in cells if (a := self.get_axes(cell.row, cell.col))]

    def delaxes(self) -> Self:
        for ax in self.empty_axes:
            self.figure.delaxes(ax)
        return self

    def __iter__(self) -> Iterator[Axes]:
        yield from self.data_axes

    def items(self) -> Iterator[tuple[Axes, Facet]]:
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

    def _display_(self) -> Figure:
        return self.figure
