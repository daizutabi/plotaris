from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, Self

import matplotlib.pyplot as plt

from .data import FacetData

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


class Plottable(Protocol):
    def __call__(
        self,
        data: pl.DataFrame,
        *args: Any,
        ax: Axes,
        **kwargs: Any,
    ) -> Any: ...


class FacetGrid:
    data: pl.DataFrame
    facet_data: FacetData
    figure: Figure
    axes: NDArray[Any]

    def __init__(
        self,
        data: pl.DataFrame,
        row: str | Iterable[str] | None = None,
        col: str | Iterable[str] | None = None,
        *,
        wrap: int | None = None,
        sharex: bool | Literal["none", "all", "row", "col"] = False,
        sharey: bool | Literal["none", "all", "row", "col"] = False,
        width_ratios: Sequence[float] | None = None,
        height_ratios: Sequence[float] | None = None,
        subplot_kw: dict[str, Any] | None = None,
        gridspec_kw: dict[str, Any] | None = None,
        **fig_kw: Any,
    ) -> None:
        self.data = data
        self.facet_data = FacetData(data, row, col, wrap)

        self.figure, self.axes = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
            self.nrows,
            self.ncols,
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw,
            **fig_kw,
        )

    @property
    def nrows(self) -> int:
        return self.facet_data.nrows

    @property
    def ncols(self) -> int:
        return self.facet_data.ncols

    def map_dataframe(self, plot: Plottable, /, *args: Any, **kwargs: Any) -> Self:
        for facet in self.facet_data.iter_facets():
            ax = self.axes[facet.row, facet.col]
            plot(facet.data, *args, ax=ax, **kwargs)

        return self
