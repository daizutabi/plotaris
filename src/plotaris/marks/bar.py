from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from .base import Mark

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes


class BarMark(Mark):
    @override
    def plot(self, ax: Axes, *, x: pl.Series, y: pl.Series, **kwargs: Any) -> None:
        ax.bar(x, y, **self.kwargs, **kwargs)  # pyright: ignore[reportUnknownMemberType]
