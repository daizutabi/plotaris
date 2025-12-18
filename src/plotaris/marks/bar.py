from __future__ import annotations

from typing import TYPE_CHECKING, override

from .base import Mark

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes

    from plotaris.core.encoding import Encoding


class BarMark(Mark):
    @override
    def plot(self, ax: Axes, data: pl.DataFrame, encoding: Encoding) -> None:
        x, y = data.select(x=encoding.x, y=encoding.y)
        ax.bar(x, y, **self.kwargs)  # pyright: ignore[reportUnknownMemberType]
