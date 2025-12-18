from __future__ import annotations

from typing import TYPE_CHECKING, override

from .base import Mark

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes


class BarMark(Mark):
    @override
    def plot(self, ax: Axes, data: pl.DataFrame, encodings: dict[str, str]) -> None:
        x = encodings.get("x")
        y = encodings.get("y")

        if not x or not y:
            msg = "x and y encodings are required for bar plot"
            raise ValueError(msg)

        ax.bar(data[x], data[y], **self.kwargs)  # pyright: ignore[reportUnknownMemberType]
