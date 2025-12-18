from __future__ import annotations

from typing import TYPE_CHECKING, override

from plotaris.marks.base import Mark

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes


class LineMark(Mark):
    @override
    def plot(self, ax: Axes, data: pl.DataFrame, encodings: dict[str, str]) -> None:
        x = encodings.get("x")
        y = encodings.get("y")

        if not x or not y:
            msg = "x and y encodings are required for line plot"
            raise ValueError(msg)

        ax.plot(data[x], data[y], **self.kwargs)  # pyright: ignore[reportUnknownMemberType]
