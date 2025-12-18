from __future__ import annotations
from typing import TYPE_CHECKING, Any
from plotaris.marks.base import Mark

if TYPE_CHECKING:
    import matplotlib.axes
    import polars as pl

class ScatterMark(Mark):
    def plot(self, ax: matplotlib.axes.Axes, data: pl.DataFrame, encodings: dict[str, str]) -> None:
        x = encodings.get("x")
        y = encodings.get("y")
        
        if not x or not y:
            msg = "x and y encodings are required for scatter plot"
            raise ValueError(msg)
            
        ax.scatter(data[x], data[y], **self.kwargs)
