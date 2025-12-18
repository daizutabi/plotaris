from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Self

import matplotlib.pyplot as plt

from plotaris.core.data import DataHandler
from plotaris.core.encoding import Encoding
from plotaris.marks.line import LineMark
from plotaris.marks.scatter import ScatterMark

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.figure import Figure
    from polars._typing import IntoExpr

    from plotaris.marks.base import Mark


class Chart:
    data_handler: DataHandler
    encoding: Encoding
    mark: Mark | None

    def __init__(self, data: pl.DataFrame | pl.LazyFrame) -> None:
        self.data_handler = DataHandler(data)
        self.encoding = Encoding()
        self.mark = None

    def encode(
        self,
        x: IntoExpr = None,
        y: IntoExpr = None,
        color: IntoExpr = None,
        size: IntoExpr = None,
        shape: IntoExpr = None,
    ) -> Self:
        """Map variables to visual properties, updating existing encodings."""
        changes = {
            "x": x,
            "y": y,
            "color": color,
            "size": size,
            "shape": shape,
        }
        changes = {k: v for k, v in changes.items() if v is not None}

        self.encoding = replace(self.encoding, **changes)

        return self

    def mark_scatter(self, **kwargs: Any) -> Self:
        self.mark = ScatterMark(**kwargs)
        return self

    def mark_line(self, **kwargs: Any) -> Self:
        self.mark = LineMark(**kwargs)
        return self

    def display(self) -> Figure:
        if self.mark is None:
            msg = "Mark must be defined before displaying the chart"
            raise ValueError(msg)

        df = self.data_handler.collect()
        fig, ax = plt.subplots()  # pyright: ignore[reportUnknownMemberType]
        self.mark.plot(ax, df, self.encoding)

        return fig

    def _display_(self) -> Figure:
        return self.display()
