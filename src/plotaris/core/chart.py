from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Self

import matplotlib.pyplot as plt

from plotaris.marks.bar import BarMark
from plotaris.marks.line import LineMark
from plotaris.marks.point import PointMark

from .data import GroupedData, to_list
from .encoding import Encoding
from .grid import Facet

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl
    from matplotlib.axes import Axes

    from plotaris.marks.base import Mark

    from .grid import Grid


class Chart:
    data: pl.DataFrame
    encoding: Encoding
    mark: Mark | None
    grid: Grid | None

    def __init__(
        self,
        data: pl.DataFrame,
        encoding: Encoding | None = None,
        mark: Mark | None = None,
        grid: Grid | None = None,
    ) -> None:
        self.data = data
        self.encoding = encoding or Encoding()
        self.mark = mark
        self.grid = grid

    def encode(
        self,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr | None = None,
        color: str | Iterable[str] | None = None,
        size: str | Iterable[str] | None = None,
        shape: str | Iterable[str] | None = None,
    ) -> Self:
        """Map variables to visual properties, updating existing encodings."""
        changes = {
            "x": x,
            "y": y,
            "color": to_list(color) or None,
            "size": to_list(size) or None,
            "shape": to_list(shape) or None,
        }
        changes = {k: v for k, v in changes.items() if v is not None}
        self.encoding = replace(self.encoding, **changes)
        return self

    def facet(
        self,
        *,
        row: str | None = None,
        col: str | None = None,
        wrap: int | None = None,
    ) -> Self:
        """Create a facet grid of subplots."""
        self.grid = Facet(row=row, col=col, wrap=wrap)
        return self

    def mark_point(self, **kwargs: Any) -> Self:
        self.mark = PointMark(**kwargs)
        return self

    def mark_line(self, **kwargs: Any) -> Self:
        self.mark = LineMark(**kwargs)
        return self

    def mark_bar(self, **kwargs: Any) -> Self:
        self.mark = BarMark(**kwargs)
        return self

    def display(self, ax: Axes | None = None) -> Axes:
        if self.mark is None:
            msg = "Mark must be defined before displaying the chart"
            raise ValueError(msg)

        if self.grid:
            raise NotImplementedError

        ax = ax or plt.gca()

        palettes = self.encoding.create_palettes(self.data)

        mapping: dict[str, Iterable[str]] = {}

        if self.encoding.color:
            mapping["color"] = self.encoding.color
        if self.encoding.size:
            mapping["size"] = self.encoding.size
        if self.encoding.shape:
            mapping["shape"] = self.encoding.shape

        gd = GroupedData(self.data, mapping=mapping)

        for df_group in gd.data:
            kwargs: dict[str, Any] = {}

            if "color" in palettes:
                key = df_group.select(self.encoding.color).row(0)
                kwargs["color"] = palettes["color"][key]
            if "size" in palettes:
                key = df_group.select(self.encoding.size).row(0)
                kwargs["s"] = palettes["size"][key]
            if "shape" in palettes:
                key = df_group.select(self.encoding.shape).row(0)
                kwargs["marker"] = palettes["shape"][key]

            x = df_group.select(self.encoding.x).to_series()
            y = df_group.select(self.encoding.y).to_series()

            self.mark.plot(ax=ax, x=x, y=y, **kwargs)

        return ax

    def _display_(self) -> Axes:
        return self.display()
