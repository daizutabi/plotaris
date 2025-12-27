from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Self

import matplotlib.pyplot as plt

from plotaris.marks.bar import BarMark
from plotaris.marks.line import LineMark
from plotaris.marks.point import PointMark

from .data import GroupedData, to_list
from .encoding import Encoding
from .grid import FacetGrid, FacetSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    import polars as pl
    from matplotlib.axes import Axes

    from plotaris.marks.base import Mark


class Chart:
    data: pl.DataFrame
    encoding: Encoding
    mark: Mark | None
    facet_spec: FacetSpec | None

    def __init__(
        self,
        data: pl.DataFrame,
        encoding: Encoding | None = None,
        mark: Mark | None = None,
        facet_spec: FacetSpec | None = None,
    ) -> None:
        self.data = data
        self.encoding = encoding or Encoding()
        self.mark = mark
        self.facet_spec = facet_spec

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
        row: str | Iterable[str] | None = None,
        col: str | Iterable[str] | None = None,
        wrap: int | None = None,
    ) -> Self:
        """Create a facet grid of subplots."""
        self.facet_spec = FacetSpec(
            row=to_list(row) or None,
            col=to_list(col) or None,
            wrap=wrap,
        )
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

    def display(self, ax: Axes | None = None) -> Axes | np.ndarray[Any, Any]:
        if self.mark is None:
            msg = "Mark must be defined before displaying the chart"
            raise ValueError(msg)

        if self.facet_spec:
            grid = FacetGrid(self.data, self.encoding, self.facet_spec)
            grid.plot(self.mark, self.encoding)
            return grid.axes

        ax = ax or plt.gca()

        palettes = self.encoding.create_palettes(self.data)
        mapping = dict(self.encoding.items())
        gd = GroupedData(self.data, mapping)

        for df in gd.data:
            kwargs: dict[str, Any] = {}

            for name, palette in palettes.items():
                columns = self.encoding.get(name)
                key = df.select(columns).row(0)
                kwargs[name] = palette[key]

            x = df.select(self.encoding.x).to_series()
            y = df.select(self.encoding.y).to_series()

            self.mark.plot(ax=ax, x=x, y=y, **kwargs)

        return ax

    def _display_(self) -> Axes | np.ndarray[Any, Any]:
        return self.display()
