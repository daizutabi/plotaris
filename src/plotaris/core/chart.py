from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import matplotlib.pyplot as plt

from plotaris.marks.bar import BarMark
from plotaris.marks.line import LineMark
from plotaris.marks.point import PointMark

from .group import to_tuple
from .palette import Palette

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl
    from matplotlib.axes import Axes

    from plotaris.marks.base import Mark


class Chart:
    data: pl.DataFrame
    x: str | pl.Expr | None = None
    y: str | pl.Expr | None = None
    color: tuple[str, ...] = ()
    size: tuple[str, ...] = ()
    shape: tuple[str, ...] = ()
    row: tuple[str, ...] = ()
    col: tuple[str, ...] = ()
    wrap: int | None = None
    mark: Mark | None = None

    def __init__(self, data: pl.DataFrame) -> None:
        self.data = data

    def encode(
        self,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr | None = None,
        color: str | Iterable[str] | None = None,
        size: str | Iterable[str] | None = None,
        shape: str | Iterable[str] | None = None,
        row: str | Iterable[str] | None = None,
        col: str | Iterable[str] | None = None,
        wrap: int | None = None,
    ) -> Self:
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if color is not None:
            self.color = to_tuple(color)
        if size is not None:
            self.size = to_tuple(size)
        if shape is not None:
            self.shape = to_tuple(shape)

        if row or col:
            self.facet(row, col, wrap)

        return self

    def facet(
        self,
        row: str | Iterable[str] | None = None,
        col: str | Iterable[str] | None = None,
        wrap: int | None = None,
    ) -> Self:
        self.row = to_tuple(row)
        self.col = to_tuple(col)
        self.wrap = wrap
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

    @property
    def encoding(self) -> Palette:
        return Palette(self.color, self.size, self.shape)

    def display(self, ax: Axes | None = None) -> Axes:
        if ax is None:
            ax = plt.figure().add_subplot()  # pyright: ignore[reportUnknownMemberType]

        if not self.mark:
            return ax

        kwargs: dict[str, Any] = {}

        if self.x is not None:
            kwargs["x"] = self.data.select(self.x).to_series()

        if self.y is not None:
            kwargs["y"] = self.data.select(self.y).to_series()

        self.mark.plot(ax, **kwargs)

        return ax

    #     if self.mark is None:
    #         msg = "Mark must be defined before displaying the chart"
    #         raise ValueError(msg)

    #     facet_spec = self.facet_spec or FacetSpec()
    #     grid = FacetGrid(self.data, self.encoding, facet_spec, ax=ax)
    #     grid.plot(self.mark, self.encoding)

    #     return grid.axes.squeeze() if grid.axes.size == 1 else grid.axes

    def _display_(self) -> Axes:
        return self.display()
