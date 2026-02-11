from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import matplotlib.pyplot as plt

from plotaris.marks.bar import BarMark
from plotaris.marks.line import LineMark
from plotaris.marks.point import PointMark
from plotaris.utils import to_tuple

from .axisgrid import FacetGrid
from .group import Group
from .palette import Palette

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    import polars as pl
    from matplotlib.axes import Axes

    from plotaris.marks.base import Mark

    from .palette import VisualValue


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
    palette: Palette | None = None
    mark: Mark | None = None
    _plot: Callable[..., Any] | None = None
    _kwargs: dict[str, Any]

    def __init__(
        self,
        data: pl.DataFrame,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.data = data
        self._kwargs = kwargs

        if figsize is not None:
            self._kwargs["figsize"] = figsize

    @property
    def encoding(self) -> dict[str, tuple[str, ...]]:
        names = ["color", "size", "shape"]
        return {name: value for name in names if (value := getattr(self, name))}

    @property
    def has_facet(self) -> bool:
        return bool(self.row or self.col)

    def encode(
        self,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr | None = None,
        *,
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

        self.palette = Palette(**self.encoding).set(self.data)

        return self

    def mapping(
        self,
        /,
        **mapping: Mapping[Any | tuple[Any, ...], VisualValue],
    ) -> Self:
        if self.palette is not None:
            self.palette.mapping(**mapping).set(self.data)
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

    def map(self, plot: Callable[..., Any], /) -> Self:
        self._plot = plot
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

    def _get_series(self, data: pl.DataFrame) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}

        if self.palette is not None:
            kwargs.update(self.palette.get(data))

        if self.x is not None:
            kwargs["x"] = data.select(self.x).to_series()

        if self.y is not None:
            kwargs["y"] = data.select(self.y).to_series()

        return kwargs

    def _iter_series(self, data: pl.DataFrame) -> Iterator[dict[str, Any]]:
        group = Group(data, **self.encoding)
        return map(self._get_series, group)

    def _plot_series(self, data: pl.DataFrame) -> None:
        for series in self._iter_series(data):
            if self._plot:
                self._plot(**series)
            if self.mark:
                self.mark.plot(**series)

    def to_facet(self) -> FacetGrid:
        grid = FacetGrid(self.data, self.row, self.col, self.wrap, **self._kwargs)
        grid.map_dataframe(self._plot_series)
        return grid

    def display(self) -> Axes | FacetGrid:
        if self.has_facet:
            return self.to_facet()

        ax = plt.figure(**self._kwargs).add_subplot()  # pyright: ignore[reportUnknownMemberType]
        self._plot_series(self.data)
        return ax

    def _display_(self) -> Axes | FacetGrid:
        return self.display()
