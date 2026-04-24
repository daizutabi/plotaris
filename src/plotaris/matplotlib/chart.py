from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import matplotlib.pyplot as plt

from plotaris.marks.bar import BarMark
from plotaris.marks.line import LineMark
from plotaris.marks.point import PointMark
from plotaris.utils import to_tuple

from .axes import format_axes
from .axisgrid import FacetGrid
from .group import Group
from .label import Label
from .palette import Palette

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    import polars as pl
    from matplotlib.axes import Axes

    from plotaris.marks.base import Mark

    from .label import Format
    from .palette import VisualValue


class Chart:
    data: pl.DataFrame
    x: str | pl.Expr | None = None
    y: str | pl.Expr | None = None
    color: tuple[str, ...] = ()
    size: tuple[str, ...] = ()
    shape: tuple[str, ...] = ()
    palette: Palette | None = None
    mark: Mark | None = None
    plot: Callable[..., Any] | None = None
    axes: Axes | None = None
    _label: Label
    _kwargs: dict[str, Any]

    def __init__(
        self,
        data: pl.DataFrame,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.data = data
        self._label = Label()
        self._kwargs = kwargs

        if figsize is not None:
            self._kwargs["figsize"] = figsize

    @property
    def encoding(self) -> dict[str, tuple[str, ...]]:
        names = ["color", "size", "shape"]
        return {name: value for name in names if (value := getattr(self, name))}

    def encode(
        self,
        x: str | pl.Expr | None = None,
        y: str | pl.Expr | None = None,
        *,
        color: str | Iterable[str] | None = None,
        shape: str | Iterable[str] | None = None,
        size: str | Iterable[str] | None = None,
    ) -> Self:
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if color is not None:
            self.color = to_tuple(color)
        if shape is not None:
            self.shape = to_tuple(shape)
        if size is not None:
            self.size = to_tuple(size)

        self.palette = Palette(**self.encoding).set(self.data)
        return self

    def mapping(
        self,
        color: Mapping[Any, VisualValue] | None = None,
        shape: Mapping[Any, VisualValue] | None = None,
        size: Mapping[Any, VisualValue] | None = None,
    ) -> Self:
        it = [("color", color), ("shape", shape), ("size", size)]
        mapping = {k: v for k, v in it if v}
        if self.palette is not None:
            self.palette.mapping(**mapping).set(self.data)
        return self

    def map(self, plot: Callable[..., Any], /) -> Self:
        self.plot = plot
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

    def label(
        self,
        formats: dict[str, Format | tuple[str, Format]] | None = None,
        /,
        eq: str = "=",
        sep: str = ", ",
        unit_sep: str = "",
        **kwargs: Format | tuple[str, Format],
    ) -> Self:
        self._label = Label(eq=eq, sep=sep, unit_sep=unit_sep).fmt(formats, **kwargs)
        return self

    def _get_series(self, data: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:
        if self.palette is not None:
            kwargs.update(self.palette.get(data))

        if self.x is not None:
            kwargs["x"] = data.select(self.x).to_series()

        if self.y is not None:
            kwargs["y"] = data.select(self.y).to_series()

        return kwargs

    def _iter_series(self, data: pl.DataFrame) -> Iterator[dict[str, Any]]:
        group = Group(data, **self.encoding)
        for df, label_ in zip(group, group.labels(merge=True), strict=True):
            label = self._label.set(label_).format()
            yield self._get_series(df, label=label)

    def _plot_series(self, data: pl.DataFrame) -> None:
        for series in self._iter_series(data):
            if self.plot:
                self.plot(**series)
            if self.mark:
                self.mark.plot(**series)

    def facet(
        self,
        row: str | Iterable[str] | None = None,
        col: str | Iterable[str] | None = None,
        wrap: int | None = None,
    ) -> FacetGrid:
        grid = FacetGrid(self.data, row, col, wrap, **self._kwargs)
        grid.map_dataframe(self._plot_series)
        return grid

    def display(self) -> Axes:
        if self.axes is None:
            self.axes = plt.figure(**self._kwargs).add_subplot()  # pyright: ignore[reportUnknownMemberType]
            self._plot_series(self.data)
        return self.axes

    def legend(self, *args: Any, **kwargs: Any) -> Self:
        self.display().legend(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        return self

    def format_axes(
        self,
        xlabel: str | None = None,
        ylabel: str | None = None,
        fontdict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        ax = self.display()
        format_axes(ax, xlabel, ylabel, fontdict, **kwargs)
        return self

    def set(self, **kwargs: Any) -> Self:
        ax = self.display()
        ax.set(**kwargs)
        return self

    def _display_(self) -> Axes:
        return self.display()
