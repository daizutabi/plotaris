from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, override

from plotaris.marks.base import Mark

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes


class PointMark(Mark):
    key_map: ClassVar[dict[str, str]] = {"size": "s", "shape": "marker"}

    @override
    def _plot(self, ax: Axes, *, x: pl.Series, y: pl.Series, **kwargs: Any) -> None:
        ax.scatter(x, y, **kwargs)  # pyright: ignore[reportUnknownMemberType]
