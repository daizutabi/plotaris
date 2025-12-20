from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from plotaris.marks.base import Mark

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from polars import Series


class PointMark(Mark):
    @override
    def plot(self, ax: Axes, *, x: Series, y: Series, **kwargs: Any) -> None:
        ax.scatter(x, y, **self.kwargs, **kwargs)  # pyright: ignore[reportUnknownMemberType]
