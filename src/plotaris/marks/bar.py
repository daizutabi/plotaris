from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from .base import Mark

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from polars import Series


class BarMark(Mark):
    @override
    def plot(self, ax: Axes, *, x: Series, y: Series, **kwargs: Any) -> None:
        ax.bar(x, y, **self.kwargs, **kwargs)  # pyright: ignore[reportUnknownMemberType]
