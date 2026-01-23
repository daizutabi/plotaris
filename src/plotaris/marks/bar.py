from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from .base import Mark

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class BarMark(Mark):
    @override
    def _plot(self, ax: Axes, *args: Any, **kwargs: Any) -> None:
        ax.bar(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
