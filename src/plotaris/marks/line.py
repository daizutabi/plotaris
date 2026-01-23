from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from plotaris.marks.base import Mark

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class LineMark(Mark):
    @override
    def _plot(self, ax: Axes, *args: Any, **kwargs: Any) -> None:
        ax.plot(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
