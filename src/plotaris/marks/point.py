from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, override

from plotaris.marks.base import Mark

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class PointMark(Mark):
    kwargs_map: ClassVar[dict[str, str]] = {"size": "s", "shape": "marker"}

    @override
    def _plot(self, ax: Axes, *args: Any, **kwargs: Any) -> None:
        ax.scatter(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
