from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes


class Mark(ABC):
    kwargs: dict[str, Any]
    kwargs_map: ClassVar[dict[str, str]] = {}

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def plot(self, ax: Axes, *, x: pl.Series, y: pl.Series, **kwargs: Any) -> None:
        kwargs = {self.kwargs_map.get(k, k): v for k, v in kwargs.items()}
        self._plot(ax, x=x, y=y, **self.kwargs, **kwargs)

    @abstractmethod
    def _plot(self, ax: Axes, *, x: pl.Series, y: pl.Series, **kwargs: Any) -> None: ...
