from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes


class Mark(ABC):
    kwargs: dict[str, Any]

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def plot(self, ax: Axes, *, x: pl.Series, y: pl.Series, **kwargs: Any) -> None: ...
