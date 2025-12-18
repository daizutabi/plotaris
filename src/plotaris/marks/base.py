from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.axes
    import polars as pl

class Mark(ABC):
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def plot(self, ax: matplotlib.axes.Axes, data: pl.DataFrame, encodings: dict[str, str]) -> None:
        pass
