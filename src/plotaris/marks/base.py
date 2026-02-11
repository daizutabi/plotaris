from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class Mark(ABC):
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    kwargs_map: ClassVar[dict[str, str]] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def plot(self, **kwargs: Any) -> None:
        kwargs = {self.kwargs_map.get(k, k): v for k, v in kwargs.items()}
        self._plot(plt.gca(), *self.args, **self.kwargs, **kwargs)

    @abstractmethod
    def _plot(self, ax: Axes, *args: Any, **kwargs: Any) -> None: ...
