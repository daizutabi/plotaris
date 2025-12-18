from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib import rcParams


@dataclass
class Config:
    figsize: tuple[float, float] = (10, 6)
    title_fontsize: int = 14
    label_fontsize: int = 12
    grid: bool = True


_config = Config()


def get_config() -> Config:
    return _config


def init(
    style: str = "default",
    dpi: int = 160,
    figsize: tuple[float, float] = (3, 2),
    labelsize: float = 9.5,
    titlesize: float = 9.5,
    ticksize: float = 8.5,
) -> None:
    plt.style.use(style)

    rcParams["figure.dpi"] = dpi
    rcParams["figure.figsize"] = figsize
    rcParams["axes.labelsize"] = labelsize
    rcParams["axes.titlesize"] = titlesize
    rcParams["xtick.labelsize"] = ticksize
    rcParams["ytick.labelsize"] = ticksize
