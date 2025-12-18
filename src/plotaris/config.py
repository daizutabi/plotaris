from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    figsize: tuple[float, float] = (10, 6)
    title_fontsize: int = 14
    label_fontsize: int = 12
    grid: bool = True


_config = Config()


def get_config() -> Config:
    return _config
