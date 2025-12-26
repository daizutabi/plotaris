from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import polars as pl


@dataclass(frozen=True)
class Encoding:
    """Declaratively specify the mapping between data and visual properties."""

    x: str | pl.Expr | None = None
    """The encoding for the x-axis."""
    y: str | pl.Expr | None = None
    """The encoding for the y-axis."""
    color: str | pl.Expr | None = None
    """The encoding for the color property."""
    size: str | pl.Expr | None = None
    """The encoding for the size property."""
    shape: str | pl.Expr | None = None
    """The encoding for the shape property (e.g., for scatter plots)."""

    def map_color(self, data: pl.DataFrame) -> dict[str, Any]:
        """Generate color mapping based on the encoding and data."""
        if self.color is None:
            return {}

        return map_color(data, self.color)


def map_color(data: pl.DataFrame, column: str | pl.Expr) -> dict[str, Any]:
    s = data.select(column).to_series()

    if s.dtype != pl.String:
        # For now, only handle categorical (string) data
        # Future implementation will handle continuous data
        return {}

    categories = s.unique().sort()
    palette = plt.get_cmap("viridis", len(categories))
    color_map = {cat: palette(i) for i, cat in enumerate(categories)}
    colors = [color_map[x] for x in s]

    return {"color": colors}
