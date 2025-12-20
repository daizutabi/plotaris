from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

    from polars import DataFrame, Expr


def map_color(data: DataFrame, column: str | Expr) -> Mapping[str, object]:
    s = data.select(column).to_series()

    if s.dtype != pl.String:
        # For now, only handle categorical (string) data
        # Future implementation will handle continuous data
        return {}

    categories = s.unique().sort()
    palette = plt.get_cmap("viridis", len(categories))
    color_map = {cat: palette(i) for i, cat in enumerate(categories)}

    # Replace s.map_dict with Series.replace
    # Series.replace expects lists of old and new values.
    old_values = categories.to_list()
    new_values = [color_map[cat] for cat in old_values]

    colors = s.replace(old_values, new_values)

    # For now, return just the colors. Legend handling will be added later.
    return {"c": colors}
