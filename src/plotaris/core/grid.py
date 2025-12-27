from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from .data import GroupedData

if TYPE_CHECKING:
    import numpy as np
    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from plotaris.marks.base import Mark

    from .encoding import Encoding


@dataclass
class FacetSpec:
    """A specification for the facet grid."""

    row: list[str] | None = None
    col: list[str] | None = None
    wrap: int | None = None


class FacetGrid:
    data: pl.DataFrame
    encoding: Encoding
    facet_spec: FacetSpec
    gd: GroupedData
    fig: Figure
    axes: np.ndarray[Any, Any]
    nrows: int
    ncols: int

    def __init__(
        self,
        data: pl.DataFrame,
        encoding: Encoding,
        facet_spec: FacetSpec,
    ) -> None:
        self.data = data
        self.encoding = encoding
        self.facet_spec = facet_spec

        # Combine aesthetic and facet mappings for GroupedData
        full_mapping = dict(encoding.items())
        if facet_spec.row:
            full_mapping["row"] = facet_spec.row
        if facet_spec.col:
            full_mapping["col"] = facet_spec.col
        self.gd = GroupedData(self.data, full_mapping)

        # Calculate grid dimensions
        self.nrows = self.gd.n_unique("row") or 1
        self.ncols = self.gd.n_unique("col") or 1

        # Create the subplot grid
        self.fig, self.axes = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
            self.nrows,
            self.ncols,
            sharex=True,
            sharey=True,
            squeeze=False,  # Ensure axes is always a 2D array
        )

    def plot(self, mark: Mark, encoding: Encoding) -> None:
        """Plot the data on the grid."""
        # Create palettes from the original, unsplit data for consistent scales
        palettes = encoding.create_palettes(self.data)

        for i, df_group in enumerate(self.gd.data):
            # Determine which subplot (ax) to draw on
            row_idx = self.gd.group["row"][i] if "row" in self.gd.group.columns else 0
            col_idx = self.gd.group["col"][i] if "col" in self.gd.group.columns else 0
            ax = self.axes[row_idx, col_idx]

            # Calculate kwargs for this group's aesthetics
            kwargs = {}
            for name, palette in palettes.items():
                # The key for the palette is the tuple of actual categorical values
                key = df_group.select(encoding.get(name)).row(0)
                kwargs[name] = palette[key]

            # Get x,y data for this group and plot
            x = df_group.select(encoding.x).to_series()
            y = df_group.select(encoding.y).to_series()
            mark.plot(ax=ax, x=x, y=y, **kwargs)

            # Set the title for the subplot
            self._set_title(ax, i)

        # Final layout adjustments
        self.fig.tight_layout()

    def _set_title(self, ax: Axes, group_index: int) -> None:
        """Set the title for a single subplot based on its facet values."""
        parts: list[str] = []
        if "row" in self.gd.mapping:
            row_vals = self.gd.item(group_index, "row", named=True)
            parts.append(", ".join(f"{k}={v}" for k, v in row_vals.items()))
        if "col" in self.gd.mapping:
            col_vals = self.gd.item(group_index, "col", named=True)
            parts.append(", ".join(f"{k}={v}" for k, v in col_vals.items()))

        ax.set_title(" | ".join(parts))  # pyright: ignore[reportUnknownMemberType]
