# from __future__ import annotations

# from typing import TYPE_CHECKING, Any

# import matplotlib.pyplot as plt

# if TYPE_CHECKING:
#     from collections.abc import Iterable

#     import numpy as np
#     import polars as pl
#     from matplotlib.axes import Axes
#     from matplotlib.figure import Figure

#     from .grid import Facet


# class LayoutManager:
#     """Manages the layout of a faceted chart grid."""

#     def __init__(
#         self,
#         data: pl.DataFrame,
#         facet: Facet,
#         *,
#         sharex: bool = True,
#         sharey: bool = True,
#     ) -> None:
#         if facet.row is None and facet.col is None:
#             msg = "Facet information is missing."
#             raise ValueError(msg)

#         self.data = data
#         self.facet = facet
#         self.sharex = sharex
#         self.sharey = sharey

#         # Lazy properties
#         self._row_keys: list[Any] | None = None
#         self._col_keys: list[Any] | None = None

#     @property
#     def row_keys(self) -> list[Any]:
#         """Get unique, sorted keys for facet rows."""
#         if self._row_keys is None:
#             if self.facet.row:
#                 self._row_keys = (
#                     self.data.get_column(self.facet.row)
#                     .unique(maintain_order=True)
#                     .to_list()
#                 )
#             else:
#                 self._row_keys = []
#         return self._row_keys

#     @property
#     def col_keys(self) -> list[Any]:
#         """Get unique, sorted keys for facet columns."""
#         if self._col_keys is None:
#             if self.facet.col:
#                 self._col_keys = (
#                     self.data.get_column(self.facet.col)
#                     .unique(maintain_order=True)
#                     .to_list()
#                 )
#             else:
#                 self._col_keys = []
#         return self._col_keys

#     @property
#     def n_rows(self) -> int:
#         """Get the number of rows in the facet grid."""
#         return len(self.row_keys) if self.facet.row else 1

#     @property
#     def n_cols(self) -> int:
#         """Get the number of columns in the facet grid."""
#         return len(self.col_keys) if self.facet.col else 1

#     def create_figure(self) -> tuple[Figure, np.ndarray[Any, Any]]:
#         """Create the figure and axes grid for the facet plot."""
#         fig, axes = plt.subplots(
#             nrows=self.n_rows,
#             ncols=self.n_cols,
#             sharex=self.sharex,
#             sharey=self.sharey,
#             squeeze=False,  # Always return a 2D array for consistent indexing
#         )
#         return fig, axes

#     def iter_subplots(
#         self,
#         axes: np.ndarray[Any, Any],
#     ) -> Iterable[tuple[pl.DataFrame, Axes, dict[str, Any], int, int]]:
#         import polars as pl

#         row_facet = self.facet.row
#         col_facet = self.facet.col

#         if row_facet and col_facet:
#             for i, row_key in enumerate(self.row_keys):
#                 for j, col_key in enumerate(self.col_keys):
#                     ax = axes[i, j]
#                     titles = {
#                         "title": f"{col_facet} = {col_key}",
#                         "ylabel": f"{row_facet} = {row_key}",
#                     }
#                     subset = self.data.filter(
#                         (pl.col(row_facet) == row_key) & (pl.col(col_facet) == col_key),  # noqa: E501
#                     )
#                     yield subset, ax, titles, i, j
#         elif row_facet:
#             for i, row_key in enumerate(self.row_keys):
#                 ax = axes[i, 0]
#                 titles = {"ylabel": f"{row_facet} = {row_key}"}
#                 subset = self.data.filter(pl.col(row_facet) == row_key)
#                 yield subset, ax, titles, i, 0
#         elif col_facet:
#             for j, col_key in enumerate(self.col_keys):
#                 ax = axes[0, j]
#                 titles = {"title": f"{col_facet} = {col_key}"}
#                 subset = self.data.filter(pl.col(col_facet) == col_key)
#                 yield subset, ax, titles, 0, j
