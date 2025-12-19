from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grid:
    """Declaratively specify how to grid plots."""

    row: str | None = None
    """The column to use for creating rows of subplots."""
    col: str | None = None
    """The column to use for creating columns of subplots."""


@dataclass(frozen=True)
class Facet(Grid):
    """Declaratively specify how to facet plots."""

    wrap: int | None = None
    """The maximum number of columns before wrapping to a new row."""
