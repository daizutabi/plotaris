from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
