from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polars._typing import IntoExpr


@dataclass(frozen=True)
class Encoding:
    """Declaratively specify the mapping between data and visual properties."""

    x: IntoExpr = None
    """The encoding for the x-axis."""
    y: IntoExpr = None
    """The encoding for the y-axis."""
    color: IntoExpr = None
    """The encoding for the color property."""
    size: IntoExpr = None
    """The encoding for the size property."""
    shape: IntoExpr = None
    """The encoding for the shape property (e.g., for scatter plots)."""

    def to_dict(self) -> dict[str, IntoExpr]:
        """Return a dictionary of non-None encodings."""
        return {k: v for k, v in asdict(self).items() if v is not None}
