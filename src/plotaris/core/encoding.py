from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from plotaris.colors import COLORS

if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl

type Palette = dict[tuple[Any, ...], str] | dict[tuple[Any, ...], int]


@dataclass(frozen=True)
class Encoding:
    """Declaratively specify the mapping between data and visual properties."""

    x: str | pl.Expr | None = None
    """The encoding for the x-axis."""
    y: str | pl.Expr | None = None
    """The encoding for the y-axis."""
    color: list[str] = field(default_factory=list)
    """The encoding for the color property."""
    size: list[str] = field(default_factory=list)
    """The encoding for the size property."""
    shape: list[str] = field(default_factory=list)
    """The encoding for the shape property (e.g., for scatter plots)."""

    palette_names: ClassVar[list[str]] = ["color", "size", "shape"]

    def get(self, name: str) -> list[str]:
        if name in self.palette_names:
            return getattr(self, name)

        msg = f"Encoding has no aesthetic '{name}'"
        raise KeyError(msg)

    def items(self) -> Iterator[tuple[str, list[str]]]:
        for name in self.palette_names:
            if value := getattr(self, name):
                yield name, value

    def create_palettes(self, data: pl.DataFrame) -> dict[str, Palette]:
        """Create palettes (ordered lists of visual properties) for all aesthetics."""
        palettes: dict[str, Palette] = {}

        if self.color:
            palettes["color"] = create_palette(data, self.color, COLORS)

        if self.size:
            sizes = [50, 100, 150, 200, 250]
            palettes["size"] = create_palette(data, self.size, sizes)

        if self.shape:
            shapes = ["o", "s", "^", "D", "v"]
            palettes["shape"] = create_palette(data, self.shape, shapes)

        return palettes


def create_palette[T](
    data: pl.DataFrame,
    columns: list[str],
    palette: list[T],
) -> dict[tuple[Any, ...], T]:
    """Create an ordered palette of visual properties corresponding to unique data values."""  # noqa: E501
    rows = data.select(columns).unique(maintain_order=True).rows()
    return {row: palette[i % len(palette)] for i, row in enumerate(rows)}
