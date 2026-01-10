from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import cycle
from typing import TYPE_CHECKING, Any, ClassVar

from plotaris.colors import COLORS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import polars as pl

type Palette = Mapping[tuple[Any, ...], str | int]

SIZES = [50, 100, 150, 200, 250]
SHAPES = ["o", "s", "^", "D", "v"]


@dataclass(frozen=True)
class Encoding:
    """Declaratively specify the mapping between data and visual properties."""

    x: str | pl.Expr | None = None
    """The encoding for the x-axis."""
    y: str | pl.Expr | None = None
    """The encoding for the y-axis."""
    color: tuple[str, ...] = field(default_factory=tuple)
    """The encoding for the color property."""
    size: tuple[str, ...] = field(default_factory=tuple)
    """The encoding for the size property."""
    shape: tuple[str, ...] = field(default_factory=tuple)
    """The encoding for the shape property (e.g., for scatter plots)."""

    palette_names: ClassVar[tuple[str, ...]] = ("color", "size", "shape")

    def get(self, name: str) -> tuple[str, ...]:
        if name in self.palette_names:
            return getattr(self, name)

        msg = f"Encoding has no aesthetic '{name}'"
        raise KeyError(msg)

    def items(self) -> Iterator[tuple[str, tuple[str, ...]]]:
        for name in self.palette_names:
            if value := getattr(self, name):
                yield name, value

    def palettes(
        self,
        data: pl.DataFrame,
        color: Sequence[str] | Mapping[tuple[Any, ...], str] | None = None,
        size: Sequence[int] | Mapping[tuple[Any, ...], int] | None = None,
        shape: Sequence[str] | Mapping[tuple[Any, ...], str] | None = None,
    ) -> dict[str, Palette]:
        """Create palettes (ordered lists of visual properties) for all aesthetics."""
        palette_default = {
            "color": (color, COLORS),
            "size": (size, SIZES),
            "shape": (shape, SHAPES),
        }

        palettes: dict[str, Palette] = {}

        for name, columns in self.items():
            palettes[name] = create_palette(data, columns, *palette_default[name])

        return palettes


def create_palette[T](
    data: pl.DataFrame,
    columns: Iterable[str],
    palette: Sequence[T] | Mapping[tuple[Any, ...], T] | None,
    default: Sequence[T],
) -> dict[tuple[Any, ...], T]:
    """Create an ordered palette of visual properties corresponding to unique data values."""  # noqa: E501
    rows = data.select(columns).unique(maintain_order=True).rows()

    if isinstance(palette, Mapping):
        defaults = cycle(default)
        return {row: palette.get(row, next(defaults)) for row in rows}

    return dict(zip(rows, cycle(palette or default)))
