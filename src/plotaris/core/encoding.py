from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from itertools import cycle
from typing import TYPE_CHECKING, Any

from plotaris.colors import COLORS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import polars as pl


type Property = str | int
type Palette = Mapping[tuple[Any, ...], Property]

SIZES = [50, 100, 150, 200, 250]
SHAPES = ["o", "s", "^", "D", "v"]


@dataclass(frozen=True)
class Encoding:
    """Declaratively specify the mapping between data and visual properties."""

    color: tuple[str, ...] = field(default_factory=tuple)
    """The encoding for the color property."""
    size: tuple[str, ...] = field(default_factory=tuple)
    """The encoding for the size property."""
    shape: tuple[str, ...] = field(default_factory=tuple)
    """The encoding for the shape property (e.g., for scatter plots)."""

    def items(self) -> Iterator[tuple[str, tuple[str, ...]]]:
        for f in fields(self):
            if value := getattr(self, f.name):
                yield f.name, value

    def build_palettes(
        self,
        data: pl.DataFrame,
        color: Sequence[Property] | Palette | None = None,
        size: Sequence[Property] | Palette | None = None,
        shape: Sequence[Property] | Palette | None = None,
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

    def get_properties(
        self,
        row: Mapping[str, Any],
        palettes: dict[str, Palette],
    ) -> dict[str, Property]:
        """Get the visual properties based on the encoding."""
        properties: dict[str, Property] = {}

        for name, columns in self.items():
            if palette := palettes.get(name):
                values = tuple(row[c] for c in columns)
                if value := palette.get(values):
                    properties[name] = value

        return properties


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
