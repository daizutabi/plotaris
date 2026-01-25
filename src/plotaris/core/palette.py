from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from itertools import cycle
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import polars as pl


type VisualValue = str | int | float
"""Type alias for values that can be assigned to visual properties."""
type MappingVisualValue = Mapping[tuple[Any, ...], VisualValue]
"""Type alias for a mapping from data values (tuple) to a visual value."""

COLORS = [
    "#d42f7e",
    "#7c388c",
    "#1952a6",
    "#54a9cc",
    "#318c3a",
    "#f2ce00",
    "#e6820b",
    "#cf1111",
]
SIZES = [50, 100, 150, 200, 250]
SHAPES = ["o", "s", "^", "D", "v"]


@dataclass(frozen=True)
class Palette:
    """Manages the mapping of data values to visual properties."""

    color: tuple[str, ...] = field(default_factory=tuple)
    """Column(s) from the DataFrame used to determine color assignments."""
    size: tuple[str, ...] = field(default_factory=tuple)
    """Column(s) from the DataFrame used to determine size assignments."""
    shape: tuple[str, ...] = field(default_factory=tuple)
    """Column(s) from the DataFrame used to determine shape assignments."""

    def items(self) -> Iterator[tuple[str, tuple[str, ...]]]:
        """Iterate over the visual properties and their associated column names."""
        for f in fields(self):
            if value := getattr(self, f.name):
                yield f.name, value

    def build(
        self,
        data: pl.DataFrame,
        color: Sequence[VisualValue] | MappingVisualValue | None = None,
        size: Sequence[VisualValue] | MappingVisualValue | None = None,
        shape: Sequence[VisualValue] | MappingVisualValue | None = None,
    ) -> dict[str, MappingVisualValue]:
        """Create palettes (ordered lists of visual properties) for all aesthetics."""
        palette_default = {
            "color": (color, COLORS),
            "size": (size, SIZES),
            "shape": (shape, SHAPES),
        }

        palettes: dict[str, MappingVisualValue] = {}

        for name, columns in self.items():
            palette, default = palette_default[name]
            palettes[name] = create_palette(data, columns, palette, default)

        return palettes

    def get(
        self,
        row: Mapping[str, Any],
        palettes: dict[str, MappingVisualValue],
    ) -> dict[str, VisualValue]:
        """Get the visual properties based on the encoding."""
        properties: dict[str, VisualValue] = {}

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
        return {row: palette.get(row, next(defaults)) for row in rows}  # ty: ignore[no-matching-overload]

    return dict(zip(rows, cycle(palette or default)))
