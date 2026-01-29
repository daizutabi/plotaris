from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING, Any, Self

from plotaris.utils import to_tuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import polars as pl

type VisualValue = str | int | float
"""Type alias for values that can be assigned to visual properties."""


class Base:
    columns: dict[str, tuple[str, ...]]
    _default: dict[str, list[VisualValue]]
    _mapping: dict[str, dict[tuple[Any, ...], VisualValue]]
    _palettes: dict[str, dict[tuple[Any, ...], VisualValue | None]]

    def __init__(self, /, **columns: str | Iterable[str] | None) -> None:
        self.columns = {k: to_tuple(v) for k, v in columns.items() if v is not None}
        self._mapping = {}
        self._default = {}
        self._palettes = {}

    def default(self, /, **default: Iterable[VisualValue] | None) -> Self:
        self._default = {k: list(v) for k, v in default.items() if v is not None}
        return self

    def mapping(self, /, **mapping: Mapping[tuple[Any, ...], VisualValue]) -> Self:
        self._mapping = {k: dict(v) for k, v in mapping.items()}
        return self

    def set(self, data: pl.DataFrame, /) -> Self:
        palettes: dict[str, dict[tuple[Any, ...], VisualValue | None]] = {}

        for name, columns in self.columns.items():
            default = self._default.get(name, [])
            mapping = self._mapping.get(name, {})
            palettes[name] = create_palette(data, columns, default, mapping)

        self._palettes = palettes
        return self

    def get(self, data: Mapping[str, Any], /) -> dict[str, VisualValue | None]:
        """Get the visual properties based on the encoding."""
        properties: dict[str, VisualValue | None] = {}

        for name, columns in self.columns.items():
            palette = self._palettes[name]
            values = tuple(data[c] for c in columns)
            properties[name] = palette.get(values)

        return properties


def create_palette[T](
    data: pl.DataFrame,
    columns: Iterable[str],
    default: Sequence[T],
    mapping: Mapping[tuple[Any, ...], T],
) -> dict[tuple[Any, ...], T | None]:
    rows = data.select(columns).unique(maintain_order=True).rows()

    default_ = default or [None]

    if mapping:
        defaults = cycle(default_)
        return {row: mapping.get(row, next(defaults)) for row in rows}

    return dict(zip(rows, cycle(default_)))


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


class Palette(Base):
    def __init__(
        self,
        color: str | Iterable[str] | None = None,
        size: str | Iterable[str] | None = None,
        shape: str | Iterable[str] | None = None,
    ) -> None:
        super().__init__(color=color, size=size, shape=shape)
        self.default(
            color=color and COLORS,
            size=size and SIZES,
            shape=shape and SHAPES,
        )
