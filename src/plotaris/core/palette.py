from __future__ import annotations

from collections.abc import Mapping
from itertools import cycle
from typing import TYPE_CHECKING, Any, Self

import matplotlib.pyplot as plt

from plotaris.utils import to_tuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

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

    def mapping(
        self,
        /,
        **mapping: Mapping[Any | tuple[Any, ...], VisualValue],
    ) -> Self:
        def to_tuple_dict(
            x: Mapping[Any | tuple[Any, ...], VisualValue],
            /,
        ) -> dict[tuple[Any, ...], VisualValue]:
            return {k if isinstance(k, tuple) else (k,): v for k, v in x.items()}

        self._mapping = {k: to_tuple_dict(v) for k, v in mapping.items()}
        return self

    def set(self, data: pl.DataFrame, /) -> Self:
        palettes: dict[str, dict[tuple[Any, ...], VisualValue | None]] = {}

        for name, columns in self.columns.items():
            default = self._default.get(name)
            mapping = self._mapping.get(name)
            palettes[name] = create_palette(data, columns, default, mapping)

        self._palettes = palettes
        return self

    def get(
        self,
        data: Mapping[str, Any] | pl.DataFrame,
        /,
    ) -> dict[str, VisualValue | None]:
        """Get the visual properties based on the encoding.

        Args:
            data: A single row (as a Mapping) or a subset of a DataFrame.
                If a DataFrame is provided, it must contain only one unique
                combination of values for the columns that define the palette.
        """
        properties: dict[str, VisualValue | None] = {}

        for name, columns in self.columns.items():
            palette = self._palettes[name]
            key = _get_palette_key(data, columns)
            properties[name] = palette.get(key)

        return properties


def _get_palette_key(
    data: Mapping[str, Any] | pl.DataFrame,
    columns: Iterable[str],
) -> tuple[Any, ...]:
    """Get the key for palette lookup from a row or a DataFrame.

    - If data is a Mapping (a single row), the key is the tuple of its values.
    - If data is a DataFrame and contains a single unique combination of values
      for the given columns, the key is that unique tuple.
    - Otherwise, returns an empty tuple.
    """
    if isinstance(data, Mapping):
        return tuple(data[c] for c in columns)

    unique_rows = data.select(columns).unique()
    if len(unique_rows) == 1:
        return tuple(unique_rows.row(0))

    return ()


def create_palette[T](
    data: pl.DataFrame,
    columns: Iterable[str],
    default: Sequence[T] | None = None,
    mapping: Mapping[tuple[Any, ...], T] | None = None,
) -> dict[tuple[Any, ...], T | None]:
    rows = data.select(columns).unique(maintain_order=True).rows()

    default_ = default or [None]

    if mapping:
        defaults = cycle(default_)
        return {row: mapping.get(row, next(defaults)) for row in rows}

    return dict(zip(rows, cycle(default_)))


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
            color=color and plt.rcParams["axes.prop_cycle"].by_key()["color"],
            size=size and SIZES,
            shape=shape and SHAPES,
        )
