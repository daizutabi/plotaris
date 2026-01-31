from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, override

from matplotlib.ticker import EngFormatter

if TYPE_CHECKING:
    from collections.abc import Callable


def get_unit_seperator(label: str) -> Literal["(", "["] | None:
    if "[" in label and label.endswith("]"):
        return "["
    if "(" in label and label.endswith(")"):
        return "("
    return None


def split_places(label: str) -> tuple[str, int | None]:
    sep = get_unit_seperator(label)

    if not sep:
        return label, None

    _, unit = label.rsplit(sep, 1)

    if ":" not in unit:
        return label, None

    suffix = label[-1]
    label, places = label[:-1].rsplit(":", 1)

    return f"{label}{suffix}", int(places)


def split_unit(label: str) -> tuple[str, str, int | None]:
    sep = get_unit_seperator(label)

    if not sep:
        return label, "", None

    label, places = split_places(label)
    label, unit = label.rsplit(sep, 1)
    return label.rstrip(), unit[:-1], places


type Format = str | Callable[[Any], str]


def _format(value: Any, fmt: Format | None, sep: str = "") -> str | tuple[str, str]:
    if fmt is None:
        return str(value)
    if callable(fmt):
        return fmt(value)
    if "{" in fmt and "}" in fmt:
        return fmt.format(value)

    label, unit, places = split_unit(fmt)
    return label, EngFormatter(unit, places, sep)(value)


@dataclass
class Label:
    data: dict[str, Any]
    eq: str = "="
    sep: str = ", "

    @override
    def __str__(self) -> str:
        return self.sep.join(f"{k}{self.eq}{v}" for k, v in self.data.items())

    def format(
        self,
        formats: dict[str, Format | tuple[str, Format]] | None = None,
        /,
        **kwargs: Format | tuple[str, Format],
    ) -> str:
        formats = (formats or {}) | kwargs

        parts: list[str] = []

        for key, value in self.data.items():
            fmt = formats.get(key)
            key_, fmt = fmt if isinstance(fmt, tuple) else (key, fmt)
            formatted = _format(value, fmt)  # ty:ignore[invalid-argument-type]

            if isinstance(formatted, tuple):
                parts.append(f"{formatted[0] or key_}{self.eq}{formatted[1]}")
            else:
                parts.append(f"{key_}{self.eq}{formatted}")

        return self.sep.join(parts)
