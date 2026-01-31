"""Provides a flexible and powerful way to format key-value data into labels.

This module is designed for creating formatted string representations from a
dictionary of data, which is particularly useful for plot legends, titles, and
annotations. It includes a custom mini-language for format strings that can
specify a new label, units, and precision for use with Matplotlib's EngFormatter,
allowing for concise representation of scientific and engineering data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from matplotlib.ticker import EngFormatter

from plotaris.utils import split_unit_precision

if TYPE_CHECKING:
    from collections.abc import Callable


type Format = str | Callable[[Any], str]


def _format(value: Any, fmt: Format | None, sep: str = "") -> str | tuple[str, str]:
    """Formats a single value based on the provided format specifier.

    This function acts as a dispatcher, handling four types of formats:
    1. None: Converts the value to a string.
    2. Callable: Calls the function with the value.
    3. f-string: Uses the string's .format() method.
    4. Custom unit string: Parses the string for a new label, unit, and
       precision, and formats the value using EngFormatter.

    Args:
        value: The value to format.
        fmt: The format specifier.
        sep: The separator to use between the number and the unit for
            EngFormatter.

    Returns:
        A formatted string, or a tuple of (label, formatted_value) if the
        custom unit string format is used.
    """
    if fmt is None:
        return str(value)
    if callable(fmt):
        return fmt(value)
    if "{" in fmt and "}" in fmt:
        return fmt.format(value)

    label, unit, precision = split_unit_precision(fmt)
    return label, EngFormatter(unit, precision, sep)(value)


@dataclass
class Label:
    """Represents a set of key-value pairs for flexible string formatting.

    This class takes a dictionary of data and provides methods to format it into
    a string, with customizable separators and advanced formatting rules per key.

    Attributes:
        data: A dictionary of key-value pairs to be formatted.
        eq: The string to use as an equals sign between keys and values.
        sep: The string to use to separate key-value pairs.
    """

    data: dict[str, Any] = field(default_factory=dict)
    eq: str = "="
    sep: str = ", "
    unit_sep: str = ""

    @override
    def __str__(self) -> str:
        """Returns a simple string representation of the label data."""
        return self.sep.join(f"{k}{self.eq}{v}" for k, v in self.data.items())

    def format(
        self,
        formats: dict[str, Format | tuple[str, Format]] | None = None,
        /,
        **kwargs: Format | tuple[str, Format],
    ) -> str:
        """Formats the label data into a single string with custom rules.

        This method provides a highly flexible way to format the label's data.
        For each key in the data, a specific format can be applied.

        The format for a key can be specified in several ways:
        1.  **As a format string**: e.g., `"{:.2f}"`.
        2.  **As a callable**: e.g., a lambda function `lambda x: f"0x{x:X}"`.
        3.  **As a custom unit string**: A string that is parsed to define a new
            label, a unit, and an optional precision for `EngFormatter`.
            For example, `"Voltage [V:2]"` would rename the key to "Voltage"
            and format its value as a number with the "V" unit and 2 decimal
            places (e.g., "12.34 V", "1.23 kV").
        4.  **As a tuple `(new_key, format)`**: This allows explicitly renaming
            the key in the output string while applying any of the above
            format types.

        Args:
            formats: A dictionary mapping data keys to format specifiers.
            **kwargs: Per-key format specifiers can also be provided as keyword
                arguments, which will be merged with the `formats` dict.

        Returns:
            The final formatted string.
        """
        formats = (formats or {}) | kwargs

        parts: list[str] = []

        for key, value in self.data.items():
            fmt = formats.get(key)
            key_, fmt = fmt if isinstance(fmt, tuple) else (key, fmt)
            formatted = _format(value, fmt, self.unit_sep)  # ty: ignore[invalid-argument-type]

            if isinstance(formatted, tuple):
                parts.append(f"{formatted[0] or key_}{self.eq}{formatted[1]}")
            else:
                parts.append(f"{key_}{self.eq}{formatted}")

        return self.sep.join(parts)
