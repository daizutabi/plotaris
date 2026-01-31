from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable


def to_tuple(values: str | Iterable[str] | None, /) -> tuple[str, ...]:
    """Convert a value to a tuple of strings.

    This utility function handles None, a single string, or an iterable of
    strings and ensures the output is always a tuple of strings.

    Args:
        values: The input value to convert.

    Returns:
        A tuple of strings.
    """
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(values)


def get_unit_seperator(label: str) -> Literal["(", "["] | None:
    """Finds the opening separator of a unit string at the end of a label.

    A unit string is assumed to be enclosed in parentheses or square brackets
    at the very end of the label string.

    Args:
        label: The label string to inspect, e.g., "Voltage (V)".

    Returns:
        The opening bracket character ("(" or "[") if a valid unit suffix is
        found, otherwise None.
    """
    if "[" in label and label.endswith("]"):
        return "["
    if "(" in label and label.endswith(")"):
        return "("
    return None


def split_precision(label: str, sep: str | None = None) -> tuple[str, int | None]:
    """Splits a precision specifier from a label string.

    The precision is expected to be in the format ":<digits>" within the unit
    part of the label, e.g., "Current [A:2]".

    Args:
        label: The label string, potentially containing a precision specifier.
        sep: The opening separator for the unit, e.g., "(". If None, it will be
            auto-detected.

    Returns:
        A tuple containing:
            - The label string with the precision part removed (e.g., "Current [A]").
            - The integer value of the precision, or None if not found.
    """
    sep = sep or get_unit_seperator(label)

    if not sep:
        return label, None

    _, unit = label.rsplit(sep, 1)

    if ":" not in unit:
        return label, None

    suffix = label[-1]
    label, places = label[:-1].rsplit(":", 1)

    return f"{label}{suffix}", int(places)


def split_unit_precision(label: str) -> tuple[str, str, int | None]:
    """Splits a label string into its constituent parts: text, unit, and precision.

    This function parses a string that may contain a unit and a precision
    specifier, e.g., "Label Text (unit:precision)".

    Args:
        label: The full label string to parse.

    Returns:
        A tuple containing:
            - The main label text.
            - The unit string (e.g., "V", "m/s").
            - The integer precision, or None.
    """
    sep = get_unit_seperator(label)

    if not sep:
        return label, "", None

    label, precision = split_precision(label, sep)
    label, unit = label.rsplit(sep, 1)
    return label.rstrip(), unit[:-1], precision
