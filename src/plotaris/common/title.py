from __future__ import annotations

from typing import Literal


class Title:
    text: str
    label: str
    unit: str
    power: int | None
    fmt: str | None
    precision: int | None

    def __init__(self, text: str) -> None:
        self.text = text
        self.label, self.unit, self.precision = split_unit_precision(text)
        self.power = get_power(self.unit)


def get_unit_seperator(text: str) -> Literal["(", "["] | None:
    """Find the opening separator of a unit string at the end of a string.

    A unit string is assumed to be enclosed in parentheses or square brackets
    at the very end of the string.

    Args:
        text: The string to inspect, e.g., "Voltage (V)".

    Returns:
        The opening bracket character ("(" or "[") if a valid unit suffix is
        found, otherwise None.

    Examples:
        >>> get_unit_seperator("Voltage (V)")
        '('
        >>> get_unit_seperator("Current [A]")
        '['
        >>> get_unit_seperator("Resistance Ω") is None
        True
    """
    if "[" in text and text.endswith("]"):
        return "["
    if "(" in text and text.endswith(")"):
        return "("
    return None


def split_precision(text: str, sep: str | None = None) -> tuple[str, int | None]:
    """Split a precision specifier from a string.

    The precision is expected to be in the format ":<digits>" within the unit
    part of the string, e.g., "Current [A:2]".

    Args:
        text: The label string, potentially containing a precision specifier.
        sep: The opening separator for the unit, e.g., "(". If None, it will be
            auto-detected.

    Returns:
        A tuple containing:
            - The string with the precision part removed (e.g., "Current [A]").
            - The integer value of the precision, or None if not found.

    Examples:
        >>> split_precision("Voltage (V:2)")
        ('Voltage (V)', 2)
    """
    sep = sep or get_unit_seperator(text)

    if not sep:
        return text, None

    _, unit = text.rsplit(sep, 1)

    if ":" not in unit:
        return text, None

    suffix = text[-1]
    text, places = text[:-1].rsplit(":", 1)

    return f"{text}{suffix}", int(places)


def split_unit_precision(text: str) -> tuple[str, str, int | None]:
    """Splits a string into its constituent parts: text, unit, and precision.

    This function parses a string that may contain a unit and a precision
    specifier, e.g., "Label Text (unit:precision)".

    Args:
        text: The full label string to parse.

    Returns:
        A tuple containing:
            - The main text.
            - The unit string (e.g., "V", "m/s").
            - The integer precision, or None.

    Examples:
        >>> split_unit_precision("Voltage (V:2)")
        ('Voltage', 'V', 2)
    """
    sep = get_unit_seperator(text)

    if not sep:
        return text, "", None

    text, precision = split_precision(text, sep)
    text, unit = text.rsplit(sep, 1)
    return text.rstrip(), unit[:-1], precision


def get_power(unit: str) -> int | None:
    if len(unit) < 2:
        return None

    prefix = unit[0]
    match prefix:
        case "G":
            return 9
        case "M":
            return 6
        case "k":
            return 3
        case "m":
            return -3
        case "µ":
            return -6
        case "n":
            return -9
        case "p":
            return -12
        case "f":
            return -15
        case _:
            return None
