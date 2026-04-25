from __future__ import annotations

from typing import Literal, override


class Title:
    text: str
    label: str
    unit: str
    power: int
    fmt: str | int | None

    def __init__(self, text: str) -> None:
        self.text = text
        self.label, self.unit, self.fmt = split_unit_format(text)
        self.power = get_power(self.unit)

    @override
    def __str__(self) -> str:
        if not self.unit:
            return self.label

        if "[" in self.text:
            return f"{self.label} [{self.unit}]"
        return f"{self.label} ({self.unit})"


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


def split_format(text: str) -> tuple[str, str | int | None]:
    """Split a format specifier from a string.

    The format is expected to be in the format ":<fmt>",
    e.g., "Current [A]:2" or "Voltage (V):~s".

    Args:
        text: The label string, potentially containing a precision specifier.

    Returns:
        A tuple containing:
            - The string with the format part removed (e.g., "Current [A]").
            - The string or integer value of the format, or None if not found.

    Examples:
        >>> split_format("Voltage (V):2")
        ('Voltage (V)', 2)
        >>> split_format("Current (A):~s")
        ('Current (A)', '~s')
    """
    if ":" not in text:
        return text, None

    prefix, fmt = text.rsplit(":", 1)

    if ")" in fmt or "]" in fmt:
        return text, None

    if fmt.isdigit():
        return prefix, int(fmt)
    return prefix, fmt


def split_unit_format(text: str) -> tuple[str, str, str | int | None]:
    """Splits a string into its constituent parts: text, unit, and format.

    Args:
        text: The full label string to parse.

    Returns:
        A tuple containing:
            - The main text.
            - The unit string (e.g., "V", "m/s").
            - The format, or None.

    Examples:
        >>> split_unit_format("Voltage (V):2")
        ('Voltage', 'V', 2)
    """
    text, fmt = split_format(text)

    sep = get_unit_seperator(text)

    if not sep:
        return text, "", fmt

    text, unit = text.rsplit(sep, 1)
    return text.rstrip(), unit[:-1], fmt


def get_power(unit: str) -> int:
    if "/" in unit:
        a, b = unit.split("/", maxsplit=1)
        ap = get_power(a)
        bp = get_power(b)
        return ap - bp

    if len(unit) < 2:
        return 0

    prefix = unit[0]
    m = int(unit[-1]) if unit[-1].isdigit() else 1
    match prefix:
        case "G":
            return 9 * m
        case "M":
            return 6 * m
        case "k":
            return 3 * m
        case "m":
            return -3 * m
        case "\u00b5":
            return -6 * m
        case "\u03bc":
            return -6 * m
        case "n":
            return -9 * m
        case "p":
            return -12 * m
        case "f":
            return -15 * m
        case _:
            return 0
