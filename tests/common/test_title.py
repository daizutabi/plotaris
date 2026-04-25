from __future__ import annotations

import pytest

from plotaris.common.title import (
    Title,
    get_power,
    get_unit_seperator,
    split_format,
    split_unit_format,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Voltage (V)", "("),
        ("Current [A]", "["),
        ("Time s", None),
        ("Power (W)s", None),
    ],
)
def test_get_unit_seperator(text: str, expected: str | None) -> None:
    assert get_unit_seperator(text) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("", ("", None)),
        ("a", ("a", None)),
        ("a (b)", ("a (b)", None)),
        ("a [b]:2", ("a [b]", 2)),
        ("[b]:~s", ("[b]", "~s")),
        ("[b:2]", ("[b:2]", None)),
        ("(b:2)", ("(b:2)", None)),
        ("(b:2):3", ("(b:2)", 3)),
    ],
)
def test_split_format(label: str, expected: tuple[str, str | int | None]) -> None:
    assert split_format(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("a", ("a", "", None)),
        ("a (b)", ("a", "b", None)),
        ("a [b]:2", ("a", "b", 2)),
        ("a:~s", ("a", "", "~s")),
        ("[b]:2", ("", "b", 2)),
        (":2", ("", "", 2)),
    ],
)
def test_split_unit_format(
    label: str,
    expected: tuple[str, str, int | None],
) -> None:
    assert split_unit_format(label) == expected


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("a", 0),
        ("invalid", 0),
        ("GB", 9),
        ("MΩ", 6),
        ("km", 3),
        ("mg", -3),
        ("µs", -6),
        ("μs", -6),
        ("ns", -9),
        ("pA", -12),
        ("fF", -15),
        ("/mm", 3),
        ("km/kg", 0),
        ("km/nA", 12),
        ("km2", 6),
        ("nm3", -27),
    ],
)
def test_get_power(unit: str, expected: int) -> None:
    assert get_power(unit) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a", "a"),
        ("text [m]", "text [m]"),
        ("text (m):.3f", "text (m)"),
        ("text [m]:3", "text [m]"),
    ],
)
def test_title_str(text: str, expected: str) -> None:
    assert str(Title(text)) == expected
