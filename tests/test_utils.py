from __future__ import annotations

import pytest

from plotaris.utils import (
    get_unit_seperator,
    split_precision,
    split_unit_precision,
    to_tuple,
)


@pytest.mark.parametrize(
    ("values", "expected"),
    [(None, ()), ("abc", ("abc",)), (["abc"], ("abc",))],
)
def test_to_tuple(values: str | list[str] | None, expected: tuple[str, ...]) -> None:
    assert to_tuple(values) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Voltage (V)", "("),
        ("Current [A]", "["),
        ("Time s", None),
        ("Power (W)s", None),
    ],
)
def test_get_unit_seperator(label: str, expected: str | None) -> None:
    assert get_unit_seperator(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("", ("", None)),
        ("a", ("a", None)),
        ("a (b)", ("a (b)", None)),
        ("a [b:2]", ("a [b]", 2)),
        ("[b:2]", ("[b]", 2)),
    ],
)
def test_split_precision(label: str, expected: tuple[str, int | None]) -> None:
    assert split_precision(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("a", ("a", "", None)),
        ("a (b)", ("a", "b", None)),
        ("a [b:2]", ("a", "b", 2)),
        ("a [:2]", ("a", "", 2)),
        ("[b:2]", ("", "b", 2)),
        ("[:2]", ("", "", 2)),
    ],
)
def test_split_unit_precision(
    label: str,
    expected: tuple[str, str, int | None],
) -> None:
    assert split_unit_precision(label) == expected
