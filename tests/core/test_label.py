from __future__ import annotations

import pytest

from plotaris.core.label import Label, get_unit_seperator, split_places, split_unit


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
def test_split_places(label: str, expected: tuple[str, str | None]) -> None:
    assert split_places(label) == expected


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
def test_split_unit(label: str, expected: tuple[str, str, int | None]) -> None:
    assert split_unit(label) == expected


def test_str_default() -> None:
    label = Label({"a": 1, "b": "test"})
    assert str(label) == "a=1, b=test"


def test_str() -> None:
    label = Label({"a": 1, "b": "test"}, sep="; ", eq=" -> ")
    assert str(label) == "a -> 1; b -> test"


def test_format_default() -> None:
    label = Label({"a": 1e-3})
    assert label.format() == "a=0.001"


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("_{}_", "a=_1e-06_"),
        ("_{:.6f}_", "a=_0.000001_"),
        ("A [V]", "A=1µV"),
        ("B(V)", "B=1µV"),
        (("b", "{}"), "b=1e-06"),
        (("b", "A [V]"), "A=1µV"),
        (("b", "(V)"), "b=1µV"),
    ],
)
def test_format_str(fmt: str | tuple[str, str], expected: str) -> None:
    label = Label({"a": 1e-6})
    assert label.format(a=fmt) == expected


def test_format_callable() -> None:
    label = Label({"a": 10})
    assert label.format({"a": lambda x: str(10 * x)}) == "a=100"
    assert label.format({"a": ("b", lambda x: str(10 * x))}) == "b=100"
