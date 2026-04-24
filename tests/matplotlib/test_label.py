from __future__ import annotations

import pytest

from plotaris.matplotlib.label import Label


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


def test_format_unit_sep() -> None:
    label = Label({"a": 1e-6}, unit_sep=" ")
    assert label.format(a="A [V]") == "A=1 µV"


def test_format_callable() -> None:
    label = Label({"a": 10})
    assert label.format({"a": lambda x: str(10 * x)}) == "a=100"
    assert label.format({"a": ("b", lambda x: str(10 * x))}) == "b=100"
