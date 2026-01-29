from __future__ import annotations

import pytest

from plotaris.utils import to_tuple


@pytest.mark.parametrize(
    ("values", "expected"),
    [(None, ()), ("abc", ("abc",)), (["abc"], ("abc",))],
)
def test_to_tuple(values: str | list[str] | None, expected: tuple[str, ...]) -> None:
    assert to_tuple(values) == expected
