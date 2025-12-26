from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from plotaris.core.encoding import Encoding, map_color

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_encoding_map_color(mocker: MockerFixture) -> None:
    mock_map_color = mocker.patch("plotaris.core.encoding.map_color")

    data = pl.DataFrame()
    encoding = Encoding(color="category")

    encoding.map_color(data)

    mock_map_color.assert_called_once_with(data, "category")


def test_encoding_map_color_no_color(mocker: MockerFixture) -> None:
    mock_map_color = mocker.patch("plotaris.core.encoding.map_color")

    data = pl.DataFrame()
    encoding = Encoding(color=None)

    result = encoding.map_color(data)

    assert result == {}
    mock_map_color.assert_not_called()


def test_map_color_string() -> None:
    data = pl.DataFrame({"category": ["A", "B", "A", "C", "B"]})

    result = map_color(data, "category")

    colors = result["color"]
    assert len(colors) == 5
    # assert all(isinstance(color, tuple) and len(color) == 4 for color in colors)
