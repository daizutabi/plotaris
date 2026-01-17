from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest

from plotaris.config import init

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture(autouse=True)
def _clear_matplotlib_settings():  # pyright: ignore[reportUnusedFunction]
    """Clear matplotlib settings before each test."""
    original_style = plt.rcParams.copy()
    yield
    plt.rcParams.update(original_style)
    plt.style.use("default")


def test_init_default_values(mocker: MockerFixture):
    """Test init with default values."""
    mock_style_use = mocker.patch("matplotlib.pyplot.style.use")
    init()

    mock_style_use.assert_called_once_with("default")
    assert plt.rcParams["figure.dpi"] == 160
    assert plt.rcParams["figure.figsize"] == [3.0, 2.0]
    assert plt.rcParams["axes.labelsize"] == 9.5
    assert plt.rcParams["axes.titlesize"] == 9.5
    assert plt.rcParams["xtick.labelsize"] == 8.5
    assert plt.rcParams["ytick.labelsize"] == 8.5


def test_init_custom_values(mocker: MockerFixture):
    """Test init with custom values."""
    mock_style_use = mocker.patch("matplotlib.pyplot.style.use")
    init(
        style="seaborn",
        dpi=300,
        figsize=(8, 6),
        labelsize=12,
        titlesize=14,
        ticksize=10,
    )

    mock_style_use.assert_called_once_with("seaborn")
    assert plt.rcParams["figure.dpi"] == 300
    assert plt.rcParams["figure.figsize"] == [8.0, 6.0]
    assert plt.rcParams["axes.labelsize"] == 12
    assert plt.rcParams["axes.titlesize"] == 14
    assert plt.rcParams["xtick.labelsize"] == 10
    assert plt.rcParams["ytick.labelsize"] == 10
