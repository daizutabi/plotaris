from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest

from plotaris.matplotlib.axes import axes_text, format_axes, format_axis, set_axes_log

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes

# pyright: reportUnknownMemberType=false


@pytest.fixture
def ax() -> Iterator[Axes]:
    fig = plt.figure(figsize=(3, 2))
    yield fig.add_subplot()
    plt.close(fig)


@pytest.mark.parametrize(
    "label",
    ["Voltage", "Voltage (V)", "Length [m]"],
)
def test_format_axis_without_prefix(ax: Axes, label: str) -> None:
    ax.set(ylim=(0, 2))
    text = format_axis(ax.yaxis, label)
    assert text.get_text() == label
    ticks = [x.get_text() for x in ax.yaxis.get_majorticklabels()]
    assert ticks == ["0", "0.5", "1", "1.5", "2"]


def test_format_axis_with_unknown_prefix(ax: Axes) -> None:
    label = "unknown [ABC]"
    ax.set(ylim=(0, 2))
    text = format_axis(ax.yaxis, label)
    assert text.get_text() == label
    ticks = [x.get_text() for x in ax.yaxis.get_majorticklabels()]
    assert ticks == ["0", "0.5", "1", "1.5", "2"]


def test_format_axis_with_prefix(ax: Axes) -> None:
    ax.set(ylim=(-2e-5, 2e-5))
    text = format_axis(ax.yaxis, "Voltage (µV)")
    assert text.get_text() == "Voltage (µV)"
    ticks = [x.get_text() for x in ax.yaxis.get_majorticklabels()]
    assert ticks == ["-20", "-10", "0", "10", "20"]


def test_format_axis_with_place(ax: Axes) -> None:
    ax.set(xlim=(0, 3000))
    text = format_axis(ax.xaxis, "Voltage [kV]:1")
    assert text.get_text() == "Voltage [kV]"
    ticks = [x.get_text() for x in ax.xaxis.get_majorticklabels()]
    assert ticks == ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]


def test_format_axes(ax: Axes) -> None:
    ax.set(xlim=(-1e-4, 1e-4), ylim=(-1e-7, 1e-7))
    format_axes(ax, "Voltage (mV):.2f", "Current (nA)")
    assert ax.xaxis.get_label().get_text() == "Voltage (mV)"
    ticks = [x.get_text() for x in ax.xaxis.get_majorticklabels()]
    assert ticks == ["-0.10", "-0.05", "0.00", "0.05", "0.10"]
    assert ax.yaxis.get_label().get_text() == "Current (nA)"
    ticks = [x.get_text() for x in ax.yaxis.get_majorticklabels()]
    assert ticks == ["-100", "-50", "0", "50", "100"]


def test_set_axes_log(ax: Axes) -> None:
    set_axes_log(ax, xlim=(1e-9, 1e-6), ylim=(1e3, 1e6))
    ticks = [x.get_text() for x in ax.xaxis.get_majorticklabels()]
    assert ticks == ["100p", "1n", "10n", "100n", "1µ", "10µ", "100µ"]
    ticks = [x.get_text() for x in ax.yaxis.get_majorticklabels()]
    assert ticks == ["10", "100", "1k", "10k", "100k", "1M", "10M"]


@pytest.mark.parametrize(
    ("x", "y", "lim", "expected_ha", "expected_va"),
    [
        (0.1, 0.5, 0.2, "left", "center"),
        (0.9, 0.5, 0.2, "right", "center"),
        (0.5, 0.5, 0.2, "center", "center"),
        (0.5, 0.1, 0.2, "center", "bottom"),
        (0.5, 0.9, 0.2, "center", "top"),
        (0.1, 0.1, 0.2, "left", "bottom"),
        (0.9, 0.9, 0.2, "right", "top"),
        (0.3, 0.5, 0.4, "left", "center"),
    ],
)
def test_axes_text_alignment(
    ax: Axes,
    x: float,
    y: float,
    lim: float,
    expected_ha: str,
    expected_va: str,
) -> None:
    text = axes_text(ax, x, y, "test", lim=lim)
    assert text.get_horizontalalignment() == expected_ha
    assert text.get_verticalalignment() == expected_va
