from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib.axis import XAxis
from matplotlib.ticker import EngFormatter, FuncFormatter

from plotaris.utils import get_unit_seperator, split_precision

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.axis import YAxis
    from matplotlib.text import Text


def _get_power(unit: str) -> int:
    if len(unit) == 1:
        return 0

    prefix = unit[0].replace("μ", EngFormatter.ENG_PREFIXES[-6])

    for power, value in EngFormatter.ENG_PREFIXES.items():
        if prefix == value:
            return power

    return 0


def format_axis(
    axis: XAxis | YAxis,
    /,
    label: str,
    fontdict: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Text:
    sep = get_unit_seperator(label)
    label, precision = split_precision(label, sep)
    fmt = "g" if precision is None else f".{precision}f"

    text = axis.set_label_text(label, fontdict, **kwargs)  # pyright: ignore[reportUnknownMemberType]

    if sep:
        _, unit = label[:-1].rsplit(sep, 1)
        scale = 10 ** _get_power(unit)
    else:
        scale = 1

    func = FuncFormatter(lambda x, _: f"{x / scale:{fmt}}")  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    axis.set_major_formatter(func)

    return text


def format_axes(
    axes: Axes,
    /,
    xlabel: str | None = None,
    ylabel: str | None = None,
    fontdict: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Axes:
    if xlabel:
        format_axis(axes.xaxis, xlabel, fontdict, **kwargs)
    if ylabel:
        format_axis(axes.yaxis, ylabel, fontdict, **kwargs)
    return axes


def set_axis_log(
    axis: XAxis | YAxis,
    /,
    lim: tuple[float, float] | None = None,
    *,
    unit: str = "",
    places: int | None = None,
    sep: str = "",
) -> None:
    if isinstance(axis, XAxis):
        axis.axes.set(xscale="log", xlim=lim)
    else:
        axis.axes.set(yscale="log", ylim=lim)
    axis.set_major_formatter(EngFormatter(unit, places, sep))


def set_axes_log(
    axes: Axes,
    /,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    *,
    unit: str = "",
    xunit: str = "",
    yunit: str = "",
    places: int | None = None,
    xplaces: int | None = None,
    yplaces: int | None = None,
    sep: str = "",
) -> Axes:
    if xlim:
        unit = xunit or unit
        places = xplaces if xplaces is not None else places
        set_axis_log(axes.xaxis, xlim, unit=unit, places=places, sep=sep)
    if ylim:
        unit = yunit or unit
        places = yplaces if yplaces is not None else places
        set_axis_log(axes.yaxis, ylim, unit=unit, places=places, sep=sep)
    return axes


def axes_text(
    ax: Axes,
    /,
    x: float,
    y: float,
    s: str,
    fontdict: dict[str, Any] | None = None,
    *,
    lim: float = 0.2,
    **kwargs: Any,
) -> Text:
    ha = "left" if x < lim else "right" if x > 1 - lim else "center"
    va = "bottom" if y < lim else "top" if y > 1 - lim else "center"
    return ax.text(x, y, s, fontdict, ha=ha, va=va, transform=ax.transAxes, **kwargs)  # pyright: ignore[reportUnknownMemberType]
