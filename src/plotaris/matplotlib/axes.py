from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib.axis import XAxis
from matplotlib.ticker import EngFormatter, FuncFormatter

from plotaris.common.title import Title

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.axis import YAxis
    from matplotlib.text import Text


def format_axis(
    axis: XAxis | YAxis,
    /,
    label: str,
    fontdict: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Text:
    title = Title(label)
    if title.fmt is None:
        fmt = "g"
    elif isinstance(title.fmt, int):
        fmt = f".{title.fmt}f"
    else:
        fmt = title.fmt

    text = axis.set_label_text(str(title), fontdict, **kwargs)  # pyright: ignore[reportUnknownMemberType]

    scale = 10**title.power
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
