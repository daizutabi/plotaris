from __future__ import annotations

from typing import cast

import altair as alt

from plotaris.common.title import Title


def get_title(text: str | Title | None, /) -> Title | None:
    if not text:
        return None
    return Title(text) if isinstance(text, str) else text


def set_title[T: alt.X | alt.Y | alt.Color | alt.Shape](x: T, title: Title) -> T:
    z = x.title(title.text) if isinstance(x, (alt.X, alt.Y)) else x.title(title.label)

    return cast("T", z)


def get_label_expr(title: Title, fmt: str | None = None) -> str | None:
    labelExpr = "datum.value"

    if title.power:
        labelExpr = f"{labelExpr}*1e{-title.power}"
    if isinstance(title.fmt, int):
        labelExpr = f"format({labelExpr},'.{title.fmt}f')"
    elif title.fmt:
        labelExpr = f"format({labelExpr},'{title.fmt}')"
    elif fmt:
        labelExpr = f"format({labelExpr},'{fmt}')"

    return None if labelExpr == "datum.value" else labelExpr


def set_axis[T: alt.X | alt.Y](x: T, title: Title) -> T:
    if labelExpr := get_label_expr(title):
        z = x.axis(labelExpr=labelExpr, labelFlush=False)
    else:
        z = x.axis(labelFlush=False)

    return cast("T", z)


def set_legend[T: alt.Color | alt.Shape](x: T, title: Title) -> T:
    if labelExpr := get_label_expr(title, "~s"):
        if title.unit:
            labelExpr = f"{labelExpr}+'{title.unit}'"
        z = x.legend(labelExpr=labelExpr)
        return cast("T", z)

    return x


def X(shorthand: str, title: str | Title | None = None) -> alt.X:
    x = alt.X(shorthand)
    if title := get_title(title):
        x = set_title(x, title)
        x = set_axis(x, title)
    return x


def Y(shorthand: str, title: str | Title | None = None) -> alt.Y:
    y = alt.Y(shorthand)
    if title := get_title(title):
        y = set_title(y, title)
        y = set_axis(y, title)
    return y


def Color(shorthand: str, title: str | Title | None = None) -> alt.Color:
    color = alt.Color(shorthand)
    if title := get_title(title):
        color = set_title(color, title)
        color = set_legend(color, title)
    return color


def Shape(shorthand: str, title: str | Title | None = None) -> alt.Shape:
    shape = alt.Shape(shorthand)
    if title := get_title(title):
        shape = set_title(shape, title)
        shape = set_legend(shape, title)
    return shape
