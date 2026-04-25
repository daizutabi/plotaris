from __future__ import annotations

from typing import Any, Literal, cast

import altair as alt

from plotaris.common.title import Title


def get_title(text: str | Title | None, /) -> Title | None:
    if not text:
        return None
    return Title(text) if isinstance(text, str) else text


def set_title[T: alt.X | alt.Y | alt.Color | alt.Shape | alt.Column | alt.Row](
    x: T,
    name: Literal["axis", "legend", "header"],
    title: Title,
) -> T:
    text = str(title) if name == "axis" else title.label
    if text:
        return cast("T", x.title(text))
    return x


def get_label_expr(
    title: Title,
    fmt: str | int | None = None,
    *,
    unit: bool = False,
) -> str | None:
    labelExpr = "datum.value"

    if title.power:
        labelExpr = f"{labelExpr}*1e{-title.power}"

    fmt = title.fmt or fmt

    if isinstance(fmt, int):
        labelExpr = f"format({labelExpr},'.{fmt}f')"
    elif fmt:
        labelExpr = f"format({labelExpr},'{fmt}')"

    if unit and title.unit:
        labelExpr = f"{labelExpr}+'{title.unit}'"

    has_si_prefix = isinstance(fmt, str) and "s" in fmt
    has_unit = unit and title.unit
    if has_si_prefix or has_unit:
        labelExpr = f"replace({labelExpr}, /^([0-9.]+)(.+)$/, '$1\u2009$2')"

    return None if labelExpr == "datum.value" else labelExpr


def _get[T: alt.X | alt.Y | alt.Color | alt.Shape | alt.Column | alt.Row](
    cls: type[T],
    name: Literal["axis", "legend", "header"],
    shorthand: str,
    title: str | Title | None = None,
    **kwargs: Any,
) -> T:
    x = cls(shorthand)
    if title := get_title(title):
        x = set_title(x, name, title)
        fmt = None if name == "axis" else "~s"
        unit = name != "axis"
        if labelExpr := get_label_expr(title, fmt, unit=unit):
            z = getattr(x, name)(labelExpr=labelExpr, **kwargs)
            return cast("T", z)
    return x


def X(shorthand: str, title: str | None = None, **kwargs: Any) -> alt.X:
    return _get(alt.X, "axis", shorthand, title, **kwargs)


def Y(shorthand: str, title: str | None = None, **kwargs: Any) -> alt.Y:
    return _get(alt.Y, "axis", shorthand, title, **kwargs)


def Color(shorthand: str, title: str | None = None, **kwargs: Any) -> alt.Color:
    return _get(alt.Color, "legend", shorthand, title, **kwargs)


def Shape(shorthand: str, title: str | None = None, **kwargs: Any) -> alt.Shape:
    return _get(alt.Shape, "legend", shorthand, title, **kwargs)


def Column(shorthand: str, title: str | None = None, **kwargs: Any) -> alt.Column:
    return _get(alt.Column, "header", shorthand, title, **kwargs)


def Row(shorthand: str, title: str | None = None, **kwargs: Any) -> alt.Row:
    return _get(alt.Row, "header", shorthand, title, **kwargs)
