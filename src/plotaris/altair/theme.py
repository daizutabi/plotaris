from __future__ import annotations

from altair import theme


def set_theme(font: str) -> None:
    @theme.register("custom_theme", enable=True)
    def custom_theme() -> theme.ThemeConfig:  # pyright: ignore[reportUnusedFunction]
        return {
            "config": {
                "font": font,
                "axis": {"labelFlush": False},
                "headerColumn": {"titlePadding": 0, "labelPadding": 0},
                "headerRow": {"titlePadding": 0, "labelPadding": 8},
            },
        }
