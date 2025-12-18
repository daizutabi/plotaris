from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from plotaris.core.data import DataHandler
from plotaris.marks.bar import BarMark
from plotaris.marks.line import LineMark
from plotaris.marks.scatter import ScatterMark

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.figure import Figure
    from polars._typing import IntoExpr

    from plotaris.marks.base import Mark


class Chart:
    data_handler: DataHandler
    encodings: dict[str, IntoExpr]

    def __init__(self, data: pl.DataFrame | pl.LazyFrame) -> None:
        self.data_handler = DataHandler(data)
        self.encodings = {}
        self.mark: Mark | None = None

    def encode(self, **kwargs: str | pl.Expr) -> Chart:
        self.encodings.update(kwargs)
        return self

    def mark_scatter(self, **kwargs: Any) -> Chart:
        self.mark = ScatterMark(**kwargs)
        return self

    def mark_line(self, **kwargs: Any) -> Chart:
        self.mark = LineMark(**kwargs)
        return self

    def mark_bar(self, **kwargs: Any) -> Chart:
        self.mark = BarMark(**kwargs)
        return self

    def display(self) -> Figure:
        if self.mark is None:
            msg = "Mark must be defined before displaying the chart"
            raise ValueError(msg)

        # Resolve expressions (mock implementation)
        # Real implementation needs to evaluate expressions against the LazyFrame

        # For now, let's just collect the data and assume encodings are column names
        # In a real scenario, we need to handle pl.Expr by selecting/aliasing them

        df = self.data_handler.collect()
        df.select()

        # Handle Expr in encodings (simplified: convert to string alias if possible, or raise)  # noqa: E501
        # This part requires more complex logic to actually modify the query
        final_encodings: dict[str, str] = {}

        # This is a placeholder. Real implementation should update self.data_handler's query  # noqa: E501
        # to include the expressions and generate aliases.
        for k, v in self.encodings.items():
            if isinstance(v, str):
                final_encodings[k] = v
            else:
                # Fallback or Todo
                pass

        fig, ax = plt.subplots()  # pyright: ignore[reportUnknownMemberType]
        self.mark.plot(ax, df, final_encodings)
        return fig

    def _display_(self) -> Figure:
        return self.display()
