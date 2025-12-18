from __future__ import annotations

import polars as pl


class DataHandler:
    data: pl.LazyFrame

    def __init__(self, data: pl.DataFrame | pl.LazyFrame) -> None:
        self.data = data.lazy() if isinstance(data, pl.DataFrame) else data

    def collect(self) -> pl.DataFrame:
        return self.data.collect()
