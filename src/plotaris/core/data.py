from __future__ import annotations
import polars as pl

class DataHandler:
    def __init__(self, data: pl.DataFrame | pl.LazyFrame):
        self.data = data.lazy() if isinstance(data, pl.DataFrame) else data

    def collect(self) -> pl.DataFrame:
        return self.data.collect()
