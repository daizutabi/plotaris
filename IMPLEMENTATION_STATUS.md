# Plotaris Implementation Status

**Date:** 2025-12-18
**Context:** Initial scaffolding complete. Transitioning to DevContainer.

## Project Overview
**Plotaris** is a visualization library combining **Polars** (speed/expressions), **Matplotlib** (rendering power), and a **Modern Declarative API** (Method Chaining).

## Completed Tasks

### 1. Infrastructure
- [x] Project structure created (`src/plotaris`, `core`, `marks`).
- [x] Dependencies updated (`pyproject.toml`): Removed `seaborn`, added `polars>=1.0.0`.
- [x] Type hints applied (`from __future__ import annotations`, strict typing).

### 2. Core Implementation
- [x] **`Chart` Class** (`src/plotaris/chart.py`):
    - Entry point for the API.
    - Implemented `encode()` for mapping variables.
    - Implemented `mark_scatter()`, `mark_line()`, `mark_bar()`.
    - Placeholder `show()` method (basic matplotlib rendering).
- [x] **Marks System** (`src/plotaris/marks/`):
    - `Mark` (Abstract Base Class).
    - `ScatterMark`, `LineMark`, `BarMark` (Basic implementations).
- [x] **Data Handling** (`src/plotaris/core/data.py`):
    - `DataHandler` class to wrap `pl.DataFrame` / `pl.LazyFrame`.
- [x] **Config** (`src/plotaris/config.py`):
    - Basic configuration class.

## Current State
The library can currently define a chart structure and render simple plots where encodings are direct column names (strings).
**Limitation:** It does not yet support `pl.Expr` evaluation or Faceting.

## Next Steps (ToDo)

### 1. Polars Expressions Support (Priority High)
- **Goal:** Allow `Chart(df).encode(y=pl.col("sales").sum())`.
- **Task:**
    - Update `DataHandler` or `Chart.show()` logic to inspect `encodings`.
    - Extract `pl.Expr` objects from encodings.
    - Apply these expressions to the `LazyFrame` context to generate result columns.
    - Alias expressions automatically or manually if needed to map them to plotting axes.

### 2. FacetGrid Implementation (Priority High)
- **Goal:** Support `.facet(col="category")`.
- **Task:**
    - Implement `LayoutManager` in `src/plotaris/core/layout.py`.
    - Add `.facet()` method to `Chart`.
    - Logic to calculate rows/cols and create `plt.subplots`.
    - Logic to filter data for each subplot and call `mark.plot()` on specific Axes.

### 3. Rendering Logic Refinement
- **Goal:** Return `Figure` objects instead of just calling `plt.show()`.
- **Task:**
    - Refactor `show()` to `render()` which returns `(Figure, Axes)`.
    - Make `show()` a wrapper around `render()`.

### 4. Testing
- **Goal:** Ensure stability.
- **Task:**
    - Add unit tests for `Chart` construction.
    - Add integration tests with `polars` data.

## API Reference (Draft)

```python
import plotaris as plts
import polars as pl

# Target API
(
    plts.Chart(df)
    .facet(col="island")  # To be implemented
    .encode(
        x="bill_length_mm",
        y=pl.col("bill_depth_mm") * 2,  # Expression support to be implemented
        color="species"
    )
    .mark_scatter()
    .show()
)
```
