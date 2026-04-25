import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import matplotlib.pyplot as plt
    import polars as pl

    from plotaris import FacetGrid

    plt.style.use(["default", "matplotlibrc"])


@app.cell
def _():
    data = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            # "b": [3, 4, 5, 3, 4, 5],
            "b": [3, 3, 4, 4, 5, 5],
            "x": range(6),
            "y": range(10, 16),
        },
    )
    return (data,)


@app.cell
def _(data):
    FacetGrid(data, row="a", col="b", figsize=(4, 2)).set_titles(
        {"b": "{:.1f}"},
        a="-(m)",
        margin_titles=True,
    )
    return


@app.cell
def _(data):
    FacetGrid(
        data.with_columns(pl.col("a") * 1e-6),
        col="a",
        wrap=2,
        figsize=(4, 2),
    ).set_titles(a="A(m)")
    return


if __name__ == "__main__":
    app.run()
