import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")


@app.cell
def _():
    import plotaris as plts
    import polars as pl
    from matplotlib.axes import Axes

    plts.init()
    return Axes, pl, plts


@app.cell
def _(pl):
    data = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [3, 3, 4, 4, 5, 5],
            "x": range(6),
            "y": range(10, 16),
        },
    )
    return (data,)


@app.cell
def _(Axes, pl):
    def plot(data: pl.DataFrame, *, ax: Axes) -> None:
        ax.scatter(data["x"], data["y"])
    return (plot,)


@app.cell
def _(data, plot, plts):
    grid = plts.FacetGrid(data, row="a", col="b", sharex=True).map_dataframe(plot)
    grid.map_axes(lambda ax: ax.set(xlabel="a"))
    grid
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
