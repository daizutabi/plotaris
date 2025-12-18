import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import plotaris as plts
    import polars as pl

    plts.init()
    return pl, plts


@app.cell
def _(pl):
    data = pl.DataFrame({"x": [1, 2, 3], "y": [11, 20, 30]})
    return (data,)


@app.cell
def _(data, plts):
    plts.Chart(data).encode(x="x", y="y").mark_line()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
