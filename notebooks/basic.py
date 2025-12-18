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
    data = pl.DataFrame({"x": [1, 2, 3], "y": [15, 20, 30]})
    return (data,)


@app.cell
def _(data, plts):
    chart = plts.Chart(data).encode("x", "y").mark_line(marker="o", ms=2)
    chart
    return (chart,)


@app.cell
def _(chart):
    from dataclasses import fields, asdict

    fields(chart.encoding)
    return


@app.cell
def _(data):
    x, y = data.select(a="x", b="y")
    return x, y


@app.cell
def _(plt, x, y):
    plt.plot(x, y)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
