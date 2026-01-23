import marimo

__generated_with = "0.19.5"
app = marimo.App(width="full")

with app.setup:
    import matplotlib.pyplot as plt
    import polars as pl

    from plotaris import Chart

    plt.style.use(["default", "matplotlibrc"])


@app.cell
def _():
    data = pl.DataFrame({"a": [1, 1, 2, 2], "x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})
    return (data,)


@app.cell
def _(data):
    chart = Chart(data).encode("x", "y", color="a")
    encoding = chart.encoding
    palettes = encoding.build_palettes(data)
    return chart, encoding, palettes


@app.cell
def _(chart, data):
    from plotaris.core.data import GroupedData

    gd = GroupedData(data, dict(chart.encoding.items()))
    return (gd,)


@app.cell
def _(gd):
    gd.data[0].row(0, named=True)
    return


@app.cell
def _(encoding, gd, palettes):
    encoding.get_properties(gd.item(0, "color", named=True), palettes)
    return


@app.cell
def _(palettes):
    palettes
    return


@app.cell
def _(gd):
    gd.get_label(0, named=True)
    return


@app.cell
def _(gd):
    gd.item(0, "color", named=True)
    return


@app.cell
def _(encoding):
    list(encoding.items())
    return


@app.cell
def _(palettes):
    palettes.get("color")[(1,)]
    return


@app.cell
def _(encoding):
    list(encoding.items())
    return


if __name__ == "__main__":
    app.run()
