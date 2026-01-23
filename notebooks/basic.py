import marimo

__generated_with = "0.19.5"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import plotaris as plts
    import polars as pl
    from matplotlib.ticker import EngFormatter

    plt.style.use(["default", "matplotlibrc"])
    return EngFormatter, pl, plt


@app.cell
def _(EngFormatter, pl, plt):
    data = pl.DataFrame({"x": [1e-6, 2e-6, 3e-6], "y": [15e-9, 20e-9, 30e-9]})
    fig, ax = plt.subplots()
    ax.plot(data["x"], data["y"])
    xformatter = EngFormatter(unit="m")
    ax.xaxis.set_major_formatter(xformatter)
    yformatter = EngFormatter(places=1, sep="\N{THIN SPACE}")
    ax.yaxis.set_major_formatter(yformatter)
    ax.set(xlabel="position", ylabel="diameter (m)")
    ax
    return xformatter, yformatter


@app.cell
def _(xformatter):
    xformatter(50e-3)
    return


@app.cell
def _(yformatter):
    yformatter._get_val_and_prefix(2.4e-3)
    return


@app.cell
def _(EngFormatter):
    EngFormatter.__module__
    return


@app.cell
def _():
    from matplotlib import ticker
    ticker.__file__
    return


if __name__ == "__main__":
    app.run()
