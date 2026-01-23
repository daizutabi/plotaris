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
    data = pl.DataFrame({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})
    return (data,)


@app.cell
def _(data):
    Chart(data).encode("x", "y").mark_point(color="red")
    return


if __name__ == "__main__":
    app.run()
