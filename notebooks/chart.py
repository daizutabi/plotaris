import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

with app.setup:
    import matplotlib.pyplot as plt
    import polars as pl

    from plotaris import Chart

    plt.style.use(["default", "matplotlibrc"])


@app.cell
def _():
    data = pl.DataFrame({"a": [1, 1, 2, 2], "x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})
    return (data,)


@app.function
def func(x, y, **kwargs):
    print(kwargs)
    ax = plt.gca()
    ax.scatter(x, y)


@app.cell
def _(data):
    chart = (
        Chart(data, figsize=(4, 2))
        .encode("x", "y", color="x", shape="y")
        .mapping(color={3: "pink"}, shape={5: "+"})
        .mark_point()
        .map(func)
        .facet("a", "x")
        # .to_facet()
        # .delaxes()
        # .set_titles()
    )
    chart
    return (chart,)


@app.cell
def _(chart):
    type(chart)
    return


if __name__ == "__main__":
    app.run()
