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
def func(x, y, label, **kwargs):
    print(label)
    ax = plt.gca()
    ax.scatter(x, y, label=f"{kwargs['color']}")
    ax.legend(fontsize=4)


@app.cell
def _(data):
    chart = (
        Chart(data, figsize=(4, 2))
        .encode("x", "y", color="x", shape="y")
        .mapping(color={1: "blue", 3: "pink"}, shape={5: "+"})
        # .map(func)
        .mark_point(s=5)
        .label(eq="")
        # .facet("a", "x")
        # .select(row=0, col=0)
        .legend(fontsize=6)
        # .set_titles()
    )
    chart
    return (chart,)


@app.cell
def _(chart):
    chart.encoding
    return


if __name__ == "__main__":
    app.run()
