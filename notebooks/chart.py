import marimo

__generated_with = "0.19.7"
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
    chart = (
        Chart(data, figsize=(4, 2))
        .encode("x", "y", color="x", shape="y")
        .facet(row="a", col="x")
        .mapping(color={3: "pink"})
        .mark_point()
        .to_facet()
        # .delaxes()
        .set_titles()
    )
    chart
    return (chart,)


@app.cell
def _(chart):
    chart.facet_data.nrows
    return


if __name__ == "__main__":
    app.run()
