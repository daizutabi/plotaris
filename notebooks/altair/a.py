import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import altair as alt
    import polars as pl
    import plotaris as pts

    alt.renderers.enable("png", ppi=200)


@app.cell
def _():
    data = pl.DataFrame({
        "x": [1e-3, 2e-3, 3e-3],
        "y": [1e3, 2e3, 3e3],
        "c": [1e-9, 2.5e-6, 3e-3],
        "s": [1e9, 2.5e6, 3e3],
    })
    return (data,)


@app.cell
def _(data):
    alt.Chart(data).mark_point().encode(
        x=pts.X("x", "X label (mm)").scale(domain=[0, 4e-3]),
        y=pts.Y("y", "Y label (km)"),
        color=pts.Color("c:N", "Color (m)"),
        shape=pts.Shape("s:N", "Shape (Hz)"),
    ).properties(width=200, height=200)
    return


@app.cell
def _():
    c = pts.Color("c:N", "Color (m)")
    return (c,)


@app.cell
def _(c):
    c.to_dict()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
