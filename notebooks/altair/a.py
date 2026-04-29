import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:
    import altair as alt
    import polars as pl

    import plotaris as pt
    from plotaris.altair.theme import set_theme

    alt.renderers.enable("svg", ppi=200)
    set_theme('"Noto Sans CJK JP", Meiryo')


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
        x=pt.X("x", "Xラベル (mm)", domain=(0, 4e-3)),
        y=pt.Y("y", "Yラベル (km)", domain=(0, 4e3)),
        color=pt.Color("c:N", "Color (m)"),
        shape=pt.Shape("s:N", "Shape (Hz)"),
        row=pt.Row("s:N", "Row (s)"),
        column=pt.Column("c:N", "Column (s)"),
    ).properties(width=80, height=80)
    return


@app.cell
def _():
    data2 = pl.DataFrame({
        "x": [1e-9, 1e-6, 1e-3],
        "y": [1e3, 1e6, 1e9],
    })
    return (data2,)


@app.cell
def _(data2):
    alt.Chart(data2).mark_point().encode(
        x=pt.Xlog("x", "Xラベル", domain=(1e-10, 1e-2)),
        y=pt.Ylog("y", "Yラベル", domain=(1e2, 1e10)),
    ).properties(width=200, height=200)
    return


@app.cell
def _():
    alt.Chart().mark_rule().encode(x=alt.datum(1))._repr_mimebundle_()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
