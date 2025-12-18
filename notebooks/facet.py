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
def _(pl, plts):
    # Load penguins dataset
    # Drop rows with nulls as some mark implementations might not handle them well yet
    df = pl.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv").drop_nulls()

    # Test faceting: col by species, row by sex
    facet_chart = (
        plts.Chart(df)
        .facet(col="species", row="sex")
        .encode(x="bill_length_mm", y="bill_depth_mm")
        .mark_point()
    )
    return df, facet_chart

@app.cell
def _(facet_chart):
    # Display the faceted chart
    facet_chart
    return
