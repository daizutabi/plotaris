
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotaris as plts
import polars as pl

print("Initializing plotaris...")
plts.init()

print("Loading penguins dataset...")
df = pl.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv").drop_nulls()

print("Building faceted chart...")
facet_chart = (
    plts.Chart(df)
    .facet(col="species", row="sex")
    .encode(
        x="bill_length_mm",
        y="bill_depth_mm",
        color="island",
        shape="island",
    )
    .mark_point()
)

print("Rendering faceted chart...")
try:
    facet_chart.display()
    print("Saving plot to gemini_facet_test_plot.png...")
    plt.savefig("gemini_facet_test_plot.png")
    print("Plot saved successfully.")
except Exception as e:
    print(f"An error occurred during chart rendering: {e}")
    import traceback
    traceback.print_exc()
