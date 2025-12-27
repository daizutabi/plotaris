
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotaris as plts
import polars as pl

print("Initializing plotaris...")
plts.init()

print("Creating sample data...")
df = pl.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8],
    "y": [10, 12, 15, 11, 14, 18, 16, 19],
    "category_a": ["A", "A", "A", "A", "B", "B", "B", "B"],
    "category_b": ["X", "X", "Y", "Y", "X", "X", "Y", "Y"],
})

print("Building chart...")
chart = (
    plts.Chart(df)
    .encode(
        x="x",
        y="y",
        color="category_a",
        size="category_b",
        shape="category_a",  # Re-using category_a for shape
    )
    .mark_point()
)

print("Rendering chart...")
try:
    chart.display()
    print("Saving plot to gemini_test_plot.png...")
    plt.savefig("gemini_test_plot.png")
    print("Plot saved successfully.")
except Exception as e:
    print(f"An error occurred during chart rendering: {e}")
    import traceback
    traceback.print_exc()
