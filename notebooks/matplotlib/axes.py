import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")

with app.setup:
    import matplotlib.pyplot as plt

    from plotaris.matplotlib.axes import (
        axes_text,
        format_axes,
        format_axis,
        set_axes_log,
    )

    plt.style.use(["default", "matplotlibrc"])


@app.cell
def _():
    _ax = plt.figure(figsize=(3, 2)).add_subplot()
    _ax.set(xlim=(-1e-5, 1e-5))
    format_axis(_ax.xaxis, "Voltage (µV)")
    _ax
    return


@app.cell
def _():
    _ax = plt.figure(figsize=(3, 2)).add_subplot()
    _ax.set(xlim=(-1e-4, 1e-4), ylim=(-1e-7, 1e-7))
    format_axes(_ax, "Voltage (mV):2", "Current (nA)")
    return


@app.cell
def _():
    _ax = plt.figure(figsize=(3, 2)).add_subplot()
    set_axes_log(_ax, xlim=(1e-9, 1e-6), ylim=(1e3, 1e6))
    axes_text(_ax, 0.02, 0.02, "abcdef")
    axes_text(_ax, 0.02, 0.98, "abcdef")
    axes_text(_ax, 0.98, 0.02, "abcdef")
    axes_text(_ax, 0.98, 0.98, "abcdef")
    _ax
    return


if __name__ == "__main__":
    app.run()
