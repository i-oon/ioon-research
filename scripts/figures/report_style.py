"""Shared plot style for the thesis figures: colorblind-safe palette (Paul Tol bright), open axes,
light grid, no title inside the figure (the LaTeX caption carries it)."""
import matplotlib

TOL = {"blue": "#4477AA", "red": "#EE6677", "green": "#228833", "yellow": "#CCBB44",
       "cyan": "#66CCEE", "purple": "#AA3377", "grey": "#BBBBBB"}


def apply():
    matplotlib.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 1.8, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
