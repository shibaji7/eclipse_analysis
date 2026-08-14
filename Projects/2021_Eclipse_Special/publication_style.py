"""Shared plotting style for publication-ready figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_publication_style(font_size: int = 15) -> None:
    """Apply a compact SciencePlots-style publication theme."""

    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans",
                                   "Lucida Grande", "Verdana"]

    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee"])
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "figure.dpi": 300,
            "text.usetex": False,
            
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.2,
        }
    )
