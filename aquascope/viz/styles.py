"""Shared styling constants and helpers for AquaScope visualizations."""

from __future__ import annotations

from aquascope.utils.imports import require

# ── Colour palettes ─────────────────────────────────────────────────────
AQUA_PALETTE = {
    "primary": "#0077B6",
    "secondary": "#00B4D8",
    "accent": "#90E0EF",
    "warning": "#F4A261",
    "danger": "#E63946",
    "success": "#2A9D8F",
    "neutral": "#6C757D",
    "light": "#CAF0F8",
    "dark": "#023E8A",
    "background": "#F8F9FA",
}

RISK_COLOURS = {
    "NORMAL": "#2A9D8F",
    "LOW": "#90E0EF",
    "MODERATE": "#F4A261",
    "HIGH": "#E76F51",
    "EXTREME": "#E63946",
}

SPI_COLOURS = {
    "extremely_wet": "#023E8A",
    "very_wet": "#0077B6",
    "moderately_wet": "#00B4D8",
    "near_normal": "#90E0EF",
    "moderately_dry": "#F4A261",
    "severely_dry": "#E76F51",
    "extremely_dry": "#E63946",
}

# Qualitative palette for multi-line plots
SERIES_COLOURS = [
    "#0077B6", "#E63946", "#2A9D8F", "#F4A261", "#6A4C93",
    "#1982C4", "#FF595E", "#8AC926", "#FFCA3A", "#6A0572",
]

# ── Figure defaults ─────────────────────────────────────────────────────
DEFAULT_FIGSIZE = (12, 5)
WIDE_FIGSIZE = (14, 6)
SQUARE_FIGSIZE = (8, 8)
MULTI_FIGSIZE = (14, 10)

DEFAULT_DPI = 150
FONT_SIZES = {
    "title": 14,
    "label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
}


def apply_aqua_style() -> None:
    """Apply the AquaScope matplotlib style globally.

    Sets a clean, publication-friendly style with the AquaScope colour
    palette.  Safe to call multiple times.
    """
    require("matplotlib", feature="plotting")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": DEFAULT_FIGSIZE,
        "figure.dpi": DEFAULT_DPI,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": FONT_SIZES["tick"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "legend.fontsize": FONT_SIZES["legend"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
    })


def _save_or_show(fig, save_path: str | None, tight: bool = True) -> None:
    """Save figure to *save_path* if given, otherwise show it interactively."""
    require("matplotlib", feature="plotting")
    import matplotlib.pyplot as plt

    if tight:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
