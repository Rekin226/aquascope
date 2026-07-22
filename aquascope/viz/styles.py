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
    "#0077B6",
    "#E63946",
    "#2A9D8F",
    "#F4A261",
    "#6A4C93",
    "#1982C4",
    "#FF595E",
    "#8AC926",
    "#FFCA3A",
    "#6A0572",
]

# Okabe-Ito / Color Universal Design palette — colorblind-safe.
# Source: Okabe & Ito (2008), https://jfly.uni-koeln.de/color/
OKABE_ITO_COLOURS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
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


_VALID_PALETTES = {"default", "colorblind"}


def apply_aqua_style(palette: str = "default") -> None:
    """Apply the AquaScope matplotlib style globally.

    Sets a clean, publication-friendly style with the AquaScope colour
    palette.  Safe to call multiple times.

    Parameters
    ----------
    palette : str, optional
        Colour palette to use for the *axes.prop_cycle*.  ``"default"``
        keeps the existing AquaScope series colours; ``"colorblind"``
        switches to the Okabe-Ito / Color Universal Design palette.

    Raises
    ------
    ValueError
        If *palette* is not one of the recognised names.
    """
    if palette not in _VALID_PALETTES:
        raise ValueError(f"Unknown palette {palette!r}. Choose from {sorted(_VALID_PALETTES)}.")

    require("matplotlib", feature="plotting")
    import matplotlib.pyplot as plt
    from cycler import cycler

    colours = OKABE_ITO_COLOURS if palette == "colorblind" else SERIES_COLOURS

    plt.rcParams.update(
        {
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.dpi": DEFAULT_DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": cycler(color=colours),
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "font.size": FONT_SIZES["tick"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["label"],
            "legend.fontsize": FONT_SIZES["legend"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
        }
    )


def _save_or_show(fig, save_path: str | None, tight: bool = True) -> None:
    """Save figure to *save_path* if given, otherwise show it interactively."""
    require("matplotlib", feature="plotting")
    import matplotlib.pyplot as plt

    if tight:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
    elif plt.get_backend().lower() != "agg":
        plt.show()
