"""Publication figures: the style a figure needs to sit in a typeset report or a journal page.

A figure in a hydrology report is printed at a fixed width beside body text
in a serif face. The conventions that make it look typeset rather than made
in a hurry are few and unglamorous, and this module applies them in one place
so every figure of a study agrees:

* **One serif typeface.** Times New Roman when the machine has it, else the
  STIX fonts that ship with matplotlib, so a figure drawn in the browser
  (Pyodide) or on a bare Linux runner looks like one drawn on a desktop.
  Math is STIX either way.
* **Sized for the page.** ``new_figure(width="full")`` is the text width of
  an A4 page with 25 mm margins (6.3 in); ``"half"`` is half of it. Drawn at
  that size and never rescaled, the 8.5 pt figure text sits just under a
  10-11 pt body.
* **A contained frame.** Full box, ticks pointing in, minor ticks, a faint
  dotted grid behind the data.
* **No title in the frame.** The numbered caption carries the description; a
  title inside the figure repeats it and costs vertical space.
* **Colour that survives colour blindness.** The Okabe and Ito (2008)
  palette, blue first, observations in near-black.
* **Units as a reader writes them.** ``m3/s`` is drawn as ``m³/s``, ``km2``
  as ``km²``, ``deg C`` as ``°C``.
* **Nothing on top of the data.** :func:`clear_corners` finds the emptiest
  corners from the plotted points so a legend or a metric box goes where the
  data is not.
* **Raster and vector.** :func:`png_bytes` at 300 dpi, :func:`svg_bytes` and
  :func:`pdf_bytes` with the text kept editable.

It also carries the two probability axes hydrology needs: return period on
Gumbel paper (:func:`return_period_axis`) and exceedance on a normal
probability scale (:func:`exceedance_axis`).

matplotlib is imported inside the functions, so importing this module costs
nothing until a figure is drawn.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Iterable, Sequence
from statistics import NormalDist
from typing import Any

__all__ = [
    "BLUE", "GREY", "GREEN", "INK", "OKABE_ITO", "ORANGE", "PURPLE", "SKY", "VERMILLION", "WIDTHS", "YELLOW",
    "axis_label", "clear_corners", "exceedance_axis", "gumbel_variate", "headroom", "metric_box", "new_figure",
    "panel_label", "pdf_bytes", "png_bytes", "return_period_axis", "serif_font", "svg_bytes", "unit_text",
    "use_style",
]

# ── colour ──────────────────────────────────────────────────────────────────

#: Near-black for observations and text (pure black is harsh beside the blue).
INK = "#1A1A1A"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
GREY = "#8C8C8C"

#: Okabe & Ito (2008) in the order a study uses them: the main estimate, the alternative, the check.
OKABE_ITO = [BLUE, VERMILLION, GREEN, ORANGE, SKY, PURPLE, INK]

# ── size ────────────────────────────────────────────────────────────────────

#: Figure widths in inches. ``full``/``half`` fit an A4 page with 25 mm margins (160 mm of text);
#: ``single``/``onehalf``/``double`` are the usual journal column widths.
WIDTHS = {"full": 6.3, "half": 3.1, "single": 3.5, "onehalf": 5.0, "double": 7.2}

#: Raster resolution: what journals and print want, and sharp on a high-density screen.
DPI = 300

_SERIF_PREF = ("Times New Roman", "TimesNewRoman", "Times", "Nimbus Roman", "Liberation Serif", "STIXGeneral",
               "STIX Two Text", "DejaVu Serif")
_font_cache: str | None = None


def serif_font() -> str:
    """The Times-like face to draw with: Times New Roman when installed, else the bundled STIXGeneral."""
    global _font_cache
    if _font_cache is not None:
        return _font_cache
    from matplotlib import font_manager

    names = {f.name.lower(): f.name for f in font_manager.fontManager.ttflist}
    for cand in _SERIF_PREF:
        low = cand.lower()
        hit = names.get(low) or next((v for k, v in names.items() if k.startswith(low)), None)
        if hit:
            _font_cache = hit
            return hit
    _font_cache = "serif"
    return _font_cache


def use_style(base_fontsize: float = 8.5) -> None:
    """Apply the publication style to matplotlib globally. Call before drawing; idempotent."""
    import matplotlib
    from cycler import cycler

    face = serif_font()
    small = base_fontsize - 0.5
    matplotlib.rcParams.update({
        # typography
        "font.family": "serif",
        "font.serif": [face, *[f for f in _SERIF_PREF if f != face]],
        "mathtext.fontset": "stix",
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize + 0.5,
        "axes.titlesize": base_fontsize + 0.5,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small - 0.5,
        "legend.title_fontsize": small,
        "figure.titlesize": base_fontsize + 1,
        "text.color": INK,
        "axes.labelcolor": INK,
        # the frame
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.axisbelow": True,
        "axes.facecolor": "white",
        "axes.titlepad": 4.0,
        "axes.labelpad": 3.0,
        "axes.prop_cycle": cycler(color=OKABE_ITO),
        "axes.formatter.use_mathtext": True,
        "axes.formatter.limits": (-4, 5),
        "axes.xmargin": 0.02,
        # ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # grid
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.color": "#D9D9D9",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        # marks
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth": 0.5,
        "errorbar.capsize": 2.0,
        "scatter.edgecolors": "face",
        # legend: framed, compact, never shadowed
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#808080",
        "legend.fancybox": False,
        "legend.borderpad": 0.35,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.6,
        "legend.columnspacing": 1.2,
        # output
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": False,
        "savefig.dpi": DPI,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "date.autoformatter.year": "%Y",
    })


def new_figure(width: str | float = "full", height: float | None = None, *, nrows: int = 1, ncols: int = 1,
               aspect: float = 0.5, **kwargs: Any) -> tuple[Any, Any]:
    """A figure sized for the page: ``width`` a key of :data:`WIDTHS` or inches; ``height`` in inches, else
    ``width * aspect`` (0.5 suits a time axis; pass ``aspect=0.75`` for a scatter or bars)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use_style()
    w = WIDTHS.get(width, 6.3) if isinstance(width, str) else float(width)
    h = float(height) if height is not None else round(w * aspect, 2)
    kwargs.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)


# ── text ────────────────────────────────────────────────────────────────────

_UNIT_SWAPS: tuple[tuple[re.Pattern[str], str], ...] = tuple((re.compile(p), r) for p, r in (
    (r"\bm3\s*/\s*s\b|\bm3s-1\b|\bm3 s-1\b|\bm\^3/s\b|\bcumecs\b", "m³/s"),
    (r"\bft3\s*/\s*s\b|\bcfs\b", "ft³/s"),
    (r"\bkm2\b|\bkm\^2\b|\bsq km\b", "km²"),
    (r"\bm2\b", "m²"),
    (r"\bm3\b", "m³"),
    (r"\bmm3\b", "mm³"),
    (r"\bdeg\s*C\b|\bdegC\b|\bdegrees C\b|\bdeg C\b", "°C"),
    (r"\bdeg\b", "°"),
    (r"\buS/cm\b|\bmicroS/cm\b", "µS/cm"),
    (r"\bug/L\b", "µg/L"),
    (r"/\s*yr\b", "/yr"),
))


def unit_text(unit: str | None) -> str:
    """A unit as a reader writes it: ``m3/s`` -> ``m³/s``, ``km2`` -> ``km²``, ``deg C`` -> ``°C``. ``-``,
    ``""`` and None (dimensionless) give ``""``."""
    u = " ".join(str(unit or "").split())
    if u in ("", "-", "none", "None", "dimensionless", "1"):
        return ""
    for pattern, repl in _UNIT_SWAPS:
        u = pattern.sub(repl, u)
    return u


def axis_label(quantity: str, unit: str | None = None) -> str:
    """``Discharge (m³/s)``: the quantity sentence-cased, the unit in parentheses when there is one."""
    q = " ".join(str(quantity or "").replace("_", " ").split())
    q = q[:1].upper() + q[1:]
    u = unit_text(unit)
    return f"{q} ({u})" if u else q


def panel_label(ax: Any, label: str, *, inside: bool = True) -> None:
    """``(a)``, ``(b)``: bold, top-left inside the frame (``inside=False`` centres it above the axes)."""
    if inside:
        ax.text(0.015, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold",
                bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    else:
        ax.text(0.5, 1.02, label, transform=ax.transAxes, ha="center", va="bottom", fontweight="bold")


_CORNERS = {
    "upper left": (0.025, 0.965, "left", "top"),
    "upper right": (0.975, 0.965, "right", "top"),
    "lower left": (0.025, 0.035, "left", "bottom"),
    "lower right": (0.975, 0.035, "right", "bottom"),
}


def metric_box(ax: Any, text: str, loc: str = "upper left", **kwargs: Any) -> None:
    """A framed box of summary numbers (n, p, slope, R²) in a corner of the axes."""
    x, y, ha, va = _CORNERS.get(loc, _CORNERS["upper left"])
    kwargs.setdefault("fontsize", max(6.5, float(_rc("legend.fontsize", 7.5))))
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, linespacing=1.25, zorder=6,
            bbox={"boxstyle": "square,pad=0.4", "facecolor": "white", "edgecolor": "#808080", "linewidth": 0.5,
                  "alpha": 0.92}, **kwargs)


def _rc(key: str, default: Any) -> Any:
    try:
        import matplotlib

        v = matplotlib.rcParams[key]
        return float(v) if isinstance(v, (int, float)) else default
    except Exception:  # pragma: no cover - defensive
        return default


def _as_numbers(values: Iterable[Any]) -> list[float]:
    """Floats from numbers or dates (matplotlib's date numbers), so corners work on a time axis too."""
    out: list[float] = []
    for v in values:
        try:
            out.append(float(v))
            continue
        except (TypeError, ValueError):
            pass
        try:
            from matplotlib.dates import date2num

            out.append(float(date2num(v)))
        except Exception:  # noqa: BLE001 - an unreadable point is skipped
            continue
    return out


def clear_corners(ax: Any, x: Sequence[Any], y: Sequence[Any], k: int = 2, frac: float = 0.38) -> list[str]:
    """The ``k`` emptiest corners of the axes for the plotted points, emptiest first. Call after the limits are
    final; hand the result to ``ax.legend(loc=...)`` and :func:`metric_box`."""
    xs, ys = _as_numbers(x), _as_numbers(y)
    n = min(len(xs), len(ys))
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xlog, ylog = ax.get_xscale() == "log", ax.get_yscale() == "log"

    def frac_of(v: float, lo: float, hi: float, log: bool) -> float | None:
        if log:
            if v <= 0 or lo <= 0 or hi <= 0:
                return None
            v, lo, hi = math.log10(v), math.log10(lo), math.log10(hi)
        if hi == lo or not math.isfinite(v):
            return None
        return (v - lo) / (hi - lo)

    counts = dict.fromkeys(_CORNERS, 0)
    for i in range(n):
        fx, fy = frac_of(xs[i], x0, x1, xlog), frac_of(ys[i], y0, y1, ylog)
        if fx is None or fy is None:
            continue
        left, right, low, high = fx < frac, fx > 1 - frac, fy < frac, fy > 1 - frac
        if high and left:
            counts["upper left"] += 1
        if high and right:
            counts["upper right"] += 1
        if low and left:
            counts["lower left"] += 1
        if low and right:
            counts["lower right"] += 1
    preference = ["upper left", "upper right", "lower right", "lower left"]
    return sorted(counts, key=lambda c: (counts[c], preference.index(c)))[:k]


def headroom(ax: Any, frac: float = 0.18, where: str = "top") -> None:
    """Open an empty band on one side of the axes (for a legend when every corner holds data)."""
    log = (ax.get_yscale() if where in ("top", "bottom") else ax.get_xscale()) == "log"
    lo, hi = ax.get_ylim() if where in ("top", "bottom") else ax.get_xlim()
    if log and lo > 0 and hi > 0:
        llo, lhi = math.log10(lo), math.log10(hi)
        span = lhi - llo
        llo, lhi = (llo - span * frac, lhi) if where in ("bottom", "left") else (llo, lhi + span * frac)
        lo, hi = 10 ** llo, 10 ** lhi
    else:
        span = hi - lo
        lo, hi = (lo - span * frac, hi) if where in ("bottom", "left") else (lo, hi + span * frac)
    (ax.set_ylim if where in ("top", "bottom") else ax.set_xlim)(lo, hi)


# ── probability axes ────────────────────────────────────────────────────────


def gumbel_variate(return_period: Any) -> Any:
    """The Gumbel reduced variate ``-ln(-ln(1 - 1/T))`` of a return period (scalar or array): on this axis a
    Gumbel distribution is a straight line, the classic flood-frequency paper."""
    import numpy as np

    t = np.asarray(return_period, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(-np.log(1.0 - 1.0 / t))


#: Return periods labelled on the Gumbel axis.
RETURN_PERIOD_TICKS = (1.01, 1.1, 1.5, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10000)


def return_period_axis(ax: Any, t_min: float = 1.01, t_max: float = 200.0, *, aep: bool = True) -> None:
    """Make the x axis of ``ax`` a return-period axis on Gumbel paper (plot against :func:`gumbel_variate`),
    with ticks at the conventional return periods and, with ``aep``, the annual exceedance probability on top."""
    from matplotlib.ticker import FixedLocator, NullLocator

    lo, hi = float(gumbel_variate(t_min)), float(gumbel_variate(t_max))
    ax.set_xlim(lo, hi)
    ticks = [t for t in RETURN_PERIOD_TICKS if t_min <= t <= t_max * 1.0001]
    ax.xaxis.set_major_locator(FixedLocator([float(gumbel_variate(t)) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([_t_label(t) for t in ticks])
    ax.tick_params(axis="x", which="both", top=not aep)
    if aep:
        top = ax.secondary_xaxis("top")
        top.xaxis.set_major_locator(FixedLocator([float(gumbel_variate(t)) for t in ticks]))
        top.xaxis.set_minor_locator(NullLocator())
        top.set_xticklabels([_aep_label(t) for t in ticks])
        top.tick_params(axis="x", direction="in", labelsize=_rc("xtick.labelsize", 8.0) - 0.5)
        top.set_xlabel("Annual exceedance probability (%)", fontsize=_rc("xtick.labelsize", 8.0))


def _t_label(t: float) -> str:
    return f"{t:g}" if t < 10 else f"{int(round(t)):,}"


def _aep_label(t: float) -> str:
    p = 100.0 / t
    if p >= 10:
        return f"{p:.0f}"
    if p >= 1:
        return f"{p:.2g}"
    return f"{p:.2g}"


_ND = NormalDist()
#: Exceedance percentages labelled on the probability axis.
EXCEEDANCE_TICKS = (0.1, 1, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99, 99.9)


def probit(percent: Any) -> Any:
    """Exceedance percentage to the standard normal quantile (scalar or array), for a probability axis."""
    import numpy as np

    p = np.clip(np.asarray(percent, dtype=float) / 100.0, 1e-6, 1 - 1e-6)
    return np.vectorize(_ND.inv_cdf)(p)


def exceedance_axis(ax: Any, lo: float = 0.1, hi: float = 99.9) -> None:
    """Make the x axis a normal-probability exceedance axis (plot against :func:`probit`): a log-normal flow
    regime is a straight line on log-y, and both tails get room instead of being crushed into the edges."""
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.set_xlim(float(probit(lo)), float(probit(hi)))
    ticks = [t for t in EXCEEDANCE_TICKS if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator([float(probit(t)) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])


# ── output ──────────────────────────────────────────────────────────────────


def png_bytes(fig: Any, dpi: int = DPI) -> bytes:
    """The figure as PNG bytes at ``dpi`` (300 by default), tight to its content, on white."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    return buf.getvalue()


def svg_bytes(fig: Any) -> bytes:
    """The figure as SVG bytes, text kept as text (editable in Illustrator or Inkscape), no timestamp."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.03, facecolor="white",
                metadata={"Date": None})
    return buf.getvalue()


def pdf_bytes(fig: Any) -> bytes:
    """The figure as vector PDF bytes with TrueType fonts embedded."""
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.03, facecolor="white",
                metadata={"CreationDate": None, "Producer": "aquascope"})
    return buf.getvalue()
