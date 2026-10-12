"""The figures of a study: one PNG and one SVG per figure kind a step yields, from its payload.

The catalogue (:mod:`aquascope.studio.catalogue`) says which kinds each tool
yields; :func:`figures_for` draws them and returns :class:`Artifact` pairs
the Author places in the report. :func:`draw` returns the matplotlib figure
itself, for the notebook that redraws them. Every maker reads the payload
defensively through :mod:`aquascope.studio.deliverables._payload` and
returns None when there is nothing to draw, so a missing key never breaks
the Author.

Every figure is drawn in the publication style of
:mod:`aquascope.viz.publication`: one serif face (Times New Roman, else the
STIX fonts bundled with matplotlib, so the browser and the desktop agree),
a full frame with inward ticks, the Okabe-Ito colours, the text width of an
A4 page (6.3 in), units written as a reader writes them, and no title inside
the frame: the numbered caption the maker returns carries the description.
PNG at 300 dpi and SVG with the text kept as text. matplotlib is imported
inside the functions (the module imports in the Pyodide worker before the
plotting package is loaded), the Agg backend is selected and every figure is
closed after rendering.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from aquascope.studio.deliverables._payload import (
    annual_maxima_of,
    date_key,
    fdc_of,
    frame_records,
    indices_of,
    num,
    numbers,
    period_of,
    record_name,
    resolution_word,
    return_levels_of,
    series_of,
    site_point,
    stations_of,
    unit_of,
    variable_of,
    year_of,
)
from aquascope.studio.workspace import MEDIA_TYPES, Artifact
from aquascope.viz import publication as pub

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

#: The default figure: the text width of an A4 page with 25 mm margins, at a 0.52 aspect for a time axis.
FIGSIZE = (pub.WIDTHS["full"], 3.3)
DPI = pub.DPI

#: The house colours: Okabe and Ito (2008), colourblind-safe, named by the role they play in a figure.
PRIMARY = pub.BLUE          # the main estimate or the record
SECONDARY = pub.SKY         # a second record of the same kind
ACCENT = "#BBD7EA"          # bands and fills behind the main estimate
DARK = pub.INK              # observations, reference lines
DANGER = pub.VERMILLION     # thresholds, the alternative fit, the design value
WARNING = pub.ORANGE        # events, reserves
SUCCESS = pub.GREEN         # the cross-check
NEUTRAL = pub.GREY          # context

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── matplotlib plumbing ───────────────────────────────────────────────────


def _plt() -> Any:
    """pyplot on the Agg backend with the publication style applied (imported here, never at module import)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pub.use_style()
    return plt


def _figure(rows: int = 1, cols: int = 1, *, height: float | None = None, sharex: bool = False,
            width: str | float = "full") -> tuple[Any, Any]:
    """A figure at the page's text width (``width="half"`` for a small one), constrained layout."""
    _plt()
    h = height if height is not None else FIGSIZE[1]
    return pub.new_figure(width, min(h, 8.6), nrows=rows, ncols=cols, sharex=sharex)


def _dates(values: list[str]) -> Any:
    import numpy as np

    return np.array([date_key(v) for v in values], dtype="datetime64[s]")


def _floats(values: list[float | None]) -> Any:
    import numpy as np

    return np.array([np.nan if v is None else v for v in values], dtype=float)


def png_bytes(fig: Figure) -> bytes:
    """The figure as PNG bytes at 300 dpi (tight bounding box, white)."""
    return pub.png_bytes(fig)


def svg_bytes(fig: Figure) -> bytes:
    """The figure as SVG bytes, the text kept as text so it stays editable."""
    return pub.svg_bytes(fig)


def close(fig: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _plain_log_y(ax: Any) -> None:
    """Plain numbers on a log axis instead of powers of ten."""
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    ax.set_yscale("log")
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(NullFormatter())


def _ylabel(variable: str, unit: str) -> str:
    return pub.axis_label(variable, unit)


def _u(unit: str | None) -> str:
    """A unit for a legend or a caption: ``m³/s``, not ``m3/s``."""
    return pub.unit_text(unit)


def _legend(ax: Any, xs: Any = (), ys: Any = (), *, ncol: int = 1, **kw: Any) -> str | None:
    """A framed legend in the emptiest corner of the plotted points; returns the other clear corner (for a
    metric box), or None when there is nothing to label."""
    handles, labels = ax.get_legend_handles_labels()
    corners = pub.clear_corners(ax, list(xs), list(ys), k=2)
    if handles:
        ax.legend(loc=corners[0], ncol=ncol, **kw)
        return corners[1]
    return corners[0]


def _breaks(t: Any, v: Any, factor: float = 3.0) -> Any:
    """``v`` with NaN inserted at every gap longer than ``factor`` times the typical step, so a line is broken
    across missing years instead of drawn straight through them."""
    import numpy as np

    if len(t) < 3:
        return t, v
    tt = t.astype("datetime64[s]").astype("int64") if np.issubdtype(t.dtype, np.datetime64) else t.astype(float)
    dt = np.diff(tt)
    typical = float(np.median(dt)) if len(dt) else 0.0
    if typical <= 0:
        return t, v
    gaps = np.where(dt > factor * typical)[0]
    if not len(gaps):
        return t, v
    t2, v2 = list(t), list(v.astype(float))
    for k in gaps[::-1]:
        mid = t[k] + (t[k + 1] - t[k]) / 2
        t2.insert(k + 1, mid)
        v2.insert(k + 1, np.nan)
    return np.array(t2, dtype=t.dtype), np.array(v2, dtype=float)


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}g}"


# ── the makers: each returns (figure, caption) or None ────────────────────

Drawn = tuple[Any, str]


def _series(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    got = series_of(payload)
    if not got:
        return None
    import numpy as np

    dates, values = got
    t, v = _dates(dates), _floats(values)
    variable = variable_of(payload)
    u = unit_of(payload, unit)
    fig, ax = _figure(height=2.9)
    tb, vb = _breaks(t, v)
    ax.plot(tb, vb, color=PRIMARY, linewidth=0.5, label=f"Daily {variable}" if resolution_word(dates) == "Daily"
            else variable[:1].upper() + variable[1:])
    marked = False
    am = annual_maxima_of(payload)
    xs: list[Any] = []
    ys: list[float] = []
    if am:
        # The marker sits at the date of the year's highest plotted day but at the true annual maximum: the
        # plotted series may be thinned for size, and a thinned peak would sit below the value the fit used.
        years = np.array([year_of(d) or 0 for d in dates])
        for y, peak in zip(am[0], am[1]):
            idx = np.where(years == y)[0]
            if len(idx) and np.isfinite(v[idx]).any():
                j = idx[np.nanargmax(v[idx])]
                xs.append(t[j])
                ys.append(float(peak) if peak is not None else float(v[j]))
        if xs:
            ax.plot(xs, ys, linestyle="none", marker="o", markersize=3.2, markerfacecolor="white",
                    markeredgecolor=DARK, markeredgewidth=0.7, label="Annual maximum", zorder=3)
            marked = True
    ax.set_xlabel("Year")
    ax.set_ylabel(_ylabel(variable, u))
    finite = v[np.isfinite(v)]
    if len(finite) and np.nanmin(finite) >= 0:
        ax.set_ylim(bottom=0)
    if marked:
        pub.headroom(ax, 0.16)
        ax.legend(loc="upper left", ncol=2)
    res = resolution_word(dates)
    period = period_of(payload, dates)
    caption = f"{res + ' ' if res else ''}{variable} at {record_name(payload, site)}"
    caption = caption[:1].upper() + caption[1:]
    if period:
        caption += f", {period}"
    if marked:
        caption += ", with the annual maxima marked"
    if len(tb) > len(t):
        caption += "; the line is broken where the record has gaps"
    return fig, caption + "."


def _annual_maxima(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    am = annual_maxima_of(payload)
    if not am:
        return None
    years, vals = am
    variable = variable_of(payload, "discharge")
    u = unit_of(payload, unit)
    fig, ax = _figure(height=2.8)
    ax.vlines(years, 0, vals, color=PRIMARY, linewidth=1.1)
    ax.plot(years, vals, linestyle="none", marker="o", markersize=2.8, color=PRIMARY)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Year")
    ax.set_ylabel(_ylabel(f"annual maximum {variable}", u))
    caption = (f"Annual maximum {variable} at {record_name(payload, site)}, {len(years)} complete years "
               f"({min(years)} to {max(years)}).")
    return fig, caption


def _gev_quantile(params: Any, aep: Any) -> Any:
    """GEV quantile at annual exceedance probability ``aep`` from Hosking's (shape, location, scale)."""
    import numpy as np

    k, loc, scale = (float(x) for x in params)
    y = -np.log(1.0 - np.asarray(aep, dtype=float))
    if abs(k) < 1e-6:
        return loc - scale * np.log(y)
    return loc + scale / k * (1.0 - y ** k)


def _lp3_quantile(params: Any, aep: Any) -> Any:
    """Log-Pearson III quantile from (skew, mean, standard deviation) of the base-10 logarithms."""
    import numpy as np
    from scipy.stats import pearson3

    skew, mu, sigma = (float(x) for x in params)
    return 10.0 ** pearson3.ppf(1.0 - np.asarray(aep, dtype=float), skew, loc=mu, scale=sigma)


def _fit_params(payload: dict[str, Any], name: str) -> list[float] | None:
    ffa = payload.get("ffa") if isinstance(payload.get("ffa"), dict) else payload
    fit = (ffa.get("fits") or {}).get(name) if isinstance(ffa.get("fits"), dict) else None
    params = fit.get("params") if isinstance(fit, dict) else None
    if isinstance(params, list) and len(params) == 3 and all(num(p) is not None for p in params):
        return [float(p) for p in params]
    return None


def _band_label(band: str | None) -> str:
    """``90 % interval, Log-Pearson III`` from the reader's ``Log-Pearson III 90 %``."""
    if not band:
        return "Confidence interval"
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*%", band)
    name = re.sub(r"\s*\d+(?:\.\d+)?\s*%", "", band).strip()
    name = name.replace("GEV MLE/L-moments bootstrap", "GEV (MLE)").replace(" bootstrap", "")
    return f"{m.group(1)} % interval, {name}" if m else f"Interval, {name}"


def _span_km(pts: list[dict[str, Any]], site: tuple[float, float] | None) -> float:
    """The largest distance from the site to any point, in km (0 without a site)."""
    import math

    if not site:
        return 0.0
    la0, lo0 = math.radians(site[0]), math.radians(site[1])
    best = 0.0
    for p in pts:
        la, lo = math.radians(p["lat"]), math.radians(p["lon"])
        h = math.sin((la - la0) / 2) ** 2 + math.cos(la0) * math.cos(la) * math.sin((lo - lo0) / 2) ** 2
        best = max(best, 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h))))
    return best


def _frequency_curve(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    rl = return_levels_of(payload)
    if not rl:
        return None
    import numpy as np

    t = _floats(rl["T"])
    variable = variable_of(payload, "discharge")
    u = unit_of(payload, unit or "m3/s")
    t_top = float(np.nanmax(t)) if np.isfinite(t).any() else 100.0
    emp = rl["empirical"]
    if emp:
        t_top = max(t_top, float(np.nanmax(_floats(emp[0]))))
    t_max = next(x for x in (200.0, 500.0, 1000.0, 10000.0, 1e5) if x >= t_top * 1.4)
    fig, ax = _figure(height=3.6)
    gx = pub.gumbel_variate
    if rl["lower"] is not None and rl["upper"] is not None:
        lo, hi = _floats(rl["lower"]), _floats(rl["upper"])
        ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(t)
        if ok.sum() >= 2:
            ax.fill_between(gx(t[ok]), lo[ok], hi[ok], color=ACCENT, alpha=0.75, linewidth=0,
                            label=_band_label(rl["band"]))
    grid_t = np.geomspace(1.0101, t_max, 200)
    drawn: list[str] = []
    curves = (("gev_lmoments", rl["gev"], "GEV (L-moments)", PRIMARY, "-", _gev_quantile),
              ("lp3", rl["lp3"], "Log-Pearson III", DANGER, "--", _lp3_quantile),
              ("gev_bootstrap", rl["boot"], "GEV (MLE)", SUCCESS, ":", _gev_quantile))
    for name, q, label, colour, style, quantile in curves:
        if q is None:
            continue
        if rl["distribution"] and name == "gev_lmoments":
            label = rl["distribution"].upper()
        params = _fit_params(payload, name)
        curve = None
        if params is not None:
            try:
                curve = quantile(params, 1.0 / grid_t)
                # Only trust the analytic curve if it reproduces the tabulated levels (same convention).
                check = quantile(params, 1.0 / t[np.isfinite(t)])
                qq = _floats(q)[np.isfinite(t)]
                if not np.allclose(check, qq, rtol=0.01, equal_nan=True):
                    curve = None
            except Exception:  # noqa: BLE001 - fall back to the tabulated points
                curve = None
        if curve is not None:
            ax.plot(gx(grid_t), curve, style, color=colour, linewidth=1.3, label=label)
        else:
            ax.plot(gx(t), _floats(q), style, color=colour, linewidth=1.3, marker="s", markersize=2.6, label=label)
        drawn.append(label)
    px: Any = []
    py: Any = []
    if emp:
        px, py = _floats(emp[0]), _floats(emp[1])
        ok = np.isfinite(px) & np.isfinite(py) & (px > 1.0)
        px, py = px[ok], py[ok]
        ax.plot(gx(px), py, linestyle="none", marker="o", markersize=3.0, markerfacecolor="white",
                markeredgecolor=DARK, markeredgewidth=0.7, label="Observed annual maxima", zorder=4)
    ex = payload.get("annual_max_excluded") if isinstance(payload.get("annual_max_excluded"), dict) else None
    n_ex = 0
    if ex and ex.get("v") and len(py):
        # An excluded year sits where it would rank in the whole sample, so the reader sees what was left out.
        ex_vals = [float(v) for v in ex["v"] if v is not None]
        allv = sorted(list(py) + ex_vals, reverse=True)
        ex_t = [(len(allv) + 1) / (allv.index(v) + 1) for v in ex_vals]
        ax.plot(gx(np.array(ex_t)), ex_vals, linestyle="none", marker="x", markersize=5, color=NEUTRAL,
                markeredgewidth=1.0, label="Excluded from the fit", zorder=4)
        n_ex = len(ex_vals)
    pub.return_period_axis(ax, 1.0101 if not len(px) or np.nanmin(px) < 1.5 else 1.1, t_max)
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel(_ylabel(f"annual maximum {variable}", u))
    xs = list(gx(px)) if len(px) else list(gx(t[np.isfinite(t)]))
    ys = list(py) if len(py) else [x for x in (rl["gev"] or rl["lp3"] or []) if x is not None]
    _legend(ax, xs, ys)
    fits = (", ".join(drawn[:-1]) + " and " + drawn[-1]) if len(drawn) > 1 else (drawn[0] if drawn else "fitted")
    caption = (f"Flood frequency curve of annual maximum {variable} at {record_name(payload, site)} on Gumbel "
               f"probability paper: the {fits} fit{'s' if len(drawn) > 1 else ''}")
    if rl["band"]:
        pct, _, owner = _band_label(rl["band"]).partition(" interval, ")
        kind = "bootstrap interval" if "bootstrap" in str(rl["band"]).lower() else "interval"
        caption += f", the {pct} {kind} of the {owner} fit (shaded)" if owner else f", with its {kind} (shaded)"
    if emp:
        caption += f", and the {len(py)} observed annual maxima at their Weibull plotting positions, T = (n + 1) / rank"
    if n_ex:
        years = ", ".join(str(y) for y in (ex or {}).get("year") or [])
        caption += f"; the crosses are the {n_ex} annual maxima left out of the fit ({years})"
    return fig, caption + "."


def _fdc(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    fdc = fdc_of(payload)
    if not fdc:
        return None
    import numpy as np

    variable = variable_of(payload, "discharge")
    u = unit_of(payload, unit or "m3/s")
    fig, ax = _figure(width="full", height=3.2)
    allq = [v for v in (fdc["q"] or list(fdc["percentiles"].values())) if v is not None]
    positive = bool(allq) and min(allq) > 0
    if fdc["q"]:
        ex, q = _floats(fdc["exceedance"]), _floats(fdc["q"])
        ok = np.isfinite(ex) & np.isfinite(q) & (ex > 0) & (ex < 100)
        ax.plot(pub.probit(ex[ok]), q[ok], color=PRIMARY, linewidth=1.3, label="Flow-duration curve")
        how = "the ranked daily flows"
    else:
        pct = fdc["percentiles"]
        ks = [k for k in pct if 0 < k < 100]
        ax.plot(pub.probit(ks), [pct[k] for k in ks], "o-", color=PRIMARY, linewidth=1.2, label="Percentiles")
        how = f"the {len(pct)} percentiles the tool reported"
    pub.exceedance_axis(ax, 0.1, 99.9)
    if positive:
        _plain_log_y(ax)
    marks = []
    for key, colour in ((10.0, DANGER), (50.0, DARK), (95.0, SUCCESS)):
        val = fdc["percentiles"].get(key)
        if val is not None and (val > 0 or not positive):
            x = float(pub.probit(key))
            ax.plot([x], [val], marker="D", markersize=4.0, color=colour, linestyle="none", zorder=4,
                    label=f"Q{int(key)} = {_fmt(val)} {_u(u)}")
            marks.append(key)
    ax.set_xlabel("Percentage of time flow is equalled or exceeded (%)")
    ax.set_ylabel(_ylabel(variable, u))
    ax.legend(loc="upper right")
    caption = (f"Flow-duration curve of {variable} at {record_name(payload, site)} from {how}, on a normal "
               f"probability axis" + (" with a logarithmic flow axis" if positive else "") +
               ("; " + ", ".join(f"Q{int(k)}" for k in marks) + " marked" if marks else "") + ".")
    return fig, caption


def _trend(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    from aquascope.trend_series import reported_trend

    tr = reported_trend(payload)
    if isinstance(tr, dict) and tr.get("unavailable"):
        return None
    if isinstance(tr, dict) and tr.get("on") == "annual maxima":
        return _trend_on_maxima(payload, tr, unit, site)
    tr = payload.get("trend")
    got = series_of(payload)
    if not isinstance(tr, dict) or not got:
        return None
    import numpy as np

    dates, values = got
    years = np.array([year_of(d) or 0 for d in dates])
    v = _floats(values)
    ok = np.isfinite(v) & (years > 0)
    if not ok.any():
        return None
    uniq = np.unique(years[ok])
    counts = np.array([(years[ok] == y).sum() for y in uniq])
    typical = float(np.median(counts)) if len(counts) else 0.0
    keep = [y for y, c in zip(uniq, counts) if typical and c >= 0.8 * typical]
    if len(keep) < 3:
        keep = list(uniq)
    xs = np.array(keep, dtype=float)
    ys = np.array([np.nanmean(v[(years == y) & ok]) for y in keep])
    variable = variable_of(payload)
    u = unit_of(payload, unit)
    fig, ax = _figure()
    ax.plot(xs, ys, "o-", color=PRIMARY, linewidth=1, markersize=4, label=f"annual mean {variable}")
    slope = num(tr.get("sens_slope_per_year"))
    if slope is not None:
        intercept = float(np.median(ys) - slope * np.median(xs))
        ax.plot(xs, intercept + slope * xs, "--", color=DANGER, linewidth=1.6,
                label=f"Sen slope {slope:+.3g} {u}/yr" if u else f"Sen slope {slope:+.3g} per yr")
    verdict = str(tr.get("trend") or "no trend").replace("_", " ")
    p = num(tr.get("p_value"))
    ax.set_xlabel("Year")
    ax.set_ylabel(_ylabel(f"annual mean {variable}", u))
    ax.legend(loc="best")
    caption = (f"Annual mean {variable} at {record_name(payload, site)} with the Sen slope line; the Mann-Kendall "
               f"test finds {verdict}" + (f" (p = {p:.3f}, {int(tr.get('n_years') or len(xs))} years)"
                                          if p is not None else "") + ".")
    return fig, caption


def _trend_on_maxima(payload: dict[str, Any], tr: dict[str, Any], unit: str | None,
                     site: dict[str, Any] | None) -> Drawn | None:
    """The trend figure for a flood question: the annual maxima the Mann-Kendall test ran on, with the Sen
    slope line."""
    import numpy as np

    am = payload.get("annual_max") if isinstance(payload.get("annual_max"), dict) else {}
    xs = _floats(am.get("year") or [])
    ys = _floats(am.get("v") or [])
    if len(xs) != len(ys):
        return None
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 3:
        return None
    xs, ys = xs[ok], ys[ok]
    variable = variable_of(payload)
    u = unit_of(payload, unit)
    fig, ax = _figure(height=3.0)
    ax.vlines(xs, 0, ys, color=ACCENT, linewidth=1.0, zorder=1)
    ax.plot(xs, ys, linestyle="none", marker="o", markersize=3.0, color=PRIMARY, label=f"Annual maximum {variable}",
            zorder=3)
    slope = num(tr.get("sens_slope_per_year"))
    if slope is not None:
        intercept = float(np.median(ys) - slope * np.median(xs))
        line_x = np.array([xs.min(), xs.max()])
        ax.plot(line_x, intercept + slope * line_x, "--", color=DANGER, linewidth=1.3,
                label=f"Sen slope, {slope:+.3g} {_u(u)} per year".replace(" , ", ", "))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Year")
    ax.set_ylabel(_ylabel(f"annual maximum {variable}", u))
    verdict = str(tr.get("trend") or "no trend").replace("_", " ")
    p = num(tr.get("p_value"))
    n_years = int(tr.get("n_years") or len(xs))
    title = "Mann-Kendall: " + verdict + (", p < 0.001" if p is not None and p < 0.001 else
                                          f", p = {p:.2f}" if p is not None else "") + f", n = {n_years}"
    pub.headroom(ax, 0.24)
    corner = pub.clear_corners(ax, xs, ys, k=1)[0]
    ax.legend(loc=corner, title=title, alignment="left")
    caption = (f"Annual maximum {variable} at {record_name(payload, site)} with the Sen slope line; the "
               f"Mann-Kendall test on the annual maxima finds {verdict}"
               + (f" (p = {p:.3f}, {int(tr.get('n_years') or len(xs))} years)" if p is not None else "") + ".")
    return fig, caption


def _drought_strip(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    panels = indices_of(payload)
    if not panels:
        return None
    import numpy as np

    n = len(panels)
    fig, axes = _figure(n, 1, height=1.6 * n + 1.0, sharex=True)
    axes = list(np.atleast_1d(axes))
    threshold = num(payload.get("threshold"))
    for ax, panel in zip(axes, panels):
        t = _dates(panel["dates"])
        if panel.get("sgi") is not None:
            main, other, name, other_name = _floats(panel["sgi"]), None, "SGI", ""
        elif panel["spei"] is not None:
            main, name = _floats(panel["spei"]), f"SPEI-{panel['timescale']}"
            other, other_name = (_floats(panel["spi"]), f"SPI-{panel['timescale']}") if panel["spi"] else (None, "")
        else:
            main, other, name, other_name = _floats(panel["spi"]), None, f"SPI-{panel['timescale']}", ""
        zero = np.zeros_like(main)
        ax.fill_between(t, zero, main, where=main >= 0, color=PRIMARY, alpha=0.85, interpolate=True, linewidth=0)
        ax.fill_between(t, zero, main, where=main < 0, color=DANGER, alpha=0.85, interpolate=True, linewidth=0)
        if other is not None:
            ax.plot(t, other, color=NEUTRAL, linewidth=0.5, alpha=0.6, label=other_name)
            ax.legend(loc="upper left")
        for level in (-2.0, -1.5, -1.0, 1.0, 1.5, 2.0):
            ax.axhline(level, color=NEUTRAL, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(name)
        ax.set_ylim(-3.2, 3.2)
    events = payload.get("events")
    if isinstance(events, list) and panels and panels[0].get("sgi") is not None:
        for e in events:
            if isinstance(e, dict) and e.get("start") and e.get("end"):
                axes[0].axvspan(_dates([e["start"]])[0], _dates([e["end"]])[0], color=WARNING, alpha=0.2)
    axes[-1].set_xlabel("Date")
    scales = [str(p["timescale"]) for p in panels if p.get("timescale") is not None]
    if scales:
        what = "SPEI (bars) with SPI (grey line)" if any(p["spei"] for p in panels) else "SPI"
        caption = (f"{what} at {record_name(payload, site)} for the {', '.join(scales)} month accumulations, "
                   f"{period_of(payload, panels[0]['dates'])}: blue above zero is wetter than normal, red below is "
                   f"drier; the dashed lines mark the moderate (1), severe (1.5) and extreme (2) classes.")
    else:
        caption = (f"Standardised Groundwater Index at {record_name(payload, site)}, "
                   f"{period_of(payload, panels[0]['dates'])}: blue above zero is above the monthly norm, red "
                   f"below; shaded spans are the droughts at or below " + (f"{threshold:g}." if threshold is not None
                                                                          else "the threshold."))
    return fig, caption


def _propagation(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    s = payload.get("series")
    if not isinstance(s, dict) or not isinstance(s.get("index"), list) or not isinstance(s.get("sgi"), list):
        return None
    t = _dates([date_key(x) for x in s["index"]])
    sgi = _floats(numbers(s["sgi"]))
    spi = _floats(numbers(s["spi"])) if isinstance(s.get("spi"), list) else None
    prop = payload.get("propagation") if isinstance(payload.get("propagation"), dict) else {}
    best = prop.get("best") if isinstance(prop.get("best"), dict) else {}
    scale, lag, corr = best.get("timescale"), best.get("lag_months"), num(best.get("correlation"))
    fig, ax = _figure()
    if spi is not None:
        ax.plot(t, spi, color=NEUTRAL, linewidth=0.8, label=f"SPI-{scale}" if scale else "SPI")
    ax.plot(t, sgi, color=PRIMARY, linewidth=1.4, label="SGI")
    ax.axhline(0, color="black", linewidth=0.5)
    thr = num(payload.get("sgi", {}).get("threshold")) if isinstance(payload.get("sgi"), dict) else None
    if thr is not None:
        ax.axhline(thr, color=DANGER, linestyle="--", linewidth=0.8, label=f"drought threshold {thr:g}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Standardised index")
    ax.legend(loc="lower left")
    caption = f"Standardised Groundwater Index at {record_name(payload, site)}"
    if spi is not None:
        caption += f" with SPI-{scale} for the ERA5 cell"
    if lag is not None:
        caption += (f"; the {scale}-month accumulation leads the water table by {lag} months"
                    + (f" (cross-correlation {corr:.2f})" if corr is not None else ""))
    return fig, caption + "."


def _points_map(ax: Any, pts: list[dict[str, Any]], site: tuple[float, float] | None, *, colour: str,
                what: str) -> None:
    import math

    lat_ref = site[0] if site else (pts[0]["lat"] if pts else 0.0)
    if pts:
        ax.plot([p["lon"] for p in pts], [p["lat"] for p in pts], linestyle="none", marker="^", markersize=5,
                markerfacecolor=colour, markeredgecolor="white", markeredgewidth=0.5, zorder=3,
                label=what[:1].upper() + what[1:])
        for p in pts[:20]:
            ax.annotate(str(p["label"])[:22], (p["lon"], p["lat"]), textcoords="offset points", xytext=(4, 3),
                        fontsize=6.5, color=DARK)
    if site:
        ax.plot([site[1]], [site[0]], linestyle="none", marker="*", markersize=11, color=DANGER,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4, label="Study site")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect(1.0 / max(math.cos(math.radians(lat_ref)), 0.2), adjustable="datalim")
    ax.margins(0.12)
    ax.grid(True, which="major")
    ax.legend(loc="best")


def _donors_map(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    donors = stations_of(payload)
    if not donors:
        return None
    pt = site_point(payload, site)
    # Donors chosen by catchment likeness can sit on another continent: a map then shows two dots an ocean
    # apart and nothing else. The table carries them instead.
    if pt is not None and _span_km(donors, pt) > 1500:
        return None
    fig, ax = _figure(height=3.6)
    _points_map(ax, donors, pt, colour=PRIMARY, what="donor gauges")
    caption = (f"The study site and the {len(donors)} donor gauges the similarity search selected, in longitude "
               f"and latitude; labels are the station identifiers.")
    return fig, caption


def _site_map(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    pt = site_point(payload, site)
    stations = stations_of(payload, site)
    # A lone site on empty axes tells the reader nothing a coordinate in the text does not.
    if not stations:
        return None
    fig, ax = _figure(height=3.6)
    _points_map(ax, stations, pt, colour=PRIMARY, what="gauging stations")
    caption = (f"The study site and the {len(stations)} catalogue stations within reach, in longitude and "
               f"latitude; labels are the station identifiers.")
    return fig, caption


def _signatures_band(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    est = payload.get("estimates")
    if not isinstance(est, dict) or not est:
        sim = payload.get("similarity")
        est = sim.get("estimates") if isinstance(sim, dict) else None
    if not isinstance(est, dict) or not est:
        return None
    skill = ((payload.get("skill") or {}).get("by_signature") or {}) if isinstance(payload.get("skill"), dict) else {}
    groups: dict[str, list[tuple[str, float, float, float, float | None]]] = {}
    for name, e in est.items():
        if not isinstance(e, dict) or num(e.get("value")) is None:
            continue
        v = num(e["value"])
        lo = num(e.get("low"))
        hi = num(e.get("high"))
        label = str(e.get("label") or name)
        label = label[:1].upper() + label[1:]
        if len(label) > 46:
            label = label.split(" (")[0][:46]
        nse = num((skill.get(name) or {}).get("nse")) if isinstance(skill.get(name), dict) else None
        groups.setdefault(str(e.get("unit") or ""), []).append((label, v, lo if lo is not None else v,
                                                                hi if hi is not None else v, nse))
    if not groups:
        return None
    units = list(groups)[:3]
    total = sum(len(groups[u]) for u in units)
    import numpy as np

    _plt()
    fig, axes = pub.new_figure("full", min(8.6, 0.24 * total + 0.75 * len(units) + 0.4), nrows=len(units),
                               gridspec_kw={"height_ratios": [len(groups[u]) + 0.9 for u in units]})
    axes = list(np.atleast_1d(axes))
    for ax, u in zip(axes, units):
        items = groups[u]
        y = np.arange(len(items))
        vals = np.array([i[1] for i in items])
        lo = np.array([i[2] for i in items])
        hi = np.array([i[3] for i in items])
        ax.hlines(y, lo, hi, color=PRIMARY, linewidth=1.4)
        ax.plot(vals, y, linestyle="none", marker="o", markersize=4.2, markerfacecolor="white",
                markeredgecolor=PRIMARY, markeredgewidth=1.0, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([i[0] for i in items])
        ax.tick_params(axis="y", which="both", length=0)
        ax.yaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
        ax.set_ylim(len(items) - 0.4, -0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(_u(u) or "Dimensionless")
        if any(i[4] is not None for i in items):
            sec = ax.secondary_yaxis("right")
            sec.set_yticks(y)
            sec.set_yticklabels([f"{i[4]:.2f}" if i[4] is not None else "" for i in items])
            sec.tick_params(axis="y", which="both", length=0)
            sec.yaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
            sec.set_ylabel("Leave-one-out NSE", fontsize=7.5)
    k = payload.get("k") or (payload.get("similarity") or {}).get("k")
    if not k:
        k = next((e.get("n_donors") for e in est.values() if isinstance(e, dict) and e.get("n_donors")), None)
    caption = (f"Flow signatures transferred to {record_name(payload, site)} from {k or 'the'} donor catchments: "
               f"the circle is the estimate and the line spans the band across donors" +
               (", with the leave-one-out skill (Nash-Sutcliffe efficiency) on the right" if skill else "") + ".")
    return fig, caption


def _monthly_climate(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    clim = payload.get("climate") if isinstance(payload.get("climate"), dict) else payload
    p = numbers(clim.get("monthly_precipitation_mm"))
    if len(p) != 12:
        return None
    import numpy as np

    et0 = numbers(clim.get("monthly_et0_mm"))
    temp = numbers(clim.get("monthly_temperature_c"))
    x = np.arange(12)
    fig, ax = _figure(height=2.9)
    ax.bar(x, _floats(p), width=0.62, color=PRIMARY, label="Precipitation")
    if len(et0) == 12:
        ax.plot(x, _floats(et0), "o-", color=WARNING, linewidth=1.3, markersize=3.5, label="Reference ET$_0$")
    ax.set_ylabel("Depth (mm per month)")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.set_xlim(-0.6, 11.6)
    ax.set_ylim(bottom=0)
    pub.headroom(ax, 0.22)
    handles, labels = ax.get_legend_handles_labels()
    if len(temp) == 12:
        ax2 = ax.twinx()
        ax2.plot(x, _floats(temp), "s--", color=DANGER, linewidth=1.0, markersize=3.0, label="Mean temperature")
        ax2.set_ylabel("Temperature (°C)")
        ax2.grid(False)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, loc="upper center", ncol=3)
    years = payload.get("years") or clim.get("years")
    caption = ("Mean monthly precipitation (bars)" + (" and FAO-56 reference evapotranspiration (line)"
                                                      if len(et0) == 12 else "") +
               (" with mean temperature (dashed)" if len(temp) == 12 else "") +
               f" for the ERA5 cell at {record_name(payload, site)}" +
               (f", {years} years ending {payload.get('end')}" if years and payload.get("end") else "") + ".")
    return fig, caption


def _glofas_series(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    g = payload.get("glofas") if isinstance(payload.get("glofas"), dict) else None
    if g is None:
        return None
    u = unit_of(g, unit or "m3/s")
    got = series_of(g)
    fig, ax = _figure(height=2.9)
    if got:
        t, v = _dates(got[0]), _floats(got[1])
        tb, vb = _breaks(t, v)
        ax.plot(tb, vb, color=SUCCESS, linewidth=0.5)
        ax.set_xlabel("Year")
        what = "Daily modelled discharge"
    else:
        am = annual_maxima_of(g)
        if not am:
            close(fig)
            return None
        from matplotlib.ticker import MaxNLocator

        ax.vlines(am[0], 0, am[1], color=ACCENT, linewidth=1.0)
        ax.plot(am[0], am[1], linestyle="none", marker="o", markersize=3.0, color=SUCCESS)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Year")
        what = "Annual maxima of the modelled discharge"
    ax.set_ylim(bottom=0)
    ax.set_ylabel(_ylabel("modelled discharge", u))
    caption = (f"{what} from GloFAS v4 (Open-Meteo) for the grid cell at {record_name(payload, site)}, "
               f"{period_of(g)}: a model output, indicative only, not a gauge reading.")
    return fig, caption


def _reliability_curve(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    fdc = fdc_of(payload)
    if not fdc:
        return None
    u = unit_of(payload, unit or "m3/s")
    required = num(payload.get("required_flow_m3s"))
    reserve = num(payload.get("reserve_m3s"))
    rel = payload.get("reliability") if isinstance(payload.get("reliability"), dict) else {}
    daily = num(rel.get("daily"))
    fig, ax = _figure()
    if fdc["q"]:
        ax.plot(_floats(fdc["exceedance"]), _floats(fdc["q"]), color=PRIMARY, linewidth=1.8, label="flow-duration")
    else:
        pct = fdc["percentiles"]
        ax.plot(list(pct), list(pct.values()), "o-", color=PRIMARY, linewidth=1.5, label="flow-duration (percentiles)")
    q95 = fdc["percentiles"].get(95.0)
    if q95 is not None:
        ax.axhline(q95, color=NEUTRAL, linestyle="--", linewidth=0.9, label=f"Q95 = {_fmt(q95)} {u}")
    if reserve is not None:
        ax.axhline(reserve, color=WARNING, linestyle="-.", linewidth=1.0, label=f"reserve = {_fmt(reserve)} {u}")
    if required is not None:
        ax.axhline(required, color=DANGER, linewidth=1.4, label=f"required = {_fmt(required)} {u}")
    allq = [v for v in (fdc["q"] or list(fdc["percentiles"].values())) if v > 0]
    if allq and min(allq) > 0:
        _plain_log_y(ax)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel(_ylabel("flow", u))
    ax.legend(loc="upper right")
    caption = (f"The flow-duration curve at {record_name(payload, site)} with the flow the demand needs (red), "
               f"the reserve left in the river (orange) and Q95 (dashed)" +
               (f"; the demand is met on {daily:.0%} of days" if daily is not None else "") + ".")
    return fig, caption


def _demand_monthly(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    import numpy as np

    sched = frame_records(payload.get("schedule"))
    seasons = payload.get("per_season")
    fig, ax = _figure()
    if sched and "date" in sched[0]:
        cols, rows = sched
        ci = {c: i for i, c in enumerate(cols)}
        wanted = [c for c in ("etc", "effective_rain", "net_irrigation", "gross_irrigation") if c in ci]
        if not wanted:
            close(fig)
            return None
        months: dict[str, dict[str, float]] = {}
        for r in rows:
            key = str(r[ci["date"]])[:7]
            m = months.setdefault(key, {c: 0.0 for c in wanted})
            for c in wanted:
                v = num(r[ci[c]])
                if v is not None:
                    m[c] += v
        keys = sorted(months)
        x = np.arange(len(keys))
        w = 0.8 / len(wanted)
        colours = {"etc": NEUTRAL, "effective_rain": SECONDARY, "net_irrigation": PRIMARY, "gross_irrigation": DARK}
        for i, c in enumerate(wanted):
            ax.bar(x + i * w - 0.4 + w / 2, [months[k][c] for k in keys], width=w, color=colours[c],
                   label=c.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Month")
        caption = (f"Monthly crop evapotranspiration, effective rain and net and gross irrigation over the season "
                   f"for {payload.get('crop') or 'the crop'} planted on {payload.get('planting_date') or '?'}, "
                   f"from the FAO-56 schedule.")
    elif isinstance(seasons, list) and seasons and all(isinstance(s, dict) for s in seasons):
        wanted = [c for c in ("etc_mm", "effective_rain_mm", "net_irrigation_mm", "gross_irrigation_mm")
                  if any(num(s.get(c)) is not None for s in seasons)]
        if not wanted:
            close(fig)
            return None
        years = [str(s.get("year")) for s in seasons]
        x = np.arange(len(years))
        w = 0.8 / len(wanted)
        colours = {"etc_mm": NEUTRAL, "effective_rain_mm": SECONDARY, "net_irrigation_mm": PRIMARY,
                   "gross_irrigation_mm": DARK}
        for i, c in enumerate(wanted):
            ax.bar(x + i * w - 0.4 + w / 2, _floats([num(s.get(c)) for s in seasons]), width=w, color=colours[c],
                   label=c.removesuffix("_mm").replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45 if len(years) > 12 else 0, fontsize=8)
        ax.set_xlabel("Season (year of planting)")
        d = payload.get("demand") if isinstance(payload.get("demand"), dict) else {}
        caption = (f"Crop evapotranspiration, effective rain and net and gross irrigation per season for "
                   f"{str(payload.get('crop') or 'the crop').replace('_', ' ')} on {payload.get('area_ha') or '?'} ha "
                   f"planted on the first of month {payload.get('planting_month') or '?'}; the mean gross depth is "
                   f"{_fmt(num(d.get('gross_irrigation_mm')), 4)} mm over the season.")
    else:
        close(fig)
        return None
    ax.set_ylabel("mm")
    ax.legend(loc="upper right")
    return fig, caption


def _et0_monthly(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    import numpy as np

    eto = payload.get("eto")
    clim = payload.get("climate") if isinstance(payload.get("climate"), dict) else payload
    got = series_of({"series": eto}) if isinstance(eto, dict) else None
    fig, ax = _figure()
    x = np.arange(12)
    if got:
        months = np.array([int(d[5:7]) for d in got[0]])
        v = _floats(got[1])
        means = [float(np.nanmean(v[months == m])) if (months == m).any() else np.nan for m in range(1, 13)]
        ax.bar(x, means, color=PRIMARY)
        ax.set_ylabel("ET0 (mm per day)")
        caption = (f"Mean FAO-56 reference evapotranspiration by calendar month from the daily series, "
                   f"{period_of(payload, got[0])}.")
    elif len(numbers(clim.get("monthly_et0_mm"))) == 12:
        ax.bar(x, _floats(numbers(clim["monthly_et0_mm"])), color=PRIMARY)
        ax.set_ylabel("ET0 (mm per month)")
        caption = f"Mean monthly FAO-56 reference evapotranspiration for the ERA5 cell at {record_name(payload, site)}."
    else:
        close(fig)
        return None
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    return fig, caption


def _samples_by_parameter(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    rows = payload.get("samples")
    if not isinstance(rows, list) or not rows:
        return None
    by: dict[str, list[float]] = {}
    units: dict[str, str] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        v = num(r.get("value"))
        p = r.get("parameter")
        if v is None or p is None:
            continue
        by.setdefault(str(p), []).append(v)
        if r.get("unit") and str(p) not in units:
            units[str(p)] = str(r["unit"])
    if not by:
        return None
    names = sorted(by, key=lambda k: -len(by[k]))[:8]
    cols = min(4, len(names))
    nrows = (len(names) + cols - 1) // cols
    fig, axes = _figure(nrows, cols, height=2.6 * nrows + 0.6)
    import numpy as np

    flat = list(np.atleast_1d(axes).ravel())
    for ax, name in zip(flat, names):
        ax.boxplot(by[name], widths=0.5, patch_artist=True,
                   boxprops={"facecolor": ACCENT, "color": PRIMARY}, medianprops={"color": DANGER},
                   whiskerprops={"color": PRIMARY}, capprops={"color": PRIMARY},
                   flierprops={"marker": ".", "markerfacecolor": NEUTRAL, "markersize": 3})
        ax.set_xticks([1])
        ax.set_xticklabels([f"n = {len(by[name])}"], fontsize=8)
        ax.set_ylabel(units.get(name, ""), fontsize=8)
    for ax in flat[len(names):]:
        ax.set_visible(False)
    caption = (f"Distribution of the sampled values per parameter at {record_name(payload, site)} "
               f"({len(rows)} samples, {period_of(payload)}): box is the interquartile range, the line the median, "
               f"points beyond 1.5 IQR shown singly.")
    return fig, caption


def _who_exceedances(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    items = [(str(r.get("parameter")), num(r.get("pct")) or 0.0, str(r.get("status") or ""), r.get("rule"))
             for r in rows if isinstance(r, dict) and r.get("parameter") is not None]
    if not items:
        return None
    import numpy as np

    items.sort(key=lambda i: -i[1])
    fig, ax = _figure(height=max(3.0, 0.4 * len(items) + 1.2))
    y = np.arange(len(items))
    colours = [DANGER if i[2] == "Alert" else WARNING if i[2] == "Warning" else SUCCESS for i in items]
    ax.barh(y, [i[1] for i in items], color=colours)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i[0]} ({i[3]})" if i[3] else i[0] for i in items], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(10, color=NEUTRAL, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Samples outside the WHO guideline (%)")
    caption = ("Share of samples outside the WHO drinking-water guideline per parameter; red is an alert (over "
               "10 %), orange a warning (any exceedance), green within the guideline.")
    return fig, caption


def _wqi_bars(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    ccme = payload.get("ccme") if isinstance(payload.get("ccme"), dict) else {}
    nsf = payload.get("nsf") if isinstance(payload.get("nsf"), dict) else {}
    scores = [(name, num(block.get("score")), block.get("category"))
              for name, block in (("CCME WQI", ccme), ("NSF WQI", nsf)) if num(block.get("score")) is not None]
    if not scores:
        return None
    factors = [(f"F{i} {label}", num(ccme.get(f"f{i}"))) for i, label in ((1, "scope"), (2, "frequency"),
                                                                           (3, "amplitude"))]
    factors = [f for f in factors if f[1] is not None]
    fig, axes = _figure(1, 2 if factors else 1)
    import numpy as np

    axes = list(np.atleast_1d(axes))
    ax = axes[0]
    x = np.arange(len(scores))
    ax.bar(x, [s[1] for s in scores], color=PRIMARY, width=0.5)
    for i, s in enumerate(scores):
        ax.text(i, s[1] + 1.5, f"{s[1]:.0f}\n{s[2] or ''}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in scores])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Index (0 to 100)")
    if factors:
        ax2 = axes[1]
        x2 = np.arange(len(factors))
        ax2.bar(x2, [f[1] for f in factors], color=WARNING, width=0.5)
        ax2.set_xticks(x2)
        ax2.set_xticklabels([f[0] for f in factors], fontsize=8)
        ax2.set_ylim(0, 100)
    head = scores[0]
    caption = (f"{head[0]} of {head[1]:.0f} ({head[2]}) over {payload.get('n_samples') or 'the'} samples against the "
               f"{payload.get('guideline_set') or payload.get('use') or 'drinking'} guidelines" +
               ("; the CCME factors are the scope, frequency and amplitude of the exceedances" if factors else "")
               + ".")
    return fig, caption


def _baseflow(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    s = payload.get("series")
    if not isinstance(s, dict) or not isinstance(s.get("index"), list) or not isinstance(s.get("total"), list):
        return None
    t = _dates([date_key(x) for x in s["index"]])
    total = _floats(numbers(s["total"]))
    base = _floats(numbers(s.get("baseflow") or []))
    if len(base) != len(total):
        return None
    u = unit_of(payload, unit or "m3/s")
    fig, ax = _figure()
    ax.plot(t, total, color=PRIMARY, linewidth=0.8, label="total flow")
    ax.fill_between(t, 0, base, color=ACCENT, alpha=0.8, label="baseflow")
    ax.plot(t, base, color=DARK, linewidth=0.7)
    bfi = num(payload.get("bfi"))
    ax.set_xlabel("Date")
    ax.set_ylabel(_ylabel("discharge", u))
    ax.legend(loc="upper right")
    caption = (f"Total flow and the separated baseflow at {record_name(payload, site)} by the "
               f"{str(payload.get('method') or 'digital filter').replace('_', ' ')} method" +
               (f"; the baseflow index is {bfi:.2f}" if bfi is not None else "") + ".")
    return fig, caption


def _recharge(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    got = series_of(payload) or series_of(payload, "levels")
    value = num(payload.get("value_mm_per_year"))
    unc = num(payload.get("uncertainty"))
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    sy = num(meta.get("specific_yield"))
    if got:
        t, v = _dates(got[0]), _floats(got[1])
        fig, ax = _figure()
        ax.plot(t, v, color=PRIMARY, linewidth=0.9, label="water table")
        events = payload.get("events") or payload.get("rises") or []
        n_ev = 0
        for e in events if isinstance(events, list) else []:
            if isinstance(e, dict) and e.get("start") and e.get("end"):
                ax.axvspan(_dates([e["start"]])[0], _dates([e["end"]])[0], color=ACCENT, alpha=0.5)
                n_ev += 1
        ax.set_xlabel("Date")
        ax.set_ylabel(_ylabel("water level", unit_of(payload, unit or "m")))
        caption = (f"The water table at {record_name(payload, site)} with the rise events the water-table "
                   f"fluctuation method sums" + (f" ({n_ev} marked)" if n_ev else "") +
                   (f"; recharge {value:.0f} mm/yr" if value is not None else "") +
                   (f" at a specific yield of {sy:g}" if sy is not None else "") + ".")
        return fig, caption
    if value is None:
        return None
    fig, ax = _figure(height=3.2)
    ax.bar([0], [value], yerr=[unc] if unc is not None else None, color=PRIMARY, width=0.4,
           error_kw={"ecolor": DARK, "capsize": 6})
    ax.set_xticks([0])
    ax.set_xticklabels([str(payload.get("method") or "water-table fluctuation")])
    ax.set_ylabel("Recharge (mm per year)")
    caption = (f"Recharge of {value:.0f} mm/yr" + (f" (uncertainty {unc:.0f} mm/yr)" if unc is not None else "") +
               " by the water-table fluctuation method" + (f" at a specific yield of {sy:g}" if sy is not None
                                                            else "") + "; the level record itself was not in the "
               "payload, so only the estimate is drawn.")
    return fig, caption


# ── advanced study steps (aquascope.advanced) ─────────────────────────────


def _change_points(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    ann = payload.get("annual") or {}
    years, vals = ann.get("year") or [], _floats(ann.get("value") or [])
    if len(years) < 5:
        return None
    import numpy as np

    u = unit_of(payload, unit)
    what = "annual maximum" if payload.get("tested") == "annual_max" else "annual mean"
    fig, ax = _figure()
    ax.plot(years, vals, "o-", color=PRIMARY, markersize=3, linewidth=1, label=what)
    for seg in (payload.get("pelt") or {}).get("segments") or []:
        if seg.get("mean") is not None:
            ax.hlines(seg["mean"], seg["start_year"], seg["end_year"], colors=DARK, linewidth=2)
    pet = payload.get("pettitt") or {}
    if pet.get("change_year"):
        ax.axvline(pet["change_year"], color=DANGER if pet.get("significant") else NEUTRAL, linestyle="--",
                   label=f"Pettitt change {pet['change_year']} (p {_fmt(pet.get('p_value'), 2)})")
    mk = payload.get("mann_kendall") or {}
    slope = num(mk.get("sen_slope_per_year"))
    if slope is not None:
        x = np.asarray(years, dtype=float)
        mid = float(np.nanmedian(vals))
        ax.plot(x, mid + slope * (x - np.median(x)), ":", color=WARNING,
                label=f"Sen's slope (MK p {_fmt(mk.get('p_value'), 2)})")
    ax.set_xlabel("Year")
    ax.set_ylabel(_ylabel(what, u))
    ax.legend(loc="upper left")
    verdict = "stationary" if payload.get("stationary") else "not stationary"
    caption = (f"{what.capitalize()} at {record_name(payload, site)} ({len(years)} years) with the PELT segment "
               f"means (solid), the Pettitt change year (dashed) and Sen's slope (dotted); the record reads as "
               f"{verdict} at the {_fmt(payload.get('alpha') or 0.05, 2)} level.")
    return fig, caption


def _nonstationary_levels(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    tt = payload.get("through_time") or {}
    years = tt.get("year") or []
    keys = [k for k in tt if k != "year"]
    if not years or not keys:
        return None
    u = unit_of(payload, unit or "m3/s")
    fig, ax = _figure()
    am = payload.get("annual_maxima") or {}
    if am.get("year"):
        ax.scatter(am["year"], _floats(am.get("value") or []), s=10, color=NEUTRAL, label="annual maxima")
    colours = [SECONDARY, PRIMARY]
    for i, k in enumerate(keys):
        ax.plot(years, _floats(tt[k]), color=colours[i % 2], linewidth=2, label=f"{k}-year level (nonstationary)")
        st = ((payload.get("stationary") or {}).get("q_by_T") or {}).get(k)
        if st is not None:
            ax.axhline(st, color=colours[i % 2], linestyle="--", linewidth=1, label=f"{k}-year level (stationary)")
    last = payload.get("last_year")
    if last and payload.get("horizon_year") and payload["horizon_year"] > last:
        ax.axvspan(last, payload["horizon_year"], color=ACCENT, alpha=0.3, label="beyond the record")
    ci = (payload.get("nonstationary") or {}).get("ci_last_year")
    if ci and last:
        ax.errorbar([last], [(ci["lower"] + ci["upper"]) / 2], yerr=[[(ci["upper"] - ci["lower"]) / 2]],
                    color=DARK, capsize=4, label=f"90% interval, {ci['T']:g}-year in {last}")
    ax.set_xlabel("Year")
    ax.set_ylabel(_ylabel("discharge", u))
    ax.legend(loc="upper left")
    lr = payload.get("likelihood_ratio") or {}
    caption = (f"T-year daily-mean flows at {record_name(payload, site)} from a GEV whose location moves with time "
               f"(solid) against the stationary GEV (dashed); the trend term has likelihood-ratio p = "
               f"{_fmt(lr.get('p_value'), 2)} and the preferred model is the {payload.get('preferred')} one.")
    return fig, caption


def _pot_frequency(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    rows = payload.get("table") or []
    if not rows:
        return None
    u = unit_of(payload, unit or "m3/s")
    t = _floats([r["T"] for r in rows])
    fig, ax = _figure()
    ax.plot(t, _floats([r.get("pot_gpd") for r in rows]), "o-", color=PRIMARY,
            label=f"peaks over threshold (GPD, {payload.get('n_peaks')} peaks)")
    if any(r.get("annual_max_gev") is not None for r in rows):
        ax.plot(t, _floats([r.get("annual_max_gev") for r in rows]), "s--", color=DARK, label="annual maxima (GEV)")
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel(_ylabel("discharge", u))
    ax.legend(loc="upper left")
    caption = (f"Return levels at {record_name(payload, site)} from {payload.get('n_peaks')} independent peaks over "
               f"{_fmt(payload.get('threshold'))} {u} ({_fmt(payload.get('peaks_per_year'), 2)} a year) fitted to a "
               "Generalised Pareto, against the annual-maximum GEV.")
    return fig, caption


def _model_fit(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    mo = payload.get("monthly") or {}
    if not mo.get("month"):
        return None
    x = _dates(mo["month"])
    fig, ax = _figure()
    ax.plot(x, _floats(mo.get("observed") or []), color=NEUTRAL, linewidth=1.2, label="observed")
    ax.plot(x, _floats(mo.get("simulated") or []), color=PRIMARY, linewidth=1.2, label="GR4J")
    val = payload.get("validation") or {}
    if val.get("start"):
        ax.axvspan(_dates([val["start"]])[0], _dates([val["end"]])[0], color=ACCENT, alpha=0.25,
                   label="validation years")
    ax.set_ylabel(_ylabel("monthly mean discharge", "m3/s"))
    ax.legend(loc="upper left")
    snow = " with a degree-day snow store" if (payload.get("snow") or {}).get("used") else ""
    caption = (f"Monthly mean flow at {record_name(payload, site)}, observed and simulated by GR4J{snow} on ERA5 "
               f"forcing; validation KGE {_fmt(val.get('kge'), 2)} and NSE {_fmt(val.get('nse'), 2)} on the shaded "
               "years the model was not calibrated on.")
    return fig, caption


def _scenario_bars(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    runs = payload.get("scenarios") or []
    if not runs:
        return None
    import numpy as np

    labels = [r.get("label") or "" for r in runs]
    metrics = [("mean_flow_change_pct", "mean flow", PRIMARY), ("q95_change_pct", "low flow (Q95)", WARNING),
               ("amax_median_change_pct", "median annual max", DANGER)]
    fig, ax = _figure(height=max(3.0, 0.7 * len(runs) + 1.5))
    y = np.arange(len(runs))
    h = 0.25
    for i, (key, name, colour) in enumerate(metrics):
        ax.barh(y + (i - 1) * h, _floats([r.get(key) for r in runs]), height=h, color=colour, label=name)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Change against the model's own baseline (%)")
    ax.legend(loc="lower right")
    caption = (f"Change in mean flow, low flow and the median annual maximum at {record_name(payload, site)} when "
               "the calibrated GR4J runs on scaled rainfall and evaporation (and a warmer snow store); a sensitivity, "
               "not a projection.")
    return fig, caption


def _projection_spread(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    rows = payload.get("models") or []
    ens = payload.get("ensemble") or {}
    keys = [(k, n) for k, n in (("precip_change_pct", "annual rainfall"), ("pet_change_pct", "evaporation"),
                                ("wettest_day_change_pct", "wettest day"), ("mean_flow_change_pct", "mean flow"),
                                ("q95_change_pct", "low flow (Q95)"), ("flood_change_pct", "flood"))
            if (ens.get(k) or {}).get("n")]
    if not rows or not keys:
        return None
    fig, ax = _figure(height=max(3.0, 0.55 * len(keys) + 1.5))
    for i, (k, _name) in enumerate(keys):
        vals = _floats([r.get(k) for r in rows])
        ax.scatter(vals, [i] * len(vals), color=SECONDARY, s=22, zorder=3)
        med = ens[k].get("median")
        if med is not None:
            ax.scatter([med], [i], marker="D", color=DARK, s=40, zorder=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(keys)), [n for _k, n in keys])
    ax.set_xlabel(f"Change {payload.get('baseline', ['', ''])[0]}-{payload.get('baseline', ['', ''])[1]} to "
                  f"{payload.get('future', ['', ''])[0]}-{payload.get('future', ['', ''])[1]} (%)")
    caption = (f"Change factors from {payload.get('n_models')} CMIP6 HighResMIP models at the site (Open-Meteo, "
               "bias-corrected onto ERA5-Land): one dot per model, the diamond the ensemble median. The spread is the "
               "result; the median alone is not.")
    return fig, caption


def _regional_growth(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    rf = payload.get("regional_frequency") or {}
    reg = rf.get("regional") or {}
    growth = reg.get("growth") or reg.get("growth_curve") or {}
    if not isinstance(growth, dict) or not growth:
        return None
    t = sorted(float(k) for k in growth)
    fig, ax = _figure()
    ax.plot(t, _floats([growth.get(str(int(x)) if x.is_integer() else str(x)) for x in t]), "o-", color=PRIMARY,
            linewidth=2, label=f"pooled growth curve ({rf.get('n_sites')} gauges)")
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Flood / index flood (-)")
    het = rf.get("heterogeneity") or {}
    ax.legend(loc="upper left")
    caption = (f"The pooled GEV growth curve from {rf.get('n_sites')} gauges within {_fmt(payload.get('radius_km'))} "
               f"km of the site (Hosking and Wallis index flood); heterogeneity H = {_fmt(het.get('H'), 2)} "
               f"({het.get('class') or 'n/a'}).")
    return fig, caption


MAKERS: dict[str, Callable[[dict[str, Any], str | None, dict[str, Any] | None], Drawn | None]] = {
    "series": _series,
    "annual_maxima": _annual_maxima,
    "frequency_curve": _frequency_curve,
    "fdc": _fdc,
    "trend": _trend,
    "drought_strip": _drought_strip,
    "propagation": _propagation,
    "donors_map": _donors_map,
    "signatures_band": _signatures_band,
    "monthly_climate": _monthly_climate,
    "glofas_series": _glofas_series,
    "reliability_curve": _reliability_curve,
    "demand_monthly": _demand_monthly,
    "et0_monthly": _et0_monthly,
    "site_map": _site_map,
    "samples_by_parameter": _samples_by_parameter,
    "who_exceedances": _who_exceedances,
    "wqi_bars": _wqi_bars,
    "baseflow": _baseflow,
    "recharge": _recharge,
    "change_points": _change_points,
    "nonstationary_levels": _nonstationary_levels,
    "pot_frequency": _pot_frequency,
    "model_fit": _model_fit,
    "scenario_bars": _scenario_bars,
    "projection_spread": _projection_spread,
    "regional_growth": _regional_growth,
}


def kinds() -> list[str]:
    """Every figure kind a maker exists for."""
    return sorted(MAKERS)


def make(kind: str, payload: dict[str, Any], *, unit: str | None = None,
         site: dict[str, Any] | None = None) -> Drawn | None:
    """``(figure, caption)`` for one kind, or None when the payload holds nothing to draw or the kind is unknown."""
    maker = MAKERS.get(kind)
    if maker is None or not isinstance(payload, dict):
        return None
    return maker(payload, unit, site)


def draw(kind: str, payload: dict[str, Any], *, unit: str | None = None,
         site: dict[str, Any] | None = None) -> Figure | None:
    """The matplotlib figure for one kind (the notebook redraws figures with this), or None."""
    drawn = make(kind, payload, unit=unit, site=site)
    return drawn[0] if drawn else None


def kinds_of(tool: str) -> list[str]:
    """The figure kinds the catalogue lists for a tool."""
    from aquascope.studio import catalogue

    entry = catalogue.get(tool)
    return list(entry.figures) if entry else []


def figures_for(step_id: str, tool: str, payload: dict[str, Any], *, unit: str | None = None,
                site: dict[str, Any] | None = None, kinds: list[str] | None = None) -> list[Artifact]:
    """One PNG and one SVG artifact per figure kind the tool yields (or per ``kinds``) that the payload supports.

    Ids are ``fig-{step_id}-{kind}`` and ``fig-{step_id}-{kind}-svg``; names ``figures/{step_id}_{kind}.png``
    and ``.svg``; ``meta`` carries the kind and the tool. A maker that finds nothing to draw yields no artifact,
    and a maker that trips on an unexpected shape is logged and skipped rather than raised.
    """
    wanted = list(kinds) if kinds is not None else kinds_of(tool)
    out: list[Artifact] = []
    if not isinstance(payload, dict) or payload.get("error"):
        return out
    for kind in wanted:
        try:
            drawn = make(kind, payload, unit=unit, site=site)
        except Exception as exc:  # noqa: BLE001 - a figure that cannot be drawn is not a failed study
            logger.warning("figure %s for step %s (%s) skipped: %s", kind, step_id, tool, exc)
            drawn = None
        if drawn is None:
            continue
        fig, caption = drawn
        try:
            png, svg = png_bytes(fig), svg_bytes(fig)
        finally:
            close(fig)
        meta = {"kind": kind, "tool": tool}
        out.append(Artifact(id=f"fig-{step_id}-{kind}", kind="figure", name=f"figures/{step_id}_{kind}.png",
                            data=png, media_type=MEDIA_TYPES["png"], caption=caption, step=step_id, meta=dict(meta)))
        out.append(Artifact(id=f"fig-{step_id}-{kind}-svg", kind="figure", name=f"figures/{step_id}_{kind}.svg",
                            data=svg, media_type=MEDIA_TYPES["svg"], caption=caption, step=step_id,
                            meta={**meta, "format": "svg"}))
    return out
