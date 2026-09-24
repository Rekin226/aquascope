"""The figures of a study: one PNG and one SVG per figure kind a step yields, from its payload.

The catalogue (:mod:`aquascope.studio.catalogue`) says which kinds each tool
yields; :func:`figures_for` draws them and returns :class:`Artifact` pairs
the Author places in the report. :func:`draw` returns the matplotlib figure
itself, for the notebook that redraws them. Every maker reads the payload
defensively through :mod:`aquascope.studio.deliverables._payload` and
returns None when there is nothing to draw, so a missing key never breaks
the Author.

matplotlib is imported inside the functions (the module imports in the
Pyodide worker before the plotting package is loaded), the Agg backend is
selected, every figure is closed after rendering, and only the default
DejaVu fonts are used. The look mirrors :func:`aquascope.viz.styles.apply_aqua_style`:
one accent colour, light dashed grid, no top or right spine, 7 by 4 inches.
"""

from __future__ import annotations

import io
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

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

FIGSIZE = (7.0, 4.0)
DPI = 150

#: The house colours (the same values as :data:`aquascope.viz.styles.AQUA_PALETTE`, kept here so this module
#: imports nothing that imports matplotlib).
PRIMARY = "#0077B6"
SECONDARY = "#00B4D8"
ACCENT = "#90E0EF"
DARK = "#023E8A"
DANGER = "#E63946"
WARNING = "#F4A261"
SUCCESS = "#2A9D8F"
NEUTRAL = "#6C757D"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── matplotlib plumbing ───────────────────────────────────────────────────


def _plt() -> Any:
    """pyplot on the Agg backend with the house style applied (imported here, never at module import)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aquascope.viz.styles import apply_aqua_style

    apply_aqua_style()
    plt.rcParams.update({"figure.figsize": FIGSIZE, "figure.dpi": DPI, "font.family": "DejaVu Sans",
                         "savefig.dpi": DPI})
    return plt


def _figure(rows: int = 1, cols: int = 1, *, height: float | None = None, sharex: bool = False) -> tuple[Any, Any]:
    plt = _plt()
    size = (FIGSIZE[0], height if height is not None else FIGSIZE[1])
    return plt.subplots(rows, cols, figsize=size, sharex=sharex)


def _dates(values: list[str]) -> Any:
    import numpy as np

    return np.array([date_key(v) for v in values], dtype="datetime64[s]")


def _floats(values: list[float | None]) -> Any:
    import numpy as np

    return np.array([np.nan if v is None else v for v in values], dtype=float)


def png_bytes(fig: Figure) -> bytes:
    """The figure as PNG bytes at 150 dpi (tight bounding box)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def svg_bytes(fig: Figure) -> bytes:
    """The figure as SVG bytes (text as paths, so no font is needed to view it)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor="white", metadata={"Date": None})
    return buf.getvalue()


def close(fig: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _plain_log_y(ax: Any) -> None:
    """Plain numbers on a log axis instead of powers of ten."""
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    ax.set_yscale("log")
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(NullFormatter())


def _ylabel(variable: str, unit: str) -> str:
    v = variable[:1].upper() + variable[1:]
    return f"{v} ({unit})" if unit else v


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
    dates, values = got
    t, v = _dates(dates), _floats(values)
    variable = variable_of(payload)
    u = unit_of(payload, unit)
    fig, ax = _figure()
    ax.plot(t, v, color=PRIMARY, linewidth=0.8, label=variable)
    marked = False
    am = annual_maxima_of(payload)
    if am:
        import numpy as np

        years = np.array([year_of(d) or 0 for d in dates])
        xs, ys = [], []
        for y in am[0]:
            idx = np.where(years == y)[0]
            if len(idx) and np.isfinite(v[idx]).any():
                j = idx[np.nanargmax(v[idx])]
                xs.append(t[j])
                ys.append(v[j])
        if xs:
            ax.scatter(xs, ys, color=DANGER, s=14, zorder=3, label="annual maximum")
            marked = True
    ax.set_xlabel("Date")
    ax.set_ylabel(_ylabel(variable, u))
    ax.set_title(f"{variable[:1].upper()}{variable[1:]} at {record_name(payload, site)}")
    if marked:
        # Below the axis, so the legend never covers the maxima it names (the PNG is saved with a tight box).
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    res = resolution_word(dates)
    period = period_of(payload, dates)
    caption = f"{res + ' ' if res else ''}{variable} at {record_name(payload, site)}"
    caption = caption[:1].upper() + caption[1:]
    if period:
        caption += f", {period}"
    if marked:
        caption += ", with the annual maxima marked"
    return fig, caption + "."


def _annual_maxima(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    am = annual_maxima_of(payload)
    if not am:
        return None
    years, vals = am
    variable = variable_of(payload, "discharge")
    u = unit_of(payload, unit)
    fig, ax = _figure()
    ax.bar(years, vals, color=PRIMARY, width=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel(_ylabel(f"annual maximum {variable}", u))
    ax.set_title(f"Annual maxima at {record_name(payload, site)}")
    caption = (f"Annual maximum {variable} at {record_name(payload, site)}, {len(years)} years "
               f"({min(years)} to {max(years)}).")
    return fig, caption


def _frequency_curve(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    rl = return_levels_of(payload)
    if not rl:
        return None
    import numpy as np

    t = _floats(rl["T"])
    variable = variable_of(payload, "discharge")
    u = unit_of(payload, unit or "m3/s")
    fig, ax = _figure()
    if rl["lower"] is not None and rl["upper"] is not None:
        lo, hi = _floats(rl["lower"]), _floats(rl["upper"])
        ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(t)
        if ok.any():
            ax.fill_between(t[ok], lo[ok], hi[ok], color=ACCENT, alpha=0.5, label=f"{rl['band']} band")
    curves = []
    if rl["gev"] is not None:
        curves.append(("GEV (L-moments)" if rl["distribution"] is None else rl["distribution"].upper(),
                       _floats(rl["gev"]), PRIMARY, "-"))
    if rl["lp3"] is not None:
        curves.append(("Log-Pearson III", _floats(rl["lp3"]), DARK, "--"))
    if rl["boot"] is not None:
        curves.append(("GEV (bootstrap)", _floats(rl["boot"]), SECONDARY, ":"))
    for label, q, colour, style in curves:
        ax.plot(t, q, style, color=colour, linewidth=1.8, marker="o", markersize=4, label=label)
    emp = rl["empirical"]
    if emp:
        et, ev = _floats(emp[0]), _floats(emp[1])
        ax.scatter(et, ev, marker="x", color=WARNING, s=28, zorder=4, label="observed (Weibull)")
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel(_ylabel(variable, u))
    ax.set_title(f"Flood frequency at {record_name(payload, site)}")
    ax.legend(loc="upper left", frameon=False)
    fits = " and ".join(c[0] for c in curves[:2]) if curves else "the fitted"
    caption = f"Return levels of annual maximum {variable} at {record_name(payload, site)}: {fits} fits"
    if rl["band"]:
        caption += f" with the {rl['band']} band"
    if emp:
        caption += ", and the observed annual maxima at their Weibull plotting positions"
    return fig, caption + "."


def _fdc(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    fdc = fdc_of(payload)
    if not fdc:
        return None
    import numpy as np

    variable = variable_of(payload, "discharge")
    u = unit_of(payload, unit or "m3/s")
    fig, ax = _figure()
    if fdc["q"]:
        ex, q = _floats(fdc["exceedance"]), _floats(fdc["q"])
        ax.plot(ex, q, color=PRIMARY, linewidth=1.8, label="flow-duration curve")
        ax.fill_between(ex, np.nanmin(q[q > 0]) if (q > 0).any() else 0, q, color=ACCENT, alpha=0.3)
        how = "the ranked daily flows"
    else:
        pct = fdc["percentiles"]
        ax.plot(list(pct), list(pct.values()), "o-", color=PRIMARY, linewidth=1.5, label="percentiles")
        how = f"the {len(pct)} percentiles the tool reported"
    for key, colour in ((95.0, DANGER), (50.0, NEUTRAL), (10.0, SECONDARY)):
        val = fdc["percentiles"].get(key)
        if val is not None:
            ax.axhline(val, color=colour, linestyle="--", linewidth=0.9, label=f"Q{int(key)} = {_fmt(val)} {u}")
    allq = [v for v in (fdc["q"] or list(fdc["percentiles"].values())) if v is not None and v > 0]
    if allq and min(allq) > 0:
        _plain_log_y(ax)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel(_ylabel(variable, u))
    ax.set_title(f"Flow duration at {record_name(payload, site)}")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    caption = (f"Flow-duration curve of {variable} at {record_name(payload, site)} from {how}, with Q95, Q50 and "
               f"Q10 marked (log scale).")
    return fig, caption


def _trend(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    from aquascope.trend_series import reported_trend

    tr = reported_trend(payload)
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
    ax.set_title(f"Mann-Kendall: {verdict}" + (f" (p = {p:.3f})" if p is not None else ""))
    ax.set_xlabel("Year")
    ax.set_ylabel(_ylabel(f"annual mean {variable}", u))
    ax.legend(loc="best", frameon=False)
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
    fig, ax = _figure()
    ax.plot(xs, ys, "o-", color=PRIMARY, linewidth=1, markersize=4, label=f"annual maximum {variable}")
    slope = num(tr.get("sens_slope_per_year"))
    if slope is not None:
        intercept = float(np.median(ys) - slope * np.median(xs))
        ax.plot(xs, intercept + slope * xs, "--", color=DANGER, linewidth=1.6,
                label=f"Sen slope {slope:+.3g} {u}/yr" if u else f"Sen slope {slope:+.3g} per yr")
    verdict = str(tr.get("trend") or "no trend").replace("_", " ")
    p = num(tr.get("p_value"))
    ax.set_title(f"Mann-Kendall on the annual maxima: {verdict}" + (f" (p = {p:.3f})" if p is not None else ""))
    ax.set_xlabel("Year")
    ax.set_ylabel(_ylabel(f"annual maximum {variable}", u))
    ax.legend(loc="best", frameon=False)
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
            ax.legend(loc="upper left", frameon=False, fontsize=7)
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
        axes[0].set_title(f"Standardised drought indices at {record_name(payload, site)}")
        what = "SPEI (bars) with SPI (grey line)" if any(p["spei"] for p in panels) else "SPI"
        caption = (f"{what} at {record_name(payload, site)} for the {', '.join(scales)} month accumulations, "
                   f"{period_of(payload, panels[0]['dates'])}: blue above zero is wetter than normal, red below is "
                   f"drier; the dashed lines mark the moderate (1), severe (1.5) and extreme (2) classes.")
    else:
        axes[0].set_title(f"Standardised Groundwater Index at {record_name(payload, site)}")
        caption = (f"Standardised Groundwater Index at {record_name(payload, site)}, "
                   f"{period_of(payload, panels[0]['dates'])}: blue above zero is above the monthly norm, red "
                   f"below; shaded spans are the droughts at or below " + (f"{threshold:g}." if threshold is not None
                                                                          else "the threshold."))
    fig.tight_layout()
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
    title = f"Drought propagation at {record_name(payload, site)}"
    if lag is not None:
        title += f": SPI-{scale} leads SGI by {lag} months" + (f" (r = {corr:.2f})" if corr is not None else "")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Date")
    ax.set_ylabel("Standardised index")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
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
        ax.scatter([p["lon"] for p in pts], [p["lat"] for p in pts], color=colour, s=30, zorder=3, label=what)
        for p in pts[:30]:
            ax.annotate(p["label"][:18], (p["lon"], p["lat"]), textcoords="offset points", xytext=(4, 3),
                        fontsize=7, color=NEUTRAL)
    if site:
        ax.scatter([site[1]], [site[0]], marker="*", color=DANGER, s=160, zorder=4, label="site")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect(1.0 / max(math.cos(math.radians(lat_ref)), 0.2))
    ax.margins(0.15)
    ax.legend(loc="best", frameon=False, fontsize=8)


def _donors_map(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    donors = stations_of(payload)
    if not donors:
        return None
    pt = site_point(payload, site)
    fig, ax = _figure(height=4.6)
    _points_map(ax, donors, pt, colour=PRIMARY, what="donor gauges")
    ax.set_title(f"Donor gauges for {record_name(payload, site)}")
    caption = (f"The site and the {len(donors)} donor gauges the similarity search selected, in longitude and "
               f"latitude (no basemap); labels are the station ids.")
    return fig, caption


def _site_map(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    pt = site_point(payload, site)
    stations = stations_of(payload, site)
    if pt is None and not stations:
        return None
    fig, ax = _figure(height=4.6)
    _points_map(ax, stations, pt, colour=PRIMARY, what="stations within reach")
    sb = payload.get("sub_basin") if isinstance(payload.get("sub_basin"), dict) else None
    title = f"The site at {record_name(payload, site).removeprefix('the site at ')}"
    if sb and sb.get("hybas_id"):
        title += f" (BasinATLAS sub-basin {sb['hybas_id']})"
    ax.set_title(title, fontsize=9)
    if stations:
        caption = (f"The site and the {len(stations)} catalogue stations within reach, in longitude and latitude "
                   f"(no basemap); labels are the station ids.")
    else:
        caption = "The site, in longitude and latitude (no basemap); no catalogue station was listed with it."
    return fig, caption


def _signatures_band(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    est = payload.get("estimates")
    if not isinstance(est, dict) or not est:
        sim = payload.get("similarity")
        est = sim.get("estimates") if isinstance(sim, dict) else None
    if not isinstance(est, dict) or not est:
        return None
    skill = ((payload.get("skill") or {}).get("by_signature") or {}) if isinstance(payload.get("skill"), dict) else {}
    groups: dict[str, list[tuple[str, float, float, float]]] = {}
    for name, e in est.items():
        if not isinstance(e, dict) or num(e.get("value")) is None:
            continue
        v = num(e["value"])
        lo = num(e.get("low"))
        hi = num(e.get("high"))
        label = str(e.get("label") or name)
        nse = num((skill.get(name) or {}).get("nse")) if isinstance(skill.get(name), dict) else None
        if nse is not None:
            label += f" (NSE {nse:.2f})"
        groups.setdefault(str(e.get("unit") or ""), []).append((label, v, lo if lo is not None else v,
                                                                hi if hi is not None else v))
    if not groups:
        return None
    units = list(groups)[:4]
    total = sum(len(groups[u]) for u in units)
    # One panel per unit, stacked, so every label has its own row and nothing overlaps.
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(len(units), 1, figsize=(7.0, max(3.2, 0.36 * total + 0.9 * len(units) + 0.8)),
                             gridspec_kw={"height_ratios": [len(groups[u]) + 0.6 for u in units]})
    axes = list(np.atleast_1d(axes))
    for ax, u in zip(axes, units):
        items = groups[u]
        y = np.arange(len(items))
        vals = np.array([i[1] for i in items])
        err = np.array([[max(i[1] - i[2], 0.0) for i in items], [max(i[3] - i[1], 0.0) for i in items]])
        ax.barh(y, vals, xerr=err, color=PRIMARY, alpha=0.85, height=0.6, error_kw={"ecolor": DARK, "capsize": 3})
        ax.set_yticks(y)
        ax.set_yticklabels([i[0] if len(i[0]) <= 58 else i[0][:55] + "..." for i in items], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(u or "value", fontsize=9)
        ax.grid(axis="y", visible=False)
    k = payload.get("k") or (payload.get("similarity") or {}).get("k")
    if not k:
        k = next((e.get("n_donors") for e in est.values() if isinstance(e, dict) and e.get("n_donors")), None)
    fig.suptitle(f"Transferred flow signatures at {record_name(payload, site)}", fontsize=11)
    fig.tight_layout()
    caption = (f"Flow signatures transferred to the site from {k or 'the'} donor catchments, with the "
               f"one-standard-deviation band across donors as error bars" +
               (" and the leave-one-out skill (NSE) where published" if skill else "") + ".")
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
    fig, ax = _figure()
    ax.bar(x, _floats(p), color=PRIMARY, label="precipitation")
    if len(et0) == 12:
        ax.plot(x, _floats(et0), "o-", color=WARNING, linewidth=1.6, markersize=4, label="reference ET0")
    ax.set_ylabel("mm per month")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    handles, labels = ax.get_legend_handles_labels()
    if len(temp) == 12:
        ax2 = ax.twinx()
        ax2.plot(x, _floats(temp), "s-", color=DANGER, linewidth=1.4, markersize=4, label="temperature")
        ax2.set_ylabel("deg C")
        ax2.grid(False)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    ax.set_title(f"Monthly climate at {record_name(payload, site)}")
    years = payload.get("years") or clim.get("years")
    caption = ("Mean monthly precipitation (bars)" + (" and FAO-56 reference evapotranspiration (line)"
                                                      if len(et0) == 12 else "") +
               (" with mean temperature" if len(temp) == 12 else "") +
               f" for the ERA5 cell at {record_name(payload, site)}" +
               (f", {years} years ending {payload.get('end')}" if years and payload.get("end") else "") + ".")
    return fig, caption


def _glofas_series(payload: dict[str, Any], unit: str | None, site: dict[str, Any] | None) -> Drawn | None:
    g = payload.get("glofas") if isinstance(payload.get("glofas"), dict) else None
    if g is None:
        return None
    u = unit_of(g, unit or "m3/s")
    got = series_of(g)
    fig, ax = _figure()
    if got:
        t, v = _dates(got[0]), _floats(got[1])
        ax.plot(t, v, color=PRIMARY, linewidth=0.8)
        ax.set_xlabel("Date")
        what = "Daily modelled discharge"
    else:
        am = annual_maxima_of(g)
        if not am:
            close(fig)
            return None
        from matplotlib.ticker import MaxNLocator

        ax.plot(am[0], am[1], "o-", color=PRIMARY, linewidth=1.2, markersize=4)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Year")
        what = "Annual maxima of the modelled discharge"
    ax.set_ylabel(_ylabel("discharge", u))
    ax.set_title(f"GloFAS modelled discharge at {record_name(payload, site)}")
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
    title = f"Supply reliability at {record_name(payload, site)}"
    if daily is not None:
        title += f": {daily:.0%} of days"
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
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
        what = "Monthly crop water demand"
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
        what = "Seasonal crop water demand"
        d = payload.get("demand") if isinstance(payload.get("demand"), dict) else {}
        caption = (f"Crop evapotranspiration, effective rain and net and gross irrigation per season for "
                   f"{str(payload.get('crop') or 'the crop').replace('_', ' ')} on {payload.get('area_ha') or '?'} ha "
                   f"planted on the first of month {payload.get('planting_month') or '?'}; the mean gross depth is "
                   f"{_fmt(num(d.get('gross_irrigation_mm')), 4)} mm over the season.")
    else:
        close(fig)
        return None
    ax.set_ylabel("mm")
    ax.set_title(f"{what} ({str(payload.get('crop') or '').replace('_', ' ')})".replace(" ()", ""))
    ax.legend(loc="upper right", frameon=False, fontsize=8)
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
    ax.set_title("Reference evapotranspiration by month")
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
        ax.set_title(name[:28], fontsize=9)
        ax.set_ylabel(units.get(name, ""), fontsize=8)
    for ax in flat[len(names):]:
        ax.set_visible(False)
    fig.suptitle(f"Sampled water quality at {record_name(payload, site)}", fontsize=11)
    fig.tight_layout()
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
    ax.set_title("WHO drinking-water screen")
    fig.tight_layout()
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
    ax.set_title(f"Water quality index ({payload.get('use') or 'drinking'})", fontsize=10)
    if factors:
        ax2 = axes[1]
        x2 = np.arange(len(factors))
        ax2.bar(x2, [f[1] for f in factors], color=WARNING, width=0.5)
        ax2.set_xticks(x2)
        ax2.set_xticklabels([f[0] for f in factors], fontsize=8)
        ax2.set_ylim(0, 100)
        ax2.set_title("CCME factors (higher is worse)", fontsize=10)
    fig.tight_layout()
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
    ax.set_title(f"Baseflow separation ({payload.get('method') or 'filter'})" +
                 (f", BFI = {bfi:.2f}" if bfi is not None else ""))
    ax.set_xlabel("Date")
    ax.set_ylabel(_ylabel("discharge", u))
    ax.legend(loc="upper right", frameon=False, fontsize=8)
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
        ax.set_title("Water-table fluctuation" + (f": recharge {value:.0f} mm/yr" if value is not None else ""))
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
    ax.set_title("Recharge estimate")
    caption = (f"Recharge of {value:.0f} mm/yr" + (f" (uncertainty {unc:.0f} mm/yr)" if unc is not None else "") +
               " by the water-table fluctuation method" + (f" at a specific yield of {sy:g}" if sy is not None
                                                            else "") + "; the level record itself was not in the "
               "payload, so only the estimate is drawn.")
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
