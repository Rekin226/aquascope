"""Plotly chart builders for the AquaScope dashboard.

All figures share one visual system:

* categorical series colors come from a fixed, colorblind-validated order
  (never cycled or re-assigned when series counts change),
* magnitude uses a single-hue blue ramp,
* polarity (SPI, correlation) uses a blue↔red diverging scale with a neutral
  gray midpoint,
* status colors (good/warning/serious/critical) are reserved for alerts and
  always shipped with an icon + label, never color alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Fixed categorical order — validated for adjacent-pair CVD separation.
CAT = [
    "#2a78d6",  # blue
    "#008300",  # green
    "#e87ba4",  # magenta
    "#eda100",  # yellow
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# Single-hue sequential ramp (magnitude).
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]

# Diverging: blue ↔ red with neutral gray midpoint (polarity).
DIVERGING = [[0.0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#e88a89"], [1.0, "#b53230"]]

# Reserved status palette — never used for data series.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

_GRID = "rgba(49, 51, 63, 0.12)"
_TEXT = "#31333F"
_MUTED = "#6f7480"


def register_template() -> None:
    """Register and activate the shared 'aquascope' plotly template."""
    if "aquascope" in pio.templates:
        pio.templates.default = "aquascope"
        return

    template = go.layout.Template()
    template.layout = go.Layout(
        colorway=CAT,
        font=dict(family='"Source Sans Pro", "Source Sans 3", sans-serif', size=13, color=_TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=76, b=8),
        hoverlabel=dict(font=dict(size=12), namelength=-1),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None),
        title=dict(font=dict(size=16, color=_TEXT), x=0, xanchor="left", y=0.97, yanchor="top"),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID, ticks="outside", tickcolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID, ticks="outside", tickcolor=_GRID),
    )
    pio.templates["aquascope"] = template
    pio.templates.default = "aquascope"


def _finish(fig: go.Figure, title: str | None = None, ylab: str | None = None, xlab: str | None = None) -> go.Figure:
    if title:
        fig.update_layout(title=title)
    if ylab:
        fig.update_yaxes(title_text=ylab)
    if xlab:
        fig.update_xaxes(title_text=xlab)
    return fig


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


def timeseries(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str | None = None,
    ylab: str | None = None,
) -> go.Figure:
    """Interactive line chart; one color per entity in fixed palette order."""
    fig = px.line(df, x=x, y=y, color=color)
    fig.update_traces(line=dict(width=2))
    if color is None:
        fig.update_layout(showlegend=False)
    else:
        fig.update_layout(legend_title_text="")
    return _finish(fig, title, ylab or y, None)


def box(df: pd.DataFrame, value: str, group: str, title: str | None = None) -> go.Figure:
    fig = px.box(df, x=group, y=value, color=group, points="outliers")
    fig.update_layout(showlegend=False, hovermode="closest")
    return _finish(fig, title, value, None)


def corr_heatmap(df: pd.DataFrame, title: str | None = None) -> go.Figure:
    num = df.select_dtypes(include="number")
    corr = num.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        color_continuous_scale=DIVERGING,
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(hovermode="closest", coloraxis_colorbar=dict(title="r"))
    return _finish(fig, title)


def histogram(series: pd.Series, title: str | None = None, xlab: str | None = None) -> go.Figure:
    fig = px.histogram(x=series, nbins=40, color_discrete_sequence=[CAT[0]])
    fig.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,1)")
    fig.update_layout(showlegend=False, bargap=0.02, hovermode="closest")
    return _finish(fig, title, "count", xlab)


# ---------------------------------------------------------------------------
# Hydrology
# ---------------------------------------------------------------------------


def fdc(series: pd.Series, title: str | None = None) -> go.Figure:
    """Flow-duration curve: exceedance probability vs discharge (log y)."""
    q = np.sort(np.asarray(series.dropna(), dtype=float))[::-1]
    n = q.size
    prob = 100.0 * np.arange(1, n + 1) / (n + 1)
    fig = go.Figure(
        go.Scatter(x=prob, y=q, mode="lines", line=dict(color=CAT[0], width=2), name="Discharge")
    )
    for p, label in ((50, "Q50"), (95, "Q95")):
        val = float(np.interp(p, prob, q))
        fig.add_trace(
            go.Scatter(
                x=[p], y=[val], mode="markers+text", text=[f"{label} = {val:.2f}"],
                textposition="top right", textfont=dict(size=12, color=_MUTED),
                marker=dict(size=9, color=CAT[0], line=dict(width=2, color="#ffffff")),
                showlegend=False, hoverinfo="skip",
            )
        )
    fig.update_yaxes(type="log")
    fig.update_layout(showlegend=False, hovermode="closest")
    return _finish(fig, title or "Flow Duration Curve", "Discharge", "Exceedance probability (%)")


def hydrograph(
    dates,
    total: pd.Series,
    baseflow: pd.Series | None = None,
    title: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=total, name="Total flow", mode="lines", line=dict(color=CAT[0], width=2))
    )
    if baseflow is not None:
        fig.add_trace(
            go.Scatter(
                x=dates, y=baseflow, name="Baseflow", mode="lines",
                line=dict(color=CAT[4], width=2), fill="tozeroy",
                fillcolor="rgba(27, 175, 122, 0.18)",
            )
        )
    return _finish(fig, title or "Hydrograph", "Discharge")


def spi_bars(dates, values: pd.Series, title: str | None = None) -> go.Figure:
    """SPI timeline — diverging polarity: blue = wet, red = dry."""
    vals = np.asarray(values, dtype=float)
    colors = np.where(vals >= 0, "#2a78d6", "#b53230")
    fig = go.Figure(go.Bar(x=dates, y=vals, marker_color=colors, name="SPI"))
    fig.add_hline(y=0, line_color=_GRID)
    for level, label in ((-1.0, "moderate"), (-1.5, "severe"), (-2.0, "extreme")):
        fig.add_hline(y=level, line_dash="dot", line_color="rgba(176,50,48,0.45)",
                      annotation_text=label, annotation_font_size=11)
    fig.update_layout(showlegend=False, bargap=0.0)
    return _finish(fig, title or "SPI Timeline", "SPI")


def return_curve(
    periods,
    levels,
    lower=None,
    upper=None,
    emp_periods=None,
    emp_values=None,
    conf: float = 0.95,
    title: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=list(periods) + list(periods)[::-1],
                y=list(upper) + list(lower)[::-1],
                fill="toself", fillcolor="rgba(42, 120, 214, 0.15)",
                line=dict(width=0), name=f"{int(conf * 100)}% CI", hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=list(periods), y=list(levels), mode="lines+markers",
            line=dict(color=CAT[0], width=2), marker=dict(size=8), name="Return level",
        )
    )
    if emp_periods is not None and emp_values is not None:
        fig.add_trace(
            go.Scatter(
                x=list(emp_periods), y=list(emp_values), mode="markers",
                marker=dict(size=8, color=CAT[5], symbol="diamond",
                            line=dict(width=1, color="#ffffff")),
                name="Observed (Weibull)",
            )
        )
    fig.update_xaxes(type="log")
    fig.update_layout(hovermode="closest")
    return _finish(fig, title or "Return-level curve", "Magnitude", "Return period (years)")


# ---------------------------------------------------------------------------
# Water quality
# ---------------------------------------------------------------------------


def exceedance_bar(rows: list[dict], title: str | None = None) -> go.Figure:
    """Horizontal % exceedance bars, colored by reserved status palette."""
    df = pd.DataFrame(rows)
    status_color = {
        "OK": STATUS["good"],
        "Warning": STATUS["warning"],
        "Alert": STATUS["critical"],
    }
    colors = [status_color.get(r, STATUS["warning"]) for r in df["status_plain"]]
    fig = go.Figure(
        go.Bar(
            x=df["pct"], y=df["parameter"], orientation="h",
            marker_color=colors, text=[f"{p:.1f}%" for p in df["pct"]],
            textposition="outside", cliponaxis=False,
        )
    )
    fig.add_vline(x=10, line_dash="dot", line_color="rgba(208,59,59,0.5)",
                  annotation_text="10% alert threshold", annotation_font_size=11)
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_xaxes(range=[0, max(100.0, float(df["pct"].max()) * 1.15)])
    return _finish(fig, title or "WHO guideline exceedances", None, "Samples exceeding guideline (%)")


def station_map(df: pd.DataFrame, lat: str, lon: str, hover: str | None = None, color: str | None = None):
    """Interactive station map (MapLibre, no API token needed)."""
    plot_df = df.dropna(subset=[lat, lon])
    # One point per station keeps the map light even for big datasets.
    subset_cols = [c for c in (lat, lon, hover) if c]
    plot_df = plot_df.drop_duplicates(subset=subset_cols)
    fig = px.scatter_map(
        plot_df,
        lat=lat,
        lon=lon,
        hover_name=hover,
        color=color,
        zoom=5,
        map_style="carto-positron",
        color_discrete_sequence=CAT,
    )
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=520, hovermode="closest")
    return fig


# ---------------------------------------------------------------------------
# Agricultural water
# ---------------------------------------------------------------------------


def water_demand(dates, sched: pd.DataFrame, crop: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=dates, y=sched["effective_rain"], name="Effective rain",
               marker_color="rgba(0, 131, 0, 0.45)")
    )
    fig.add_trace(
        go.Scatter(x=dates, y=sched["etc"], name="Crop ET (ETc)", mode="lines",
                   line=dict(color=CAT[5], width=2))
    )
    fig.add_trace(
        go.Scatter(x=dates, y=sched["eto"], name="Reference ET₀", mode="lines",
                   line=dict(color=CAT[0], width=2, dash="dash"))
    )
    return _finish(fig, f"{crop} — daily water demand vs effective rainfall", "mm/day")


def kc_curves(dates, sched: pd.DataFrame, method: str, crop: str) -> go.Figure:
    fig = go.Figure()
    if method == "dual":
        fig.add_trace(go.Scatter(x=dates, y=sched["kcb"], name="Kcb (transpiration)",
                                 line=dict(color=CAT[1], width=2)))
        fig.add_trace(go.Scatter(x=dates, y=sched["ke"], name="Ke (soil evaporation)",
                                 line=dict(color=CAT[5], width=2)))
        fig.add_trace(go.Scatter(x=dates, y=sched["kc_dual"], name="Kc = Kcb + Ke",
                                 line=dict(color=CAT[0], width=2.5)))
        title = f"{crop} — dual crop coefficients"
    else:
        fig.add_trace(go.Scatter(x=dates, y=sched["kc"], name="Kc (single)",
                                 line=dict(color=CAT[0], width=2.5)))
        title = f"{crop} — single crop coefficient"
    return _finish(fig, title, "Coefficient")


def cumulative_irrigation(dates, sched: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=sched["gross_irrigation"].cumsum(), name="Cumulative gross",
                   fill="tozeroy", fillcolor="rgba(74, 58, 167, 0.15)",
                   line=dict(color=CAT[6], width=2))
    )
    fig.add_trace(
        go.Scatter(x=dates, y=sched["net_irrigation"].cumsum(), name="Cumulative net",
                   line=dict(color=CAT[0], width=2))
    )
    return _finish(fig, "Cumulative irrigation requirement", "mm")
