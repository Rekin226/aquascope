"""Smart insights — the dashboard's automatic intelligence layer.

Whenever a dataset lands in the workspace, this module:

1. profiles it (records, stations, time span, completeness),
2. scores its analysis-readiness,
3. quietly checks WHO guideline compliance,
4. recommends the next analytical steps — as one-click navigation chips.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from aquascope.dashboard import _state

WHO_GUIDELINES: dict[str, tuple[float, float, str]] = {
    "ph": (6.5, 8.5, "pH units"),
    "dissolved_oxygen": (5.0, float("inf"), "mg/L"),
    "turbidity": (0, 5.0, "NTU"),
    "nitrate": (0, 50.0, "mg/L"),
    "e_coli": (0, 0, "CFU/100mL"),
    "arsenic": (0, 0.01, "mg/L"),
    "lead": (0, 0.01, "mg/L"),
    "mercury": (0, 0.001, "mg/L"),
}


@dataclass
class Suggestion:
    label: str
    page_key: str
    reason: str


@dataclass
class Insights:
    quality_score: int = 100
    quality_notes: list[str] = field(default_factory=list)
    who_alerts: int = 0
    who_checked: int = 0
    n_duplicates: int = 0
    suggestions: list[Suggestion] = field(default_factory=list)


def who_exceedances(df: pd.DataFrame, prof: _state.DataProfile) -> list[dict]:
    """Per-parameter WHO exceedance stats for long-format quality data."""
    if not prof.has_params:
        return []
    results = []
    params = df[prof.param_col].astype(str).str.lower().unique()
    for param in params:
        if param not in WHO_GUIDELINES:
            continue
        lo, hi, unit = WHO_GUIDELINES[param]
        subset = df[df[prof.param_col].astype(str).str.lower() == param][prof.value_col].dropna()
        if subset.empty:
            continue
        n = len(subset)
        if hi == float("inf"):
            n_exceed = int((subset < lo).sum())
            rule = f"≥ {lo} {unit}"
        elif lo == 0:
            n_exceed = int((subset > hi).sum())
            rule = f"≤ {hi} {unit}"
        else:
            n_exceed = int(((subset < lo) | (subset > hi)).sum())
            rule = f"{lo}–{hi} {unit}"
        pct = n_exceed / n * 100 if n else 0.0
        plain = "Alert" if pct > 10 else "Warning" if pct > 0 else "OK"
        results.append(
            {
                "parameter": param,
                "rule": rule,
                "n": n,
                "n_exceed": n_exceed,
                "pct": round(pct, 1),
                "status_plain": plain,
            }
        )
    return results


def build(df: pd.DataFrame, prof: _state.DataProfile) -> Insights:
    """Compute the smart-insight bundle for the active dataset."""
    ins = Insights()

    # --- quality score -----------------------------------------------------
    score = 100.0
    missing = 100.0 - prof.completeness_pct
    if missing > 0:
        score -= min(40.0, missing)
        ins.quality_notes.append(f"{missing:.1f}% missing values")
    try:
        ins.n_duplicates = int(df.duplicated().sum())
    except Exception:  # noqa: BLE001 — unhashable cells
        ins.n_duplicates = 0
    if ins.n_duplicates:
        dup_pct = 100.0 * ins.n_duplicates / max(len(df), 1)
        score -= min(20.0, dup_pct)
        ins.quality_notes.append(f"{ins.n_duplicates} duplicate rows")
    if prof.has_time and prof.span_years < 1 / 12:
        score -= 10
        ins.quality_notes.append("very short time span")
    if not prof.has_time:
        score -= 10
        ins.quality_notes.append("no datetime column detected")
    ins.quality_score = int(max(0, round(score)))

    # --- WHO quick check ---------------------------------------------------
    rows = who_exceedances(df, prof)
    ins.who_checked = len(rows)
    ins.who_alerts = sum(1 for r in rows if r["status_plain"] != "OK")

    # --- next-step suggestions ----------------------------------------------
    sugg: list[Suggestion] = []
    if ins.who_alerts:
        sugg.append(
            Suggestion("⚠️ Review quality alerts", "alerts",
                       f"{ins.who_alerts} parameter(s) exceed WHO guidelines")
        )
    if prof.discharge_col and prof.has_time:
        if prof.span_years >= 3:
            sugg.append(
                Suggestion("🌀 Flood frequency analysis", "extremes",
                           f"{prof.span_years:.0f} years of discharge — enough for GEV/LP3 return levels")
            )
        sugg.append(
            Suggestion("🌊 Baseflow & flow signatures", "hydrology",
                       f"daily `{prof.discharge_col}` series detected")
        )
    if ins.quality_score < 85:
        sugg.append(
            Suggestion("🔬 Clean & preprocess", "analysis",
                       "; ".join(ins.quality_notes) or "quality score below 85")
        )
    if prof.has_time and prof.value_col:
        sugg.append(Suggestion("📈 Plot time series", "visualize", "datetime + numeric columns detected"))
    if prof.has_geo:
        sugg.append(Suggestion("🗺️ Map the stations", "visualize", "coordinates detected"))
    sugg.append(Suggestion("🤖 AI methodology advice", "ai", "profile-based research methodology matching"))
    ins.suggestions = sugg[:4]
    return ins


def render_panel(df: pd.DataFrame, key_prefix: str = "ins") -> None:
    """Render the Smart Insights panel: metrics, quality score, next steps."""
    prof = _state.profile(df)
    ins = build(df, prof)

    with st.container(border=True):
        st.markdown("##### 💡 Smart insights")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Records", f"{prof.n_records:,}")
        m2.metric("Stations", f"{prof.n_stations:,}" if prof.n_stations else "—")
        if prof.has_time and prof.date_min is not None:
            span = f"{prof.span_years:.1f} yr" if prof.span_years >= 1 else f"{(prof.span_years * 365.25):.0f} d"
            m3.metric("Time span", span)
        else:
            m3.metric("Time span", "—")
        m4.metric("Completeness", f"{prof.completeness_pct:.1f}%")
        m5.metric("Quality score", f"{ins.quality_score}/100")

        detected = []
        if prof.datetime_col:
            detected.append(f"time: `{prof.datetime_col}`")
        if prof.discharge_col:
            detected.append(f"discharge: `{prof.discharge_col}`")
        if prof.has_params:
            detected.append(f"{len(prof.parameters)} parameter(s)")
        if prof.has_geo:
            detected.append("coordinates ✓")
        if detected:
            st.caption("Auto-detected — " + " · ".join(detected))
        if ins.quality_notes:
            st.caption("Quality notes — " + "; ".join(ins.quality_notes))
        if ins.who_checked:
            if ins.who_alerts:
                st.caption(
                    f"⚠️ WHO quick check — **{ins.who_alerts} of {ins.who_checked}** monitored "
                    "parameter(s) show exceedances."
                )
            else:
                st.caption(f"✅ WHO quick check — all {ins.who_checked} monitored parameter(s) within guidelines.")

        if ins.suggestions:
            st.markdown("**Suggested next steps**")
            cols = st.columns(len(ins.suggestions))
            for i, (col, s) in enumerate(zip(cols, ins.suggestions)):
                with col:
                    if st.button(s.label, key=f"{key_prefix}_sugg_{i}", width="stretch", help=s.reason):
                        _state.goto(s.page_key)
