"""Regression tests against CAMELS benchmark catchment statistics.

Validates that AquaScope's hydrological computations produce results
consistent with published CAMELS attributes for 10 diverse US catchments.
Synthetic daily discharge (generated with a fixed seed) is used as input,
so tolerances are intentionally wide to catch gross errors rather than
demand exact matches.

References
----------
- Addor et al. (2017), doi:10.5194/hess-21-5293-2017
- Newman et al. (2015), doi:10.5194/hess-19-209-2015
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from aquascope.hydrology.baseflow import lyne_hollick
from aquascope.hydrology.flow_duration import flow_duration_curve
from aquascope.hydrology.signatures import compute_signatures

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "data" / "camels_benchmark"
CATCHMENTS_FILE = BENCHMARK_DIR / "catchments.json"

# ── tolerances ────────────────────────────────────────────────────────
REL_TOL = 0.25  # ±25 % relative
BFI_ABS_TOL = 0.15  # absolute for baseflow index
PEAK_MONTH_TOL = 2  # ±2 calendar months (circular)


# ── helpers ───────────────────────────────────────────────────────────

def _load_catchments() -> list[dict]:
    with open(CATCHMENTS_FILE) as f:
        return json.load(f)


def _load_series(gauge_id: str) -> tuple[pd.Series, pd.Series]:
    """Return (discharge, precipitation) Series with DatetimeIndex."""
    df = pd.read_csv(BENCHMARK_DIR / f"{gauge_id}.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df["discharge_cms"], df["precipitation_mm"]


def _compute_fdc_slope(discharge: pd.Series) -> float:
    """FDC slope between 33rd and 66th exceedance percentiles (log10)."""
    fdc = flow_duration_curve(discharge, percentiles=[33, 66])
    q33 = fdc.percentiles[33]
    q66 = fdc.percentiles[66]
    if q33 > 0 and q66 > 0:
        return (np.log(q66) - np.log(q33)) / (0.66 - 0.33)
    return float("nan")


# ── precomputed results (module-level for performance) ────────────────

_CATCHMENTS: list[dict] = []
_SIGNATURES: dict[str, object] = {}
_BF_RESULTS: dict[str, object] = {}
_FDC_SLOPES: dict[str, float] = {}
_DISCHARGES: dict[str, pd.Series] = {}


def _ensure_computed() -> None:
    """Lazily compute signatures for all catchments once."""
    if _CATCHMENTS:
        return
    _CATCHMENTS.extend(_load_catchments())
    for c in _CATCHMENTS:
        gid = c["gauge_id"]
        discharge, precipitation = _load_series(gid)
        _DISCHARGES[gid] = discharge
        _SIGNATURES[gid] = compute_signatures(
            discharge, precipitation=precipitation, area_km2=c["area_km2"],
        )
        _BF_RESULTS[gid] = lyne_hollick(discharge)
        _FDC_SLOPES[gid] = _compute_fdc_slope(discharge)


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _setup():
    """Compute all signatures before tests run."""
    _ensure_computed()


# ── tests ─────────────────────────────────────────────────────────────


class TestMeanDischarge:
    """q_mean within ±25 % for all 10 catchments."""

    def test_mean_discharge_within_tolerance(self) -> None:
        for c in _CATCHMENTS:
            gid = c["gauge_id"]
            computed = _SIGNATURES[gid].mean_flow
            published = c["published_q_mean"]
            err = abs(computed - published) / abs(published) if published else 0.0
            assert err <= REL_TOL, (
                f"{gid}: mean_flow {computed:.2f} vs published {published:.2f} "
                f"(err={err:.1%}, tol={REL_TOL:.0%})"
            )


class TestBaseflowIndex:
    """BFI within ±0.15 for all 10 catchments (using Lyne–Hollick)."""

    def test_baseflow_index_within_tolerance(self) -> None:
        for c in _CATCHMENTS:
            gid = c["gauge_id"]
            computed = _BF_RESULTS[gid].bfi
            published = c["published_baseflow_index"]
            diff = abs(computed - published)
            assert diff <= BFI_ABS_TOL, (
                f"{gid}: BFI {computed:.3f} vs published {published:.3f} "
                f"(diff={diff:.3f}, tol={BFI_ABS_TOL})"
            )


class TestFDCSlope:
    """FDC slope is negative for all catchments."""

    def test_fdc_slope_sign_correct(self) -> None:
        for c in _CATCHMENTS:
            gid = c["gauge_id"]
            slope = _FDC_SLOPES[gid]
            assert not np.isnan(slope), f"{gid}: FDC slope is NaN"
            assert slope < 0, f"{gid}: FDC slope {slope:.4f} is not negative"


class TestFlowPercentiles:
    """Q5 < Q50 < Q95 for all catchments (statistical percentile convention)."""

    def test_flow_percentiles_ordered(self) -> None:
        for c in _CATCHMENTS:
            gid = c["gauge_id"]
            sig = _SIGNATURES[gid]
            assert sig.q5 < sig.median_flow < sig.q95, (
                f"{gid}: percentiles not ordered: "
                f"q5={sig.q5:.3f}, median={sig.median_flow:.3f}, q95={sig.q95:.3f}"
            )


class TestRunoffRatio:
    """Runoff ratio between 0 and 1 for all catchments."""

    def test_runoff_ratio_reasonable(self) -> None:
        for c in _CATCHMENTS:
            gid = c["gauge_id"]
            rr = _SIGNATURES[gid].runoff_ratio
            assert rr is not None, f"{gid}: runoff_ratio is None"
            assert 0 < rr < 1, f"{gid}: runoff_ratio {rr:.3f} out of (0, 1)"


class TestSignaturesComplete:
    """All 21 signature fields are populated (non-None)."""

    def test_signatures_complete(self) -> None:
        expected_fields = [
            "mean_flow", "median_flow", "q5", "q95", "q5_q95_ratio",
            "cv", "iqr",
            "high_flow_frequency", "high_flow_duration", "q_peak_mean",
            "low_flow_frequency", "low_flow_duration", "baseflow_index",
            "zero_flow_fraction",
            "peak_month", "seasonality_index",
            "rising_limb_density", "flashiness_index",
            "mean_recession_constant",
            "runoff_ratio", "elasticity",
        ]
        for c in _CATCHMENTS:
            gid = c["gauge_id"]
            sig = _SIGNATURES[gid]
            for field in expected_fields:
                val = getattr(sig, field)
                assert val is not None, (
                    f"{gid}: signature field '{field}' is None"
                )


class TestSeasonalPattern:
    """Peak month within ±2 months of expected (circular)."""

    def test_seasonal_pattern_detected(self) -> None:
        for c in _CATCHMENTS:
            gid = c["gauge_id"]
            computed = _SIGNATURES[gid].peak_month
            published = c["published_peak_month"]
            diff = abs(computed - published)
            diff = min(diff, 12 - diff)  # circular distance
            assert diff <= PEAK_MONTH_TOL, (
                f"{gid}: peak_month {computed} vs published {published} "
                f"(diff={diff}, tol=±{PEAK_MONTH_TOL})"
            )


class TestRecessionConstant:
    """Mean recession constant is positive for all catchments."""

    def test_recession_constant_positive(self) -> None:
        for c in _CATCHMENTS:
            gid = c["gauge_id"]
            k = _SIGNATURES[gid].mean_recession_constant
            assert k > 0, (
                f"{gid}: mean_recession_constant {k:.4f} is not positive"
            )
