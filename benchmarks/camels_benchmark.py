"""CAMELS benchmark harness: run AquaScope's own methods against the benchmark data.

For each of the 10 CAMELS catchments in ``data/camels_benchmark``, this harness
runs AquaScope's flood-frequency, baseflow and signature implementations and
compares the numbers to:

- the published CAMELS attributes (``daily_catchments.json``), and
- the flood frequency reference quantiles (``ffa_reference.json``).

It emits ``results.json`` as the single source of truth; ``results.md`` and
``results.html`` are renderings of that JSON (they read the results, they
never recompute). Per-stage execution time is recorded.

Usage::

    python -m benchmarks.camels_benchmark [--output-dir benchmark-output]

Notes
-----
- No network access: everything reads the local files under
  ``data/camels_benchmark/``.
- The cached peak series is one value per **USGS water year** (Oct 1 - Sep 30).
  AquaScope's ``flood_analysis`` resamples a dated input to calendar-year
  annual maxima, which would silently collapse the ~20 % of peaks that share a
  calendar year across two water years (e.g. a Jan peak in WY2000 and a Dec
  peak in WY2001). The peaks are therefore passed on a synthetic unique-year
  index so the resampler is a no-op and no peak is dropped -- the fits run on
  exactly the same annual-maxima set as the reference.
- GEV is compared estimator-to-estimator against the reference MLE even when
  the reference fit is unstable (``mle_stable: false``). Those mismatches are
  a genuine cross-implementation finding (the data is insufficient for a
  stable GEV-MLE, not a defect in either implementation) and are classified as
  ``"data_limitation"`` in the results rather than hidden.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import ValidationError

from aquascope.analysis.metrics import pbias, rmse
from aquascope.analysis.metrics import r2 as r2_score
from aquascope.api import baseflow_analysis, compute_all_signatures, flood_analysis
from aquascope.hydrology.flow_duration import flow_duration_curve
from aquascope.reporting.builder import ReportBuilder
from benchmarks import _paths
from benchmarks.results_models import RETURN_PERIODS, SCHEMA_VERSION
from benchmarks.results_models import Results as ResultsModel

BASE = pathlib.Path(__file__).resolve().parent
BENCHMARK_DIR = _paths.BENCHMARK_DIR
DAILY_CATCHMENTS_FILE = _paths.DAILY_CATCHMENTS_FILE
DAILY_DIR = _paths.DAILY_DIR
PEAKS_DIR = _paths.PEAKS_DIR
FFA_REFERENCE_FILE = _paths.FFA_REFERENCE_FILE
SCHEMA_FILE = BASE / "results.schema.json"

DEFAULT_OUTPUT_DIR = "benchmark-output"

# ---------------------------------------------------------------------------
# Tolerances (decided up front, not loosened when a check fails)
# ---------------------------------------------------------------------------

#: Relative error (as a fraction) allowed for signatures computed on synthetic
#: daily series that are calibrated *to*, not measured from, CAMELS attributes.
SIGNATURE_REL_TOL = 0.25

#: Absolute difference allowed for a baseflow index against the published
#: CAMELS BFI, which was computed with a different separation algorithm than
#: AquaScope's digital filters.
BFI_ABS_TOL = 0.15

#: Circular difference allowed for the (month-of-year) dominant peak month.
PEAK_MONTH_TOL = 2

#: Relative error allowed between an AquaScope flood-frequency quantile and the
#: flood-frequency reference quantile for the same distribution.
FFA_REL_TOL = 0.20

#: Headline aggregate gates, expressed as percentages.
Q_MEAN_NRMSE_TOL = 25.0  # RMSE / mean(published q_mean) across gauges
BFI_PBIAS_TOL = 25.0  # PBIAS(%) across gauges
FFA_MEAN_REL_TOL = 20.0  # mean relative error (%) of dependable cross-checks

#: Reference JSON keys per AquaScope method name.
_FFA_REFERENCE_KEYS = {
    "gev": "scipy_gev",
    "gev_lmoments": "scipy_gev_lmoments",
    "lp3": "scipy_lp3",
}

TOLERANCES: dict[str, dict] = {
    "signature_relative": {
        "value": SIGNATURE_REL_TOL,
        "unit": "fraction",
        "rationale": (
            "Synthetic daily series are calibrated to approximate the published "
            "CAMELS attributes; they are not measurements of them."
        ),
    },
    "baseflow_absolute": {
        "value": BFI_ABS_TOL,
        "unit": "fraction",
        "rationale": (
            "Published CAMELS baseflow index comes from a different separation "
            "algorithm than AquaScope's Lyne-Hollick/Eckhardt digital filters."
        ),
    },
    "peak_month_circular": {
        "value": PEAK_MONTH_TOL,
        "unit": "months",
        "rationale": "Peak month is month-of-year and circular (Jan == Dec + 1 month).",
    },
    "ffa_relative": {
        "value": FFA_REL_TOL,
        "unit": "fraction",
        "rationale": (
            "Wider than the +/-10% used in the Potomac federal-standard "
            "validation because several benchmark gauges are semi-arid or "
            "heavy-tailed, where MLE vs L-moments spread is larger."
        ),
    },
    "q_mean_nrmse_gate": {
        "value": Q_MEAN_NRMSE_TOL,
        "unit": "%",
        "rationale": "Aggregate q_mean gate: normalized RMSE (%) across the 10 gauges.",
    },
    "bfi_pbias_gate": {
        "value": BFI_PBIAS_TOL,
        "unit": "%",
        "rationale": "Aggregate baseflow gate: PBIAS (%) across the 10 gauges.",
    },
    "ffa_gate": {
        "value": FFA_MEAN_REL_TOL,
        "unit": "%",
        "rationale": (
            "Aggregate flood-frequency gate over the dependable reference cross-checks "
            "(GEV-L-moments and LP3); GEV-MLE mismatches are reported separately as "
            "data-limitation findings, not folded into the implementation gate."
        ),
    },
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_catchments() -> list[dict]:
    """Load the benchmark daily catchment metadata."""
    with open(DAILY_CATCHMENTS_FILE) as f:
        return json.load(f)


def load_reference() -> dict:
    """Load the flood frequency reference quantiles."""
    with open(FFA_REFERENCE_FILE) as f:
        return json.load(f)


def _require_data_files(gauge_id: str) -> None:
    """Raise an error when committed benchmark data is missing."""
    expected = [DAILY_DIR / f"{gauge_id}_daily.csv", PEAKS_DIR / f"{gauge_id}_peaks.csv"]
    missing = [p.name for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Missing benchmark data for {gauge_id}: {missing} under {BENCHMARK_DIR}. "
            "The synthetic series and USGS peaks are committed in the AquaScope repository; "
            "restore them (`git checkout -- data/camels_benchmark`) or regenerate with "
            "`python data/camels_benchmark/generate_synthetic.py` and "
            "`python benchmarks/fetch_peak_flows.py`."
        )


def load_synthetic_series(gauge_id: str) -> tuple[pd.Series, pd.Series]:
    """Load the synthetic daily discharge (m^3/s) and precipitation (mm)."""
    df = pd.read_csv(DAILY_DIR / f"{gauge_id}_daily.csv", parse_dates=["date"]).set_index("date")
    return df["discharge_cms"], df["precipitation_mm"]


def load_peaks_series(gauge_id: str) -> pd.Series:
    """Load the cached USGS annual peak series in m^3/s (one value per water year)."""
    df = pd.read_csv(PEAKS_DIR / f"{gauge_id}_peaks.csv", parse_dates=["date"])
    index = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return pd.Series(df["peak_va"].astype(float).values, index=index).sort_index()


def _annual_maxima_series(values: np.ndarray) -> pd.Series:
    """Present an already-annual series as dates with unique years.

    ``flood_analysis`` resamples a dated input to calendar-year annual maxima;
    this is a no-op when every value sits in its own year, so the whole
    water-year series is preserved and matches the reference annual-max set.
    The datetimes are synthetic and carry no meaning beyond "unique year".
    """
    index = pd.DatetimeIndex([datetime(2000 + i, 1, 1) for i in range(len(values))])
    return pd.Series(values, index=index)


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _relative_error(computed: float, reference: float) -> float:
    """Absolute relative error as a fraction (0.25 === 25 %)."""
    if reference == 0:
        return float("inf") if computed != 0 else 0.0
    return abs(computed - reference) / abs(reference)


def _circular_month_diff(computed: float, reference: float) -> float:
    """Signed-free circular difference between month-of-year values."""
    diff = abs(computed - reference) % 12
    return float(min(diff, 12 - diff))


def _check(metric: str, computed: float, published: float, error_type: str, tolerance: float) -> dict:
    """Build one recorded check with its outcome."""
    if error_type == "relative":
        error = _relative_error(computed, published)
    elif error_type == "absolute":
        error = abs(computed - published)
    elif error_type == "circular_months":
        error = _circular_month_diff(computed, published)
    else:
        raise ValueError(f"Unknown error_type {error_type!r}")
    error = round(float(error), 6)
    computed = round(float(computed), 6)
    published = round(float(published), 6)
    return {
        "metric": metric,
        "computed": computed,
        "published": published,
        "error_type": error_type,
        "error": error,
        "tolerance": tolerance,
        "check_passes": bool(np.isfinite(error) and error <= tolerance),
    }


# ---------------------------------------------------------------------------
# Analysis runners
# ---------------------------------------------------------------------------


def run_signatures(discharge: pd.Series, precipitation: pd.Series, catchment: dict) -> dict:
    """Compute AquaScope signatures and compare them to the published values."""
    start = time.perf_counter()
    sig = compute_all_signatures(discharge, precipitation=precipitation, area_km2=catchment["area_km2"])
    seconds = time.perf_counter() - start

    fdc = _camels_fdc_slope(discharge)
    published = catchment

    checks = [
        _check("q_mean", sig.mean_flow, published["published_q_mean"], "relative", SIGNATURE_REL_TOL),
        _check("q5", sig.q5, published["published_q5"], "relative", SIGNATURE_REL_TOL),
        _check("q95", sig.q95, published["published_q95"], "relative", SIGNATURE_REL_TOL),
        _check("runoff_ratio", sig.runoff_ratio, published["published_runoff_ratio"], "relative", SIGNATURE_REL_TOL),
        _check("fdc_slope", fdc, published["published_fdc_slope"], "relative", SIGNATURE_REL_TOL),
        _check(
            "peak_month",
            int(sig.peak_month),
            int(published["published_peak_month"]),
            "circular_months",
            PEAK_MONTH_TOL,
        ),
        _check("baseflow_index", sig.baseflow_index, published["published_baseflow_index"], "absolute", BFI_ABS_TOL),
    ]

    integrity = _signature_integrity(sig, fdc)

    return {
        "checks": checks,
        "values": {
            "q_mean": float(sig.mean_flow),
            "q5": float(sig.q5),
            "q95": float(sig.q95),
            "median_flow": float(sig.median_flow),
            "runoff_ratio": float(sig.runoff_ratio) if sig.runoff_ratio is not None else None,
            "mean_recession_constant": float(sig.mean_recession_constant),
            "fdc_slope": fdc,
            "peak_month": int(sig.peak_month),
            "baseflow_index": float(sig.baseflow_index),
        },
        "integrity": integrity,
        "seconds": seconds,
    }


def _signature_integrity(sig, fdc: float) -> list[dict]:
    """Internal sanity checks on the signature object (mirrors the regression tests).

    These are invariants on the *signature object itself*, not comparisons
    against published CAMELS attributes, so a failure is a software defect
    rather than a benchmark finding. Recorded in ``results.json`` and rendered
    in the report, but never folded into the headline gates.
    """
    missing = [f.name for f in dataclasses.fields(sig) if getattr(sig, f.name) is None]
    return [
        _integrity_check(
            "order",
            sig.q5 < sig.median_flow < sig.q95,
            f"q5={sig.q5:.3f} < median={sig.median_flow:.3f} < q95={sig.q95:.3f}",
        ),
        _integrity_check(
            "runoff_ratio",
            sig.runoff_ratio is not None and 0 < sig.runoff_ratio < 1,
            f"runoff_ratio={sig.runoff_ratio}",
        ),
        _integrity_check(
            "recession_constant",
            sig.mean_recession_constant > 0,
            f"mean_recession_constant={sig.mean_recession_constant:.4f}",
        ),
        _integrity_check("fdc_slope", np.isfinite(fdc) and fdc < 0, f"fdc_slope={fdc:.4f}"),
        _integrity_check(
            "complete",
            not missing,
            "all signature fields populated" if not missing else f"missing: {', '.join(missing)}",
        ),
    ]


def _integrity_check(metric: str, passes: bool, detail: str) -> dict:
    """A recorded internal-invariant check (no published reference exists)."""
    return {"metric": metric, "detail": detail, "check_passes": bool(passes)}


def _camels_fdc_slope(discharge: pd.Series) -> float:
    """FDC slope between 33 % and 66 % exceedance, signed like CAMELS (negative)."""
    fdc = flow_duration_curve(discharge, percentiles=[33, 66])
    q33 = fdc.percentiles[33]
    q66 = fdc.percentiles[66]
    if q33 > 0 and q66 > 0:
        return float((np.log(q66) - np.log(q33)) / (0.66 - 0.33))
    return float("nan")


def run_baseflow(discharge: pd.Series, catchment: dict) -> dict:
    """Separate baseflow with both filters and compare BFI to the published value."""
    results: dict = {"methods": {}, "seconds": 0.0}
    for method in ("lyne_hollick", "eckhardt"):
        start = time.perf_counter()
        result = baseflow_analysis(discharge, method=method)
        seconds = time.perf_counter() - start
        results["methods"][method] = {
            "bfi": round(float(result.bfi), 6),
            "published_bfi": round(float(catchment["published_baseflow_index"]), 6),
            "seconds": seconds,
            "check": _check(
                f"bfi_{method}",
                result.bfi,
                catchment["published_baseflow_index"],
                "absolute",
                BFI_ABS_TOL,
            ),
        }
        results["seconds"] += seconds
    return results


def run_flood_frequency(peaks: pd.Series, reference_entry: dict) -> dict:
    """Fit each distribution and compare quantiles to the flood-frequency reference."""
    series = _annual_maxima_series(peaks.values)
    results: dict = {"methods": {}, "seconds": 0.0}

    for method, reference_key in _FFA_REFERENCE_KEYS.items():
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if method == "gev":
                fit = flood_analysis(series, method="gev", return_periods=RETURN_PERIODS, ci_level=None)
            else:
                fit = flood_analysis(series, method=method, return_periods=RETURN_PERIODS)
        seconds = time.perf_counter() - start

        quantiles = fit.return_periods
        reference_quantiles = reference_entry[reference_key]
        reference_mle_unstable = reference_entry.get("mle_stable") is False

        fits: list[dict] = []
        errors: list[float] = []
        for rp in RETURN_PERIODS:
            computed = float(quantiles[rp])
            reference = float(reference_quantiles[str(rp)])
            err = _relative_error(computed, reference)
            errors.append(err)
            passes = np.isfinite(err) and err <= FFA_REL_TOL
            classification = (
                "data_limitation" if method == "gev" and reference_mle_unstable and not passes else "implementation"
            )
            fits.append(
                {
                    "return_period": rp,
                    "computed_m3s": round(computed, 2),
                    "reference_m3s": round(reference, 2),
                    "relative_error_pct": round(err * 100, 2),
                    "check_passes": bool(passes),
                    "classification": classification,
                }
            )

        results["methods"][method] = {
            "reference_key": reference_key,
            "reference_mle_unstable": bool(reference_mle_unstable) if method == "gev" else False,
            "mean_relative_error_pct": round(float(np.mean(errors)) * 100, 2),
            "max_relative_error_pct": round(float(np.max(errors)) * 100, 2),
            "warnings": [str(w.message) for w in caught],
            "seconds": seconds,
            "fits": fits,
        }
        results["seconds"] += seconds

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_SIGNATURE_KEYS = ["q_mean", "q5", "q95", "runoff_ratio", "fdc_slope"]


def _aggregate(catchment_results: dict[str, dict], catchments: dict[str, dict]) -> dict:
    """Cross-catchment metrics and headline gates."""
    aggregate: dict[str, dict] = {}
    for key in _SIGNATURE_KEYS:
        computed_vec: list[float] = []
        published_vec: list[float] = []
        for gid, res in catchment_results.items():
            value = res["signatures"]["values"].get(key)
            if value is None or not np.isfinite(value):
                continue
            computed_vec.append(value)
            published_vec.append(catchments[gid][f"published_{key}"])
        aggregate[key] = _metrics_pair(computed_vec, published_vec)

    for method in ("lyne_hollick", "eckhardt"):
        computed_vec = [r["baseflow"]["methods"][method]["bfi"] for r in catchment_results.values()]
        published_vec = [r["baseflow"]["methods"][method]["published_bfi"] for r in catchment_results.values()]
        aggregate[f"bfi_{method}"] = _metrics_pair(computed_vec, published_vec)

    # The signature object's own baseflow index is also a published-value check and is
    # counted in n_unmet (alongside the two digital filters); tabulate it as well so the
    # aggregate table and the summary's check count stay in step.
    aggregate["bfi_signatures"] = _metrics_pair(
        [r["signatures"]["values"]["baseflow_index"] for r in catchment_results.values()],
        [catchments[gid]["published_baseflow_index"] for gid in catchment_results],
    )

    # Headline gates
    q_mean_rmse = aggregate["q_mean"]["rmse"]
    q_mean_published = [catchments[gid]["published_q_mean"] for gid in catchment_results]
    q_mean_nrmse_pct = q_mean_rmse / np.mean(q_mean_published) * 100 if q_mean_published else None

    bfi_pbias_pct = aggregate["bfi_lyne_hollick"]["pbias"]

    ffa_errors: list[float] = []
    for res in catchment_results.values():
        for method in ("gev_lmoments", "lp3"):
            ffa_errors.extend([fit["relative_error_pct"] for fit in res["flood_frequency"]["methods"][method]["fits"]])
    ffa_mean_rel_pct = float(np.mean(ffa_errors)) if ffa_errors else float("nan")

    # Every published-value comparison is counted once: the seven signature checks, the
    # two baseflow-filter checks, and the signature object's own BFI (per catchment),
    # plus each FFA fit. The single-criterion per-fit classification is a labelling of
    # the same fit, not an extra finding, so it is not counted twice.
    n_unmet = sum(
        1
        for res in catchment_results.values()
        for check in res["signatures"]["checks"] + [m["check"] for m in res["baseflow"]["methods"].values()]
        if not check["check_passes"]
    )
    n_unmet += sum(
        1
        for res in catchment_results.values()
        for method in res["flood_frequency"]["methods"].values()
        for fit in method["fits"]
        if not fit["check_passes"]
    )

    n_data_limitation = sum(
        1
        for res in catchment_results.values()
        for method in res["flood_frequency"]["methods"].values()
        for fit in method["fits"]
        if fit["classification"] == "data_limitation"
    )

    n_integrity_failures = sum(
        1 for res in catchment_results.values() for check in res["signatures"]["integrity"] if not check["check_passes"]
    )

    total_s = sum(
        res["signatures"]["seconds"] + res["baseflow"]["seconds"] + res["flood_frequency"]["seconds"]
        for res in catchment_results.values()
    )

    return {
        "n_catchments": len(catchment_results),
        "n_unmet": n_unmet,
        "n_data_limitation_findings": n_data_limitation,
        "n_integrity_failures": n_integrity_failures,
        "gates": {
            "q_mean_nrmse_pct": round(q_mean_nrmse_pct, 2),
            "q_mean_nrmse_tolerance_pct": Q_MEAN_NRMSE_TOL,
            "q_mean_gate_met": bool(q_mean_nrmse_pct <= Q_MEAN_NRMSE_TOL),
            "bfi_pbias_pct": round(bfi_pbias_pct, 2),
            "bfi_pbias_tolerance_pct": BFI_PBIAS_TOL,
            "bfi_gate_met": bool(abs(bfi_pbias_pct) <= BFI_PBIAS_TOL),
            "ffa_cross_method_mean_relative_error_pct": round(ffa_mean_rel_pct, 2),
            "ffa_tolerance_pct": FFA_MEAN_REL_TOL,
            "ffa_gate_met": bool(ffa_mean_rel_pct <= FFA_MEAN_REL_TOL),
        },
        "aggregate": aggregate,
        "timings": {"total_seconds": round(total_s, 3)},
    }


def _strict_failed(results: dict) -> bool:
    """Whether ``--strict`` should exit non-zero for a results dict."""
    fences = results["summary"]["gates"]
    return bool(results["summary"]["n_unmet"] or results["summary"]["n_integrity_failures"]) or not all(
        fences[k] for k in fences if k.endswith("_met")
    )


def _metrics_pair(computed: list[float], published: list[float]) -> dict:
    """RMSE / PBIAS / R2 between matched vectors (NaN when insufficient data)."""
    if not computed or not published:
        return {"rmse": None, "pbias": None, "r2": None}
    obs = np.asarray(published, dtype=float)
    sim = np.asarray(computed, dtype=float)
    return {
        "rmse": round(float(rmse(obs, sim)), 4) if len(obs) else None,
        "pbias": round(float(pbias(obs, sim)), 2) if len(obs) else None,
        "r2": round(float(r2_score(obs, sim)), 4) if len(obs) else None,
    }


# ---------------------------------------------------------------------------
# Results assembly + writing
# ---------------------------------------------------------------------------


def validate_results(results: dict) -> list[str]:
    """Validate a results dict against the Pydantic models that define its shape.

    Returns the list of violation messages (empty when valid). The same models
    back ``Results.model_json_schema()``, committed as
    ``benchmarks/results.schema.json`` for external consumers.
    """
    try:
        ResultsModel.model_validate(results)
    except ValidationError as exc:
        return [_validation_message(e) for e in exc.errors()]
    return []


def _validation_message(error: dict) -> str:
    """One human-readable schema violation from a pydantic ``Error`` dict."""
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}"


def build_results(gauge_ids: list[str] | None = None) -> dict:
    """Run the full harness and return the results dict."""
    for path in (DAILY_CATCHMENTS_FILE, FFA_REFERENCE_FILE):
        if not path.exists():
            raise RuntimeError(
                f"missing benchmark metadata file: {path.name}. daily_catchments.json and "
                "ffa_reference.json are committed in the repository; restore them "
                "(`git checkout -- data/camels_benchmark`) or regenerate via "
                "`python benchmarks/fetch_peak_flows.py`."
            )
    catchments = load_catchments()
    if gauge_ids:
        catchments = [c for c in catchments if c["gauge_id"] in gauge_ids]
        missing = set(gauge_ids) - {c["gauge_id"] for c in catchments}
        if missing:
            raise ValueError(f"unknown gauge_id(s): {sorted(missing)}")

    reference = load_reference()
    catchment_results: dict[str, dict] = {}

    for c in catchments:
        gid = c["gauge_id"]
        _require_data_files(gid)
        discharge, precipitation = load_synthetic_series(gid)
        peaks = load_peaks_series(gid)
        reference_entry = reference["catchments"].get(gid, {})

        signatures = run_signatures(discharge, precipitation, c)
        baseflow = run_baseflow(discharge, c)
        flood_frequency = run_flood_frequency(peaks, reference_entry)

        catchment_results[gid] = {
            "name": c["name"],
            "climate": c["climate"],
            "signatures": signatures,
            "baseflow": baseflow,
            "flood_frequency": flood_frequency,
            "timings": {
                "signatures_s": round(signatures["seconds"], 3),
                "baseflow_s": round(baseflow["seconds"], 3),
                "flood_frequency_s": round(flood_frequency["seconds"], 3),
                "total_s": round(signatures["seconds"] + baseflow["seconds"] + flood_frequency["seconds"], 3),
            },
        }

    summary = _aggregate(catchment_results, {c["gauge_id"]: c for c in catchments})

    builder = _report_builder({"metadata": {"schema_version": SCHEMA_VERSION}})
    cff_version, cff_doi, cff_author = _read_citation_cff()
    if cff_version:
        builder.metadata.software_version = cff_version
    if cff_doi:
        builder.metadata.doi = cff_doi
    if cff_author:
        builder.metadata.author = cff_author
    software = {
        "version": builder.metadata.software_version,
        "author": builder.metadata.author,
        "doi": builder.metadata.doi or "",
        "citation": builder.software_citation(),
    }

    results = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "generated": datetime.now().strftime("%Y-%m-%d"),
            "aquascope_version": builder.metadata.software_version,
            "note": (
                "AquaScope's own implementations compared against published CAMELS "
                "attributes and the flood-frequency reference quantiles. Per-catchment "
                "integrity checks record internal invariants on the signature object "
                "and are surfaced in the report but never folded into the gates. "
                "results.md and results.html are renderings of this file. This file "
                "is validated against benchmarks/results.schema.json, generated from "
                "the Pydantic models in benchmarks/results_models.py. Execution "
                "times are recorded and reported, never asserted."
            ),
            "tolerances": TOLERANCES,
            "timings_not_asserted": True,
        },
        "catchments": catchment_results,
        "summary": summary,
        "software": software,
    }

    try:
        results = ResultsModel.model_validate(results).model_dump(mode="json")
    except ValidationError as exc:
        raise ValueError(
            "results do not conform to benchmarks/results.schema.json "
            f"(schema_version {SCHEMA_VERSION}):\n  " + "\n  ".join(_validation_message(e) for e in exc.errors()[:10])
        ) from exc
    return results


_REPO_ROOT = _paths.REPO_ROOT


def _read_citation_cff() -> tuple[str | None, str | None, str | None]:
    """Read ``version``, ``doi`` and ``authors`` from the repo-root ``CITATION.cff``.

    Author names are formatted "given-names name-particle family-names
    name-suffix" per the CFF conventions; entries with a plain ``name`` are
    used verbatim.

    Returns:
        ``(version, doi, author)`` — each may be ``None`` if the file is
        missing or the field is absent.
    """
    cff = _REPO_ROOT / "CITATION.cff"
    try:
        text = cff.read_text(encoding="utf-8")
    except OSError:
        return None, None, None
    version = doi = None
    authors: list[dict[str, str]] = []
    entry: dict[str, str] | None = None
    in_authors = False
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)
        if indent == 0:
            if in_authors:
                authors.append(entry) if entry else None
            in_authors = stripped.startswith("authors:")
            entry = None
            if not in_authors:
                key, _, val = stripped.partition(":")
                key = key.strip()
                val = val.strip().strip('"')
                if key == "version" and version is None:
                    version = val
                elif key == "doi" and doi is None:
                    doi = val
            continue
        if not in_authors:
            continue
        if indent == 2 and stripped.startswith("- "):
            if entry:
                authors.append(entry)
            entry = {}
            item = stripped[2:].strip()
            if item and ":" in item:
                key, _, val = item.partition(":")
                entry[key.strip()] = val.strip().strip('"')
            continue
        if entry is not None:
            key, _, val = stripped.partition(":")
            entry[key.strip()] = val.strip().strip('"')
    if entry:
        authors.append(entry)
    if authors:
        author = ", ".join(
            " ".join(
                (
                    a.get("given-names", "") + " "
                    + a.get("name-particle", "") + " "
                    + a.get("family-names", "") + " "
                    + a.get("name-suffix", "")
                ).split()
            )
            if "name" not in a
            else a["name"]
            for a in authors
        )
        return version, doi, author
    return version, doi, None


def _report_builder(results: dict) -> ReportBuilder:
    """A ReportBuilder configured from a results dict."""
    builder = ReportBuilder(
        "CAMELS Benchmark: AquaScope against published and reference values",
        author="AquaScope",
        description="Flood frequency, baseflow and signature verification over 10 CAMELS catchments.",
    )
    builder.metadata.data_sources = [
        "Synthetic daily series (data/camels_benchmark/daily)",
        "USGS annual peaks (data/camels_benchmark/peaks)",
        "CAMELS attributes (Addor et al., 2017)",
        "Flood-frequency reference quantiles (ffa_reference.json)",
    ]
    builder.metadata.version = results["metadata"]["schema_version"]
    # Restore the software identity recorded at run time, so a re-rendered
    # report (from-json) carries the same version and DOI it was produced under.
    software = results.get("software") or {}
    if software.get("author"):
        builder.metadata.author = software["author"]
    if software.get("doi"):
        builder.metadata.doi = software["doi"]
    if software.get("version"):
        builder.metadata.software_version = software["version"]
    return builder


def _write_json_atomic(data: dict, path: pathlib.Path) -> None:
    """Write ``data`` (a JSON-serialisable dict) to ``path`` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(str(path) + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


# ---------------------------------------------------------------------------
# Report renderers (consume the results dict only)
# ---------------------------------------------------------------------------


def _check_rows(checks: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "Metric": c["metric"],
            "Computed": c["computed"],
            "Published": c["published"],
            "Error": c["error"],
            "Tolerance": c["tolerance"],
            "Meets tolerance": "yes" if c["check_passes"] else "no",
        }
        for c in checks
    ]
    return pd.DataFrame(rows)


def _ffa_rows(method: dict) -> pd.DataFrame:
    rows = [
        {
            "Return period (yr)": fit["return_period"],
            "Computed (m3/s)": fit["computed_m3s"],
            "Reference (m3/s)": fit["reference_m3s"],
            "Rel. error (%)": fit["relative_error_pct"],
            "Meets tolerance": "yes" if fit["check_passes"] else "no",
            "Classification": fit["classification"],
        }
        for fit in method["fits"]
    ]
    return pd.DataFrame(rows)


def render_markdown(results: dict, path: str | pathlib.Path) -> pathlib.Path:
    """Render the results dict to a Markdown report."""
    builder = _report_builder(results)
    _populate_report(builder, results)
    return builder.to_markdown(path)


def render_html(results: dict, path: str | pathlib.Path) -> pathlib.Path:
    """Render the results dict to a self-contained HTML report."""
    builder = _report_builder(results)
    _populate_report(builder, results)
    return builder.to_html(path)


def _populate_report(builder: ReportBuilder, results: dict) -> None:
    """Lay out every section from the results dict."""
    summary = results["summary"]
    metadata = results["metadata"]

    # Headline gates
    builder.add_heading("Summary", level=2)
    gates = summary["gates"]
    builder.add_metric("Catchments", summary["n_catchments"])
    builder.add_metric(
        "Q mean normalized RMSE (%)", gates["q_mean_nrmse_pct"], "%", gates["q_mean_nrmse_tolerance_pct"]
    )
    builder.add_metric("BFI PBIAS (%)", gates["bfi_pbias_pct"], "%", gates["bfi_pbias_tolerance_pct"])
    builder.add_metric(
        "FFA mean relative error (%)",
        gates["ffa_cross_method_mean_relative_error_pct"],
        "%",
        gates["ffa_tolerance_pct"],
    )
    builder.add_paragraph(
        f"{summary['n_unmet']} check(s) unmet across "
        f"{summary['n_catchments']} catchments; "
        f"{summary['n_data_limitation_findings']} GEV-MLE mismatch(es) classified as data "
        "limitations (unstable reference MLE), not implementation defects."
    )
    if summary["n_integrity_failures"]:
        builder.add_paragraph(
            f"{summary['n_integrity_failures']} signature-integrity check(s) unmet "
            "(internal invariants on the signature object, not published-value comparisons)."
        )

    # Tolerances + rationale
    builder.add_heading("Tolerances and rationale", level=2)
    tol_df = pd.DataFrame(
        [
            {"Check": name, "Value": value["value"], "Unit": value["unit"], "Rationale": value["rationale"]}
            for name, value in metadata["tolerances"].items()
        ]
    )
    builder.add_dataframe(tol_df, caption="Decided up front; not loosened when a check fails.")

    # Aggregate performance across catchments
    builder.add_heading(f"Aggregate performance (across {summary['n_catchments']} catchments)", level=2)
    agg_rows = []
    for name, vals in summary["aggregate"].items():
        agg_rows.append({"Signature": name, "RMSE": vals["rmse"], "PBIAS %": vals["pbias"], "R2": vals["r2"]})
    builder.add_dataframe(pd.DataFrame(agg_rows), caption="RMSE / PBIAS / R2 vs published values")

    # Per-catchment detail
    builder.add_heading("Per-catchment results", level=2)
    for gid, res in results["catchments"].items():
        builder.add_heading(f"{gid} - {res['name']} ({res['climate']})", level=3)
        builder.add_dataframe(_check_rows(res["signatures"]["checks"]), caption=f"{gid} signatures")

        integrity_rows = [
            {"Metric": c["metric"], "Detail": c["detail"], "Passes": "yes" if c["check_passes"] else "no"}
            for c in res["signatures"]["integrity"]
        ]
        builder.add_dataframe(pd.DataFrame(integrity_rows), caption=f"{gid} signature integrity")

        bf_rows = [
            {
                "Method": method,
                "BFI": bf["bfi"],
                "Published": bf["published_bfi"],
                "Error": bf["check"]["error"],
                "Meets tolerance": "yes" if bf["check"]["check_passes"] else "no",
            }
            for method, bf in res["baseflow"]["methods"].items()
        ]
        builder.add_dataframe(pd.DataFrame(bf_rows), caption=f"{gid} baseflow index")

        for method, mres in res["flood_frequency"]["methods"].items():
            caption = f"{gid} {method} vs {mres['reference_key']}"
            if mres["reference_mle_unstable"]:
                caption += " (reference MLE unstable)"
            builder.add_dataframe(_ffa_rows(mres), caption=caption)
            warnings = mres["warnings"]
            if warnings:
                builder.add_paragraph("<br>".join(f"- {w}" for w in warnings))

        timing_row = {"Gauge": gid, **res["timings"]}
        builder.add_dataframe(pd.DataFrame([timing_row]), caption=f"{gid} stage timings (s)")

    # Timing summary
    builder.add_heading("Execution timings", level=2)
    builder.add_paragraph(
        "Wall-clock seconds per stage, per catchment. Recorded and reported; "
        "never asserted, so a slow shared CI runner cannot produce a false failure."
    )
    timing_rows = []
    for gid, res in results["catchments"].items():
        timing_rows.append({"Gauge": gid, **res["timings"]})
    builder.add_dataframe(pd.DataFrame(timing_rows), caption="Recorded wall-clock timings (s)")
    builder.add_metric("Total runtime (s)", summary["timings"]["total_seconds"], "s")

    # Software citation (#334)
    builder.add_heading("Data and methods", level=2)
    builder.add_paragraph(
        "Discharge: synthetic daily series calibrated to CAMELS; peaks: USGS NWIS annual "
        "peak series, one value per USGS water year, screened for qualifier codes and "
        "censored values. Reference quantiles: GEV-MLE and LP3 via scipy, GEV-L-moments "
        "via lmoments3 (Hosking 1997). Full provenance per catchment in results.json."
    )
    builder.add_software_citation()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the CAMELS benchmark harness.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR!r}).",
    )
    parser.add_argument(
        "--gauge-id",
        nargs="*",
        default=None,
        help="Restrict the run to these gauge ids (default: all 10).",
    )
    parser.add_argument(
        "--from-json",
        default=None,
        metavar="PATH",
        help=(
            "Skip the run: render an existing results.json to results.md / results.html "
            "in the output directory (plus a copy of the JSON). results.json stays the "
            "single source of truth; the reports are built from its numbers."
        ),
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Write only results.json (skip results.md / results.html).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any check is unmet.",
    )
    args = parser.parse_args(argv)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    if args.from_json:
        if args.no_reports:
            raise ValueError("--from-json renders reports; it cannot be combined with --no-reports.")
        with open(args.from_json) as f:
            results = json.load(f)
        try:
            ResultsModel.model_validate(results)
        except ValidationError as exc:
            raise ValueError(
                f"{args.from_json} does not conform to benchmarks/results.schema.json "
                f"(schema_version {SCHEMA_VERSION}):\n  "
                + "\n  ".join(_validation_message(e) for e in exc.errors()[:10])
            ) from exc
        results_path = output_dir / "results.json"
        _write_json_atomic(results, results_path)
        render_markdown(results, output_dir / "results.md")
        render_html(results, output_dir / "results.html")
        print(f"Rendered {results_path.name}, results.md and results.html from {args.from_json} into {output_dir}/")
        print(
            f"  Unmet checks: {results['summary']['n_unmet']}  "
            f"data-limitation findings: {results['summary']['n_data_limitation_findings']}"
        )
        return 0

    results = build_results(gauge_ids=args.gauge_id)
    runtime = time.perf_counter() - start

    results_path = output_dir / "results.json"
    _write_json_atomic(results, results_path)

    paths = [results_path]
    if not args.no_reports:
        paths.append(render_markdown(results, output_dir / "results.md"))
        paths.append(render_html(results, output_dir / "results.html"))

    gates = results["summary"]["gates"]
    print(f"Wrote {len(results['catchments'])} catchment results to {output_dir}/")
    print(f"  Q mean NRMSE: {gates['q_mean_nrmse_pct']:.2f} % (tol {gates['q_mean_nrmse_tolerance_pct']:.0f} %)")
    print(f"  BFI PBIAS:    {gates['bfi_pbias_pct']:.2f} % (tol {gates['bfi_pbias_tolerance_pct']:.0f} %)")
    print(
        f"  FFA mean rel: {gates['ffa_cross_method_mean_relative_error_pct']:.2f} % "
        f"(tol {gates['ffa_tolerance_pct']:.0f} %)"
    )
    print(
        f"  Unmet checks: {results['summary']['n_unmet']}  "
        f"data-limitation findings: {results['summary']['n_data_limitation_findings']}"
    )
    print(f"  Runtime: {results['summary']['timings']['total_seconds']:.1f} s (harness total {runtime:.1f} s)")

    if args.strict and _strict_failed(results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
