"""Water quality indices over sampled parameters (#62).

Three indices, each a plain function over a tidy table of samples (the shape
:class:`aquascope.schemas.water_data.WaterQualitySample` records produce:
``parameter``, ``value``, ``unit``, ``sample_datetime``):

* :func:`wqi_ccme`: the CCME Water Quality Index 1.0 (CCME 2001), three
  factors over an explicit guideline table: scope (F1, the share of variables
  that fail), frequency (F2, the share of tests that fail) and amplitude (F3,
  from the normalised sum of excursions), combined into a 0 to 100 score and
  five categories. Guidelines are an input, never hard-coded to one
  jurisdiction; :data:`GUIDELINE_SETS` ships three defaults (WHO 2022
  drinking water, FAO 29 irrigation, CCME freshwater aquatic life).
* :func:`wqi_nsf`: the NSF Water Quality Index (Brown et al. 1970): nine
  parameters, published weights, and sub-index curves reproduced here as
  piecewise-linear digitised approximations of the published rating curves.
* :func:`iwqi`: irrigation suitability after FAO Irrigation and Drainage
  Paper 29 (Ayers and Westcot 1985): the sodium adsorption ratio, sodium
  percentage and residual sodium carbonate with their classic classes, and the
  FAO degree of restriction on use (none, slight to moderate, severe) for
  salinity, infiltration, ion toxicity and miscellaneous effects.

Every index is computed over the parameters that were sampled and nothing
else; a result says which parameters it saw, how many samples of each, over
what period, and which parameters drove the verdict. Units are normalised to
one canonical unit per parameter (mg/L, uS/cm, deg C, NTU, pH units,
CFU/100mL) before any comparison; samples in a unit that cannot be converted
are dropped and counted, never silently compared.

Outputs are plain dicts (no numpy scalars, no NaN) so they cross the browser
worker boundary and become tool results unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from aquascope.utils.parameters import (
    CANONICAL_UNITS,
    EQUIVALENT_WEIGHTS,
    resolve_parameter,
)
from aquascope.utils.parameters import convert_value as _convert
from aquascope.utils.parameters import normalise_unit as _norm_unit
from aquascope.workbench import WHO_GUIDELINES

__all__ = [
    "CANONICAL_UNITS",
    "CCME_AQUATIC_LIFE_GUIDELINES",
    "CCME_CATEGORIES",
    "FAO29_IRRIGATION_GUIDELINES",
    "GUIDELINE_SETS",
    "NSF_CATEGORIES",
    "NSF_CURVES",
    "NSF_WEIGHTS",
    "WHO_DRINKING_GUIDELINES",
    "guideline_set",
    "iwqi",
    "normalise_samples",
    "resolve_parameter",
    "wqi_ccme",
    "wqi_nsf",
]

# ── parameters and units ────────────────────────────────────────────────────
# The vocabulary (names, aliases, canonical units, conversions) lives in
# aquascope.utils.parameters so the WHO screen reads the same names.

_PARAM_COLS = ("parameter", "variable", "characteristic_name", "characteristic", "analyte", "param",
               "result_characteristic")
_VALUE_COLS = ("value", "result_value", "result_measure", "result", "measurement", "resultmeasurevalue")
_UNIT_COLS = ("unit", "units", "result_unit", "result_measureunit", "unit_of_measure", "uom")
_TIME_COLS = ("sample_datetime", "reading_datetime", "datetime", "date", "timestamp", "time",
              "activity_startdate", "observation_datetime")


def _col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def normalise_samples(
    samples: pd.DataFrame, *, extra_parameters: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A tidy frame of recognised samples in canonical units, and what was left out.

    Accepts the long form (``parameter``, ``value``, ``unit``, a datetime
    column) or a wide table whose numeric columns are parameter names.
    Returns ``(frame, report)``: the frame has ``datetime``, ``parameter``
    (canonical key), ``value`` (canonical unit), ``unit`` (canonical) and
    ``reported`` (the name as given); the report counts what was dropped
    (``unrecognised``, ``unconvertible_units``) and which units were assumed.
    ``extra_parameters`` are names outside the canonical vocabulary (a
    pesticide in a user's guideline table) kept as given, lower-cased, with no
    unit conversion.
    """
    report: dict[str, Any] = {"n_in": 0, "n_used": 0, "unrecognised": [], "unconvertible_units": {},
                              "assumed_units": {}}
    extra = {str(p).strip().lower() for p in (extra_parameters or ())}
    if samples is None or samples.empty:
        return pd.DataFrame(columns=["datetime", "parameter", "value", "unit", "reported"]), report
    df = samples
    pcol, vcol = _col(df, _PARAM_COLS), _col(df, _VALUE_COLS)
    ucol, tcol = _col(df, _UNIT_COLS), _col(df, _TIME_COLS)
    if pcol is None or vcol is None:
        # wide: melt the numeric columns whose names are parameters
        numeric = [c for c in df.columns if c != tcol and pd.api.types.is_numeric_dtype(df[c])
                   and (resolve_parameter(c)[0] is not None or str(c).strip().lower() in extra)]
        if not numeric:
            return pd.DataFrame(columns=["datetime", "parameter", "value", "unit", "reported"]), report
        long = df.melt(id_vars=[tcol] if tcol else None, value_vars=numeric, var_name="parameter",
                       value_name="value")
        long["unit"] = ""
        pcol, vcol, ucol = "parameter", "value", "unit"
        tcol = tcol if tcol else None
        df = long
    report["n_in"] = int(len(df))
    values = pd.to_numeric(df[vcol], errors="coerce")
    times = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_localize(None) if tcol else None
    units = df[ucol] if ucol else pd.Series([""] * len(df), index=df.index)
    rows: list[tuple[Any, str, float, str, str]] = []
    unrec: dict[str, int] = {}
    for i, (name, unit, value) in enumerate(zip(df[pcol].tolist(), units.tolist(), values.tolist())):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        key, factor = resolve_parameter(name, unit)
        raw = str(name).strip().lower()
        if key is None and raw in extra:
            t = times.iloc[i] if times is not None else pd.NaT
            rows.append((t, raw, float(value), _norm_unit(unit), str(name)))
            continue
        if key is None:
            unrec[str(name)] = unrec.get(str(name), 0) + 1
            continue
        conv, note = _convert(key, float(value) * factor, unit)
        if conv is None:
            bad = report["unconvertible_units"].setdefault(key, {})
            bad[note or ""] = bad.get(note or "", 0) + 1
            continue
        if note:
            report["assumed_units"][key] = note
        t = times.iloc[i] if times is not None else pd.NaT
        rows.append((t, key, conv, CANONICAL_UNITS[key], str(name)))
    out = pd.DataFrame(rows, columns=["datetime", "parameter", "value", "unit", "reported"])
    report["n_used"] = int(len(out))
    report["unrecognised"] = sorted(unrec, key=lambda k: -unrec[k])[:20]
    return out, report


def _period(frame: pd.DataFrame) -> dict[str, Any]:
    t = frame["datetime"].dropna() if "datetime" in frame else pd.Series([], dtype="datetime64[ns]")
    if t.empty:
        return {"start": None, "end": None, "years": None}
    start, end = t.min(), t.max()
    return {"start": start.date().isoformat(), "end": end.date().isoformat(),
            "years": round((end - start).days / 365.25, 2)}


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame.groupby("parameter").size().sort_index().items()}


def _f(x: Any, digits: int = 4) -> float | None:
    if x is None:
        return None
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, digits)


# ── guideline tables ────────────────────────────────────────────────────────


def _who_from_screen() -> dict[str, dict[str, Any]]:
    """The WHO drinking-water screen's table (``aquascope.workbench.WHO_GUIDELINES``) in guideline form."""
    out: dict[str, dict[str, Any]] = {}
    for name, (lo, hi, unit) in WHO_GUIDELINES.items():
        entry: dict[str, Any] = {"unit": unit}
        if lo > 0 or math.isinf(hi):
            entry["min"] = lo
        if not math.isinf(hi):
            entry["max"] = hi
        out[name] = entry
    return out


#: WHO (2022) Guidelines for drinking-water quality, 4th edition with addenda: the values the WHO screen
#: already carries, plus the guideline values for further common inorganic constituents (mg/L).
WHO_DRINKING_GUIDELINES: dict[str, dict[str, Any]] = {
    **_who_from_screen(),
    "nitrite": {"max": 3.0, "unit": "mg/L"},
    "fluoride": {"max": 1.5, "unit": "mg/L"},
    "cadmium": {"max": 0.003, "unit": "mg/L"},
    "chromium": {"max": 0.05, "unit": "mg/L"},
    "copper": {"max": 2.0, "unit": "mg/L"},
    "boron": {"max": 2.4, "unit": "mg/L"},
    "manganese": {"max": 0.08, "unit": "mg/L"},
    "nickel": {"max": 0.07, "unit": "mg/L"},
    "selenium": {"max": 0.04, "unit": "mg/L"},
    "uranium": {"max": 0.03, "unit": "mg/L"},
    "antimony": {"max": 0.02, "unit": "mg/L"},
    "barium": {"max": 1.3, "unit": "mg/L"},
}

#: FAO Irrigation and Drainage Paper 29 (Ayers and Westcot 1985), Table 1: the thresholds above which the
#: restriction on use is "severe". An exceedance here means severe restriction; the graded classes
#: (none, slight to moderate, severe) are the irrigation index's job.
FAO29_IRRIGATION_GUIDELINES: dict[str, dict[str, Any]] = {
    "conductivity": {"max": 3000.0, "unit": "uS/cm"},   # 3.0 dS/m
    "tds": {"max": 2000.0, "unit": "mg/L"},
    "chloride": {"max": 355.0, "unit": "mg/L"},         # 10 meq/L, surface irrigation
    "boron": {"max": 3.0, "unit": "mg/L"},
    "nitrate": {"max": 133.0, "unit": "mg/L"},          # 30 mg/L nitrate-N as NO3
    "bicarbonate": {"max": 519.0, "unit": "mg/L"},      # 8.5 meq/L, sprinkler irrigation
    "ph": {"min": 6.5, "max": 8.4, "unit": "pH units"},
}

#: CCME Canadian Water Quality Guidelines for the Protection of Aquatic Life, freshwater long-term values;
#: hardness-dependent metals take the least strict bound of their range.
CCME_AQUATIC_LIFE_GUIDELINES: dict[str, dict[str, Any]] = {
    "ph": {"min": 6.5, "max": 9.0, "unit": "pH units"},
    "dissolved_oxygen": {"min": 5.5, "unit": "mg/L"},   # warm-water biota, other life stages
    "nitrate": {"max": 13.0, "unit": "mg/L"},            # as NO3 (about 3 mg/L nitrate-N)
    "chloride": {"max": 120.0, "unit": "mg/L"},
    "arsenic": {"max": 0.005, "unit": "mg/L"},
    "lead": {"max": 0.007, "unit": "mg/L"},
    "mercury": {"max": 0.000026, "unit": "mg/L"},
    "cadmium": {"max": 0.00009, "unit": "mg/L"},
    "copper": {"max": 0.004, "unit": "mg/L"},
    "zinc": {"max": 0.007, "unit": "mg/L"},
}

GUIDELINE_SETS: dict[str, dict[str, dict[str, Any]]] = {
    "drinking": WHO_DRINKING_GUIDELINES,
    "irrigation": FAO29_IRRIGATION_GUIDELINES,
    "aquatic life": CCME_AQUATIC_LIFE_GUIDELINES,
}

GUIDELINE_CITATIONS: dict[str, str] = {
    "drinking": "World Health Organization (2022). Guidelines for drinking-water quality, 4th edition, "
                "incorporating the first and second addenda. WHO, Geneva.",
    "irrigation": "Ayers, R. S. and Westcot, D. W. (1985). Water quality for agriculture. FAO Irrigation and "
                  "Drainage Paper 29, Rev. 1. FAO, Rome.",
    "aquatic life": "CCME. Canadian Water Quality Guidelines for the Protection of Aquatic Life (freshwater, "
                    "long-term). Canadian Council of Ministers of the Environment, Winnipeg.",
}


def guideline_set(name: str) -> dict[str, dict[str, Any]]:
    """One of the shipped guideline tables by use (``drinking``, ``irrigation``, ``aquatic life``)."""
    key = str(name or "").strip().lower().replace("_", " ")
    if key in ("aquatic", "aquatic_life", "ecology", "ecological"):
        key = "aquatic life"
    if key not in GUIDELINE_SETS:
        raise ValueError(f"unknown guideline set {name!r}; one of {sorted(GUIDELINE_SETS)}")
    return GUIDELINE_SETS[key]


def _normalise_guidelines(guidelines: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """A user's guideline table keyed by canonical parameter, its bounds in the canonical unit.

    Keys may be any recognised spelling (``TP``, ``Total phosphorus``); a bound
    given with a unit (mercury ``0.1 ug/L``) is converted like a sample would
    be. A name outside the vocabulary (a pesticide) is kept as given,
    lower-cased, and compared in whatever unit its samples report.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, g in guidelines.items():
        key, factor = resolve_parameter(name)
        entry = dict(g or {})
        if key is None:
            out[str(name).strip().lower()] = entry
            continue
        for bound in ("min", "max"):
            if entry.get(bound) is not None:
                conv, _ = _convert(key, float(entry[bound]) * factor, entry.get("unit"))
                entry[bound] = float(entry[bound]) * factor if conv is None else conv
        entry["unit"] = CANONICAL_UNITS[key]
        out[key] = entry
    return out


def _rule_text(g: dict[str, Any]) -> str:
    unit = g.get("unit", "")
    if g.get("min") is not None and g.get("max") is not None:
        return f"{g['min']:g} to {g['max']:g} {unit}".strip()
    if g.get("min") is not None:
        return f"at least {g['min']:g} {unit}".strip()
    return f"at most {g['max']:g} {unit}".strip()


# ── CCME WQI 1.0 ────────────────────────────────────────────────────────────

#: CCME (2001) categories: the lower bound of each band.
CCME_CATEGORIES: tuple[tuple[float, str], ...] = (
    (95.0, "Excellent"), (80.0, "Good"), (65.0, "Fair"), (45.0, "Marginal"), (0.0, "Poor"),
)

CCME_CITATION = ("CCME (2001). Canadian water quality guidelines for the protection of aquatic life: CCME Water "
                 "Quality Index 1.0, User's Manual. Canadian Council of Ministers of the Environment, Winnipeg.")


def ccme_category(score: float) -> str:
    for floor, label in CCME_CATEGORIES:
        if score >= floor:
            return label
    return "Poor"


def wqi_ccme(
    samples: pd.DataFrame,
    guidelines: dict[str, dict[str, Any]] | str = "drinking",
    *,
    min_variables: int = 4,
    min_tests_per_variable: int = 4,
) -> dict[str, Any]:
    """CCME Water Quality Index 1.0 over the sampled parameters that have a guideline.

    ``guidelines`` is ``{parameter: {"min": x, "max": y, "unit": u}}`` (either
    bound optional, values in the canonical unit) or the name of a shipped set.
    F1 (scope) is the share of variables with at least one failed test, F2
    (frequency) the share of tests that fail, F3 (amplitude) comes from the
    normalised sum of excursions, and the index is
    ``100 - sqrt(F1^2 + F2^2 + F3^2) / 1.732``. CCME recommends at least four
    variables sampled at least four times each; ``meets_minimum_design`` says
    whether this table does. A zero guideline (E. coli) has no ratio to a
    guideline of zero, so its excursions are computed against 1 CFU/100 mL,
    the smallest countable value.
    """
    if isinstance(guidelines, str):
        set_name = guidelines
        table = guideline_set(guidelines)
    else:
        set_name = "custom"
        table = _normalise_guidelines(guidelines or {})
    frame, report = normalise_samples(samples, extra_parameters=[k for k in table if k not in CANONICAL_UNITS])
    frame = frame[frame["parameter"].isin(table)]
    notes: list[str] = []
    variables: list[dict[str, Any]] = []
    n_tests = failed_tests = 0
    excursions_sum = 0.0
    for key in sorted(frame["parameter"].unique()):
        g = table[key]
        vals = frame.loc[frame["parameter"] == key, "value"].to_numpy(dtype=float)
        lo, hi = g.get("min"), g.get("max")
        exc = np.zeros(len(vals))
        if hi is not None:
            over = vals > hi
            divisor = hi if hi > 0 else 1.0
            exc[over] = vals[over] / divisor - 1.0
        if lo is not None:
            under = vals < lo
            exc[under] = lo / np.clip(vals[under], 1e-9, None) - 1.0
        failed = exc > 0
        n_tests += len(vals)
        failed_tests += int(failed.sum())
        excursions_sum += float(exc[failed].sum())
        variables.append({
            "parameter": key, "guideline": _rule_text(g), "unit": CANONICAL_UNITS.get(key, g.get("unit", "")),
            "n": int(len(vals)), "n_failed": int(failed.sum()),
            "pct_failed": _f(100.0 * failed.sum() / len(vals), 1),
            "worst_excursion": _f(exc.max(), 3) if failed.any() else 0.0,
            "min": _f(vals.min()), "median": _f(np.median(vals)), "max": _f(vals.max()),
        })
    n_variables = len(variables)
    if n_variables == 0 or n_tests == 0:
        notes.append("No sampled parameter has a guideline in this set; the index is not computed.")
        return {
            "index": "ccme_wqi", "guideline_set": set_name, "score": None, "category": None,
            "f1": None, "f2": None, "f3": None, "nse": None,
            "n_variables": 0, "n_tests": 0, "n_failed_variables": 0, "n_failed_tests": 0,
            "variables": [], "drivers": [], "meets_minimum_design": False,
            "period": _period(frame), "sample_counts": {}, "input": report, "notes": notes,
            "citation": CCME_CITATION,
        }
    failed_vars = [v for v in variables if v["n_failed"]]
    f1 = 100.0 * len(failed_vars) / n_variables
    f2 = 100.0 * failed_tests / n_tests
    nse = excursions_sum / n_tests
    f3 = nse / (0.01 * nse + 0.01)
    score = 100.0 - math.sqrt(f1 ** 2 + f2 ** 2 + f3 ** 2) / 1.732
    score = max(0.0, min(100.0, score))
    thin = [v["parameter"] for v in variables if v["n"] < min_tests_per_variable]
    meets = n_variables >= min_variables and not thin
    if n_variables < min_variables:
        notes.append(f"Only {n_variables} parameter(s) with a guideline were sampled; CCME recommends at least "
                     f"{min_variables}.")
    if thin:
        notes.append(f"Fewer than {min_tests_per_variable} samples for {', '.join(thin)}; CCME recommends at "
                     f"least {min_tests_per_variable} per parameter.")
    if report["unconvertible_units"]:
        notes.append("Samples in units that could not be converted were left out: "
                     + "; ".join(f"{k} in {', '.join(u for u in v)}" for k, v in report["unconvertible_units"].items())
                     + ".")
    notes.append("The index covers the sampled parameters that have a guideline in this set and nothing else.")
    drivers = sorted(failed_vars, key=lambda v: (-v["n_failed"] * (1 + v["worst_excursion"]), v["parameter"]))
    return {
        "index": "ccme_wqi",
        "guideline_set": set_name,
        "score": round(score, 1),
        "category": ccme_category(score),
        "f1": round(f1, 2), "f2": round(f2, 2), "f3": round(f3, 2), "nse": _f(nse, 5),
        "n_variables": n_variables, "n_tests": n_tests,
        "n_failed_variables": len(failed_vars), "n_failed_tests": failed_tests,
        "variables": variables,
        "drivers": [{"parameter": v["parameter"], "n_failed": v["n_failed"], "n": v["n"],
                     "worst_excursion": v["worst_excursion"], "guideline": v["guideline"]} for v in drivers],
        "meets_minimum_design": meets,
        "period": _period(frame),
        "sample_counts": _counts(frame),
        "input": report,
        "notes": notes,
        "citation": CCME_CITATION,
    }


# ── NSF WQI ─────────────────────────────────────────────────────────────────

#: Brown et al. (1970) weights for the nine parameters (they sum to one).
NSF_WEIGHTS: dict[str, float] = {
    "dissolved_oxygen_saturation": 0.17,
    "fecal_coliform": 0.15,
    "ph": 0.12,
    "bod": 0.10,
    "temperature_change": 0.10,
    "phosphate": 0.10,
    "nitrate": 0.10,
    "turbidity": 0.08,
    "total_solids": 0.08,
}

#: Sub-index rating curves as (value, Q) points, linearly interpolated (fecal coliform in log10 of the
#: count). These are digitised approximations of the published curves, not the curves themselves.
NSF_CURVES: dict[str, tuple[tuple[float, float], ...]] = {
    "dissolved_oxygen_saturation": ((0, 2), (10, 6), (20, 15), (30, 22), (40, 30), (50, 44), (60, 57), (70, 75),
                                    (80, 87), (90, 95), (100, 99), (110, 95), (120, 85), (130, 78), (140, 50)),
    "fecal_coliform": ((1, 97), (5, 88), (10, 80), (50, 60), (100, 50), (500, 33), (1000, 25), (5000, 12),
                       (10000, 8), (100000, 2)),
    "ph": ((2, 0), (3, 2), (4, 8), (5, 25), (6, 55), (7, 90), (7.3, 93), (7.5, 92), (8, 80), (8.5, 65), (9, 48),
           (10, 20), (11, 6), (12, 2)),
    "bod": ((0, 100), (1, 90), (2, 80), (3, 70), (4, 60), (5, 55), (6, 50), (8, 40), (10, 34), (15, 22), (20, 17),
            (25, 12), (30, 8)),
    "temperature_change": ((0, 93), (2.5, 88), (5, 75), (7.5, 60), (10, 45), (12.5, 35), (15, 28), (20, 21),
                           (25, 17), (30, 10)),
    "phosphate": ((0, 99), (0.1, 92), (0.25, 80), (0.5, 60), (1, 40), (2, 22), (3, 15), (5, 8), (10, 3)),
    "nitrate": ((0, 98), (1, 95), (3, 85), (5, 75), (10, 55), (20, 40), (30, 30), (40, 25), (50, 22), (100, 10)),
    "turbidity": ((0, 97), (5, 90), (10, 80), (20, 65), (30, 55), (40, 48), (50, 40), (60, 34), (80, 25),
                  (100, 18)),
    "total_solids": ((0, 80), (50, 86), (100, 80), (200, 72), (300, 56), (400, 43), (500, 32)),
}

NSF_CATEGORIES: tuple[tuple[float, str], ...] = (
    (91.0, "Excellent"), (71.0, "Good"), (51.0, "Medium"), (26.0, "Bad"), (0.0, "Very bad"),
)

NSF_CITATION = ("Brown, R. M., McClelland, N. I., Deininger, R. A. and Tozer, R. G. (1970). A water quality "
                "index: do we dare? Water and Sewage Works 117, 339-343. Sub-index curves are digitised "
                "approximations of the published rating curves.")

#: How many of the nine parameters must be present for a (renormalised) score to be reported.
NSF_MIN_PARAMETERS = 5


def nsf_sub_index(parameter: str, value: float) -> float:
    """The 0 to 100 rating of one value on its NSF curve (linear interpolation, log scale for coliforms)."""
    curve = NSF_CURVES[parameter]
    xs = np.array([p[0] for p in curve], dtype=float)
    ys = np.array([p[1] for p in curve], dtype=float)
    x = float(value)
    if parameter == "fecal_coliform":
        x = math.log10(max(x, xs[0]))
        xs = np.log10(xs)
    if parameter == "temperature_change":
        x = abs(x)
    return float(np.interp(x, xs, ys))


def nsf_category(score: float) -> str:
    for floor, label in NSF_CATEGORIES:
        if score >= floor:
            return label
    return "Very bad"


def oxygen_saturation_mg_l(temperature_c: float) -> float:
    """Dissolved-oxygen saturation concentration in fresh water at sea level (Elmore and Hayes 1960)."""
    t = float(temperature_c)
    return 14.652 - 0.41022 * t + 0.007991 * t ** 2 - 0.000077774 * t ** 3


def _daily_mean(frame: pd.DataFrame, key: str) -> pd.Series:
    sub = frame[frame["parameter"] == key].dropna(subset=["datetime"])
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby(sub["datetime"].dt.normalize())["value"].mean()


def wqi_nsf(samples: pd.DataFrame, *, reference_temperature: float | None = None) -> dict[str, Any]:
    """NSF Water Quality Index (Brown et al. 1970) over the nine parameters that are present.

    Each sample is rated on its parameter's curve and the ratings are averaged
    per parameter; the index is the weight-averaged rating. When some of the
    nine are missing the weights are renormalised over the ones present and the
    result says so (``complete`` is false, ``missing`` lists them); with fewer
    than :data:`NSF_MIN_PARAMETERS` present no score is reported.

    Dissolved oxygen is rated on percent saturation: taken as reported, or
    computed from mg/L and the water temperature sampled the same day (else the
    record's median temperature, else 20 deg C, each noted). Temperature change
    needs a reference: ``temperature_change`` samples, or ``reference_temperature``
    against the mean water temperature; otherwise the parameter is left out.
    Fecal coliform falls back to E. coli counts, total solids to total dissolved
    solids, and phosphate to total phosphorus (as P, converted to PO4), each
    with a note.
    """
    frame, report = normalise_samples(samples)
    notes: list[str] = []
    sub: dict[str, dict[str, Any]] = {}

    def rate(param: str, values: np.ndarray, *, source: str, unit: str) -> None:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        q = np.array([nsf_sub_index(param, v) for v in values])
        sub[param] = {"q": round(float(q.mean()), 1), "weight": NSF_WEIGHTS[param], "value": _f(values.mean()),
                      "unit": unit, "n": int(values.size), "from": source}

    present = set(frame["parameter"].unique())
    # dissolved oxygen as percent saturation
    if "dissolved_oxygen_saturation" in present:
        rate("dissolved_oxygen_saturation",
             frame.loc[frame["parameter"] == "dissolved_oxygen_saturation", "value"].to_numpy(float),
             source="reported percent saturation", unit="%")
    elif "dissolved_oxygen" in present:
        do = _daily_mean(frame, "dissolved_oxygen")
        temp = _daily_mean(frame, "temperature")
        if do.empty:
            do_vals = frame.loc[frame["parameter"] == "dissolved_oxygen", "value"].to_numpy(float)
            temps = np.full(do_vals.shape, 20.0)
            notes.append("Dissolved oxygen has no dates; saturation was computed at an assumed 20 deg C.")
        else:
            do_vals = do.to_numpy(float)
            if not temp.empty:
                paired = temp.reindex(do.index)
                fill = float(temp.median())
                n_missing = int(paired.isna().sum())
                temps = paired.fillna(fill).to_numpy(float)
                if n_missing:
                    notes.append(f"{n_missing} dissolved-oxygen day(s) had no same-day temperature; the record's "
                                 f"median ({fill:.1f} deg C) was used for those.")
            else:
                temps = np.full(do_vals.shape, 20.0)
                notes.append("No water temperature sampled; dissolved-oxygen saturation assumes 20 deg C.")
        sat = 100.0 * do_vals / np.array([oxygen_saturation_mg_l(t) for t in temps])
        rate("dissolved_oxygen_saturation", sat, source="computed from mg/L and temperature", unit="%")
    # fecal coliform, or E. coli as a stand-in
    if "fecal_coliform" in present:
        rate("fecal_coliform", frame.loc[frame["parameter"] == "fecal_coliform", "value"].to_numpy(float),
             source="fecal coliform", unit="CFU/100mL")
    elif "e_coli" in present:
        rate("fecal_coliform", frame.loc[frame["parameter"] == "e_coli", "value"].to_numpy(float),
             source="E. coli as a stand-in for fecal coliform", unit="CFU/100mL")
        notes.append("E. coli counts stand in for fecal coliform on the NSF curve.")
    for key in ("ph", "bod", "nitrate", "turbidity"):
        if key in present:
            rate(key, frame.loc[frame["parameter"] == key, "value"].to_numpy(float), source=key,
                 unit=CANONICAL_UNITS[key])
    # temperature change
    if "temperature_change" in present:
        rate("temperature_change", frame.loc[frame["parameter"] == "temperature_change", "value"].to_numpy(float),
             source="reported temperature change", unit="deg C")
    elif reference_temperature is not None and "temperature" in present:
        mean_t = float(frame.loc[frame["parameter"] == "temperature", "value"].mean())
        rate("temperature_change", np.array([mean_t - float(reference_temperature)]),
             source=f"mean water temperature against a reference of {float(reference_temperature):g} deg C",
             unit="deg C")
    else:
        notes.append("Temperature change needs a reference temperature; the parameter is left out.")
    # phosphate, or total phosphorus converted
    if "phosphate" in present:
        rate("phosphate", frame.loc[frame["parameter"] == "phosphate", "value"].to_numpy(float),
             source="phosphate as PO4", unit="mg/L")
    elif "total_phosphorus" in present:
        rate("phosphate", frame.loc[frame["parameter"] == "total_phosphorus", "value"].to_numpy(float) * 3.066,
             source="total phosphorus as P, converted to PO4", unit="mg/L")
        notes.append("Total phosphorus (as P) converted to phosphate (as PO4, factor 3.066) stands in for total "
                     "phosphate.")
    # total solids, or TDS
    if "total_solids" in present:
        rate("total_solids", frame.loc[frame["parameter"] == "total_solids", "value"].to_numpy(float),
             source="total solids", unit="mg/L")
    elif "tds" in present:
        rate("total_solids", frame.loc[frame["parameter"] == "tds", "value"].to_numpy(float),
             source="total dissolved solids as a stand-in for total solids", unit="mg/L")
        notes.append("Total dissolved solids stand in for total solids on the NSF curve.")

    missing = [k for k in NSF_WEIGHTS if k not in sub]
    n_present = len(sub)
    complete = not missing
    score = category = None
    weight_sum = sum(v["weight"] for v in sub.values())
    if n_present >= NSF_MIN_PARAMETERS and weight_sum > 0:
        score = round(sum(v["q"] * v["weight"] for v in sub.values()) / weight_sum, 1)
        category = nsf_category(score)
        if not complete:
            notes.append(f"{n_present} of the nine NSF parameters are present; the weights were renormalised over "
                         f"them (missing: {', '.join(missing)}).")
    elif n_present:
        notes.append(f"Only {n_present} of the nine NSF parameters are present (fewer than {NSF_MIN_PARAMETERS}); "
                     "no score is reported.")
    else:
        notes.append("None of the nine NSF parameters is present; no score is reported.")
    notes.append("The NSF sub-index curves used here are digitised approximations of the published curves.")
    drivers = sorted(sub.items(), key=lambda kv: kv[1]["q"])[:3]
    return {
        "index": "nsf_wqi",
        "score": score,
        "category": category,
        "complete": complete,
        "n_parameters": n_present,
        "missing": missing,
        "weights_renormalised": bool(sub) and not complete,
        "sub_indices": sub,
        "drivers": [{"parameter": k, "q": v["q"], "value": v["value"], "unit": v["unit"]} for k, v in drivers],
        "period": _period(frame),
        "sample_counts": _counts(frame),
        "input": report,
        "notes": notes,
        "citation": NSF_CITATION,
    }


# ── irrigation suitability (FAO 29) ─────────────────────────────────────────

IWQI_CITATION = ("Ayers, R. S. and Westcot, D. W. (1985). Water quality for agriculture. FAO Irrigation and Drainage "
                 "Paper 29, Rev. 1. FAO, Rome; Richards, L. A. (ed.) (1954). Diagnosis and improvement of saline and "
                 "alkali soils. USDA Handbook 60; Wilcox, L. V. (1955). Classification and use of irrigation waters. "
                 "USDA Circular 969; Eaton, F. M. (1950). Significance of carbonates in irrigation waters. Soil "
                 "Science 69, 123-134.")

RESTRICTION_ORDER = ("none", "slight to moderate", "severe")


def _restriction(value: float | None, lo: float, hi: float) -> str | None:
    """FAO 29 degree of restriction: below ``lo`` none, ``lo`` to ``hi`` slight to moderate, above severe."""
    if value is None:
        return None
    if value < lo:
        return "none"
    if value <= hi:
        return "slight to moderate"
    return "severe"


def _infiltration_restriction(sar: float | None, ec_ds_m: float | None) -> str | None:
    """FAO 29 Table 1: the infiltration hazard reads SAR together with EC (low salinity worsens it)."""
    if sar is None or ec_ds_m is None:
        return None
    for top, none_above, severe_below in ((3, 0.7, 0.2), (6, 1.2, 0.3), (12, 1.9, 0.5), (20, 2.9, 1.3), (40, 5.0, 2.9)):
        if sar <= top or top == 40:
            if ec_ds_m > none_above:
                return "none"
            if ec_ds_m >= severe_below:
                return "slight to moderate"
            return "severe"
    return None


def rsc_class(rsc: float | None) -> str | None:
    """Residual sodium carbonate (Eaton 1950): below 1.25 safe, 1.25 to 2.5 marginal, above 2.5 unsuitable."""
    if rsc is None:
        return None
    if rsc < 1.25:
        return "safe"
    if rsc <= 2.5:
        return "marginal"
    return "unsuitable"


def sodium_percent_class(pct: float | None) -> str | None:
    """Sodium percentage (Wilcox 1955): excellent, good, permissible, doubtful, unsuitable at 20/40/60/80."""
    if pct is None:
        return None
    for top, label in ((20, "excellent"), (40, "good"), (60, "permissible"), (80, "doubtful")):
        if pct < top:
            return label
    return "unsuitable"


def ussl_class(ec_us_cm: float | None, sar: float | None) -> str | None:
    """USSL (Richards 1954) salinity and sodium hazard class, C1 to C4 and S1 to S4."""
    if ec_us_cm is None and sar is None:
        return None
    c = s = ""
    if ec_us_cm is not None:
        c = "C1" if ec_us_cm < 250 else "C2" if ec_us_cm < 750 else "C3" if ec_us_cm < 2250 else "C4"
    if sar is not None:
        s = "S1" if sar < 10 else "S2" if sar < 18 else "S3" if sar < 26 else "S4"
    return "-".join(x for x in (c, s) if x)


def iwqi(samples: pd.DataFrame, *, statistic: str = "median") -> dict[str, Any]:
    """Irrigation water quality after FAO 29 (Ayers and Westcot 1985) over the sampled ions and salinity.

    Each parameter is reduced to one value over the period (its median by
    default, or the mean); ions are converted to meq/L. Reports the sodium
    adsorption ratio ``SAR = Na / sqrt((Ca + Mg) / 2)``, the sodium percentage
    ``(Na + K) / (Ca + Mg + Na + K)`` and the residual sodium carbonate
    ``(HCO3 + CO3) - (Ca + Mg)`` with their classic classes (USSL, Wilcox,
    Eaton), the FAO degree of restriction on use for salinity (EC, TDS),
    infiltration (SAR read with EC), ion toxicity (sodium, chloride, boron) and
    miscellaneous effects (nitrate-N, bicarbonate, pH), and the overall
    restriction, the worst of them, with the parameters that drove it. What was
    not sampled is listed under ``missing`` and is not judged.
    """
    frame, report = normalise_samples(samples)
    notes: list[str] = []
    agg = "mean" if str(statistic).lower() == "mean" else "median"
    stat = frame.groupby("parameter")["value"].agg(agg) if not frame.empty else pd.Series(dtype=float)
    values: dict[str, float] = {str(k): float(v) for k, v in stat.items()}

    def meq(key: str) -> float | None:
        v = values.get(key)
        return None if v is None else v / EQUIVALENT_WEIGHTS[key]

    na, k_, ca, mg = meq("sodium"), meq("potassium"), meq("calcium"), meq("magnesium")
    hco3, co3, cl = meq("bicarbonate"), meq("carbonate"), meq("chloride")
    if hco3 is None and values.get("alkalinity") is not None:
        hco3 = values["alkalinity"] / 50.04
        notes.append("Bicarbonate taken from total alkalinity (as CaCO3, 50.04 mg per meq).")
    if co3 is None and hco3 is not None:
        co3 = 0.0
    ec_us = values.get("conductivity")
    ec_ds = ec_us / 1000.0 if ec_us is not None else None
    tds = values.get("tds")
    boron = values.get("boron")
    nitrate_n = values["nitrate"] / 4.4268 if values.get("nitrate") is not None else None
    ph = values.get("ph")

    sar = sodium_pct = rsc = None
    if na is not None and ca is not None and mg is not None and (ca + mg) > 0:
        sar = na / math.sqrt((ca + mg) / 2.0)
        total = ca + mg + na + (k_ or 0.0)
        sodium_pct = 100.0 * (na + (k_ or 0.0)) / total if total > 0 else None
        if k_ is None:
            notes.append("Potassium was not sampled; the sodium percentage is over Na, Ca and Mg alone.")
    if hco3 is not None and ca is not None and mg is not None:
        rsc = (hco3 + (co3 or 0.0)) - (ca + mg)

    components: dict[str, dict[str, Any]] = {}

    def component(name: str, value: float | None, unit: str, lo: float, hi: float, basis: str,
                  restriction: str | None = None) -> None:
        r = restriction if restriction is not None else _restriction(value, lo, hi)
        components[name] = {"value": _f(value), "unit": unit, "restriction": r,
                            "thresholds": {"none_below": lo, "severe_above": hi}, "basis": basis}

    component("salinity_ec", ec_ds, "dS/m", 0.7, 3.0, "FAO 29 Table 1, ECw")
    component("salinity_tds", tds, "mg/L", 450.0, 2000.0, "FAO 29 Table 1, TDS")
    components["infiltration"] = {
        "value": _f(sar), "unit": "SAR with EC", "restriction": _infiltration_restriction(sar, ec_ds),
        "thresholds": {"note": "SAR bands 0-3, 3-6, 6-12, 12-20, 20-40 read against EC (dS/m)"},
        "basis": "FAO 29 Table 1, infiltration",
    }
    component("sodium_toxicity", sar, "SAR", 3.0, 9.0, "FAO 29 Table 1, sodium (surface irrigation)")
    component("chloride_toxicity", cl, "meq/L", 4.0, 10.0, "FAO 29 Table 1, chloride (surface irrigation)")
    component("boron_toxicity", boron, "mg/L", 0.7, 3.0, "FAO 29 Table 1, boron")
    component("nitrate_nitrogen", nitrate_n, "mg/L as N", 5.0, 30.0, "FAO 29 Table 1, nitrate-nitrogen")
    component("bicarbonate", hco3, "meq/L", 1.5, 8.5, "FAO 29 Table 1, bicarbonate (sprinkler irrigation)")
    ph_r = None if ph is None else ("none" if 6.5 <= ph <= 8.4 else "slight to moderate")
    component("ph", ph, "pH units", 6.5, 8.4, "FAO 29 Table 1, normal range 6.5 to 8.4", restriction=ph_r)

    judged = {k: v for k, v in components.items() if v["restriction"] is not None}
    overall = None
    if judged:
        overall = max((v["restriction"] for v in judged.values()), key=RESTRICTION_ORDER.index)
    drivers = sorted((k for k, v in judged.items() if v["restriction"] == overall and overall != "none"),
                     key=lambda k: -RESTRICTION_ORDER.index(judged[k]["restriction"]))
    needed = ("sodium", "calcium", "magnesium", "conductivity", "bicarbonate", "chloride", "boron", "nitrate",
              "ph", "tds", "potassium", "carbonate")
    missing = [p for p in needed if values.get(p) is None and not (p == "bicarbonate" and hco3 is not None)]
    if sar is None:
        notes.append("Sodium, calcium and magnesium are needed for SAR and the sodium percentage; not all were "
                     "sampled.")
    if not judged:
        notes.append("None of the parameters FAO 29 reads was sampled; no restriction is judged.")
    notes.append(f"Each parameter enters as its {agg} over the period; ions in meq/L.")
    notes.append("Only the sampled parameters are judged; a parameter under `missing` is not cleared, it is unknown.")
    return {
        "index": "iwqi",
        "restriction": overall,
        "class": overall,
        "drivers": drivers,
        "components": components,
        "indices": {
            "sar": {"value": _f(sar, 2), "class": ("S1" if sar is not None and sar < 10 else "S2" if sar is not None
                                                     and sar < 18 else "S3" if sar is not None and sar < 26 else
                                                     "S4" if sar is not None else None),
                    "unit": "(meq/L)^0.5", "basis": "USSL sodium hazard S1 to S4 (Richards 1954)"},
            "sodium_percent": {"value": _f(sodium_pct, 1), "class": sodium_percent_class(sodium_pct), "unit": "%",
                               "basis": "Wilcox (1955)"},
            "rsc": {"value": _f(rsc, 2), "class": rsc_class(rsc), "unit": "meq/L", "basis": "Eaton (1950)"},
            "ussl_class": ussl_class(ec_us, sar),
        },
        "ions_meq_l": {k: _f(v, 3) for k, v in (("sodium", na), ("potassium", k_), ("calcium", ca),
                                                 ("magnesium", mg), ("bicarbonate", hco3), ("carbonate", co3),
                                                 ("chloride", cl)) if v is not None},
        "values": {k: _f(v) for k, v in sorted(values.items())},
        "units": {k: CANONICAL_UNITS.get(k, "") for k in sorted(values)},
        "statistic": agg,
        "missing": missing,
        "n_samples": int(len(frame)),
        "period": _period(frame),
        "sample_counts": _counts(frame),
        "input": report,
        "notes": notes,
        "citation": IWQI_CITATION,
    }
