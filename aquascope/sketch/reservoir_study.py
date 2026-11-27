"""Client-supplied reservoir scenarios for a single flood event in Studio."""

from __future__ import annotations

import math
from typing import Any

from aquascope import routing
from aquascope.sketch import reservoir


def _number(value: Any, field: str, upper: float) -> float:
    if type(value) not in (int, float) or not 0 < value <= upper:
        raise ValueError(f"{field} must be a number above 0 and at most {upper:g}")
    return float(value)


def _polygon(points: Any) -> None:
    if not isinstance(points, list) or not all(
        isinstance(p, (list, tuple)) and len(p) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in p)
        for p in points
    ):
        raise ValueError("Supply a polygon as a ring of [longitude, latitude] points.")
    ring = reservoir._ring(points)
    if not any((x - ring[0][0]) * (b - ring[0][1]) != (y - ring[0][1]) * (a - ring[0][0])
               for (x, y), (a, b) in zip(ring[1:], ring[2:])):
        raise ValueError("Draw a polygon enclosing an area.")


def validate_scenario(params: Any) -> None:
    if not isinstance(params, dict):
        raise ValueError("Supply a reservoir polygon, dam dimensions and flood scenario in reservoir_routing, "
                         "or route a drawn reservoir on the map and choose Study this reservoir flood.")
    allowed = {"polygon", "crest_m", "top_m", "width_m", "duration_h", "peak_cms", "return_period",
               "from_river_id", "hours", "storm"}
    if extras := params.keys() - allowed:
        raise ValueError(f"Unknown reservoir inputs: {', '.join(sorted(extras))}")
    _polygon(params.get("polygon"))
    crest = _number(params.get("crest_m"), "crest_m", 300)
    top = _number(params.get("top_m"), "top_m", 320)
    if top <= crest:
        raise ValueError("The dam top must be above the spillway crest.")
    _number(params.get("width_m"), "width_m", 2000)
    if "hours" in params and (type(params["hours"]) is not int or not 1 <= params["hours"] <= routing.MAX_HOURS):
        raise ValueError(f"hours must be an integer from 1 to {routing.MAX_HOURS}")
    if sum(key in params for key in ("storm", "peak_cms", "return_period")) != 1:
        raise ValueError("Supply exactly one of peak_cms, return_period or storm.")
    if "storm" in params:
        storm = params["storm"]
        if (not isinstance(storm, dict) or storm.keys() - {"polygon", "rain_mm", "duration_h", "runoff_coefficient"}
                or "from_river_id" in params or "duration_h" in params):
            raise ValueError("Supply the storm polygon, rain and duration inside storm.")
        _polygon(storm.get("polygon"))
        _number(storm.get("rain_mm"), "rain_mm", 2000)
        _number(storm.get("duration_h", 6), "storm duration_h", routing.MAX_HOURS)
        _number(storm.get("runoff_coefficient", 0.3), "runoff_coefficient", 1)
    else:
        if _number(params.get("duration_h"), "duration_h", routing.MAX_HOURS) <= 1:
            raise ValueError("The triangular wave must last over one hour to sample positive flow.")
        if "peak_cms" in params:
            _number(params["peak_cms"], "peak_cms", 1_000_000)
        elif _number(params["return_period"], "return_period", 500) < 2:
            raise ValueError("return_period must be at least 2 years.")
        if "from_river_id" in params:
            rid = params["from_river_id"]
            if type(rid) is not int or not 100_000_000 <= rid <= 999_999_999:
                raise ValueError("from_river_id must be a nine-digit GEOGLOWS reach ID.")


def study_reservoir(**scenario: Any) -> dict[str, Any]:
    """Route one drawn reservoir and retain its inflow/outflow over the sampled window."""
    validate_scenario(scenario)
    run = reservoir.route_reservoir(**scenario, frames=False)
    if not run.get("ok"):
        raise ValueError(run.get("error") or "The reservoir routing failed.")
    res = run["reservoir"]
    inflow, outflow = res["inflow_cms"], res["outflow_cms"]
    if (not inflow or len(inflow) != len(outflow)
            or not all(math.isfinite(q) and q >= 0 for series in (inflow, outflow) for q in series)):
        raise ValueError("The dam needs aligned, finite, non-negative inflow and outflow.")
    check = reservoir.neh_check()
    check["ok"] = (check["max_abs_diff_cfs"] <= 1
                   and abs(check["peak_cfs"] - reservoir.CHECK["book_peak_cfs"]) <= 1
                   and check["peak_day"] in reservoir.CHECK["book_peak_days"])
    if not check["ok"]:
        check["summary"] = "The level-pool solver failed the NRCS NEH Example 17-2 reference check."
    return {
        "hydrograph": {"hour": [i * res["dt_h"] for i in range(len(inflow))],
                       "inflow_cms": inflow, "outflow_cms": outflow},
        "summary": {k: res[k] for k in ("peak_in_cms", "peak_out_cms", "peak_in_h", "peak_out_h",
                                        "reduction_pct", "delay_h", "overtopped", "max_depth_m")},
        "message": run["message"], "notes": run["notes"], "attribution": run["attribution"],
        "check": check, "grade": "screening",
    }
