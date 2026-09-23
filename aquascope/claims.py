"""Machine-readable identities for scientific results, shared by every output.

An interval belongs to a particular fitted result, not to another result with
similar words or even an identical rounded point estimate. Unknown provenance
stays unknown; a content digest identifies inputs but is not a data DOI.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

from aquascope import __version__


def content_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def dataset_identity(payload: dict[str, Any]) -> dict[str, Any]:
    """Actual analysis coverage, distinct from the catalog's advertised coverage."""
    return {
        "source": payload.get("source"), "station_id": payload.get("station_id"),
        "snapshot": payload.get("data_snapshot"), "variable": payload.get("variable"),
        "unit": payload.get("unit"), "start": payload.get("start"), "end": payload.get("end"),
        "observations": payload.get("n"), "software_version": __version__,
        "software_revision": payload.get("software_revision") or os.environ.get("AQUASCOPE_REVISION"),
        "archive_revision": payload.get("archive_revision"),
    }


def flood_result(payload: dict[str, Any], step: str, fit_name: str, index: int) -> dict[str, Any]:
    """Identity and uncertainty for one quantile of one fitted distribution."""
    ffa = payload["ffa"]
    fit = ffa["fits"][fit_name]
    period = ffa["return_periods"][index]
    path = f"ffa.fits.{fit_name}"
    result_id = f"{step}.{path}.q.{index}"
    interval = None
    cis = fit.get("ci") or []
    if index < len(cis) and isinstance(cis[index], (list, tuple)) and len(cis[index]) == 2:
        bounds = cis[index]
        if (all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in bounds)
                and bounds[0] <= bounds[1]):
            interval = {"result_id": result_id, "bounds": list(bounds),
                        "basis": [f"{step}.{path}.ci.{index}.0", f"{step}.{path}.ci.{index}.1"],
                        "method": fit.get("interval_method"), "level": fit.get("ci_level")}
    return {
        "result_id": result_id, "basis": f"{path}.q.{index}",
        "dataset": dataset_identity(payload), "variable": payload.get("variable"),
        "aggregation": "annual maxima of daily mean discharge; years with at least 292 observed days",
        "estimator": fit.get("estimator") or fit_name, "return_period_years": period,
        "sample_years": ffa.get("n_years"), "interval": interval,
        "assumptions": ["The annual maxima are independent and drawn from a stationary distribution.",
                        "Daily mean maxima can understate instantaneous flood peaks."],
    }
