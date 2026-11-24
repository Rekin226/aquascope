"""Read the checked routing series for both study findings and optional exports."""

import math
from typing import Any


def route_hydrograph_of(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The full-resolution hourly series at both ends of the route, when usable."""
    hydro = payload.get("hydrograph")
    check = payload.get("hydrograph_check")
    if not isinstance(hydro, dict) or not isinstance(check, dict) or check.get("ok") is not True:
        return None
    if any(hydro.get(k) is None for k in ("start_river_id", "end_river_id")):
        return None
    hours = hydro.get("hours")
    if not isinstance(hours, list) or not hours:
        return None
    for key in ("hours", "start_cms", "end_cms"):
        values = hydro.get(key)
        if not isinstance(values, list) or len(values) != len(hours):
            return None
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values):
            return None
    return hydro
