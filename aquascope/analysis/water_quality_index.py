"""Canadian Council of Ministers of the Environment Water Quality Index.

This module implements the CCME WQI using user-provided parameter
guidelines rather than hardcoded jurisdiction-specific thresholds.

References
----------
Canadian Council of Ministers of the Environment. CCME Water Quality
Index User's Manual 2017 Update.
"""
from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real

import pandas as pd


@dataclass
class CCMEWQIResult:
    """The score, category, and factors from a CCME WQI calculation."""
    score: float
    category: str
    f1: float
    f2: float
    f3: float


def wqi_category(score: float) -> str:
    """Return the CCME category for a score between 0 and 100."""
    if not 0 <= score <= 100:
        raise ValueError("WQI score must be between 0 and 100.")

    if score >= 95:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 65:
        return "Fair"
    if score >= 45:
        return "Marginal"
    return "Poor"

def ccme_wqi(
        measurements: pd.DataFrame,
        guidelines: dict[str, dict[str, float]],
) -> CCMEWQIResult:
    """Calculate CCME WQI for a selected body of water and reporting period.

    Parameters
    ----------
    measurements : pd.DataFrame
        Long-form measurements with ``parameter`` and ``value`` columns.
        Additional WaterQualitySample columns are accepted.
    guidelines : dict
        Parameter names mapped to ``min`` and/or ``max`` limits.
        Names must match the table exactly. This implementation requires
        finite limits, with min >= 0 and max > 0.

    Returns
    -------
    CCMEWQIResult
        Score from 0 to 100, category, and the scope (f1), frequency (f2),
        and amplitude (f3) factors.

    Notes
    -----
    Select the water body and reporting period before calling.
    Measurement values and guideline limits must use matching units;
    this function does not convert units or assess sampling adequacy.
    Missing values and parameters without guidelines are excluded.
    Categories are assigned using the unrounded score.

    Raises
    ------
    ValueError
        For invalid inputs, including nonfinite measurements or nonpositive
        measurements that fail a minimum guideline.

    References
    ----------
    CCME (2017), Water Quality Index User's Manual, 2017 Update.
    https://ccme.ca/en/res/wqimanualen.pdf
    """

    required_columns = {"parameter", "value"}
    missing_columns = required_columns - set(measurements.columns)

    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {names}")

    for parameter, limits in guidelines.items():
        if not limits:
            raise ValueError(f"Empty guideline for {parameter}.")

        unknown_keys = set(limits) - {"min", "max"}
        if unknown_keys:
            raise ValueError(
                f"Invalid guideline keys for {parameter}: {sorted(unknown_keys)}"
            )

        for bound, limit in limits.items():
            if isinstance(limit, bool) or not isinstance(limit, Real):
                raise ValueError(
                    f"Invalid guideline for {parameter}: {bound} must be a number."
                )

            if not isfinite(limit):
                raise ValueError(
                    f"Invalid guideline for {parameter}: {bound} must be finite."
                )

            if limit < 0 or (bound == "max" and limit == 0):
                raise ValueError(
                    f"Invalid guideline for {parameter}: min must be >= 0 and max > 0."
                )

        if "min" in limits and "max" in limits:
            if limits["min"] > limits["max"]:
                raise ValueError(
                    f"Invalid guideline for {parameter}: min must not exceed max."
                )

    data = measurements[
        measurements["parameter"].isin(guidelines)].dropna(subset=["value"])

    if data.empty:
        raise ValueError("No measurements selected.")

    failed_parameters = set()
    failed_tests = 0
    excursion_sum = 0.0

    for parameter, raw_value in data[["parameter", "value"]].itertuples(index=False, name=None):
        value = float(raw_value)

        if not isfinite(value):
            raise ValueError(
                f"Invalid measurement for {parameter}: value must be finite."
            )
        limits = guidelines[parameter]
        excursion = 0.0

        if "min" in limits and value < limits["min"]:
            if value <= 0:
                raise ValueError(
                    f"Invalid measurement for {parameter}: a value below "
                    "a minimum guideline must be positive to calculate its excursion."
                )
            excursion = limits["min"] / value - 1.0
        elif "max" in limits and value > limits["max"]:
            excursion = value / limits["max"] - 1.0

        if excursion > 0:
            failed_parameters.add(parameter)
            failed_tests += 1
            excursion_sum += excursion

    total_parameters  = data["parameter"].nunique()
    total_tests = len(data)

    f1 = 100 * len(failed_parameters) / total_parameters
    f2 = 100 * failed_tests / total_tests

    nse = excursion_sum / total_tests
    f3 = nse / (0.01 * nse + 0.01)

    score = 100.0 - sqrt(f1**2 + f2**2 + f3**2) / 1.732
    score = max(0.0, score)

    return CCMEWQIResult(
        score=score,
        category=wqi_category(score),
        f1=f1,
        f2=f2,
        f3=f3,
    )
