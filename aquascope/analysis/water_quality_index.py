"""Water Quality Index (WQI) computation.

Implements the Canadian Council of Ministers of the Environment (CCME) WQI,
the most widely used standardized water quality index in Canada and
internationally. The CCME WQI aggregates three factors — scope, frequency,
and amplitude of guideline exceedances — into a single dimensionless score
from 0 (worst) to 100 (best).

References
----------
Canadian Council of Ministers of the Environment (2001). Canadian water
    quality guidelines for the protection of aquatic life: CCME Water
    Quality Index 1.0, User's Manual. In: Canadian environmental quality
    guidelines, 1999, Canadian Council of Ministers of the Environment,
    Winnipeg.
    https://ccme.ca/en/res/wqi-techrpt-en.pdf
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CCME WQI category bands (CCME 2001, Table 1).
CCME_CATEGORIES: list[tuple[float, float, str]] = [
    (95.0, 100.0, "Excellent"),
    (80.0, 95.0, "Good"),
    (65.0, 80.0, "Fair"),
    (45.0, 65.0, "Marginal"),
    (0.0, 45.0, "Poor"),
]


@dataclass
class CCMEWQIResult:
    """Result of a CCME WQI computation.

    Attributes
    ----------
    wqi : float
        CCME WQI score in [0, 100]. Higher is better.
    category : str
        Qualitative category: Excellent / Good / Fair / Marginal / Poor.
    scope : float
        F1 — fraction of parameters that failed at least one test (0-100).
    frequency : float
        F2 — fraction of individual tests that failed (0-100).
    amplitude : float
        F3 — magnitude of exceedances relative to the guidelines (0-100).
    n_parameters : int
        Total number of parameters evaluated.
    n_failed_parameters : int
        Number of parameters with at least one guideline exceedance.
    n_tests : int
        Total number of parameter-sample pairs evaluated.
    n_failed_tests : int
        Number of parameter-sample pairs that exceeded their guideline.
    """

    wqi: float
    category: str
    scope: float
    frequency: float
    amplitude: float
    n_parameters: int
    n_failed_parameters: int
    n_tests: int
    n_failed_tests: int


def ccme_wqi(
    measurements: pd.DataFrame,
    guidelines: dict[str, float],
    parameter_col: str = "parameter",
    value_col: str = "value",
    objective: str = "maximum",
) -> CCMEWQIResult:
    """Compute the CCME Water Quality Index.

    Aggregates scope (F1), frequency (F2), and amplitude (F3) of
    guideline exceedances into a single score from 0 to 100.

    Parameters
    ----------
    measurements : pd.DataFrame
        Tidy DataFrame of water quality measurements. Must contain at
        least ``parameter_col`` and ``value_col`` columns. Matches the
        schema used by :func:`aquascope.analysis.quality.assess_quality`.
    guidelines : dict[str, float]
        Mapping of parameter name to guideline threshold value. Parameters
        not present in this dict are ignored.
    parameter_col : str
        Column name for the parameter identifier. Default ``"parameter"``.
    value_col : str
        Column name for the measured value. Default ``"value"``.
    objective : str
        Direction of the guideline: ``"maximum"`` (exceedance = value >
        threshold, e.g. contaminants) or ``"minimum"`` (exceedance =
        value < threshold, e.g. dissolved oxygen). Default ``"maximum"``.

    Returns
    -------
    CCMEWQIResult
        CCME WQI score, category, and component factors.

    Raises
    ------
    ValueError
        If ``measurements`` is empty, ``guidelines`` is empty, or
        ``objective`` is not ``"maximum"`` or ``"minimum"``.

    Examples
    --------
    CCME (2001) User's Manual worked example (Appendix II):

    >>> import pandas as pd
    >>> from aquascope.analysis.water_quality_index import ccme_wqi
    >>> data = pd.DataFrame({
    ...     "parameter": ["pH"] * 4 + ["DO"] * 4,
    ...     "value": [7.0, 6.5, 6.0, 5.5, 8.0, 7.5, 6.0, 5.0],
    ... })
    >>> guidelines = {"pH": 6.5, "DO": 6.5}
    >>> result = ccme_wqi(data, guidelines, objective="minimum")
    >>> 0 <= result.wqi <= 100
    True

    References
    ----------
    CCME (2001). CCME Water Quality Index 1.0, User's Manual. Winnipeg.
    """
    if objective not in ("maximum", "minimum"):
        raise ValueError(
            f"objective must be 'maximum' or 'minimum', got '{objective}'"
        )
    if measurements.empty:
        raise ValueError("measurements DataFrame is empty.")
    if not guidelines:
        raise ValueError("guidelines dict is empty.")

    # Filter to parameters that have guidelines.
    df = measurements[[parameter_col, value_col]].copy()
    df = df[df[parameter_col].isin(guidelines)].dropna(subset=[value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    if df.empty:
        raise ValueError(
            "No measurements match the provided guidelines after filtering."
        )

    # --- F1: Scope ---
    # Percentage of parameters that fail at least one test.
    params = df[parameter_col].unique()
    n_parameters = len(params)
    failed_params = set()

    for param in params:
        vals = df.loc[df[parameter_col] == param, value_col].values
        threshold = guidelines[param]
        if objective == "maximum":
            if np.any(vals > threshold):
                failed_params.add(param)
        else:
            if np.any(vals < threshold):
                failed_params.add(param)

    n_failed_parameters = len(failed_params)
    f1 = 100.0 * n_failed_parameters / n_parameters

    # --- F2: Frequency ---
    # Percentage of individual tests that fail.
    n_tests = len(df)
    if objective == "maximum":
        failed_mask = df.apply(
            lambda row: row[value_col] > guidelines[row[parameter_col]], axis=1
        )
    else:
        failed_mask = df.apply(
            lambda row: row[value_col] < guidelines[row[parameter_col]], axis=1
        )

    n_failed_tests = int(failed_mask.sum())
    f2 = 100.0 * n_failed_tests / n_tests

    # --- F3: Amplitude ---
    # Normalised magnitude of exceedances.
    # excursion_i = (failed_value / guideline) - 1  [maximum objective]
    # excursion_i = (guideline / failed_value) - 1  [minimum objective]
    failed_df = df[failed_mask].copy()

    if failed_df.empty:
        nse = 0.0
        f3 = 0.0
    else:
        if objective == "maximum":
            excursions = failed_df.apply(
                lambda row: (row[value_col] / guidelines[row[parameter_col]]) - 1,
                axis=1,
            )
        else:
            excursions = failed_df.apply(
                lambda row: (guidelines[row[parameter_col]] / row[value_col]) - 1,
                axis=1,
            )
        # Normalised sum of excursions (nse).
        nse = float(excursions.sum()) / n_tests
        # F3 = nse / (0.01 * nse + 0.01)  — CCME formula (bounded 0-100).
        f3 = float(nse / (0.01 * nse + 0.01))

    # --- CCME WQI ---
    wqi = 100.0 - (np.sqrt(f1**2 + f2**2 + f3**2) / 1.732)
    wqi = float(np.clip(wqi, 0.0, 100.0))

    # Determine category.
    category = "Poor"
    for lo, hi, cat in CCME_CATEGORIES:
        if lo <= wqi <= hi:
            category = cat
            break

    logger.info(
        "CCME WQI=%.1f (%s): F1=%.1f F2=%.1f F3=%.1f | "
        "%d/%d params failed, %d/%d tests failed",
        wqi, category, f1, f2, f3,
        n_failed_parameters, n_parameters,
        n_failed_tests, n_tests,
    )

    return CCMEWQIResult(
        wqi=wqi,
        category=category,
        scope=f1,
        frequency=f2,
        amplitude=f3,
        n_parameters=n_parameters,
        n_failed_parameters=n_failed_parameters,
        n_tests=n_tests,
        n_failed_tests=n_failed_tests,
    )


def wqi_category(score: float) -> str:
    """Return the CCME WQI category label for a given score.

    Parameters
    ----------
    score : float
        A CCME WQI score in [0, 100].

    Returns
    -------
    str
        One of: ``'Excellent'``, ``'Good'``, ``'Fair'``,
        ``'Marginal'``, or ``'Poor'``.
    """
    for lo, hi, cat in CCME_CATEGORIES:
        if lo <= score <= hi:
            return cat
    return "Poor"
