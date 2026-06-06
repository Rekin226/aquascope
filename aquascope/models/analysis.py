"""Result types for extreme-value and return-period analysis.

These lightweight dataclasses are the structured return types for the
functions in :mod:`aquascope.analysis.extreme_events`. They mirror the
dataclass style used across the ``analysis`` package (see
``ChangePointResult``, ``CopulaResult``) rather than Pydantic so they stay
dependency-free and cheap to construct in tight numerical loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GEVParameters:
    """Fitted parameters of a Generalised Extreme Value distribution.

    Attributes:
        shape: GEV shape parameter (``xi``). ``< 0`` Weibull, ``0`` Gumbel,
            ``> 0`` Frechet. Note this uses the classic hydrology sign
            convention, the negative of SciPy's ``genextreme`` ``c``.
        location: Location parameter (``mu``).
        scale: Scale parameter (``sigma``), strictly positive.
        method: Estimation method used to fit the parameters.
    """

    shape: float
    location: float
    scale: float
    method: str = "mle"


@dataclass
class DistributionFit:
    """Goodness-of-fit summary for a single distribution fitted to block maxima.

    Attributes:
        distribution: Distribution identifier (``"gev"``, ``"lp3"`` or
            ``"gumbel"``).
        parameters: Named distribution parameters as a mapping.
        aic: Akaike Information Criterion (lower is better).
        ks_pvalue: Kolmogorov-Smirnov goodness-of-fit p-value.
        n_samples: Number of block-maxima observations used in the fit.
    """

    distribution: str
    parameters: dict[str, float]
    aic: float
    ks_pvalue: float
    n_samples: int


@dataclass
class ReturnPeriodResult:
    """Return levels and confidence bounds for a set of return periods.

    Attributes:
        distribution: Distribution used to estimate the return levels.
        return_periods: Return periods ``T`` in years.
        return_levels: Estimated magnitude for each return period.
        lower_bound: Lower confidence bound for each return level.
        upper_bound: Upper confidence bound for each return level.
        confidence_level: Two-sided confidence level of the bounds (e.g. 0.95).
        fit: The underlying distribution fit the levels were derived from.
    """

    distribution: str
    return_periods: list[float]
    return_levels: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    confidence_level: float
    fit: DistributionFit
    units: str = "value"
    extra: dict[str, float] = field(default_factory=dict)
