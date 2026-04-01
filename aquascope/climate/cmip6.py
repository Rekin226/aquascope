"""
CMIP6 data processing and analysis tools.

Provides utilities for working with already-downloaded CMIP6 climate model
output (as pandas DataFrames or numpy arrays).  Does **not** fetch data from
ESGF or CDS — callers are expected to supply pre-downloaded datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Period look-up table ────────────────────────────────────────────────
_PERIODS: dict[str, tuple[int, int]] = {
    "historical": (1950, 2014),
    "near_future": (2015, 2050),
    "mid_century": (2041, 2070),
    "end_century": (2071, 2100),
}


# ── Enums ───────────────────────────────────────────────────────────────
class SSP(str, Enum):
    """Shared Socio-economic Pathways used in CMIP6 projections."""

    SSP126 = "ssp126"
    SSP245 = "ssp245"
    SSP370 = "ssp370"
    SSP585 = "ssp585"

    @property
    def description(self) -> str:
        """Human-readable description of each scenario."""
        return {
            "ssp126": "Sustainability — low challenges to mitigation and adaptation",
            "ssp245": "Middle of the road — medium challenges",
            "ssp370": "Regional rivalry — high challenges to mitigation",
            "ssp585": "Fossil-fuelled development — high challenges to mitigation and adaptation",
        }[self.value]


# ── Result dataclasses ──────────────────────────────────────────────────
@dataclass
class EnsembleStats:
    """Multi-model ensemble statistics.

    Attributes
    ----------
    mean : pd.Series
        Ensemble mean across models.
    median : pd.Series
        Ensemble median across models.
    std : pd.Series
        Ensemble standard deviation.
    p10 : pd.Series
        10th percentile across models.
    p90 : pd.Series
        90th percentile across models.
    n_models : int
        Number of models in the ensemble.
    """

    mean: pd.Series
    median: pd.Series
    std: pd.Series
    p10: pd.Series
    p90: pd.Series
    n_models: int


@dataclass
class TrendResult:
    """Linear trend analysis result.

    Attributes
    ----------
    slope : float
        Slope of the regression line (unit per time-step).
    intercept : float
        y-intercept of the regression line.
    p_value : float
        Two-sided p-value for the slope.
    ci_lower : float
        Lower bound of the 95 % confidence interval for the slope.
    ci_upper : float
        Upper bound of the 95 % confidence interval for the slope.
    unit_per_decade : float
        Slope scaled to change per decade.
    """

    slope: float
    intercept: float
    p_value: float
    ci_lower: float
    ci_upper: float
    unit_per_decade: float


# ── Processor ───────────────────────────────────────────────────────────
class CMIP6Processor:
    """Process and analyse CMIP6 climate-model output.

    Parameters
    ----------
    variable : str
        Climate variable name (e.g. ``"tas"``, ``"pr"``, ``"tasmax"``).
    frequency : str
        Temporal frequency of the input data (default ``"monthly"``).
    """

    def __init__(self, variable: str, frequency: str = "monthly") -> None:
        self.variable = variable
        self.frequency = frequency

    # ── ensemble ────────────────────────────────────────────────────────
    def compute_ensemble_stats(self, models: dict[str, pd.DataFrame]) -> EnsembleStats:
        """Compute multi-model ensemble statistics.

        Parameters
        ----------
        models : dict[str, pd.DataFrame]
            Mapping of *model_name → DataFrame*.  Each DataFrame must have a
            ``DatetimeIndex`` and a single value column.

        Returns
        -------
        EnsembleStats
            Mean, median, std, 10th / 90th percentiles across models.

        Raises
        ------
        ValueError
            If *models* is empty.
        """
        if not models:
            raise ValueError("models dict must not be empty")

        combined = pd.DataFrame(
            {name: df.iloc[:, 0] for name, df in models.items()}
        )
        return EnsembleStats(
            mean=combined.mean(axis=1),
            median=combined.median(axis=1),
            std=combined.std(axis=1),
            p10=combined.quantile(0.1, axis=1),
            p90=combined.quantile(0.9, axis=1),
            n_models=len(models),
        )

    # ── time slicing ────────────────────────────────────────────────────
    def time_slice(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        """Extract a named or custom time period from a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with a ``DatetimeIndex``.
        period : str
            One of ``"historical"``, ``"near_future"``, ``"mid_century"``,
            ``"end_century"``, or a custom ``"YYYY-YYYY"`` string.

        Returns
        -------
        pd.DataFrame
            Subset of *data* within the requested period.

        Raises
        ------
        ValueError
            If *period* is not recognised and cannot be parsed.
        """
        if period in _PERIODS:
            start, end = _PERIODS[period]
        else:
            try:
                parts = period.split("-")
                start, end = int(parts[0]), int(parts[1])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Unrecognised period {period!r}. Use one of "
                    f"{list(_PERIODS)} or 'YYYY-YYYY'."
                ) from exc

        mask = (data.index.year >= start) & (data.index.year <= end)
        return data.loc[mask]

    # ── anomaly ─────────────────────────────────────────────────────────
    def compute_anomaly(
        self,
        data: pd.DataFrame,
        baseline_period: tuple[int, int] = (1981, 2010),
    ) -> pd.DataFrame:
        """Compute anomalies relative to a baseline climatology.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with a ``DatetimeIndex`` and one or more value columns.
        baseline_period : tuple[int, int]
            Start and end years of the baseline (inclusive).

        Returns
        -------
        pd.DataFrame
            Anomalies (data minus baseline mean).

        Raises
        ------
        ValueError
            If no data falls within the baseline period.
        """
        mask = (data.index.year >= baseline_period[0]) & (data.index.year <= baseline_period[1])
        baseline = data.loc[mask]
        if baseline.empty:
            raise ValueError(f"No data within baseline period {baseline_period}")
        return data - baseline.mean()

    # ── annual cycle ────────────────────────────────────────────────────
    def annual_cycle(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the 12-month climatological mean (annual cycle).

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with a ``DatetimeIndex``.

        Returns
        -------
        pd.DataFrame
            Monthly climatology indexed 1–12.
        """
        return data.groupby(data.index.month).mean()

    # ── trend analysis ──────────────────────────────────────────────────
    def trend_analysis(self, data: pd.DataFrame) -> TrendResult:
        """Fit a linear trend to the first column of *data*.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with a ``DatetimeIndex`` and at least one value column.

        Returns
        -------
        TrendResult
            Slope, intercept, p-value, 95 % CI, and decadal rate of change.

        Raises
        ------
        ValueError
            If *data* has fewer than 3 rows.
        """
        series = data.iloc[:, 0].dropna()
        if len(series) < 3:
            raise ValueError("Need at least 3 data points for trend analysis")

        # Convert datetime index to fractional years for regression
        t0 = series.index[0]
        x = np.array([(t - t0).total_seconds() / (365.25 * 86400) for t in series.index])
        y = series.values.astype(float)

        result = stats.linregress(x, y)
        n = len(x)
        se = result.stderr
        t_crit = stats.t.ppf(0.975, df=n - 2)

        return TrendResult(
            slope=float(result.slope),
            intercept=float(result.intercept),
            p_value=float(result.pvalue),
            ci_lower=float(result.slope - t_crit * se),
            ci_upper=float(result.slope + t_crit * se),
            unit_per_decade=float(result.slope * 10),
        )
