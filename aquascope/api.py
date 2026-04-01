"""High-level convenience API for common hydrological analyses.

Provides one-liner functions wrapping AquaScope's lower-level modules.
Designed for quick analyses in Jupyter notebooks and scripts.

Examples
--------
>>> from aquascope.api import flood_analysis, baseflow_analysis
>>> result = flood_analysis(daily_discharge, method="gev", return_periods=[10, 50, 100])
>>> bf = baseflow_analysis(daily_discharge, method="eckhardt")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from aquascope.analysis.changepoint import ChangePointResult
    from aquascope.analysis.copulas import CopulaResult
    from aquascope.hydrology.baseflow import BaseflowResult
    from aquascope.hydrology.flood_frequency import FloodFreqResult
    from aquascope.hydrology.flow_duration import FDCResult
    from aquascope.hydrology.signatures import SignatureReport
    from aquascope.models.bayesian import PosteriorResult
    from aquascope.reporting.builder import ReportBuilder

logger = logging.getLogger(__name__)

_FLOOD_METHODS = {"gev", "lp3", "gumbel", "gev_lmoments", "gpd"}
_BASEFLOW_METHODS = {"lyne_hollick", "eckhardt"}
_CHANGEPOINT_METHODS = {"pelt", "cusum", "binary_segmentation", "pettitt"}
_COPULA_FAMILIES = {"auto", "gaussian", "clayton", "gumbel", "frank"}
_ENSEMBLE_METHODS = {"weighted", "stacking", "adaptive"}


# -- Hydrology convenience functions ----------------------------------------

def flood_analysis(
    discharge: pd.Series,
    method: str = "gev",
    return_periods: list[int] | None = None,
    ci_level: float = 0.90,
    regional_skew: float | None = None,
    **kwargs,
) -> FloodFreqResult:
    """Fit a flood-frequency distribution and estimate return-period quantiles.

    Parameters
    ----------
    discharge:
        Daily (or sub-daily) discharge time-series with a
        :class:`~pandas.DatetimeIndex`.
    method:
        Distribution to fit.  One of ``"gev"``, ``"lp3"``, ``"gumbel"``,
        ``"gev_lmoments"``, or ``"gpd"``.
    return_periods:
        List of return periods (years) for which to estimate quantiles.
        Defaults to ``[2, 5, 10, 25, 50, 100]``.
    ci_level:
        Confidence level for bootstrap confidence intervals (GEV only).
    regional_skew:
        Optional regional skew coefficient (LP3 only).
    **kwargs:
        Forwarded to the underlying fitting function.

    Returns
    -------
    FloodFreqResult
        Fitted distribution parameters, quantile estimates, and
        confidence intervals.

    Raises
    ------
    ValueError
        If *method* is not one of the supported methods.
    """
    from aquascope.hydrology.flood_frequency import (
        fit_gev,
        fit_gev_lmoments,
        fit_gpd,
        fit_gumbel,
        fit_lp3,
    )

    if method not in _FLOOD_METHODS:
        msg = f"Unknown flood-frequency method {method!r}. Choose from {sorted(_FLOOD_METHODS)}."
        raise ValueError(msg)

    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100]

    if method == "gev":
        return fit_gev(discharge, return_periods=return_periods, ci_level=ci_level, **kwargs)
    if method == "lp3":
        lp3_kwargs: dict = {"return_periods": return_periods, **kwargs}
        if regional_skew is not None:
            lp3_kwargs["regional_skew"] = regional_skew
        return fit_lp3(discharge, **lp3_kwargs)
    if method == "gumbel":
        return fit_gumbel(discharge, return_periods=return_periods, **kwargs)
    if method == "gev_lmoments":
        return fit_gev_lmoments(discharge, return_periods=return_periods, **kwargs)
    # gpd
    return fit_gpd(discharge, return_periods=return_periods, **kwargs)


def baseflow_analysis(
    discharge: pd.Series,
    method: str = "lyne_hollick",
    **kwargs,
) -> BaseflowResult:
    """Separate baseflow from quickflow using a digital filter.

    Parameters
    ----------
    discharge:
        Daily discharge time-series.
    method:
        ``"lyne_hollick"`` (recursive filter, default) or ``"eckhardt"``
        (two-parameter filter).
    **kwargs:
        Forwarded to the filter function (e.g. ``alpha``, ``n_passes``).

    Returns
    -------
    BaseflowResult
        DataFrame of total / baseflow / quickflow plus BFI.

    Raises
    ------
    ValueError
        If *method* is not supported.
    """
    from aquascope.hydrology.baseflow import eckhardt, lyne_hollick

    if method not in _BASEFLOW_METHODS:
        msg = f"Unknown baseflow method {method!r}. Choose from {sorted(_BASEFLOW_METHODS)}."
        raise ValueError(msg)

    if method == "lyne_hollick":
        return lyne_hollick(discharge, **kwargs)
    return eckhardt(discharge, **kwargs)


def flow_duration(discharge: pd.Series, **kwargs) -> FDCResult:
    """Compute a flow-duration curve.

    Parameters
    ----------
    discharge:
        Daily discharge time-series.
    **kwargs:
        Forwarded to :func:`~aquascope.hydrology.flow_duration.flow_duration_curve`
        (e.g. ``percentiles``).

    Returns
    -------
    FDCResult
        Exceedance probabilities, sorted discharges, and percentile values.
    """
    from aquascope.hydrology.flow_duration import flow_duration_curve

    return flow_duration_curve(discharge, **kwargs)


def compute_all_signatures(discharge: pd.Series, **kwargs) -> SignatureReport:
    """Compute a comprehensive set of hydrological signatures.

    Parameters
    ----------
    discharge:
        Daily discharge time-series.
    **kwargs:
        Forwarded to :func:`~aquascope.hydrology.signatures.compute_signatures`
        (e.g. ``precipitation``, ``area_km2``).

    Returns
    -------
    SignatureReport
        Dataclass containing ~20 hydrological signature values.
    """
    from aquascope.hydrology.signatures import compute_signatures

    return compute_signatures(discharge, **kwargs)


# -- Statistical-analysis convenience functions ------------------------------

def detect_changepoints(
    series: np.ndarray | pd.Series,
    method: str = "pelt",
    **kwargs,
) -> ChangePointResult:
    """Detect abrupt shifts in a time-series.

    Parameters
    ----------
    series:
        One-dimensional numeric data.
    method:
        Detection algorithm.  One of ``"pelt"``, ``"cusum"``,
        ``"binary_segmentation"``, or ``"pettitt"``.
    **kwargs:
        Forwarded to the detection function.

    Returns
    -------
    ChangePointResult
        Detected change-points, segment summaries, and test statistics.

    Raises
    ------
    ValueError
        If *method* is not supported.
    """
    from aquascope.analysis.changepoint import (
        ChangePointResult as CPResult,
    )
    from aquascope.analysis.changepoint import (
        binary_segmentation,
        cusum,
        pelt,
        pettitt_test,
    )

    if method not in _CHANGEPOINT_METHODS:
        msg = f"Unknown changepoint method {method!r}. Choose from {sorted(_CHANGEPOINT_METHODS)}."
        raise ValueError(msg)

    if method == "pelt":
        return pelt(series, **kwargs)
    if method == "cusum":
        return cusum(series, **kwargs)
    if method == "binary_segmentation":
        return binary_segmentation(series, **kwargs)
    # pettitt — returns ChangePoint | None; wrap into ChangePointResult
    cp = pettitt_test(series, **kwargs)
    changepoints = [cp] if cp is not None else []
    return CPResult(
        changepoints=changepoints,
        n_changepoints=len(changepoints),
        method="pettitt",
        penalty=None,
        segments=[],
    )


def fit_copula(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    family: str = "auto",
    **kwargs,
) -> CopulaResult:
    """Fit a bivariate copula to paired observations.

    Parameters
    ----------
    x, y:
        Paired data arrays of equal length.
    family:
        Copula family — ``"auto"`` (best AIC), ``"gaussian"``,
        ``"clayton"``, ``"gumbel"``, or ``"frank"``.
    **kwargs:
        Forwarded to :func:`~aquascope.analysis.copulas.fit_copula`.

    Returns
    -------
    CopulaResult
        Fitted copula parameters, dependence measures, and AIC.

    Raises
    ------
    ValueError
        If *family* is not supported.
    """
    from aquascope.analysis.copulas import (
        compare_copulas,
        to_pseudo_observations,
    )
    from aquascope.analysis.copulas import (
        fit_copula as _fit_copula,
    )

    if family not in _COPULA_FAMILIES:
        msg = f"Unknown copula family {family!r}. Choose from {sorted(_COPULA_FAMILIES)}."
        raise ValueError(msg)

    u, v = to_pseudo_observations(x, y)

    if family == "auto":
        results = compare_copulas(u, v)
        return results[0]  # best AIC

    return _fit_copula(u, v, family=family, **kwargs)


# -- Model convenience functions ---------------------------------------------

def bayesian_regression(
    X: np.ndarray | pd.DataFrame,  # noqa: N803
    y: np.ndarray | pd.Series,
    degree: int = 1,
    **kwargs,
) -> PosteriorResult:
    """Fit a Bayesian linear or polynomial regression.

    Parameters
    ----------
    X:
        Feature matrix (degree=1) or 1-D predictor (degree>1).
    y:
        Response variable.
    degree:
        Polynomial degree.  ``1`` uses
        :class:`~aquascope.models.bayesian.BayesianLinearRegression`;
        higher values use
        :class:`~aquascope.models.bayesian.BayesianPolynomialRegression`.
    **kwargs:
        Forwarded to the model constructor (e.g. ``prior_precision``).

    Returns
    -------
    PosteriorResult
        Posterior summaries, credible intervals, and diagnostics.
    """
    from aquascope.models.bayesian import BayesianLinearRegression, BayesianPolynomialRegression

    if degree == 1:
        model = BayesianLinearRegression(**kwargs)
        return model.fit(X, y)
    model = BayesianPolynomialRegression(degree=degree, **kwargs)
    return model.fit(X, y)


def ensemble_forecast(
    models: list[tuple[str, object]],
    X_train: pd.DataFrame,  # noqa: N803
    y_train: pd.Series,
    X_test: pd.DataFrame,  # noqa: N803
    method: str = "stacking",
    **kwargs,
) -> np.ndarray:
    """Train an ensemble of models and return predictions on *X_test*.

    Parameters
    ----------
    models:
        List of ``(name, estimator)`` tuples.
    X_train:
        Training features.
    y_train:
        Training target.
    X_test:
        Test features.
    method:
        Ensemble strategy — ``"weighted"``, ``"stacking"``, or
        ``"adaptive"``.
    **kwargs:
        Forwarded to the ensemble constructor.

    Returns
    -------
    numpy.ndarray
        Predicted values for *X_test*.

    Raises
    ------
    ValueError
        If *method* is not supported.
    """
    from aquascope.models.ensemble import AdaptiveEnsemble, StackingEnsemble, WeightedEnsemble

    if method not in _ENSEMBLE_METHODS:
        msg = f"Unknown ensemble method {method!r}. Choose from {sorted(_ENSEMBLE_METHODS)}."
        raise ValueError(msg)

    if method == "weighted":
        ens = WeightedEnsemble(models, **kwargs)
        ens.fit(X_train, y_train)
        return ens.predict(X_test).predictions
    if method == "stacking":
        ens = StackingEnsemble(models, **kwargs)
        ens.fit(X_train, y_train)
        return ens.predict(X_test).predictions
    # adaptive
    ens = AdaptiveEnsemble(models, **kwargs)
    ens.fit(X_train, y_train)
    return ens.update_and_predict(X_test).predictions


# -- Reporting convenience function ------------------------------------------

def generate_report(title: str, **kwargs) -> ReportBuilder:
    """Create a pre-configured :class:`~aquascope.reporting.builder.ReportBuilder`.

    Parameters
    ----------
    title:
        Report title.
    **kwargs:
        Forwarded to :class:`~aquascope.reporting.builder.ReportBuilder`
        (e.g. ``author``, ``description``).

    Returns
    -------
    ReportBuilder
        A builder instance ready for method-chaining.
    """
    from aquascope.reporting.builder import ReportBuilder

    return ReportBuilder(title, **kwargs)


# -- Groundwater convenience functions ----------------------------------------

def groundwater_analysis(
    levels: pd.Series,
    method: str = "trend",
    **kwargs,
) -> dict:
    """Run a groundwater analysis on a water-level time series.

    Parameters
    ----------
    levels:
        Water-level measurements with :class:`~pandas.DatetimeIndex`.
    method:
        Analysis type — ``"trend"`` (Mann-Kendall trend detection),
        ``"recession"`` (aquifer recession), ``"seasonal"`` (decomposition),
        or ``"hydrograph"`` (full hydrograph summary).
    **kwargs:
        Forwarded to the underlying function.

    Returns
    -------
    dict
        Result dataclass from the chosen analysis, accessed as a dict
        or the original dataclass depending on method.

    Raises
    ------
    ValueError
        If *method* is not supported.
    """
    _gw_methods = {"trend", "recession", "seasonal", "hydrograph"}
    if method not in _gw_methods:
        msg = f"Unknown groundwater method {method!r}. Choose from {sorted(_gw_methods)}."
        raise ValueError(msg)

    if method == "trend":
        from aquascope.groundwater.wells import trend_detection
        return trend_detection(levels, **kwargs)
    if method == "recession":
        from aquascope.groundwater.wells import recession_analysis
        return recession_analysis(levels, **kwargs)
    if method == "seasonal":
        from aquascope.groundwater.wells import seasonal_decomposition
        return seasonal_decomposition(levels, **kwargs)
    # hydrograph
    from aquascope.groundwater.wells import well_hydrograph
    return well_hydrograph(levels, **kwargs)


# -- Climate convenience functions -------------------------------------------

def climate_downscale(
    obs: pd.Series,
    gcm_hist: pd.Series,
    gcm_future: pd.Series,
    method: str = "quantile_mapping",
    **kwargs,
) -> pd.Series:
    """Downscale a GCM projection using statistical bias correction.

    Parameters
    ----------
    obs:
        Observed station data.
    gcm_hist:
        GCM historical simulation (overlapping period with *obs*).
    gcm_future:
        GCM future projection to downscale.
    method:
        Downscaling method — ``"delta"`` (additive/multiplicative),
        ``"quantile_mapping"``, or ``"qdm"`` (Quantile Delta Mapping).
    **kwargs:
        Forwarded to the underlying function.

    Returns
    -------
    pandas.Series
        Bias-corrected future projection.

    Raises
    ------
    ValueError
        If *method* is not supported.
    """
    _ds_methods = {"delta", "quantile_mapping", "qdm"}
    if method not in _ds_methods:
        msg = f"Unknown downscaling method {method!r}. Choose from {sorted(_ds_methods)}."
        raise ValueError(msg)

    if method == "delta":
        from aquascope.climate.downscaling import delta_method
        return delta_method(obs, gcm_hist, gcm_future, **kwargs)
    if method == "quantile_mapping":
        from aquascope.climate.downscaling import quantile_mapping
        return quantile_mapping(obs, gcm_hist, gcm_future, **kwargs)
    # qdm
    from aquascope.climate.downscaling import quantile_delta_mapping
    return quantile_delta_mapping(obs, gcm_hist, gcm_future, **kwargs)


def climate_indices(
    precip: pd.Series | None = None,
    temperature: pd.Series | None = None,
    pet: pd.Series | None = None,
    index: str = "cdd",
    **kwargs,
) -> object:
    """Compute a climate index from meteorological data.

    Parameters
    ----------
    precip:
        Precipitation series (required for ``"cdd"``, ``"cwd"``, ``"pci"``,
        ``"drought"``, ``"pdsi"``).
    temperature:
        Maximum temperature series (required for ``"heat_wave"``).
    pet:
        Potential evapotranspiration (required for ``"pdsi"``, ``"aridity"``).
    index:
        Index to compute — ``"cdd"`` (consecutive dry days),
        ``"cwd"`` (consecutive wet days), ``"pci"`` (precipitation
        concentration), ``"heat_wave"``, ``"aridity"``, ``"pdsi"``.
    **kwargs:
        Forwarded to the underlying function.

    Returns
    -------
    object
        Result dataclass or value from the chosen index.

    Raises
    ------
    ValueError
        If *index* is not supported or required input is missing.
    """
    _idx_names = {"cdd", "cwd", "pci", "heat_wave", "aridity", "pdsi"}
    if index not in _idx_names:
        msg = f"Unknown climate index {index!r}. Choose from {sorted(_idx_names)}."
        raise ValueError(msg)

    if index == "cdd":
        from aquascope.climate.indices import consecutive_dry_days
        if precip is None:
            raise ValueError("'precip' is required for CDD index.")
        return consecutive_dry_days(precip, **kwargs)
    if index == "cwd":
        from aquascope.climate.indices import consecutive_wet_days
        if precip is None:
            raise ValueError("'precip' is required for CWD index.")
        return consecutive_wet_days(precip, **kwargs)
    if index == "pci":
        from aquascope.climate.indices import precipitation_concentration_index
        if precip is None:
            raise ValueError("'precip' is required for PCI index.")
        return precipitation_concentration_index(precip, **kwargs)
    if index == "heat_wave":
        from aquascope.climate.indices import heat_wave_index
        if temperature is None:
            raise ValueError("'temperature' is required for heat_wave index.")
        return heat_wave_index(temperature, **kwargs)
    if index == "aridity":
        from aquascope.climate.indices import aridity_index
        if precip is None or pet is None:
            raise ValueError("'precip' and 'pet' are required for aridity index.")
        return aridity_index(float(precip.sum()), float(pet.sum()), **kwargs)
    # pdsi
    from aquascope.climate.indices import palmer_drought_severity_index
    if precip is None or pet is None:
        raise ValueError("'precip' and 'pet' are required for PDSI.")
    return palmer_drought_severity_index(precip, pet, **kwargs)
