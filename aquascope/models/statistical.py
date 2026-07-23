"""
Statistical models for hydrological forecasting.

- ``ProphetModel`` — Facebook Prophet for seasonal time series
- ``ARIMAModel`` — ARIMA / SARIMA
- ``SPIModel`` — Standardised Precipitation Index (WMO drought monitoring)
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from aquascope.models.base import BaseHydroModel
from aquascope.utils.imports import require

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class ProphetModel(BaseHydroModel):
    """Facebook Prophet for hydrological time-series forecasting.

    Handles seasonality, missing data, and produces uncertainty intervals.
    Ideal for streamflow, precipitation, and drought monitoring.

    Parameters
    ----------
    yearly_seasonality : bool
        Enable yearly seasonality component.
    interval_width : float
        Width of the uncertainty interval (0–1).
    changepoint_prior_scale : float
        Flexibility of the trend changepoints.
    """

    MODEL_ID = "prophet"
    SUPPORTS_UNCERTAINTY = True
    SUPPORTS_MULTIVARIATE = False

    def __init__(
        self,
        target_variable: str = "value",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        interval_width: float = 0.95,
        changepoint_prior_scale: float = 0.05,
    ):
        super().__init__(target_variable)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.interval_width = interval_width
        self.changepoint_prior_scale = changepoint_prior_scale
        self._model = None

    def fit(self, df: pd.DataFrame, **kwargs) -> ProphetModel:
        """Fit Prophet on a time-series DataFrame."""
        require("prophet", feature="Prophet forecasting")
        from prophet import Prophet

        series = self._prepare_series(df)
        prophet_df = pd.DataFrame({"ds": series.index, "y": series.values})

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(prophet_df)

        self._is_fitted = True
        logger.info("ProphetModel fitted on %d data points", len(series))
        return self

    def predict(self, horizon: int = 30, **kwargs) -> pd.DataFrame:
        """Generate a *horizon*-day forecast with uncertainty bounds."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        future = self._model.make_future_dataframe(periods=horizon, freq="D")
        forecast = self._model.predict(future)

        result = forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result = result.rename(columns={"ds": "datetime"}).set_index("datetime")
        return result

    def predict_components(self) -> pd.DataFrame:
        """Return decomposed trend and seasonality components."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        future = self._model.make_future_dataframe(periods=0, freq="D")
        return self._model.predict(future)


class ARIMAModel(BaseHydroModel):
    """ARIMA / SARIMA model for hydrological time series.

    Auto-selects ``(p, d, q)`` order when *auto_order* is ``True``.

    Parameters
    ----------
    order : tuple
        ARIMA order ``(p, d, q)``.
    seasonal_order : tuple
        Seasonal order ``(P, D, Q, s)``.
    auto_order : bool
        Use heuristic order selection.
    """

    MODEL_ID = "arima"
    SUPPORTS_UNCERTAINTY = True
    SUPPORTS_MULTIVARIATE = False

    def __init__(
        self,
        target_variable: str = "value",
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 12),
        auto_order: bool = True,
    ):
        super().__init__(target_variable)
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_order = auto_order
        self._model_fit = None

    def fit(self, df: pd.DataFrame, **kwargs) -> ARIMAModel:
        """Fit SARIMA on a time-series DataFrame."""
        require("statsmodels", feature="ARIMA forecasting")
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        series = self._prepare_series(df)

        if self.auto_order:
            order, seasonal_order = self._auto_select_order(series)
        else:
            order, seasonal_order = self.order, self.seasonal_order

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._model_fit = model.fit(disp=False)
        self.order = order
        self.seasonal_order = seasonal_order
        self._is_fitted = True
        logger.info("ARIMAModel fitted: order=%s seasonal=%s", order, seasonal_order)
        return self

    def predict(self, horizon: int = 14, **kwargs) -> pd.DataFrame:
        """Generate a *horizon*-day forecast with confidence intervals."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        forecast = self._model_fit.get_forecast(steps=horizon)
        mean = forecast.predicted_mean
        ci = forecast.conf_int()

        result = pd.DataFrame(
            {
                "yhat": mean.values,
                "yhat_lower": ci.iloc[:, 0].values,
                "yhat_upper": ci.iloc[:, 1].values,
            },
            index=self._future_dates(horizon),
        )
        result.index.name = "datetime"
        return result

    @staticmethod
    def _auto_select_order(series: pd.Series) -> tuple:
        """Simple heuristic order selection."""
        if len(series) >= 24:
            return (1, 1, 1), (1, 1, 1, 12)
        elif len(series) >= 12:
            return (1, 1, 1), (0, 0, 0, 0)
        else:
            return (1, 1, 0), (0, 0, 0, 0)


class SPIModel(BaseHydroModel):
    """Standardised Precipitation Index (WMO drought monitoring standard).

    Computes SPI at multiple timescales (1, 3, 6, 12 months) using
    gamma-distribution fitting per calendar month.

    SPI interpretation:
      ≥ 2.0  : Extremely wet
      1.5–1.99: Very wet
      1.0–1.49: Moderately wet
      −0.99–0.99: Near normal
      −1.0–−1.49: Moderately dry
      −1.5–−1.99: Severely dry
      ≤ −2.0 : Extremely dry

    Parameters
    ----------
    timescales : list[int]
        SPI accumulation windows in months.
    """

    MODEL_ID = "spi_drought_index"
    SUPPORTS_UNCERTAINTY = False
    SUPPORTS_MULTIVARIATE = False

    def __init__(self, target_variable: str = "value", timescales: list[int] | None = None):
        super().__init__(target_variable)
        self.timescales = timescales or [1, 3, 6, 12]
        self._series: pd.Series | None = None

    def fit(self, df: pd.DataFrame, **kwargs) -> SPIModel:
        """Fit the SPI model on precipitation data."""
        self._series = self._prepare_series(df)
        if self._series.index.freqstr not in ("MS", "M", "ME"):
            self._series = self._series.resample("ME").sum()
        self._is_fitted = True
        logger.info("SPIModel fitted: %d monthly values, timescales=%s", len(self._series), self.timescales)
        return self

    def predict(self, horizon: int = 0, **kwargs) -> pd.DataFrame:
        """Compute SPI values for all configured timescales.

        Returns
        -------
        pd.DataFrame
            Columns ``SPI_<scale>`` for each timescale, plus ``drought_category``
            and ``yhat`` (alias for the first timescale).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        result = pd.DataFrame(index=self._series.index)
        for scale in self.timescales:
            result[f"SPI_{scale}"] = self._compute_spi(self._series, scale)

        if "SPI_3" in result.columns:
            result["drought_category"] = result["SPI_3"].apply(self._categorise)

        primary = f"SPI_{self.timescales[0]}"
        result["yhat"] = result[primary]
        result.index.name = "datetime"
        return result

    @staticmethod
    def _compute_spi(series: pd.Series, scale: int) -> pd.Series:
        """Compute SPI at a given timescale using gamma-distribution fitting."""
        rolling = series.rolling(window=scale, min_periods=scale).sum()
        spi = pd.Series(index=rolling.index, dtype=float)

        for month in range(1, 13):
            mask = rolling.index.month == month
            vals = rolling[mask].dropna().values

            if len(vals) < 4:
                continue

            try:
                shape, loc, scale_param = stats.gamma.fit(vals[vals > 0], floc=0)
                p_zero = (vals == 0).sum() / len(vals)
                probs = p_zero + (1 - p_zero) * stats.gamma.cdf(
                    rolling[mask].values, shape, loc=loc, scale=scale_param
                )
                probs = np.clip(probs, 0.001, 0.999)
                spi[mask] = stats.norm.ppf(probs)
            except Exception:
                pass

        return spi.round(3)

    @staticmethod
    def _categorise(spi_val: float) -> str:
        """Map an SPI value to a WMO drought category label."""
        if pd.isna(spi_val):
            return "unknown"
        if spi_val >= 2.0:
            return "extremely_wet"
        elif spi_val >= 1.5:
            return "very_wet"
        elif spi_val >= 1.0:
            return "moderately_wet"
        elif spi_val >= -1.0:
            return "normal"
        elif spi_val >= -1.5:
            return "moderately_dry"
        elif spi_val >= -2.0:
            return "severely_dry"
        else:
            return "extremely_dry"

    def current_status(self) -> dict:
        """Return the latest drought status across all timescales."""
        result = self.predict()
        latest = result.dropna().iloc[-1]
        status = {}
        for col in [c for c in result.columns if c.startswith("SPI")]:
            status[col] = {"value": round(latest[col], 3), "category": self._categorise(latest[col])}
        return status
