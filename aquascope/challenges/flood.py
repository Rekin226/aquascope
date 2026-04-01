"""
Flood challenge handler — forecasting, risk assessment, and return-period estimation.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FloodChallenge:
    """High-level interface for flood forecasting and risk assessment.

    Combines streamflow / discharge data with statistical models to produce
    flood forecasts, return-period estimates, and early-warning signals.

    Parameters
    ----------
    lat, lon : float | None
        Coordinates for the site of interest (for global data sources).
    usgs_site_id : str | None
        USGS gauge number (for US sites).
    name : str | None
        Human-readable site label.

    Example
    -------
    >>> challenge = FloodChallenge(lat=13.5, lon=2.1, name="Niger River")
    >>> challenge.load_dataframe(discharge_df)
    >>> challenge.fit(model="prophet")
    >>> forecast = challenge.forecast(days=7)
    >>> risk = challenge.assess_risk()
    """

    def __init__(
        self,
        lat: float | None = None,
        lon: float | None = None,
        usgs_site_id: str | None = None,
        name: str | None = None,
    ):
        self.lat = lat
        self.lon = lon
        self.usgs_site_id = usgs_site_id
        self.name = name or (f"({lat}, {lon})" if lat else usgs_site_id or "Unknown")
        self._discharge_df: pd.DataFrame | None = None
        self._precip_df: pd.DataFrame | None = None
        self._model = None
        self._return_periods: dict | None = None

    def load_dataframe(
        self,
        discharge_df: pd.DataFrame,
        precip_df: pd.DataFrame | None = None,
    ) -> FloodChallenge:
        """Load pre-fetched discharge (and optional precipitation) data.

        Parameters
        ----------
        discharge_df : pd.DataFrame
            Must have a DatetimeIndex and a ``value`` column.
        precip_df : pd.DataFrame | None
            Optional precipitation data (same schema).
        """
        self._discharge_df = discharge_df
        self._precip_df = precip_df
        logger.info("FloodChallenge: loaded %d discharge records", len(discharge_df))
        return self

    def fit(self, model: str = "prophet") -> FloodChallenge:
        """Fit a forecasting model to the discharge data.

        Parameters
        ----------
        model : str
            Model identifier (e.g. ``prophet``, ``arima``, ``random_forest``).
        """
        if self._discharge_df is None:
            raise RuntimeError("Call load_dataframe() first")

        from aquascope.models import get_model_map

        model_map = get_model_map()
        cls = model_map.get(model)
        if cls is None:
            raise ValueError(f"Unknown model: {model}. Available: {list(model_map)}")

        self._model = cls()
        self._model.fit(self._discharge_df)
        self._compute_return_periods()
        return self

    def forecast(self, days: int = 7) -> pd.DataFrame:
        """Generate a discharge forecast.

        Returns
        -------
        pd.DataFrame
            Columns ``yhat``, ``yhat_lower``, ``yhat_upper`` with DatetimeIndex.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first")
        return self._model.predict(horizon=days)

    def assess_risk(self, forecast: pd.DataFrame | None = None) -> dict:
        """Assess flood risk by comparing forecast peaks to historical percentiles.

        Returns
        -------
        dict
            Keys: ``risk_level``, ``description``, ``peak_forecast``,
            ``threshold_75/90/95/99``, ``return_periods``.
        """
        if forecast is None:
            forecast = self.forecast(days=7)

        if self._discharge_df is None or forecast is None:
            return {"risk": "unknown", "reason": "insufficient data"}

        historical = self._discharge_df["value"]
        q_75 = historical.quantile(0.75)
        q_90 = historical.quantile(0.90)
        q_95 = historical.quantile(0.95)
        q_99 = historical.quantile(0.99)

        peak_forecast = forecast["yhat"].max()

        if peak_forecast >= q_99:
            risk, description = "EXTREME", "Forecast exceeds 99th percentile — extreme flood risk"
        elif peak_forecast >= q_95:
            risk, description = "HIGH", "Forecast exceeds 95th percentile — high flood risk"
        elif peak_forecast >= q_90:
            risk, description = "MODERATE", "Forecast exceeds 90th percentile — moderate flood risk"
        elif peak_forecast >= q_75:
            risk, description = "LOW", "Above average flows expected"
        else:
            risk, description = "NORMAL", "No significant flood risk detected"

        return {
            "location": self.name,
            "risk_level": risk,
            "description": description,
            "peak_forecast": round(float(peak_forecast), 2),
            "threshold_75": round(float(q_75), 2),
            "threshold_90": round(float(q_90), 2),
            "threshold_95": round(float(q_95), 2),
            "threshold_99": round(float(q_99), 2),
            "return_periods": self._return_periods or {},
            "forecast_days": len(forecast),
        }

    def _compute_return_periods(self) -> None:
        """Estimate discharge at 2/5/10/20/50/100-year return periods via GEV."""
        if self._discharge_df is None:
            return
        try:
            from scipy.stats import genextreme

            annual_max = self._discharge_df["value"].resample("YE").max().dropna()
            if len(annual_max) < 5:
                return
            c, loc, scale = genextreme.fit(annual_max.values)
            rps: dict[str, float] = {}
            for rp in [2, 5, 10, 20, 50, 100]:
                rps[f"{rp}yr"] = round(float(genextreme.ppf(1 - 1 / rp, c, loc, scale)), 2)
            self._return_periods = rps
        except Exception as e:
            logger.debug("Return-period computation failed: %s", e)

    @property
    def discharge_data(self) -> pd.DataFrame | None:
        """Return the loaded discharge DataFrame."""
        return self._discharge_df

    @property
    def precipitation_data(self) -> pd.DataFrame | None:
        """Return the loaded precipitation DataFrame."""
        return self._precip_df
