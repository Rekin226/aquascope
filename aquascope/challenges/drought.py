"""
Drought challenge handler — SPI-based monitoring, forecasting, and water balance.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DroughtChallenge:
    """High-level interface for drought monitoring and forecasting.

    Uses SPI (Standardised Precipitation Index) for monitoring and Prophet
    for precipitation forecasting.

    Parameters
    ----------
    lat, lon : float
        Site coordinates (for labelling and optional data loading).
    name : str | None
        Human-readable site label.

    Example
    -------
    >>> challenge = DroughtChallenge(lat=15.0, lon=0.0, name="Sahel")
    >>> challenge.load_dataframe(precip_df, et_df=et_df)
    >>> spi = challenge.compute_spi()
    >>> print(challenge.current_status())
    """

    DROUGHT_THRESHOLDS = {
        "normal": (float("-inf"), -1.0),
        "moderate": (-1.0, -1.5),
        "severe": (-1.5, -2.0),
        "extreme": (-2.0, float("inf")),
    }

    def __init__(self, lat: float, lon: float, name: str | None = None):
        self.lat = lat
        self.lon = lon
        self.name = name or f"({lat}, {lon})"
        self._precip_df: pd.DataFrame | None = None
        self._et_df: pd.DataFrame | None = None
        self._soil_df: pd.DataFrame | None = None
        self._spi_df: pd.DataFrame | None = None

    def load_dataframe(
        self,
        precip_df: pd.DataFrame,
        et_df: pd.DataFrame | None = None,
        soil_df: pd.DataFrame | None = None,
    ) -> DroughtChallenge:
        """Load pre-fetched precipitation (and optional ET / soil moisture) data.

        Parameters
        ----------
        precip_df : pd.DataFrame
            Precipitation with DatetimeIndex and ``value`` column.
        et_df : pd.DataFrame | None
            Evapotranspiration data (same schema).
        soil_df : pd.DataFrame | None
            Soil moisture data (same schema).
        """
        self._precip_df = precip_df
        self._et_df = et_df
        self._soil_df = soil_df
        logger.info(
            "DroughtChallenge: loaded %d precipitation records for %s",
            len(precip_df) if precip_df is not None else 0,
            self.name,
        )
        return self

    def compute_spi(self, timescales: list[int] | None = None) -> pd.DataFrame:
        """Compute SPI at multiple timescales.

        Parameters
        ----------
        timescales : list[int] | None
            SPI accumulation windows in months (default ``[1, 3, 6, 12]``).

        Returns
        -------
        pd.DataFrame
            SPI values with ``drought_category`` column.
        """
        if self._precip_df is None:
            raise RuntimeError("Call load_dataframe() first")

        from aquascope.models.statistical import SPIModel

        timescales = timescales or [1, 3, 6, 12]
        model = SPIModel(timescales=timescales)
        model.fit(self._precip_df)
        self._spi_df = model.predict()
        return self._spi_df

    def current_status(self) -> dict:
        """Return the latest drought status across all SPI timescales.

        Returns
        -------
        dict
            Per-timescale values + ``overall`` drought category.
        """
        if self._spi_df is None:
            self.compute_spi()

        from aquascope.models.statistical import SPIModel

        spi_cols = [c for c in self._spi_df.columns if c.startswith("SPI")]
        valid_rows = self._spi_df[spi_cols].dropna(how="all")
        if valid_rows.empty:
            return {"location": self.name, "overall": "INSUFFICIENT DATA"}

        latest = valid_rows.iloc[-1]

        status: dict = {
            "location": self.name,
            "date": str(valid_rows.index[-1].date()),
        }
        for col in spi_cols:
            val = latest[col]
            if pd.isna(val):
                continue
            val = float(val)
            status[col] = {"value": round(val, 3), "category": SPIModel._categorise(val)}

        spi3 = status.get("SPI_3", {}).get("value", 0)
        if spi3 <= -2.0:
            status["overall"] = "EXTREME DROUGHT"
        elif spi3 <= -1.5:
            status["overall"] = "SEVERE DROUGHT"
        elif spi3 <= -1.0:
            status["overall"] = "MODERATE DROUGHT"
        else:
            status["overall"] = "NORMAL / WET"

        return status

    def forecast_precipitation(self, days: int = 90) -> pd.DataFrame:
        """Forecast future precipitation using Prophet.

        Returns
        -------
        pd.DataFrame
            Forecast with ``yhat``, ``yhat_lower``, ``yhat_upper``.
        """
        if self._precip_df is None:
            raise RuntimeError("Call load_dataframe() first")

        from aquascope.models.statistical import ProphetModel

        model = ProphetModel(yearly_seasonality=True)
        model.fit(self._precip_df)
        return model.predict(horizon=days)

    def water_balance(self) -> pd.DataFrame:
        """Compute monthly P − ET water balance.

        Returns
        -------
        pd.DataFrame
            Columns ``precipitation_mm``, ``et_mm``, ``water_balance_mm``,
            ``surplus_deficit``.

        Raises
        ------
        RuntimeError
            If ET data is not loaded.
        """
        if self._precip_df is None or self._et_df is None:
            raise RuntimeError("Load data with et_df to compute water balance")

        p = self._precip_df["value"].resample("ME").sum().rename("precipitation_mm")
        et = self._et_df["value"].resample("ME").sum().rename("et_mm")
        balance = pd.concat([p, et], axis=1).dropna()
        balance["water_balance_mm"] = balance["precipitation_mm"] - balance["et_mm"]
        balance["surplus_deficit"] = balance["water_balance_mm"].apply(lambda x: "surplus" if x >= 0 else "deficit")
        return balance

    @property
    def precipitation_data(self) -> pd.DataFrame | None:
        """Return loaded precipitation DataFrame."""
        return self._precip_df

    @property
    def spi(self) -> pd.DataFrame | None:
        """Return computed SPI DataFrame."""
        return self._spi_df
