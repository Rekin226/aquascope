"""GRACE satellite groundwater storage estimation.

Derives groundwater storage anomalies from GRACE Total Water Storage (TWS)
observations by subtracting soil moisture and surface water components.
Provides trend analysis, anomaly detection, and depletion rate calculation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class GWSResult:
    """Result of groundwater storage computation.

    Attributes
    ----------
    gws:
        Groundwater storage anomaly series (mm).
    mean_gws:
        Mean groundwater storage over the period.
    std_gws:
        Standard deviation of groundwater storage.
    """

    gws: pd.Series
    mean_gws: float
    std_gws: float


@dataclass
class TrendResult:
    """Result of trend analysis on a groundwater storage series.

    Attributes
    ----------
    slope:
        Linear trend slope (mm/month or mm per time step).
    intercept:
        Linear trend intercept.
    r_squared:
        Coefficient of determination of the linear fit.
    p_value:
        P-value of the linear trend significance test.
    trend_line:
        Fitted trend values aligned with the input index.
    seasonal_amplitude:
        Amplitude of the dominant seasonal cycle (mm), or None.
    seasonal_phase:
        Phase of the dominant seasonal cycle (months), or None.
    """

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    trend_line: pd.Series
    seasonal_amplitude: float | None = None
    seasonal_phase: float | None = None


@dataclass
class GWSAnomaly:
    """A detected groundwater storage anomaly.

    Attributes
    ----------
    date:
        Date/time of the anomaly.
    value:
        GWS value at the anomaly.
    z_score:
        Number of standard deviations from the mean.
    anomaly_type:
        ``"depletion"`` or ``"surplus"``.
    """

    date: pd.Timestamp
    value: float
    z_score: float
    anomaly_type: str


@dataclass
class DepletionResult:
    """Result of groundwater depletion rate analysis.

    Attributes
    ----------
    rate_mm_per_year:
        Annual depletion rate in mm/year (negative = depletion).
    rate_km3_per_year:
        Annual depletion rate in km³/year, or None if area not provided.
    total_change_mm:
        Total storage change over the period (mm).
    period_years:
        Length of the analysis period in years.
    confidence_interval:
        95 % confidence interval for the rate (mm/year).
    """

    rate_mm_per_year: float
    rate_km3_per_year: float | None = None
    total_change_mm: float = 0.0
    period_years: float = 0.0
    confidence_interval: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


class GRACEProcessor:
    """Processor for GRACE satellite groundwater storage estimation.

    Uses the water balance approach: GWS = TWS − SM − SW to isolate
    the groundwater component from GRACE Total Water Storage data.

    Parameters
    ----------
    area_km2:
        Basin or region area in km² for volumetric conversions.
        If not provided, km³/year rates will be ``None``.
    """

    def __init__(self, area_km2: float | None = None) -> None:
        self.area_km2 = area_km2

    def compute_gws(
        self,
        tws: pd.Series,
        soil_moisture: pd.Series,
        surface_water: pd.Series,
    ) -> GWSResult:
        """Compute groundwater storage anomaly from GRACE components.

        Parameters
        ----------
        tws:
            Total Water Storage anomaly (mm), with DatetimeIndex.
        soil_moisture:
            Soil moisture anomaly (mm), aligned with *tws*.
        surface_water:
            Surface water anomaly (mm), aligned with *tws*.

        Returns
        -------
        GWSResult
            Groundwater storage anomaly series and summary statistics.

        Raises
        ------
        ValueError
            If input series are empty or have mismatched lengths.
        """
        if len(tws) == 0:
            raise ValueError("Input TWS series is empty.")
        if not (len(tws) == len(soil_moisture) == len(surface_water)):
            raise ValueError("All input series must have the same length.")

        gws = tws - soil_moisture - surface_water
        gws.name = "gws_mm"

        logger.info("Computed GWS for %d time steps; mean=%.2f mm", len(gws), gws.mean())
        return GWSResult(gws=gws, mean_gws=float(gws.mean()), std_gws=float(gws.std()))

    def trend_analysis(self, gws_series: pd.Series) -> TrendResult:
        """Perform linear trend and seasonal decomposition on a GWS series.

        Parameters
        ----------
        gws_series:
            Groundwater storage anomaly series with DatetimeIndex.

        Returns
        -------
        TrendResult
            Linear trend parameters and optional seasonal characteristics.

        Raises
        ------
        ValueError
            If series has fewer than 3 data points.
        """
        if len(gws_series) < 3:
            raise ValueError("Need at least 3 data points for trend analysis.")

        x = np.arange(len(gws_series), dtype=float)
        y = gws_series.values.astype(float)

        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        trend_line = pd.Series(slope * x + intercept, index=gws_series.index, name="trend")

        # Seasonal decomposition via FFT on detrended residuals
        seasonal_amplitude: float | None = None
        seasonal_phase: float | None = None
        if len(gws_series) >= 24:
            detrended = y - (slope * x + intercept)
            fft_vals = np.fft.rfft(detrended)
            magnitudes = np.abs(fft_vals)
            # Skip DC component (index 0)
            if len(magnitudes) > 1:
                peak_idx = int(np.argmax(magnitudes[1:])) + 1
                seasonal_amplitude = float(2.0 * magnitudes[peak_idx] / len(detrended))
                seasonal_phase = float(-np.angle(fft_vals[peak_idx]) / (2.0 * np.pi) * (len(detrended) / peak_idx))

        logger.info("Trend: slope=%.4f mm/step, R²=%.4f, p=%.4e", slope, r_value**2, p_value)
        return TrendResult(
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_value**2),
            p_value=float(p_value),
            trend_line=trend_line,
            seasonal_amplitude=seasonal_amplitude,
            seasonal_phase=seasonal_phase,
        )

    def anomaly_detection(
        self,
        gws_series: pd.Series,
        threshold_sigma: float = 2.0,
    ) -> list[GWSAnomaly]:
        """Detect anomalous groundwater storage values.

        Flags values exceeding *threshold_sigma* standard deviations
        from the mean as either ``"depletion"`` (negative) or
        ``"surplus"`` (positive) anomalies.

        Parameters
        ----------
        gws_series:
            Groundwater storage anomaly series with DatetimeIndex.
        threshold_sigma:
            Number of standard deviations to use as the threshold.

        Returns
        -------
        list[GWSAnomaly]
            Detected anomalies sorted chronologically.
        """
        mean = float(gws_series.mean())
        std = float(gws_series.std())
        if std == 0:
            return []

        anomalies: list[GWSAnomaly] = []
        for date, value in gws_series.items():
            z = (float(value) - mean) / std
            if abs(z) >= threshold_sigma:
                anomaly_type = "depletion" if z < 0 else "surplus"
                anomalies.append(GWSAnomaly(
                    date=pd.Timestamp(date),
                    value=float(value),
                    z_score=z,
                    anomaly_type=anomaly_type,
                ))

        logger.info("Detected %d anomalies (threshold=%.1fσ)", len(anomalies), threshold_sigma)
        return anomalies

    def depletion_rate(self, gws_series: pd.Series) -> DepletionResult:
        """Estimate annual groundwater depletion rate.

        Parameters
        ----------
        gws_series:
            Groundwater storage anomaly series with DatetimeIndex.

        Returns
        -------
        DepletionResult
            Annual depletion rate and confidence interval.

        Raises
        ------
        ValueError
            If series has fewer than 2 data points.
        """
        if len(gws_series) < 2:
            raise ValueError("Need at least 2 data points for depletion rate.")

        idx = gws_series.index
        days = (idx - idx[0]).total_seconds() / 86400.0
        years = np.array(days) / 365.25
        y = gws_series.values.astype(float)

        slope, intercept, r_value, p_value, std_err = stats.linregress(years, y)

        period_years = float(years[-1] - years[0])
        total_change = float(slope * period_years)

        # 95 % confidence interval on the slope
        t_crit = stats.t.ppf(0.975, df=max(len(y) - 2, 1))
        ci_low = float(slope - t_crit * std_err)
        ci_high = float(slope + t_crit * std_err)

        rate_km3: float | None = None
        if self.area_km2 is not None:
            # 1 mm over area_km2 = area_km2 * 1e6 m² * 1e-3 m = area_km2 * 1e3 m³ = area_km2 * 1e-6 km³
            rate_km3 = slope * self.area_km2 * 1e-6

        logger.info("Depletion rate: %.2f mm/year (p=%.4e)", slope, p_value)
        return DepletionResult(
            rate_mm_per_year=float(slope),
            rate_km3_per_year=rate_km3,
            total_change_mm=total_change,
            period_years=period_years,
            confidence_interval=(ci_low, ci_high),
        )
