"""
Water-quality challenge handler — anomaly detection, WHO guidelines, and trend analysis.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

WHO_GUIDELINES: dict[str, tuple[float, float]] = {
    "ph": (6.5, 8.5),
    "dissolved_oxygen": (5.0, float("inf")),
    "turbidity": (0, 5.0),
    "nitrate": (0, 50.0),
    "e_coli": (0, 0),
    "arsenic": (0, 0.01),
    "lead": (0, 0.01),
    "mercury": (0, 0.001),
}


class WaterQualityChallenge:
    """High-level interface for water-quality monitoring and analysis.

    Detects contamination events, trends in pollutants, and WHO/EU
    guideline exceedances.

    Parameters
    ----------
    site_id : str
        Monitoring site identifier.
    name : str | None
        Human-readable site label.

    Example
    -------
    >>> wq = WaterQualityChallenge(site_id="USGS-01589440")
    >>> wq.load_dataframes({"ph": ph_df, "nitrate": nitrate_df})
    >>> exceedances = wq.check_who_guidelines()
    >>> anomalies  = wq.detect_anomalies()
    """

    def __init__(self, site_id: str, name: str | None = None):
        self.site_id = site_id
        self.name = name or site_id
        self._data: dict[str, pd.DataFrame] = {}

    def load_dataframes(self, data: dict[str, pd.DataFrame]) -> WaterQualityChallenge:
        """Load pre-fetched variable DataFrames.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Mapping of variable name → DataFrame (DatetimeIndex + ``value`` column).
        """
        for var, df in data.items():
            if not df.empty:
                self._data[var] = df
                logger.info("Loaded %s: %d records", var, len(df))
        return self

    def detect_anomalies(self, contamination: float = 0.05) -> pd.DataFrame:
        """Run Isolation Forest anomaly detection across all loaded variables.

        Parameters
        ----------
        contamination : float
            Expected proportion of anomalous observations.

        Returns
        -------
        pd.DataFrame
            Anomalous observations with ``variable`` and ``anomaly_score`` columns.
        """
        from aquascope.models.ml import IsolationForestModel

        all_anomalies: list[pd.DataFrame] = []
        for var, df in self._data.items():
            if df.empty or len(df) < 20:
                continue
            try:
                model = IsolationForestModel(contamination=contamination)
                model.fit(df)
                result = model.get_anomalies()
                result["variable"] = var
                result["site_id"] = self.site_id
                all_anomalies.append(result)
            except Exception as e:
                logger.warning("Anomaly detection failed for %s: %s", var, e)

        if not all_anomalies:
            return pd.DataFrame()
        return pd.concat(all_anomalies).sort_values("anomaly_score", ascending=False)

    def check_who_guidelines(self) -> pd.DataFrame:
        """Compare each variable against WHO/EU drinking-water guidelines.

        Returns
        -------
        pd.DataFrame
            One row per variable with exceedance counts and status.
        """
        rows: list[dict] = []
        for var, df in self._data.items():
            if df.empty:
                continue
            guideline = WHO_GUIDELINES.get(var)
            if guideline is None:
                continue

            values = df["value"].dropna()
            low, high = guideline
            n_exceed = int(((values < low) | (values > high)).sum())
            pct_exceed = 100 * n_exceed / len(values) if len(values) > 0 else 0

            rows.append({
                "variable": var,
                "n_measurements": len(values),
                "mean": round(float(values.mean()), 4),
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
                "guideline_low": low,
                "guideline_high": high if high != float("inf") else "N/A",
                "n_exceedances": n_exceed,
                "pct_exceedances": round(pct_exceed, 1),
                "status": "EXCEEDANCE" if n_exceed > 0 else "OK",
            })
        return pd.DataFrame(rows)

    def trend_analysis(self, variable: str) -> dict:
        """Run a Mann-Kendall trend test on monthly means.

        Parameters
        ----------
        variable : str
            Variable name to analyse.

        Returns
        -------
        dict
            Keys: ``trend``, ``mann_kendall_tau``, ``p_value``, ``significant``, ``n_months``.
        """
        if variable not in self._data:
            raise ValueError(f"Variable '{variable}' not loaded")

        try:
            from scipy.stats import kendalltau
        except ImportError:
            raise ImportError("scipy required: pip install scipy")

        series = self._data[variable]["value"].resample("ME").mean().dropna()
        if len(series) < 12:
            return {"variable": variable, "trend": "insufficient data"}

        x = list(range(len(series)))
        tau, p_value = kendalltau(x, series.values)

        if p_value < 0.05:
            trend = "INCREASING" if tau > 0 else "DECREASING"
        else:
            trend = "NO SIGNIFICANT TREND"

        return {
            "variable": variable,
            "trend": trend,
            "mann_kendall_tau": round(float(tau), 4),
            "p_value": round(float(p_value), 4),
            "significant": p_value < 0.05,
            "n_months": len(series),
        }

    def summary(self) -> pd.DataFrame:
        """Return descriptive statistics for every loaded variable.

        Returns
        -------
        pd.DataFrame
            One row per variable with count, mean, std, percentiles.
        """
        rows: list[dict] = []
        for var, df in self._data.items():
            if df.empty:
                continue
            v = df["value"].dropna()
            rows.append({
                "variable": var,
                "n_records": len(v),
                "mean": round(float(v.mean()), 4),
                "std": round(float(v.std()), 4),
                "min": round(float(v.min()), 4),
                "p25": round(float(v.quantile(0.25)), 4),
                "median": round(float(v.median()), 4),
                "p75": round(float(v.quantile(0.75)), 4),
                "max": round(float(v.max()), 4),
                "unit": df["unit"].iloc[0] if "unit" in df.columns else "unknown",
            })
        return pd.DataFrame(rows)

    @property
    def variables(self) -> list[str]:
        """Return names of loaded variables."""
        return list(self._data.keys())

    @property
    def data(self) -> dict[str, pd.DataFrame]:
        """Return the internal variable → DataFrame mapping."""
        return self._data
