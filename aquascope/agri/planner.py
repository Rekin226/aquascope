"""High-level irrigation planning workflows.

Wraps AquaScope's FAO-56 functions into a single planning interface that can
consume daily ET and precipitation time series from saved files or live
Open-Meteo queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from aquascope.agri.crop_water import DEFAULT_STAGE_LENGTHS, irrigation_schedule
from aquascope.agri.water_balance import SoilProperties, SoilWaterBalance
from aquascope.collectors.openmeteo import OpenMeteoCollector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_DATE_COLUMNS = ("date", "sample_datetime", "reading_datetime", "datetime", "time")


@dataclass
class IrrigationPlan:
    """Structured result from a crop irrigation planning workflow."""

    crop: str
    planting_date: date
    season_end_date: date
    efficiency: float
    total_eto_mm: float
    total_precipitation_mm: float
    total_effective_rain_mm: float
    total_etc_mm: float
    total_net_irrigation_mm: float
    total_gross_irrigation_mm: float
    total_applied_irrigation_mm: float
    irrigation_trigger_days: int
    schedule: pd.DataFrame
    balance: pd.DataFrame

    def to_dict(self) -> dict[str, object]:
        """Convert the plan to a JSON-serializable dictionary."""
        schedule = self.schedule.copy()
        balance = self.balance.copy()
        if "date" in schedule.columns:
            schedule["date"] = schedule["date"].astype(str)
        if "date" in balance.columns:
            balance["date"] = balance["date"].astype(str)

        return {
            "crop": self.crop,
            "planting_date": self.planting_date.isoformat(),
            "season_end_date": self.season_end_date.isoformat(),
            "efficiency": self.efficiency,
            "total_eto_mm": self.total_eto_mm,
            "total_precipitation_mm": self.total_precipitation_mm,
            "total_effective_rain_mm": self.total_effective_rain_mm,
            "total_etc_mm": self.total_etc_mm,
            "total_net_irrigation_mm": self.total_net_irrigation_mm,
            "total_gross_irrigation_mm": self.total_gross_irrigation_mm,
            "total_applied_irrigation_mm": self.total_applied_irrigation_mm,
            "irrigation_trigger_days": self.irrigation_trigger_days,
            "schedule": schedule.to_dict("records"),
            "balance": balance.to_dict("records"),
        }


def _coerce_daily_series(series: pd.Series, name: str | None = None) -> pd.Series:
    """Normalize a series to a sorted daily float series."""
    import pandas as pd

    values = pd.to_numeric(series, errors="coerce")
    index = pd.to_datetime(series.index, errors="coerce")
    daily = pd.Series(values.to_numpy(), index=index, name=name or series.name)
    daily = daily[~daily.index.isna()].dropna()
    if daily.empty:
        return daily
    daily.index = daily.index.normalize()
    return daily.groupby(daily.index).mean().sort_index()


def series_from_dataframe(
    df: pd.DataFrame,
    *,
    value_columns: tuple[str, ...],
    parameter: str | None = None,
) -> pd.Series:
    """Extract a daily time series from a saved AquaScope-style data file.

    Supports both long-form saved collector output
    (``sample_datetime``, ``parameter``, ``value``) and simpler tables such as
    ``date`` + ``eto_mm``.
    """
    import pandas as pd

    data = df.copy()
    if parameter is not None and "parameter" in data.columns:
        data = data[data["parameter"] == parameter].copy()
        if data.empty:
            raise ValueError(f"No rows found for parameter {parameter!r}.")

    date_col = next((col for col in _DATE_COLUMNS if col in data.columns), None)
    if date_col is None:
        if isinstance(data.index, pd.DatetimeIndex):
            dates = pd.to_datetime(data.index, errors="coerce")
        else:
            raise ValueError(f"Could not find a date column. Expected one of {_DATE_COLUMNS}.")
    else:
        dates = pd.to_datetime(data[date_col], errors="coerce")

    value_col = next((col for col in value_columns if col in data.columns), None)
    if value_col is None and parameter is not None and parameter in data.columns:
        value_col = parameter
    if value_col is None:
        raise ValueError(f"Could not find a value column. Tried {value_columns}.")

    series = pd.Series(data[value_col].to_numpy(), index=dates, name=value_col)
    daily = _coerce_daily_series(series, name=value_col)
    if daily.empty:
        raise ValueError(f"No valid daily values found in column {value_col!r}.")
    return daily


def fetch_openmeteo_plan_inputs(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> tuple[pd.Series, pd.Series]:
    """Fetch daily ET0 and precipitation from Open-Meteo for planning."""
    import pandas as pd

    collector = OpenMeteoCollector(mode="weather")
    raw = collector.fetch_raw(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        daily=["et0_fao_evapotranspiration", "precipitation_sum"],
    )

    daily = raw.get("daily", {}) if isinstance(raw, dict) else {}
    time_index = pd.to_datetime(daily.get("time", []), errors="coerce")

    eto = _coerce_daily_series(
        pd.Series(daily.get("et0_fao_evapotranspiration", []), index=time_index, name="eto_mm"),
        name="eto_mm",
    )
    precip = _coerce_daily_series(
        pd.Series(daily.get("precipitation_sum", []), index=time_index, name="precipitation_sum"),
        name="precipitation_sum",
    )

    if eto.empty or precip.empty:
        raise ValueError("Open-Meteo did not return ET0 and precipitation values for the requested period.")

    return eto, precip


def default_season_end_date(crop: str, planting_date: date, stage_lengths: dict[str, int] | None = None) -> date:
    """Return the default end date for a crop season."""
    from datetime import timedelta

    lengths = stage_lengths or DEFAULT_STAGE_LENGTHS.get(crop)
    if lengths is None:
        raise ValueError(f"No default stage lengths for {crop!r}. Provide an explicit end date.")
    return planting_date + timedelta(days=sum(lengths.values()) - 1)


def plan_irrigation(
    crop: str,
    planting_date: date,
    eto_series: pd.Series,
    precip_series: pd.Series,
    soil: SoilProperties,
    *,
    efficiency: float = 0.7,
    depletion_fraction: float = 0.5,
    initial_depletion: float = 0.0,
    stage_lengths: dict[str, int] | None = None,
) -> IrrigationPlan:
    """Build a full irrigation plan from daily ET and precipitation series."""
    import pandas as pd

    eto_daily = _coerce_daily_series(eto_series, name="eto_mm")
    precip_daily = _coerce_daily_series(precip_series, name="precipitation_sum")
    if eto_daily.empty:
        raise ValueError("ET0 series must contain at least one valid daily value.")
    if precip_daily.empty:
        raise ValueError("Precipitation series must contain at least one valid daily value.")

    schedule = irrigation_schedule(
        eto_daily,
        precip_daily,
        crop,
        planting_date,
        efficiency=efficiency,
        stage_lengths=stage_lengths,
    )
    schedule = schedule.copy()
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")

    etc_series = pd.Series(
        pd.to_numeric(schedule["etc"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(schedule["date"]),
        name="etc_mm",
    )
    precip_for_balance = precip_daily.reindex(etc_series.index, fill_value=0.0)

    balance_model = SoilWaterBalance(
        soil,
        depletion_fraction=depletion_fraction,
        initial_depletion=initial_depletion,
    )
    balance = balance_model.auto_irrigate(etc_series, precip_for_balance, efficiency=efficiency)
    balance = balance.copy()
    balance["date"] = pd.to_datetime(balance["date"], errors="coerce")

    season_end = pd.Timestamp(schedule["date"].max()).date()
    trigger_days = int(balance["irrigation_trigger"].fillna(False).astype(bool).sum())
    applied_irrigation = round(float(pd.to_numeric(balance["irrigation_mm"], errors="coerce").fillna(0.0).sum()), 2)

    return IrrigationPlan(
        crop=crop,
        planting_date=planting_date,
        season_end_date=season_end,
        efficiency=efficiency,
        total_eto_mm=round(float(pd.to_numeric(schedule["eto"], errors="coerce").sum()), 2),
        total_precipitation_mm=round(float(precip_for_balance.sum()), 2),
        total_effective_rain_mm=round(float(pd.to_numeric(schedule["effective_rain"], errors="coerce").sum()), 2),
        total_etc_mm=round(float(pd.to_numeric(schedule["etc"], errors="coerce").sum()), 2),
        total_net_irrigation_mm=round(float(pd.to_numeric(schedule["net_irrigation"], errors="coerce").sum()), 2),
        total_gross_irrigation_mm=round(float(pd.to_numeric(schedule["gross_irrigation"], errors="coerce").sum()), 2),
        total_applied_irrigation_mm=applied_irrigation,
        irrigation_trigger_days=trigger_days,
        schedule=schedule,
        balance=balance,
    )
