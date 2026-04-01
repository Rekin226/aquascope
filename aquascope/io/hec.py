"""HEC-RAS / HEC-HMS format support for AquaScope.

Provides writers for the US Army Corps of Engineers Hydrologic Engineering
Center (HEC) data formats:

- **HEC-DSS CSV** — A CSV representation importable by HEC-DSSVue (the
  actual DSS binary format requires a proprietary library).
- **HEC-RAS unsteady flow** — Simplified ``.u##`` files for boundary
  conditions.

References
----------
- HEC-DSSVue: https://www.hec.usace.army.mil/software/hec-dssvue/
- HEC-RAS:    https://www.hec.usace.army.mil/software/hec-ras/
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── HEC-DSS path part indices ────────────────────────────────────────────────
# Pathname format: /A-Watershed/B-Location/C-Parameter/D-StartDate/E-Timestep/F-Source/
_HEC_TIMESTEP_MAP: dict[str, str] = {
    "T": "1MIN",
    "5T": "5MIN",
    "15T": "15MIN",
    "h": "1HOUR",
    "D": "1DAY",
    "MS": "1MON",
}


# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class HECDSSRecord:
    """Simplified HEC-DSS-like record.

    Parameters
    ----------
    pathname : str
        DSS pathname in ``/A/B/C/D/E/F/`` format.
    timestamps : list[datetime]
        Time stamps for each value.
    values : list[float]
        Numeric data values.
    unit : str
        Unit of measurement.
    data_type : str
        One of ``"INST-VAL"``, ``"PER-AVER"``, ``"PER-CUM"``.
    """

    pathname: str
    timestamps: list[datetime] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    unit: str = "unknown"
    data_type: str = "INST-VAL"


# ── HEC-DSS CSV ─────────────────────────────────────────────────────────────
def write_hec_dss_csv(records: list[HECDSSRecord], path: str | Path) -> None:
    """Write HEC-DSS compatible CSV.

    The output can be imported into HEC-DSSVue for further use.

    Parameters
    ----------
    records : list[HECDSSRecord]
        Records to write.
    path : str | Path
        Destination CSV file path.
    """
    path = Path(path)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pathname", "timestamp", "value", "unit", "type"])
        for rec in records:
            for ts, val in zip(rec.timestamps, rec.values):
                writer.writerow([rec.pathname, ts.strftime("%Y-%m-%d %H:%M:%S"), val, rec.unit, rec.data_type])

    logger.debug("write_hec_dss_csv: wrote %d records to %s", len(records), path)


# ── DataFrame → HEC-DSS ─────────────────────────────────────────────────────
def _infer_timestep(timestamps: pd.Series) -> str:
    """Best-effort timestep string from a timestamp series."""
    if len(timestamps) < 2:
        return "IR-YEAR"
    ts = pd.to_datetime(timestamps).sort_values()
    median_delta = ts.diff().dropna().median()
    seconds = median_delta.total_seconds()

    if seconds <= 60:
        return "1MIN"
    if seconds <= 300:
        return "5MIN"
    if seconds <= 900:
        return "15MIN"
    if seconds <= 3600:
        return "1HOUR"
    if seconds <= 86400:
        return "1DAY"
    return "IR-YEAR"


def dataframe_to_hec_format(
    df: pd.DataFrame,
    station_col: str = "station_id",
    param_col: str = "parameter",
    value_col: str = "value",
    time_col: str = "timestamp",
    watershed: str = "UNKNOWN",
    location: str = "UNKNOWN",
) -> list[HECDSSRecord]:
    """Convert a DataFrame to HEC-DSS record format.

    Pathname convention::

        /watershed/location/parameter/start_date/timestep/source/

    Groups by *station_col* + *param_col*.

    Parameters
    ----------
    df : pd.DataFrame
        Input tabular data.
    station_col : str
        Column with station identifiers (used as *location* if *location* is
        ``"UNKNOWN"``).
    param_col : str
        Column with parameter names.
    value_col : str
        Column with numeric values.
    time_col : str
        Column with timestamps.
    watershed : str
        ``A``-part of the pathname.
    location : str
        ``B``-part override.  When ``"UNKNOWN"`` the station identifier is
        used instead.

    Returns
    -------
    list[HECDSSRecord]
        One record per station–parameter combination.
    """
    records: list[HECDSSRecord] = []
    grouped = df.groupby([station_col, param_col])

    for (sid, param), group in grouped:
        loc = str(sid) if location == "UNKNOWN" else location
        ts_sorted = group.sort_values(time_col)
        timestamps = [pd.Timestamp(t).to_pydatetime() for t in ts_sorted[time_col]]

        start_date = timestamps[0].strftime("%d%b%Y").upper() if timestamps else "01JAN2000"
        timestep = _infer_timestep(ts_sorted[time_col])

        pathname = f"/{watershed}/{loc}/{param}/{start_date}/{timestep}/AQUASCOPE/"

        unit_val = "unknown"
        if "unit" in group.columns:
            unit_val = str(group["unit"].iloc[0])

        records.append(
            HECDSSRecord(
                pathname=pathname,
                timestamps=timestamps,
                values=list(ts_sorted[value_col].astype(float)),
                unit=unit_val,
                data_type="INST-VAL",
            )
        )

    logger.debug("dataframe_to_hec_format: created %d records", len(records))
    return records


# ── HEC-RAS unsteady flow ───────────────────────────────────────────────────
def write_hec_ras_flow(
    discharge: np.ndarray | pd.Series,
    timestamps: pd.DatetimeIndex,
    river_name: str,
    reach_name: str,
    station: str,
    path: str | Path,
) -> None:
    """Write a simplified HEC-RAS unsteady flow file (``.u##`` format).

    Contains a flow hydrograph that can be used as an upstream boundary
    condition.

    Parameters
    ----------
    discharge : np.ndarray | pd.Series
        Discharge values (m³/s or cfs).
    timestamps : pd.DatetimeIndex
        Corresponding timestamps.
    river_name : str
        Name of the river.
    reach_name : str
        Name of the reach.
    station : str
        Station identifier (river station).
    path : str | Path
        Destination file path.
    """
    path = Path(path)
    discharge_arr = np.asarray(discharge, dtype=float)

    lines: list[str] = [
        "Flow Title=AquaScope Generated Flow Data",
        "Program Version=6.00",
        f"Number of Profiles= {len(discharge_arr)}",
        "",
        f"River Rch & Prof={river_name},{reach_name},{station}",
        "",
        "Boundary Location=",
        f"  River={river_name}",
        f"  Reach={reach_name}",
        f"  RS={station}",
        "  Interval=",
        f"  Flow Hydrograph= {len(discharge_arr)}",
    ]

    for ts, q in zip(timestamps, discharge_arr):
        lines.append(f"  {ts.strftime('%d%b%Y %H%M').upper()}  {q:.2f}")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    logger.debug("write_hec_ras_flow: wrote %d points to %s", len(discharge_arr), path)
