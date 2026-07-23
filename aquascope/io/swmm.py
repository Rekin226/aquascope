"""EPA SWMM format support for AquaScope.

Provides writers for the US EPA Storm Water Management Model (SWMM) input
file sections:

- **SWMM timeseries** — ``[TIMESERIES]`` section entries.
- **SWMM rainfall** — ``.dat`` rain-gauge data files.

References
----------
- SWMM User's Manual: https://www.epa.gov/water-research/storm-water-management-model-swmm
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── SWMM Timeseries ─────────────────────────────────────────────────────────
def write_swmm_timeseries(
    df: pd.DataFrame,
    name: str,
    path: str | Path,
    time_col: str = "timestamp",
    value_col: str = "value",
) -> None:
    """Write an EPA SWMM timeseries input section.

    Output format::

        [TIMESERIES]
        ;;Name          Date        Time        Value
        name            MM/DD/YYYY  HH:MM       value

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing timestamps and values.
    name : str
        Timeseries identifier used by SWMM.
    path : str | Path
        Destination file path.
    time_col : str
        Column with timestamps.
    value_col : str
        Column with numeric values.
    """
    path = Path(path)
    lines: list[str] = [
        "[TIMESERIES]",
        ";;Name          Date        Time        Value",
    ]

    for _, row in df.iterrows():
        ts = pd.Timestamp(row[time_col])
        date_str = ts.strftime("%m/%d/%Y")
        time_str = ts.strftime("%H:%M")
        value = float(row[value_col])
        lines.append(f"{name:<16s}{date_str:<12s}{time_str:<12s}{value:.4f}")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    logger.debug("write_swmm_timeseries: wrote %d rows to %s", len(df), path)


# ── SWMM Rainfall .dat ──────────────────────────────────────────────────────
def write_swmm_rainfall(
    df: pd.DataFrame,
    gauge_name: str,
    path: str | Path,
    time_col: str = "timestamp",
    value_col: str = "value",
) -> None:
    """Write an EPA SWMM rainfall data file (``.dat`` format).

    Output format::

        StationID Year Month Day Hour Minute Value

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing timestamps and rainfall values.
    gauge_name : str
        Rain-gauge / station identifier.
    path : str | Path
        Destination file path.
    time_col : str
        Column with timestamps.
    value_col : str
        Column with rainfall values.
    """
    path = Path(path)
    lines: list[str] = []

    for _, row in df.iterrows():
        ts = pd.Timestamp(row[time_col])
        value = float(row[value_col])
        lines.append(
            f"{gauge_name} {ts.year:4d} {ts.month:2d} {ts.day:2d} "
            f"{ts.hour:2d} {ts.minute:2d} {value:.4f}"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    logger.debug("write_swmm_rainfall: wrote %d rows to %s", len(df), path)
