"""Engineering export bundles: hand a record to the tools practitioners already use.

None of HEC-HMS, HEC-RAS, HEC-SSP, SWMM, MODFLOW 6, Delft-FEWS or Raven runs in a
browser, but each reads plain inputs. Every exporter here is a plain function from a
pandas series (a gauge record, or the main series of a Studio study) to file contents,
so the CLI (``aquascope export --to``), the MCP tool, the Explorer's "Export for..."
menu and the Studio bundle all write the same bytes.

Formats, and the documentation each one was written against:

- **HEC-DSS** time series through HEC's own ``hecdss`` package (MIT,
  https://github.com/HydrologicEngineeringCenter/hec-dss-python), an optional extra
  (``pip install aquascope[engineering]``). ``hecdss`` ships its native library for
  Linux and Windows only, so on other systems, and always in the browser, the record is
  written as the CSV layout ``hecdss`` itself reads and writes
  (``RegularTimeSeries.read_csv``; columns ``Date/Time, Value, Units, Type, Path``),
  which turns into a DSS record with one call on any machine where ``hecdss`` loads.
  Period data (daily means, rainfall totals) are stamped at the end of the period
  (``PER-AVER`` / ``PER-CUM``), the end-of-period convention described in HEC's
  hec-python-library (https://hec-python-library.readthedocs.io/en/latest/classes/Duration.html).
- **HEC-SSP**: the annual peaks as a comma-delimited text file in the layout of the
  HEC-SSP Data Importer's "Text File" option (date and time as ``ddMMMyyyy hhmm`` in the
  first column, the value in the second), plus the DSS record, plus the Bulletin 17C
  settings and AquaScope's own Bulletin 17C result to compare against. HEC-SSP User's
  Manual, Data Importer:
  https://www.hec.usace.army.mil/confluence/sspdocs/sspum/latest/using-the-hec-ssp-data-importer/developing-a-new-data-set
- **HEC-HMS**: observed flow and precipitation gage records as DSS (or the DSS CSV),
  with the steps to attach them in the Time-Series Data Manager. HEC-HMS keeps gage data
  in DSS; its project files (``.gage``) are not a documented format, so none is written.
- **HEC-RAS**: the ``Flow Hydrograph=`` (or ``Stage Hydrograph=``) block of an unsteady
  flow file, ten values per line in eight-character columns, plus the DSS record for the
  "Use DSS" route. HEC does not publish the unsteady-file layout; the block follows the
  writer in ras-commander (MIT,
  https://github.com/gpt-cmdr/ras-commander/blob/main/ras_commander/usgs/boundary_generation.py).
- **SWMM 5**: an external time series file (``date time value``), the same data as an
  inline ``[TIMESERIES]`` block, and the ``[TIMESERIES]``/``[INFLOWS]`` (or
  ``[RAINGAGES]``) lines that wire it in. SWMM 5.2 User's Manual, section 11.6 and
  Appendix D: https://www.epa.gov/system/files/documents/2022-04/swmm-users-manual-version-5.2.pdf
- **MODFLOW 6**: a River (stage) or Well (flow) package with one stress period per time
  step, and the matching TDIS file. MODFLOW 6 Input Guide:
  https://modflow6.readthedocs.io/en/stable/_mf6io/gwf-riv.html ,
  https://modflow6.readthedocs.io/en/stable/_mf6io/gwf-wel.html ,
  https://modflow6.readthedocs.io/en/stable/_mf6io/sim-tdis.html . Written as plain text
  (works in the browser); ``engine="flopy"`` writes the same package through FloPy
  (optional extra).
- **Delft-FEWS**: a PI-XML time series following ``pi_timeseries.xsd``
  (https://fewsdocs.deltares.nl/schemas/version1.0/pi-schemas/pi_timeseries.xsd).
- **Raven**: an ``:ObservationData`` block (``HYDROGRAPH`` or ``WATER_LEVEL``) in an
  ``.rvt`` file, missing values as ``-1.2345``. Raven User's and Developer's Manual
  v3.8, section A.4.2: https://raven.uwaterloo.ca/files/v3.8/RavenManual_v3.8.pdf

Everything here imports on a bare install (numpy, pandas, the standard library), so the
Explorer's worker can run it.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: The tools, in menu order. ``kinds`` are the record kinds each one takes.
TOOLS: dict[str, dict[str, Any]] = {
    "hec-hms": {"label": "HEC-HMS", "kinds": ("flow", "precip"),
                "what": "Observed flow or precipitation gage record (DSS)."},
    "hec-ras": {"label": "HEC-RAS", "kinds": ("flow", "stage"),
                "what": "Unsteady boundary hydrograph block, plus the DSS record."},
    "hec-ssp": {"label": "HEC-SSP", "kinds": ("flow", "stage"),
                "what": "Annual peaks to import, the Bulletin 17C settings and AquaScope's result to compare."},
    "dss": {"label": "HEC-DSS", "kinds": ("flow", "stage", "head", "precip"),
            "what": "The record as a DSS time series."},
    "swmm": {"label": "SWMM", "kinds": ("flow", "precip", "stage"),
             "what": "Time series file, [TIMESERIES] block and the [INFLOWS] or [RAINGAGES] line."},
    "modflow6": {"label": "MODFLOW 6", "kinds": ("stage", "head", "flow"),
                 "what": "River (stage) or Well (flow) package by stress period, with its TDIS file."},
    "fews": {"label": "Delft-FEWS", "kinds": ("flow", "stage", "head", "precip"),
             "what": "PI-XML time series."},
    "raven": {"label": "Raven", "kinds": ("flow", "stage", "precip"),
              "what": "Observation (or forcing) block for an .rvt file."},
}

DOCS: dict[str, str] = {
    "dss": "https://github.com/HydrologicEngineeringCenter/hec-dss-python",
    "hec-hms": "https://www.hec.usace.army.mil/software/hec-hms/",
    "hec-ras": "https://github.com/gpt-cmdr/ras-commander/blob/main/ras_commander/usgs/boundary_generation.py",
    "hec-ssp": "https://www.hec.usace.army.mil/confluence/sspdocs/sspum/latest/using-the-hec-ssp-data-importer/"
               "developing-a-new-data-set",
    "swmm": "https://www.epa.gov/system/files/documents/2022-04/swmm-users-manual-version-5.2.pdf",
    "modflow6": "https://modflow6.readthedocs.io/en/stable/_mf6io/gwf-riv.html",
    "fews": "https://fewsdocs.deltares.nl/schemas/version1.0/pi-schemas/pi_timeseries.xsd",
    "raven": "https://raven.uwaterloo.ca/files/v3.8/RavenManual_v3.8.pdf",
}

_KIND_OF = {
    "discharge": "flow", "flow": "flow", "streamflow": "flow", "q": "flow",
    "water_level": "stage", "stage": "stage", "level": "stage",
    "groundwater_level": "head", "head": "head",
    "precipitation": "precip", "precip": "precip", "rainfall": "precip", "rain": "precip",
}
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
#: HEC-DSS undefined value (stored where a value is missing).
DSS_MISSING = -3.4028234663852886e38
RAVEN_MISSING = -1.2345
FEWS_MISSING = -999.0
_DSS_INTERVALS = {60: "1Minute", 120: "2Minute", 180: "3Minute", 240: "4Minute", 300: "5Minute", 360: "6Minute",
                  600: "10Minute", 720: "12Minute", 900: "15Minute", 1200: "20Minute", 1800: "30Minute",
                  3600: "1Hour", 7200: "2Hour", 10800: "3Hour", 14400: "4Hour", 21600: "6Hour", 28800: "8Hour",
                  43200: "12Hour", 86400: "1Day", 604800: "1Week"}
_RAS_INTERVALS = {60: "1MIN", 120: "2MIN", 180: "3MIN", 240: "4MIN", 300: "5MIN", 360: "6MIN", 600: "10MIN",
                  720: "12MIN", 900: "15MIN", 1200: "20MIN", 1800: "30MIN", 3600: "1HOUR", 7200: "2HOUR",
                  10800: "3HOUR", 14400: "4HOUR", 21600: "6HOUR", 28800: "8HOUR", 43200: "12HOUR", 86400: "1DAY",
                  604800: "1WEEK"}


# ── the record ───────────────────────────────────────────────────────────────


@dataclass
class Record:
    """One series with what the exporters need to label it."""

    series: pd.Series
    kind: str = "flow"
    unit: str = "m3/s"
    location: str = "GAUGE"
    name: str = ""
    source: str = ""
    lat: float | None = None
    lon: float | None = None
    utc: bool = False
    notes: list[str] = field(default_factory=list)


def kind_of(variable: str | None) -> str:
    """``flow``, ``stage``, ``head`` or ``precip`` for an AquaScope variable name (``discharge`` by default)."""
    v = str(variable or "discharge").strip().lower()
    if v not in _KIND_OF:
        raise ValueError(f"no engineering export for variable {variable!r}; "
                         f"use one of {sorted(set(_KIND_OF))}")
    return _KIND_OF[v]


def safe_id(text: str, *, max_len: int = 40) -> str:
    """An identifier every target accepts: letters, digits, ``_``, ``-`` and ``.`` only."""
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip()).strip("_")
    return (out or "GAUGE")[:max_len]


def make_record(series: pd.Series, *, variable: str | None = "discharge", unit: str | None = None,
                location: str | None = None, name: str | None = None, source: str | None = None,
                lat: float | None = None, lon: float | None = None) -> Record:
    """Clean a series for export: numeric, time-sorted, one value per time, time zone dropped (UTC kept).

    ``unit`` defaults to the AquaScope unit of the variable (m3/s, m, mm).
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series with a datetime index")
    kind = kind_of(variable)
    s = pd.to_numeric(series, errors="coerce")
    idx = pd.to_datetime(s.index)
    utc = False
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
        utc = True
    s = pd.Series(s.to_numpy(dtype=float), index=idx)
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if not s.notna().any():
        raise ValueError("the series has no values to export")
    default_unit = {"flow": "m3/s", "stage": "m", "head": "m", "precip": "mm"}[kind]
    return Record(series=s, kind=kind, unit=str(unit or default_unit), location=safe_id(location or "GAUGE"),
                  name=str(name or location or ""), source=str(source or ""), lat=lat, lon=lon, utc=utc)


def _step_seconds(s: pd.Series) -> int | None:
    """The most common spacing of the record in seconds, or None for fewer than two times."""
    if len(s) < 2:
        return None
    d = np.diff(s.index.values).astype("timedelta64[s]").astype(np.int64)
    d = d[d > 0]
    if not len(d):
        return None
    vals, counts = np.unique(d, return_counts=True)
    return int(vals[np.argmax(counts)])


def regular(rec: Record) -> tuple[pd.Series, int, list[str]]:
    """The record on a regular time step: ``(series with NaN for gaps, step in seconds, notes)``.

    A record whose spacing is not one of the standard intervals (1 minute to 1 week), or which is too
    irregular to lay on one (more than half the slots would be empty), is averaged to daily values first.
    """
    s = rec.series
    notes: list[str] = []
    step = _step_seconds(s.dropna())
    if step not in _DSS_INTERVALS:
        s = s.resample("D").mean()
        step = 86400
        notes.append("The record was not on a standard time step, so it was averaged to daily values.")
    else:
        full = pd.date_range(s.index[0], s.index[-1], freq=pd.Timedelta(seconds=step))
        if len(full) > 2 * len(s):
            s = s.resample("D").mean()
            step = 86400
            notes.append("The record was too irregular for one time step, so it was averaged to daily values.")
        else:
            # Bin onto the grid rather than reindex, so a reading a little off the grid is kept, not dropped.
            bins = s.resample(pd.Timedelta(seconds=step), origin="start")
            shared = int((bins.count() > 1).sum())
            s = bins.mean()
            if shared:
                notes.append(f"{shared} time steps held more than one reading; each was averaged into one value.")
    s = s.loc[s.first_valid_index():s.last_valid_index()]
    gaps = int(s.isna().sum())
    if gaps:
        notes.append(f"{gaps} of {len(s)} time steps have no value.")
    return s, int(step), notes


#: The longest gap (in time steps) that is bridged by interpolation for a format that cannot hold one.
MAX_FILLED_GAP = 10


def _filled(s: pd.Series, what: str, *, max_gap: int = MAX_FILLED_GAP) -> tuple[pd.Series, list[str]]:
    """Short gaps filled by linear interpolation, for formats that cannot hold a missing value.

    A gap longer than ``max_gap`` steps is not invented: the record is cut to its latest stretch without one.
    """
    notes: list[str] = []
    missing = s.isna().to_numpy()
    if missing.any():
        run_id = np.cumsum(np.r_[True, missing[1:] != missing[:-1]])
        lengths = pd.Series(missing).groupby(run_id).transform("size").to_numpy()
        long_gap = missing & (lengths > max_gap)
        if long_gap.any():
            last = int(np.flatnonzero(long_gap)[-1])
            s = s.iloc[last + 1:]
            notes.append(f"{what} cannot hold a gap, so only {s.index[0]:%Y-%m-%d} to {s.index[-1]:%Y-%m-%d} is "
                         f"written: the latest stretch with no gap longer than {max_gap} steps.")
    gaps = int(s.isna().sum())
    if gaps:
        s = s.interpolate(method="time").ffill().bfill()
        notes.append(f"{gaps} missing steps were filled by linear interpolation for {what}, which cannot hold a gap.")
    return s, notes


def _is_period(rec: Record, step: int) -> bool:
    """Daily (or coarser) means and rainfall totals are period data; sub-daily level or flow is instantaneous."""
    return rec.kind == "precip" or step >= 86400


def _dss_stamp(t: pd.Timestamp) -> str:
    """``01Jan2020 2400`` style: midnight is written as 2400 of the day before (the HEC convention)."""
    if t.hour == 0 and t.minute == 0 and t.second == 0:
        d = t - pd.Timedelta(days=1)
        return f"{d.day:02d}{_MONTHS[d.month - 1]}{d.year:04d} 2400"
    return f"{t.day:02d}{_MONTHS[t.month - 1]}{t.year:04d} {t.hour:02d}{t.minute:02d}"


def _dss_date(t: pd.Timestamp) -> str:
    return f"{t.day:02d}{_MONTHS[t.month - 1]}{t.year:04d}"


def _num(v: float, digits: int = 6) -> str:
    """A compact number: no exponent for ordinary values, no trailing zeros."""
    if v is None or not math.isfinite(float(v)):
        return ""
    return f"{float(v):.{digits}f}".rstrip("0").rstrip(".") or "0"


# ── HEC-DSS ──────────────────────────────────────────────────────────────────


def dss_units(rec: Record) -> str:
    u = rec.unit.strip().lower().replace("³", "3").replace(" ", "")
    return {"m3/s": "CMS", "cms": "CMS", "m^3/s": "CMS", "ft3/s": "CFS", "cfs": "CFS", "m": "M", "ft": "FT",
            "mm": "MM", "in": "IN", "inch": "IN"}.get(u, rec.unit.upper())


def dss_c_part(rec: Record) -> str:
    return {"flow": "FLOW", "stage": "STAGE", "head": "ELEV", "precip": "PRECIP-INC"}[rec.kind]


def dss_pathname(rec: Record, *, e_part: str, d_part: str = "", a_part: str | None = None,
                 c_part: str | None = None, f_part: str = "AQUASCOPE") -> str:
    """``/A/B/C/D/E/F/``: A the source, B the station, C the parameter, E the interval, F ``AQUASCOPE``."""
    def part(x: str) -> str:
        return str(x).replace("/", "_").upper()[:64]

    a = part(a_part if a_part is not None else rec.source)
    return f"/{a}/{part(rec.location)}/{part(c_part or dss_c_part(rec))}/{part(d_part)}/{e_part}/{part(f_part)}/"


@dataclass
class DssRecord:
    """One DSS time series ready to write: times already in the stamping convention of its data type."""

    pathname: str
    times: list[pd.Timestamp]
    values: list[float]
    units: str
    data_type: str
    regular: bool = True


def dss_record(rec: Record) -> tuple[DssRecord, list[str]]:
    """The record as a regular DSS time series (period data stamped at the end of each period)."""
    s, step, notes = regular(rec)
    period = _is_period(rec, step)
    dtype = ("PER-CUM" if rec.kind == "precip" else "PER-AVER") if period else "INST-VAL"
    times = list(s.index + pd.Timedelta(seconds=step)) if period else list(s.index)
    if period:
        notes.append(f"{dtype}: each value is stamped at the end of its period (a day's mean at 2400).")
    path = dss_pathname(rec, e_part=_DSS_INTERVALS[step], d_part=_dss_date(times[0]))
    return DssRecord(pathname=path, times=times, values=[float(v) for v in s.to_numpy()], units=dss_units(rec),
                     data_type=dtype), notes


def dss_csv_text(rec: DssRecord) -> str:
    """The CSV layout ``hecdss`` reads with ``RegularTimeSeries.read_csv`` (or ``IrregularTimeSeries``).

    Header ``Date/Time,Value,Units,Type,Path``; units, type and path on the first data row only; a missing
    value is an empty cell.
    """
    lines = ["Date/Time,Value,Units,Type,Path"]
    for i, (t, v) in enumerate(zip(rec.times, rec.values)):
        row = f"{_dss_stamp(pd.Timestamp(t))},{_num(v)}"
        if i == 0:
            row += f",{rec.units},{rec.data_type},{rec.pathname}"
        lines.append(row)
    return "\n".join(lines) + "\n"


def hecdss_available() -> bool:
    """True when ``hecdss`` imports and its native library loads on this machine."""
    try:
        from hecdss import HecDss  # noqa: F401
        from hecdss.native import _Native

        _Native()
        return True
    except Exception:  # noqa: BLE001 - not installed, or no native library for this platform
        return False


def write_dss(path: str | Path, records: list[DssRecord]) -> Path:
    """Write the records into a DSS file with HEC's ``hecdss`` (raises ImportError/OSError where it cannot load)."""
    from hecdss import HecDss, IrregularTimeSeries, RegularTimeSeries

    path = Path(path)
    with HecDss(str(path)) as dss:
        for r in records:
            times = [pd.Timestamp(t).to_pydatetime() for t in r.times]
            values = np.array([DSS_MISSING if not math.isfinite(v) else v for v in r.values], dtype=float)
            if r.regular:
                ts = RegularTimeSeries.create(values=values, times=times, units=r.units, data_type=r.data_type,
                                              path=r.pathname)
            else:
                # Minute granularity: second granularity overflows DSS's 32-bit times past about 68 years.
                ts = IrregularTimeSeries.create(values=values, times=times, units=r.units, data_type=r.data_type,
                                                path=r.pathname, time_granularity_seconds=60)
            status = dss.put(ts)
            if status != 0:
                raise OSError(f"hecdss could not store {r.pathname} (status {status})")
    return path


def _dss_files(folder: str, stem: str, records: list[DssRecord], *, binary: bool | None) -> tuple[dict[str, bytes],
                                                                                                    list[str]]:
    """A real ``.dss`` when ``hecdss`` loads here (and ``binary`` is not False), else one DSS CSV per record."""
    use_binary = hecdss_available() if binary is None else bool(binary)
    if use_binary:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                p = write_dss(Path(tmp) / f"{stem}.dss", records)
                return {f"{folder}/{stem}.dss": p.read_bytes()}, []
        except Exception as exc:  # noqa: BLE001 - fall back to the CSV every machine can convert
            logger.warning("DSS write failed, writing the DSS CSV instead: %s", exc)
            if binary:
                raise
    files = {}
    for r in records:
        c = re.sub(r"[^a-z0-9]+", "_", r.pathname.split("/")[3].lower()).strip("_")
        files[f"{folder}/{stem}_{c}.dss.csv"] = dss_csv_text(r).encode()
    note = ("DSS is written as the CSV that HEC's hecdss reads: in Python, "
            "`RegularTimeSeries.read_csv(path)` (IrregularTimeSeries for annual peaks), then `HecDss(file).put(ts)`.")
    return files, [note]


# ── HEC-SSP: annual peaks and Bulletin 17C ──────────────────────────────────


def annual_peaks(rec: Record, *, year_start_month: int = 10, min_coverage: float = 0.8) -> pd.DataFrame:
    """Annual maxima of the daily means by water year: columns ``year``, ``date``, ``peak``.

    A year counts when at least ``min_coverage`` of its days have a value (the Explorer's rule, so a partial
    first or last year does not fake a low peak). ``year_start_month`` 10 is the US water year (October to
    September, named by the year it ends in); 1 is the calendar year.
    """
    start_month = int(year_start_month)
    if not 1 <= start_month <= 12:
        raise ValueError("year_start_month must be 1 to 12")
    daily = rec.series.resample("D").mean()
    # A year that starts in October 2019 is water year 2020: shift the dates forward so it lands in 2020.
    shift = 13 - start_month if start_month != 1 else 0
    wy = (daily.index + pd.DateOffset(months=shift)).year if shift else daily.index.year
    frame = pd.DataFrame({"v": daily.to_numpy(), "wy": wy}, index=daily.index)
    rows = []
    for year, grp in frame.groupby("wy"):
        first = pd.Timestamp(year=int(year) - (1 if shift else 0), month=start_month, day=1)
        n_days = (first + pd.DateOffset(years=1) - first).days
        if grp["v"].count() < min_coverage * n_days:
            continue
        t = grp["v"].idxmax()
        rows.append({"year": int(year), "date": t, "peak": float(grp["v"].max())})
    return pd.DataFrame(rows, columns=["year", "date", "peak"])


def hec_ssp_text(peaks: pd.DataFrame, rec: Record) -> str:
    """Comma-delimited text for HEC-SSP's Data Importer (Text File): ``ddMMMyyyy hhmm`` then the value.

    Each peak is stamped at noon of its day, so no tool can round it into the neighbouring water year.

    The first row is a header to skip in the importer (right-click, Skip Row(s)).
    """
    head = "Flow" if rec.kind == "flow" else "Stage"
    lines = [f"Date/Time,{head} ({rec.unit})"]
    for _, r in peaks.iterrows():
        t = pd.Timestamp(r["date"])
        lines.append(f"{t.day:02d}{_MONTHS[t.month - 1]}{t.year:04d} 1200,{_num(r['peak'])}")
    return "\n".join(lines) + "\n"


def b17c_compare(peaks: pd.DataFrame, *, regional_skew: float | None = None,
                 regional_skew_mse: float | None = None) -> dict[str, Any]:
    """AquaScope's Bulletin 17C (EMA with the Multiple Grubbs-Beck test) on the peaks, as a JSON-able dict."""
    from aquascope.hydrology.flood_frequency import expected_moments_algorithm

    aeps = [0.5, 0.2, 0.1, 0.04, 0.02, 0.01, 0.005, 0.002]
    rps = [round(1 / p) for p in aeps]
    kw: dict[str, Any] = {"return_periods": rps}
    if regional_skew is not None:
        kw["regional_skew"] = float(regional_skew)
        if regional_skew_mse is not None:
            kw["regional_skew_mse"] = float(regional_skew_mse)
    res = expected_moments_algorithm(peaks["peak"].to_numpy(dtype=float), **kw)
    skew, mean, std = res.params
    return {
        "n_peaks": int(len(peaks)), "n_censored": int(res.n_censored),
        "low_outlier_threshold": res.low_outlier_threshold,
        "mean_log10": round(float(mean), 4), "std_log10": round(float(std), 4), "skew": round(float(skew), 3),
        "weighted_skew": None if res.weighted_skew is None else round(float(res.weighted_skew), 3),
        "quantiles": [{"aep": p, "T": t, "q": float(res.return_periods[t]),
                       "lower_5": float(res.confidence_intervals[t][0]),
                       "upper_95": float(res.confidence_intervals[t][1])} for p, t in zip(aeps, rps)],
    }


def b17c_settings_text(peaks: pd.DataFrame, rec: Record, *, regional_skew: float | None = None,
                       regional_skew_mse: float | None = None) -> str:
    """The Bulletin 17C settings to enter in HEC-SSP, and AquaScope's result on the same peaks to compare."""
    lines = [
        f"Bulletin 17C analysis input for {rec.name or rec.location}",
        "=" * 60,
        "",
        f"Peaks: {len(peaks)} annual maxima of daily means ({rec.unit}), one per water year,",
        "in annual_peaks.csv (import with the Data Importer, Text File option).",
        "These are maxima of daily means, not instantaneous peaks: an agency peak-flow file, where one exists,",
        "is the better input for a design value.",
        "",
        "Settings for the HEC-SSP Bulletin 17 analysis",
        "---------------------------------------------",
        "Procedure:            Bulletin 17C, Expected Moments Algorithm (EMA)",
        "Low outlier test:     Multiple Grubbs-Beck",
        "Perception threshold: 0 to infinity for the whole record (systematic record)",
        "Confidence limits:    0.05 and 0.95",
    ]
    if regional_skew is not None:
        lines.append(f"Skew:                 Weighted, regional skew {regional_skew}"
                     + (f", regional skew MSE {regional_skew_mse}" if regional_skew_mse is not None else ""))
    else:
        lines.append("Skew:                 Station skew (no regional skew given; enter yours to weight it)")
    lines += ["", "AquaScope's result on the same peaks", "------------------------------------"]
    try:
        r = b17c_compare(peaks, regional_skew=regional_skew, regional_skew_mse=regional_skew_mse)
    except Exception as exc:  # noqa: BLE001 - too few peaks, or a degenerate record
        lines += [f"Not computed: {exc}", ""]
        return "\n".join(lines)
    lines += [f"Mean of log10: {r['mean_log10']}   Std of log10: {r['std_log10']}   Adopted skew: {r['skew']}",
              f"Censored (zeros and low outliers): {r['n_censored']}", "",
              f"{'AEP':>7}  {'T (yr)':>7}  {'Estimate':>12}  {'5% limit':>12}  {'95% limit':>12}"]
    for q in r["quantiles"]:
        lines.append(f"{q['aep']:>7}  {q['T']:>7}  {q['q']:>12.4g}  {q['lower_5']:>12.4g}  {q['upper_95']:>12.4g}")
    lines += ["", "AquaScope's EMA is a simplified implementation. How it compares with the published Bulletin 17C",
              "examples is in the docs (Engineering exports, Bulletin 17C check); trust HEC-SSP or PeakFQ where",
              "they differ.", ""]
    return "\n".join(lines)


# ── HEC-RAS ──────────────────────────────────────────────────────────────────


def _ras_values(values: np.ndarray) -> str:
    """Ten values per line, eight characters each, as many decimals as fit."""
    top = float(np.nanmax(np.abs(values))) if len(values) else 0.0
    for dec in (2, 1, 0):
        if len(f"{-top if np.nanmin(values) < 0 else top:.{dec}f}") <= 8:
            break
    else:
        raise ValueError("values too large for HEC-RAS's eight-character columns")
    lines = []
    for i in range(0, len(values), 10):
        lines.append("".join(f"{v:>8.{dec}f}" for v in values[i:i + 10]))
    return "\n".join(lines)


def hec_ras_block(rec: Record) -> tuple[str, list[str]]:
    """The ``Interval=`` and ``Flow Hydrograph=`` (or ``Stage Hydrograph=``) lines of a ``.u##`` file."""
    s, step, notes = regular(rec)
    if step not in _RAS_INTERVALS:
        s, step = s.resample("D").mean(), 86400
        notes.append("Averaged to daily values to match a HEC-RAS interval.")
    s, fill = _filled(s, "HEC-RAS")
    table = "Flow Hydrograph=" if rec.kind == "flow" else "Stage Hydrograph="
    text = f"Interval={_RAS_INTERVALS[step]}\n{table} {len(s)} \n{_ras_values(s.to_numpy(dtype=float))}\n"
    return text, notes + fill + [f"The hydrograph starts {s.index[0]:%d%b%Y %H:%M} ({rec.unit}): set the "
                                 "simulation start to that time, or tick Use Fixed Start Time in the boundary."]


# ── SWMM ─────────────────────────────────────────────────────────────────────


def _swmm_hhmm(seconds: int) -> str:
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}"


def swmm_series_lines(s: pd.Series, name: str | None = None) -> list[str]:
    """``[name] MM/DD/YYYY HH:MM value`` lines, one value per line, gaps dropped (SWMM interpolates)."""
    out = []
    for t, v in s.dropna().items():
        row = f"{t.month:02d}/{t.day:02d}/{t.year:04d} {t.hour:02d}:{t.minute:02d} {_num(v)}"
        out.append(f"{name} {row}" if name else row)
    return out


def swmm_files(rec: Record) -> tuple[dict[str, str], list[str]]:
    """The external time series file, the inline ``[TIMESERIES]`` block and the wiring lines."""
    s, step, notes = regular(rec)
    name = rec.location
    dat = f"{name}.dat"
    head = f";{rec.name or name} ({rec.unit}), from AquaScope"
    files = {dat: "\n".join([head] + swmm_series_lines(s)) + "\n"}
    files[f"{name}_timeseries.inp"] = "\n".join(
        ["[TIMESERIES]", ";;Name           Date       Time       Value"] + swmm_series_lines(s, name)) + "\n"
    wiring = ["[TIMESERIES]", ";;Name           Date       Time       Value", f'{name} FILE "{dat}"', ""]
    gaps = int(s.isna().sum())
    if gaps:
        notes.append(f"{gaps} missing steps are left out of the SWMM files: SWMM interpolates a flow or level "
                     "across them and reads missing rain as none.")
    if rec.kind == "flow":
        wiring += ["[INFLOWS]", ";;Node  Constituent  Time Series  Type  Mfactor  Sfactor",
                   f"OUTLET1  FLOW  {name}  FLOW  1.0  1.0", ""]
        unit = dss_units(rec)
        if unit == "CMS":
            notes.append("Values are in m3/s: set FLOW_UNITS CMS in [OPTIONS], or Sfactor 35.3147 for CFS.")
        elif unit == "CFS":
            notes.append("Values are in ft3/s: set FLOW_UNITS CFS in [OPTIONS], or Sfactor 0.0283168 for CMS.")
        else:
            notes.append(f"Values are in {rec.unit}: set Sfactor to convert them to the project's FLOW_UNITS.")
        notes.append("Rename OUTLET1 to the node that takes the inflow.")
    elif rec.kind == "precip":
        wiring += ["[RAINGAGES]", ";;Name  Format  Interval  SCF  Source",
                   f"RG_{name}  VOLUME  {_swmm_hhmm(step)}  1.0  TIMESERIES  {name}", ""]
        notes.append(f"Rainfall depth per {_swmm_hhmm(step)} interval in {rec.unit}; SWMM's US units expect inches.")
    files[f"{name}_sections.inp"] = "\n".join(wiring)
    return files, notes


# ── MODFLOW 6 ────────────────────────────────────────────────────────────────


def modflow6_inputs(rec: Record, *, cellid: tuple[int, ...] = (1, 1, 1), cond: float | None = None,
                    rbot: float | None = None, period: str | None = None) -> dict[str, Any]:
    """Stress periods for a River (stage, head) or Well (flow) package: ``{"package", "perlen", "rows", ...}``.

    One stress period per time step (``period`` resamples first, e.g. ``"MS"`` for monthly means). Flow in
    m3/s becomes m3/d (the model is set up in days). ``cond`` and ``rbot`` are placeholders when not given:
    conductance 1.0 and the riverbed bottom one metre below the lowest stage.
    """
    s = rec.series
    notes: list[str] = []
    if period:
        s = s.resample(period).mean()
        s = s.loc[s.first_valid_index():s.last_valid_index()]
    else:
        s, step, notes = regular(rec)
    s, fill = _filled(s, "MODFLOW 6")
    notes += fill
    if period:
        idx = s.index
        ends = list(idx[1:]) + [idx[-1] + pd.tseries.frequencies.to_offset(period)]
        perlen = [(b - a).total_seconds() / 86400 for a, b in zip(idx, ends)]
    else:
        perlen = [step / 86400] * len(s)
    pkg = "riv" if rec.kind in ("stage", "head") else "wel"
    vals = s.to_numpy(dtype=float)
    placeholders = []
    if pkg == "wel":
        if rec.unit.lower().replace("³", "3") in ("m3/s", "cms"):
            vals = vals * 86400.0
            notes.append("Flow converted from m3/s to m3/d; positive Q is injection, so negate it for pumping.")
        rows = [[float(v)] for v in vals]
    else:
        if cond is None:
            cond = 1.0
            placeholders.append("cond")
        if rbot is None:
            rbot = float(np.nanmin(vals)) - 1.0
            placeholders.append("rbot")
        rows = [[float(v), float(cond), float(rbot)] for v in vals]
    if placeholders:
        notes.append(f"{' and '.join(placeholders)} are placeholders (conductance 1.0, riverbed bottom 1 m below "
                     "the lowest stage): set them for your river.")
    return {"package": pkg, "cellid": tuple(int(c) for c in cellid), "perlen": perlen, "rows": rows,
            "start": s.index[0], "placeholders": placeholders, "notes": notes}


def modflow6_text(rec: Record, **kw: Any) -> tuple[dict[str, str], list[str]]:
    """The package file and the TDIS file, written as MODFLOW 6 input text."""
    m = modflow6_inputs(rec, **kw)
    name = rec.location.lower()
    cell = " ".join(str(c) for c in m["cellid"])
    pkg = m["package"]
    head = [f"# MODFLOW 6 {pkg.upper()} package from AquaScope: {rec.name or rec.location} ({rec.unit})",
            f"# one stress period per time step from {m['start']:%Y-%m-%d %H:%M}"]
    if m["placeholders"]:
        head.append(f"# PLACEHOLDERS to set for your river: {', '.join(m['placeholders'])}")
    cols = "# layer row col  stage  cond  rbot  boundname" if pkg == "riv" else "# layer row col  q  boundname"
    body = head + ["BEGIN OPTIONS", "  BOUNDNAMES", "END OPTIONS", "", "BEGIN DIMENSIONS", "  MAXBOUND 1",
                   "END DIMENSIONS", ""]
    bname = safe_id(rec.location)
    bname = bname if bname[0].isalpha() else f"g{bname}"  # a boundname that cannot be read as a number
    for i, row in enumerate(m["rows"], start=1):
        body.append(f"BEGIN PERIOD {i}")
        if i == 1:
            body.append(f"  {cols}")
        body += [f"  {cell}  {'  '.join(_num(v) for v in row)}  {bname}", "END PERIOD", ""]
    pkg_text = "\n".join(body)
    tdis = [f"# MODFLOW 6 TDIS from AquaScope: {len(m['perlen'])} stress periods", "BEGIN OPTIONS",
            "  TIME_UNITS DAYS", f"  START_DATE_TIME {m['start']:%Y-%m-%dT%H:%M:%S}", "END OPTIONS", "",
            "BEGIN DIMENSIONS", f"  NPER {len(m['perlen'])}", "END DIMENSIONS", "", "BEGIN PERIODDATA"]
    tdis += [f"  {_num(p)}  1  1.0" for p in m["perlen"]] + ["END PERIODDATA", ""]
    return {f"{name}.{pkg}": pkg_text, f"{name}.tdis": "\n".join(tdis)}, m["notes"]


def modflow6_flopy(rec: Record, out_dir: str | Path, **kw: Any) -> dict[str, str]:
    """The same package written through FloPy into a one-cell simulation skeleton (optional extra)."""
    import flopy

    m = modflow6_inputs(rec, **kw)
    out = Path(out_dir)
    name = rec.location.lower()
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=str(out), exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=len(m["perlen"]),
                          perioddata=[(p, 1, 1.0) for p in m["perlen"]], filename=f"{name}.tdis")
    flopy.mf6.ModflowIms(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name)
    nlay, nrow, ncol = (max(c, 1) for c in (list(m["cellid"]) + [1, 1, 1])[:3])
    flopy.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol)
    flopy.mf6.ModflowGwfic(gwf)
    flopy.mf6.ModflowGwfnpf(gwf)
    cell = tuple(c - 1 for c in m["cellid"])
    spd = {i: [(cell, *row)] for i, row in enumerate(m["rows"])}
    if m["package"] == "riv":
        flopy.mf6.ModflowGwfriv(gwf, stress_period_data=spd, maxbound=1, filename=f"{name}.riv")
    else:
        flopy.mf6.ModflowGwfwel(gwf, stress_period_data=spd, maxbound=1, filename=f"{name}.wel")
    sim.write_simulation(silent=True)
    return {p.name: p.read_text() for p in out.iterdir() if p.is_file()}


# ── Delft-FEWS PI-XML ────────────────────────────────────────────────────────


def fews_pi_xml(rec: Record, *, parameter_id: str | None = None) -> tuple[str, list[str]]:
    """A PI-XML ``TimeSeries`` document with one series, equidistant, missing values as ``-999``."""
    s, step, notes = regular(rec)
    period = _is_period(rec, step)
    ts_type = ("accumulative" if rec.kind == "precip" else "mean") if period else "instantaneous"
    unit, mult = ("day", step // 86400) if step % 86400 == 0 else (
        ("hour", step // 3600) if step % 3600 == 0 else ("minute", step // 60))
    pid = parameter_id or {"flow": "Q.obs", "stage": "H.obs", "head": "GWL.obs", "precip": "P.obs"}[rec.kind]

    def dt(tag: str, t: pd.Timestamp) -> str:
        return f'<{tag} date="{t:%Y-%m-%d}" time="{t:%H:%M:%S}"/>'

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<TimeSeries xmlns="http://www.wldelft.nl/fews/PI" '
             'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xsi:schemaLocation="http://www.wldelft.nl/fews/PI '
             'http://fews.wldelft.nl/schemas/version1.0/pi-schemas/pi_timeseries.xsd" version="1.2">',
             "  <timeZone>0.0</timeZone>",
             f"  <!-- From AquaScope. Times as the source gave them{' (UTC)' if rec.utc else ''}; "
             f"{'each value labels the start of its period' if period else 'instantaneous values'}. -->",
             "  <series>", "    <header>",
             f"      <type>{ts_type}</type>",
             f"      <locationId>{escape(rec.location)}</locationId>",
             f"      <parameterId>{escape(pid)}</parameterId>",
             f'      <timeStep unit="{unit}" multiplier="{int(mult)}"/>',
             "      " + dt("startDate", s.index[0]),
             "      " + dt("endDate", s.index[-1]),
             f"      <missVal>{_num(FEWS_MISSING)}</missVal>"]
    if rec.name:
        lines.append(f"      <stationName>{escape(rec.name)}</stationName>")
    if rec.lat is not None and rec.lon is not None:
        lines += [f"      <lat>{_num(rec.lat)}</lat>", f"      <lon>{_num(rec.lon)}</lon>"]
    lines += [f"      <units>{escape(rec.unit)}</units>", "    </header>"]
    for t, v in s.items():
        val = _num(v) if math.isfinite(float(v)) else _num(FEWS_MISSING)
        lines.append(f'    <event date="{t:%Y-%m-%d}" time="{t:%H:%M:%S}" value={quoteattr(val)} flag="0"/>')
    lines += ["  </series>", "</TimeSeries>", ""]
    if not rec.utc:
        notes.append("The source did not give a time zone; the file says 0.0 (UTC). Change <timeZone> if the "
                     "agency reports local time.")
    return "\n".join(lines), notes


# ── Raven ────────────────────────────────────────────────────────────────────


def raven_rvt(rec: Record, *, subbasin_id: int = 1) -> tuple[str, list[str]]:
    """An ``.rvt`` observation block (``HYDROGRAPH``, ``WATER_LEVEL``) or a ``PRECIP`` forcing block."""
    s, step, notes = regular(rec)
    days = step / 86400
    if rec.kind == "precip":
        if step != 86400:
            s, step, days = s.resample("D").sum(min_count=1), 86400, 1.0
            notes.append("Rainfall summed to daily totals (Raven reads precipitation as a rate in mm/d).")
        head = ":Data PRECIP mm/d"
        tail = ":EndData"
    else:
        tag = "HYDROGRAPH" if rec.kind == "flow" else "WATER_LEVEL"
        head = f":ObservationData {tag} {int(subbasin_id)} {rec.unit}"
        tail = ":EndObservationData"
    start = s.index[0]
    lines = [f"# {rec.name or rec.location} from AquaScope ({rec.unit}); missing values are -1.2345",
             head, f"  {start:%Y-%m-%d %H:%M:%S}.0 {days:.6g}{'.0' if float(days).is_integer() else ''} {len(s)}"]
    lines += [f"  {_num(v) if math.isfinite(float(v)) else _num(RAVEN_MISSING)}" for v in s.to_numpy(dtype=float)]
    lines += [tail, ""]
    if rec.kind != "precip":
        notes.append(f"Observation for subbasin {int(subbasin_id)}: change the ID to the subbasin at the gauge.")
    return "\n".join(lines), notes


# ── the faces ────────────────────────────────────────────────────────────────


def _readme(tool: str, rec: Record, files: list[str], notes: list[str]) -> str:
    meta = TOOLS[tool]
    lines = [f"{meta['label']} input from AquaScope", "=" * 40, "",
             f"Record: {rec.name or rec.location}" + (f" ({rec.source})" if rec.source else ""),
             f"Variable: {rec.kind} in {rec.unit}, {rec.series.index[0]:%Y-%m-%d} to {rec.series.index[-1]:%Y-%m-%d}"]
    lines += _terms(rec.source)
    lines += ["", "Files", "-----"] + [f"  {f}" for f in files] + ["", "How to use", "----------"]
    lines += _HOWTO[tool]
    if notes:
        lines += ["", "Notes", "-----"] + [f"- {n}" for n in notes]
    lines += ["", f"Format reference: {DOCS[tool]}", ""]
    return "\n".join(lines)


def _terms(source: str) -> list[str]:
    """The source's licence and attribution lines from the registry (none for a CSV or an unknown source)."""
    if not source:
        return []
    from aquascope.registry import SOURCES

    meta = SOURCES.get(source)
    if meta is None:
        return []
    out = [f"Data licence: {meta.license}"]
    if meta.attribution:
        out.append(f"Attribution: {meta.attribution}")
    return out


_HOWTO: dict[str, list[str]] = {
    "dss": ["Open the .dss file in HEC-DSSVue, or convert a .dss.csv with Python:",
            "  from hecdss import HecDss, RegularTimeSeries",
            "  ts = RegularTimeSeries.read_csv('file.dss.csv')",
            "  with HecDss('out.dss') as dss: dss.put(ts)"],
    "hec-hms": ["In the Time-Series Data Manager add a Discharge Gage (or Precipitation Gage), choose",
                "Single Record HEC-DSS as the data source, and point it at the DSS file and pathname here.",
                "A .dss.csv becomes a .dss with HEC's hecdss (pip install hecdss) on Linux or Windows:",
                "  from hecdss import HecDss, RegularTimeSeries",
                "  with HecDss('gage.dss') as dss: dss.put(RegularTimeSeries.read_csv('file.dss.csv'))"],
    "hec-ras": ["Paste the block into the unsteady flow file (.u##) under the matching Boundary Location,",
                "replacing its Interval= and Flow Hydrograph= lines; or use the DSS record with Use DSS."],
    "hec-ssp": ["Data Importer, Text File: select annual_peaks.csv, skip the header row, set the date/time",
                "and data columns, then name the pathname parts. Then add a Bulletin 17 analysis with the",
                "settings in b17c_settings.txt and compare its curve with AquaScope's result in the same file."],
    "swmm": ["Copy the lines in *_sections.inp into your .inp file and keep the .dat file next to it,",
             "or paste the inline [TIMESERIES] block from *_timeseries.inp instead."],
    "modflow6": ["Add the package to your GWF model's name file and use the TDIS file (or merge its periods).",
                 "The cell is layer 1, row 1, column 1 unless you passed another: change it to the river cell."],
    "fews": ["Import the XML with a PI import module (type pi_timeseries); map locationId and parameterId",
             "to your configuration."],
    "raven": ["Add the block to your .rvt file, or keep this file and point to it with :RedirectToFile."],
}


def export_files(rec: Record, tool: str, *, dss_binary: bool | None = None,
                 regional_skew: float | None = None, regional_skew_mse: float | None = None,
                 year_start_month: int = 10, subbasin_id: int = 1, cellid: tuple[int, ...] = (1, 1, 1),
                 cond: float | None = None, rbot: float | None = None, period: str | None = None,
                 engine: str = "text") -> tuple[dict[str, bytes], list[str]]:
    """Every file for one tool, as ``{"<tool>/<file>": bytes}``, plus the notes (a README.txt is included).

    ``dss_binary`` None writes a real .dss only where ``hecdss`` loads (False always writes the DSS CSV).
    """
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}; choose from {', '.join(TOOLS)}")
    if rec.kind not in TOOLS[tool]["kinds"]:
        raise ValueError(f"{TOOLS[tool]['label']} export takes {', '.join(TOOLS[tool]['kinds'])} records, "
                         f"not {rec.kind}")
    files: dict[str, bytes] = {}
    notes: list[str] = []
    if tool in ("dss", "hec-hms"):
        r, n = dss_record(rec)
        got, n2 = _dss_files(tool, rec.location, [r], binary=dss_binary)
        files.update(got)
        notes += n + n2
        if tool == "hec-hms":
            notes.append(f"DSS pathname: {r.pathname} (units {r.units}, type {r.data_type}).")
    elif tool == "hec-ras":
        block, n = hec_ras_block(rec)
        files[f"hec-ras/{rec.location}_boundary.txt"] = block.encode()
        r, n1 = dss_record(rec)
        got, n2 = _dss_files("hec-ras", rec.location, [r], binary=dss_binary)
        files.update(got)
        notes += n + n2
    elif tool == "hec-ssp":
        peaks = annual_peaks(rec, year_start_month=year_start_month)
        if len(peaks) < 5:
            raise ValueError(f"only {len(peaks)} complete years: Bulletin 17C needs more annual peaks")
        files["hec-ssp/annual_peaks.csv"] = hec_ssp_text(peaks, rec).encode()
        files["hec-ssp/b17c_settings.txt"] = b17c_settings_text(
            peaks, rec, regional_skew=regional_skew, regional_skew_mse=regional_skew_mse).encode()
        # Noon of the peak day: inside the day whichever way a tool rounds, so the water year is never in doubt.
        times = [pd.Timestamp(d).normalize() + pd.Timedelta(hours=12) for d in peaks["date"]]
        r = DssRecord(pathname=dss_pathname(rec, e_part="IR-Century", d_part="",
                                            c_part=f"{dss_c_part(rec)}-ANNUAL PEAK"),
                      times=times, values=[float(v) for v in peaks["peak"]], units=dss_units(rec),
                      data_type="INST-VAL", regular=False)
        got, n2 = _dss_files("hec-ssp", rec.location, [r], binary=dss_binary)
        files.update(got)
        notes += n2 + [f"{len(peaks)} water years (starting month {year_start_month}) with at least 80% of days."]
    elif tool == "swmm":
        got, notes = swmm_files(rec)
        files.update({f"swmm/{k}": v.encode() for k, v in got.items()})
    elif tool == "modflow6":
        if engine == "flopy":
            with tempfile.TemporaryDirectory() as tmp:
                got = modflow6_flopy(rec, tmp, cellid=cellid, cond=cond, rbot=rbot, period=period)
            notes = modflow6_inputs(rec, cellid=cellid, cond=cond, rbot=rbot, period=period)["notes"]
            notes.append("Written through FloPy into a one-cell simulation skeleton.")
        else:
            got, notes = modflow6_text(rec, cellid=cellid, cond=cond, rbot=rbot, period=period)
        files.update({f"modflow6/{k}": v.encode() for k, v in got.items()})
    elif tool == "fews":
        xml, notes = fews_pi_xml(rec)
        files[f"fews/{rec.location}.xml"] = xml.encode()
    elif tool == "raven":
        rvt, notes = raven_rvt(rec, subbasin_id=subbasin_id)
        files[f"raven/{rec.location}.rvt"] = rvt.encode()
    seen: set[str] = set()
    notes = [n for n in notes if not (n in seen or seen.add(n))]  # type: ignore[func-returns-value]
    names = [k.split("/", 1)[1] for k in files]
    files[f"{tool}/README.txt"] = _readme(tool, rec, names, notes).encode()
    return files, notes


def tools_for(kind: str) -> list[str]:
    """The tools that take a record of this kind, in menu order."""
    return [t for t, meta in TOOLS.items() if kind in meta["kinds"]]


def menu(variable: str | None) -> dict[str, Any]:
    """The "Export for..." choices for a record of this variable: ``{"kind", "tools": [{id, label, what}]}``.

    An empty list when no tool takes the variable (water quality, say).
    """
    try:
        kind = kind_of(variable)
    except ValueError:
        return {"kind": None, "tools": []}
    return {"kind": kind, "tools": [{"id": t, "label": TOOLS[t]["label"], "what": TOOLS[t]["what"]}
                                    for t in tools_for(kind)]}


def bundle(rec: Record, tools: list[str] | None = None, **kw: Any) -> tuple[dict[str, bytes], dict[str, str]]:
    """Every tool's files for the record (``tools`` default: all that take its kind).

    Returns the files and ``{tool: reason}`` for the tools that were skipped.
    """
    files: dict[str, bytes] = {}
    skipped: dict[str, str] = {}
    for tool in tools or tools_for(rec.kind):
        try:
            got, _ = export_files(rec, tool, **kw)
            files.update(got)
        except Exception as exc:  # noqa: BLE001 - one tool failing must not lose the others
            skipped[tool] = str(exc)
    return files, skipped


def zip_bytes(files: dict[str, bytes], *, prefix: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(files):
            zf.writestr(f"{prefix}{name}", files[name])
    return buf.getvalue()


def _describe(files: dict[str, bytes], *, with_text: bool) -> list[dict[str, Any]]:
    out = []
    for name in sorted(files):
        data = files[name]
        item: dict[str, Any] = {"path": name, "bytes": len(data)}
        if with_text:
            try:
                item["text"] = data.decode("utf-8")
            except UnicodeDecodeError:
                item["base64"] = base64.b64encode(data).decode()
        out.append(item)
    return out


def export_series(series: pd.Series, tool: str, *, variable: str | None = "discharge", unit: str | None = None,
                  location: str | None = None, name: str | None = None, source: str | None = None,
                  lat: float | None = None, lon: float | None = None, with_text: bool = True,
                  as_zip: bool = False, **kw: Any) -> dict[str, Any]:
    """The face-level call: one tool (or ``"all"``) for a series, as a JSON-able dict.

    ``files`` lists each file with its text (``base64`` for a binary .dss); ``as_zip`` adds the whole set as a
    base64 zip under ``zip_base64`` with a ``filename``, which is what the Explorer downloads.
    """
    rec = make_record(series, variable=variable, unit=unit, location=location, name=name, source=source,
                      lat=lat, lon=lon)
    if tool == "all":
        files, skipped = bundle(rec, **kw)
        notes = [f"{TOOLS[t]['label']} skipped: {why}" for t, why in skipped.items()]
    else:
        files, notes = export_files(rec, tool, **kw)
    out: dict[str, Any] = {"tool": tool, "label": "All tools" if tool == "all" else TOOLS[tool]["label"],
                           "kind": rec.kind, "unit": rec.unit, "location": rec.location,
                           "files": _describe(files, with_text=with_text), "notes": notes,
                           "doc": DOCS.get(tool)}
    if as_zip:
        out["filename"] = f"aquascope-{rec.location}-{tool}.zip"
        out["zip_base64"] = base64.b64encode(zip_bytes(files)).decode()
    return out


def export_station(source: str, station_id: str, tool: str, *, years: int | None = None,
                   variable: str | None = None, **kw: Any) -> dict[str, Any]:
    """Fetch a gauge's record (as the Explorer does) and export it for ``tool``."""
    from aquascope.explore import fetch_series

    fetched = fetch_series(source, station_id, years=years, variable=variable)
    s = fetched.get("series")
    if s is None or s.empty:
        return {"error": f"{source} {station_id}: the source returned no observations"}
    return export_series(s, tool, variable=fetched.get("variable") or variable or "discharge",
                         unit=fetched.get("unit") or None, location=station_id, name=kw.pop("name", None) or station_id,
                         source=source, **kw)


def write_files(files: dict[str, bytes], out_dir: str | Path) -> list[str]:
    """Write ``{relative path: bytes}`` under ``out_dir``; returns the paths written."""
    root = Path(out_dir)
    written = []
    for name, data in files.items():
        rel = Path(*[p for p in Path(name).parts if p not in ("..", "/", "")])
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written.append(str(dest))
    return written


def files_from(result: dict[str, Any]) -> dict[str, bytes]:
    """Back from :func:`export_series`'s ``files`` list to ``{path: bytes}``."""
    out = {}
    for f in result.get("files") or []:
        if "text" in f:
            out[f["path"]] = f["text"].encode()
        elif "base64" in f:
            out[f["path"]] = base64.b64decode(f["base64"])
    return out
