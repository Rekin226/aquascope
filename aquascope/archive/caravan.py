"""Caravan-format export from the Archive: per-basin forcing + streamflow, HydroATLAS attributes, climate indices.

Caravan (Kratzert et al. 2023, Sci. Data) is the layout the large-sample /
ML hydrology community trains on: one CSV per gauge with daily ERA5-Land
forcing and area-normalised streamflow (mm/d), plus three attribute tables
per sub-dataset. Extending it normally means running Earth Engine. The
Archive already holds every ingredient, so this module writes the same
layout from it, with two honest differences that every file records:

* forcing is Open-Meteo's reanalysis blend (ERA5-Land where available, ERA5
  otherwise) **at the gauge point**, not a basin average, and only the daily
  variables Open-Meteo serves (precipitation, FAO-56 ET0, 2 m temperature
  mean/min/max, downward shortwave radiation);
* HydroATLAS attributes are BasinATLAS's own upstream-aggregated values for the
  level-12 sub-basin containing the gauge (its outlet, not the gauge itself,
  closes the catchment), written under Caravan's column names, with the raw
  sub-basin row kept next to them.

Layout written under ``out_dir`` (mirrors the Caravan tree, prefix = the
sub-dataset name, default ``aquascope_<source>``)::

    attributes/<prefix>/attributes_other_<prefix>.csv
        gauge_id, gauge_name, gauge_lat, gauge_lon, area, country (+ provenance columns)
    attributes/<prefix>/attributes_caravan_<prefix>.csv
        climate indices, computed exactly as caravan_utils.calculate_climate_indices
    attributes/<prefix>/attributes_hydroatlas_<prefix>.csv
        catchment values under Caravan/HydroATLAS names
    attributes/<prefix>/attributes_basinatlas_raw_<prefix>.csv
        the containing sub-basin's full BasinATLAS row
    timeseries/csv/<prefix>/<gauge_id>.csv
        date, forcing columns, streamflow (mm/d), rounded to 2 decimals
    licenses/<prefix>.md          per-source terms and attributions
    provenance.json               what came from where, versions, dates

Streamflow comes from the archive bundle for the source (``obs/discharge/<source>.parquet``)
and, when asked, from the agency for stations the archive has not reached yet.
Areas come from the agency where it publishes one (USGS drainage area, UK EA
``catchmentArea``, Hub'Eau ``surface_bv``), else from BasinATLAS ``up_area``
of the containing sub-basin (flagged).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aquascope import __version__
from aquascope.registry import SOURCES, build_collector

logger = logging.getLogger(__name__)

# Caravan's reference period for the climate indices (caravan_utils._CARAVAN_START_DATE/_END_DATE).
CARAVAN_START = pd.Timestamp("1981-01-01")
CARAVAN_END = pd.Timestamp("2020-12-31")

# Caravan column <- Open-Meteo daily variable, unit, note. Names follow Caravan / ERA5-Land where the
# quantity is the same; where it is not (downward instead of net radiation) the ERA5 name of what we
# actually serve is used, so nothing is mislabelled.
FORCING: list[tuple[str, str, str, str]] = [
    ("total_precipitation_sum", "precipitation_sum", "mm", "daily precipitation at the gauge point"),
    ("potential_evaporation_sum_FAO_PENMAN_MONTEITH", "et0_fao_evapotranspiration", "mm",
     "FAO-56 Penman-Monteith reference ET0 computed by Open-Meteo from the reanalysis"),
    ("temperature_2m_mean", "temperature_2m_mean", "degC", "daily mean 2 m air temperature"),
    ("temperature_2m_min", "temperature_2m_min", "degC", "daily minimum 2 m air temperature"),
    ("temperature_2m_max", "temperature_2m_max", "degC", "daily maximum 2 m air temperature"),
    ("surface_solar_radiation_downwards_mean", "shortwave_radiation_sum", "W/m2",
     "daily mean downward shortwave radiation (Open-Meteo MJ/m2/day x 11.574); "
     "Caravan's net solar radiation is not served"),
]
OPEN_METEO_DAILY = [om for _, om, _, _ in FORCING]
STREAMFLOW_COL = "streamflow"

SUPPORTED_SOURCES = ("usgs", "uk_ea", "hubeau_hydrometrie")


@dataclass
class GaugeExport:
    gauge_id: str
    source: str
    station_id: str
    ok: bool
    n_days: int = 0
    n_streamflow: int = 0
    area_km2: float | None = None
    area_source: str = ""
    hybas_id: int | None = None
    error: str = ""
    seconds: float = 0.0


@dataclass
class CaravanReport:
    prefix: str
    out_dir: str
    run_at: str
    aquascope_version: str
    gauges: list[GaugeExport] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for g in self.gauges if g.ok)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "n_ok": self.n_ok, "n_failed": len(self.gauges) - self.n_ok}


def default_prefix(source: str) -> str:
    return f"aquascope_{source}"


def gauge_id_for(prefix: str, station_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(station_id))
    return f"{prefix}_{safe}"


# ── climate indices (port of caravan_utils.calculate_climate_indices, FAO-PM PET only) ─────


def _split_runs(idx: np.ndarray) -> list[np.ndarray]:
    """Consecutive-index runs (caravan_utils._split_list)."""
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) != 1)[0] + 1
    return [r for r in np.split(idx, breaks) if r.size]


def moisture_and_seasonality(precip: pd.Series, pet: pd.Series) -> tuple[float, float]:
    """Knoben et al. (2018) annual moisture index and seasonality, as Caravan computes them."""
    mp = precip.groupby(precip.index.month).mean()
    me = pet.groupby(pet.index.month).mean()
    gt = 1 - me[mp > me] / mp[mp > me]
    eq = pd.Series(0.0, index=me.index)[mp == me]
    lt = mp[mp < me] / me[mp < me] - 1
    monthly = pd.concat([gt, eq, lt])
    return float(monthly.mean()), float(monthly.max() - monthly.min())


def climate_indices(
    df: pd.DataFrame, start: pd.Timestamp = CARAVAN_START, end: pd.Timestamp = CARAVAN_END
) -> dict[str, float]:
    """Caravan's climate indices from ``total_precipitation_sum``, the FAO-PM PET column and ``temperature_2m_mean``.

    Same definitions and thresholds as ``caravan_utils.calculate_climate_indices``; only the
    FAO Penman-Monteith variants are produced (Open-Meteo serves no ERA5-Land potential evaporation).
    """
    d = df.loc[slice(start, end)]
    p = d["total_precipitation_sum"].astype(float)
    pet = d["potential_evaporation_sum_FAO_PENMAN_MONTEITH"].astype(float)
    t = d["temperature_2m_mean"].astype(float)
    p_mean = float(p.mean())
    pet_mean = float(pet.mean())
    aridity = pet_mean / p_mean if p_mean else float("nan")
    moisture, seasonality = moisture_and_seasonality(p, pet)
    mp = p.groupby(p.index.month).mean()
    mt = t.groupby(t.index.month).mean()
    frac_snow = float(mp[mt < 0].sum() / mp.sum()) if float(mp.sum()) else float("nan")
    n = max(len(d), 1)
    high_freq = float((p >= 5 * p_mean).sum() / n)
    low_freq = float((p < 1).sum() / n)
    vals = p.to_numpy()
    low_runs = _split_runs(np.where(vals < 1)[0])
    high_runs = _split_runs(np.where(vals >= 5 * p_mean)[0])
    return {
        "p_mean": p_mean,
        "pet_mean_FAO_PM": pet_mean,
        "aridity_FAO_PM": aridity,
        "frac_snow": frac_snow,
        "moisture_index_FAO_PM": moisture,
        "seasonality_FAO_PM": seasonality,
        "high_prec_freq": high_freq,
        "high_prec_dur": float(np.mean([r.size for r in high_runs])) if high_runs else 0.0,
        "low_prec_freq": low_freq,
        "low_prec_dur": float(np.mean([r.size for r in low_runs])) if low_runs else 0.0,
    }


# ── inputs ──────────────────────────────────────────────────────────────────


RATE_LIMIT_WAIT = 65.0  # Open-Meteo's free tier is metered per minute; a 40-year request weighs a lot


def _is_rate_limited(exc: BaseException) -> bool:
    text = str(exc)
    return "429" in text or "Too Many" in text or "rate limit" in text.lower()


def fetch_forcing(
    lat: float, lon: float, start: date, end: date, *, models: str | None = "best_match", retries: int = 3
) -> tuple[pd.DataFrame, str]:
    """Daily forcing at a point from Open-Meteo, in Caravan column names (local time, like Caravan).

    Returns ``(frame, model_used)``. ``models="best_match"`` is Open-Meteo's blend (ERA5-Land where it
    has the variable, ERA5 otherwise) and the only option that serves every daily variable
    (``era5_land`` alone has no precipitation, ET0 or radiation); ERA5 via ``/era5`` is the fallback.
    A 429 waits :data:`RATE_LIMIT_WAIT` seconds and retries instead of failing the gauge.
    """
    weather = build_collector("openmeteo", mode="weather")
    attempts = [models, None] if models else [None]
    raw, used, last_exc = None, "", None
    for model in attempts:
        for _try in range(retries):
            try:
                raw = weather.fetch_raw(latitude=lat, longitude=lon, start_date=start.isoformat(),
                                        end_date=end.isoformat(), daily=OPEN_METEO_DAILY, models=model)
                used = model or "era5"
                break
            except Exception as exc:  # noqa: BLE001 - retry / fall back, then give up
                last_exc = exc
                if _is_rate_limited(exc) and _try < retries - 1:
                    logger.info("Open-Meteo rate limit, waiting %.0fs", RATE_LIMIT_WAIT)
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                break
        if raw is not None:
            break
    if raw is None:
        raise RuntimeError(f"Open-Meteo forcing failed: {last_exc}")
    daily = raw.get("daily", {})
    idx = pd.to_datetime(daily.get("time", []))
    out = pd.DataFrame(index=idx)
    out.index.name = "date"
    for col, om, unit, _ in FORCING:
        vals = pd.Series(daily.get(om, []), index=idx, dtype="float64")
        if om == "shortwave_radiation_sum":
            vals = vals * (1e6 / 86400.0)  # MJ/m2/day -> W/m2 mean
        out[col] = vals
    return out, used


def station_area_km2(source: str, station: dict[str, Any], collectors: dict[str, Any]) -> tuple[float | None, str]:
    """Catchment area from the agency where it publishes one; ``(None, "")`` otherwise."""
    sid = station["station_id"]
    known = (station.get("extra") or {}).get("catchment_area_km2")
    try:
        if known is not None and float(known) > 0:
            return float(known), f"{source}_catalog"
    except (TypeError, ValueError):
        pass
    try:
        if source == "usgs":
            c = collectors.setdefault("usgs", build_collector("usgs"))
            area = c._get_monitoring_location_catchment_area(sid)
            return (float(area), "usgs_drainage_area") if area else (None, "")
        if source == "uk_ea":
            c = collectors.setdefault("uk_ea", build_collector("uk_ea"))
            data = c.client.get_json(f"id/stations/{sid}.json")
            items = data.get("items") or []
            st = items[0] if isinstance(items, list) and items else items
            area = st.get("catchmentArea") if isinstance(st, dict) else None
            return (float(area), "uk_ea_catchmentArea") if area else (None, "")
        if source == "hubeau_hydrometrie":
            c = collectors.setdefault("hubeau_hydrometrie", build_collector("hubeau_hydrometrie"))
            code_site = (station.get("extra") or {}).get("code_site") or sid[:8]
            areas = c._get_catchment_areas({code_site})
            area = areas.get(code_site)
            return (float(area), "hubeau_surface_bv") if area else (None, "")
    except Exception as exc:  # noqa: BLE001 - area is optional, fall back to BasinATLAS
        logger.info("area lookup failed for %s/%s: %s", source, sid, exc)
    return None, ""


def _hydroatlas_catchment_row(raw_row: dict[str, Any]) -> dict[str, Any]:
    """Caravan/HydroATLAS column names filled with the catchment (upstream) value where BasinATLAS has one.

    ``xxx_yy_s??`` takes the sibling ``xxx_yy_u??`` when present (BasinATLAS's own upstream aggregate),
    else the local sub-basin value; ``_p??`` (pour point) fields are already catchment values.
    """
    out: dict[str, Any] = {}
    for col, val in raw_row.items():
        if not re.match(r"^[a-z]{3}_[a-z0-9]{2}_[sp][a-z0-9]{2}$", col):
            continue
        if col[7] == "s":
            sib = col[:7] + "u" + col[8:]
            v = raw_row.get(sib)
            out[col] = v if v is not None and not (isinstance(v, float) and math.isnan(v)) else val
        else:
            out[col] = val
    return out


# ── export ──────────────────────────────────────────────────────────────────


def export_caravan(
    source: str,
    out_dir: str | Path,
    *,
    station_ids: list[str] | None = None,
    max_stations: int | None = None,
    min_years: float = 10.0,
    start: date | None = None,
    end: date | None = None,
    prefix: str | None = None,
    forcing: bool = True,
    forcing_models: str | None = "best_match",
    fetch_missing: bool = False,
    catalog: list[dict[str, Any]] | None = None,
    observations: pd.DataFrame | None = None,
    write_netcdf: bool = False,
    pause: float = 3.0,
    on_event: Any = None,
) -> CaravanReport:
    """Write a Caravan-format sub-dataset for ``source`` from the Archive.

    Stations: ``station_ids`` when given, else every discharge station of the source that the
    archive holds observations for (``max_stations`` caps it, longest records first). A station
    needs ``min_years`` of daily streamflow to be exported. ``start``/``end`` bound the forcing
    period (defaults: 1981-01-01 to the last observation, capped at seven days ago); the
    streamflow is NaN outside its own record, like Caravan. ``fetch_missing`` fetches stations
    the archive has not reached from the agency through ``fetch_series``. ``catalog`` and
    ``observations`` (a bundle frame) can be injected (tests, offline runs). ``pause`` seconds
    separate the Open-Meteo calls (its free tier is metered per minute).
    """
    from aquascope.archive import basins as basins_mod
    from aquascope.archive.bundles import load_observations, to_series
    from aquascope.archive.catalog import load_stations

    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"{source!r} is not exportable yet; choose from {list(SUPPORTED_SOURCES)}")
    meta = SOURCES[source]
    say = on_event or (lambda _m: None)
    prefix = prefix or default_prefix(source)
    out = Path(out_dir)
    ts_dir = out / "timeseries" / "csv" / prefix
    attr_dir = out / "attributes" / prefix
    ts_dir.mkdir(parents=True, exist_ok=True)
    attr_dir.mkdir(parents=True, exist_ok=True)
    report = CaravanReport(prefix=prefix, out_dir=str(out), aquascope_version=__version__,
                           run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))

    catalog = catalog if catalog is not None else load_stations()
    by_id = {r["station_id"]: r for r in catalog if r["source"] == source}
    obs = observations if observations is not None else load_observations(source, "discharge")
    have = set(obs["station_id"].unique()) if not obs.empty else set()

    if station_ids:
        picked = [s for s in station_ids if s in by_id]
        missing_ids = [s for s in station_ids if s not in by_id]
        for s in missing_ids:
            report.gauges.append(GaugeExport(gauge_id_for(prefix, s), source, s, False, error="not in the catalog"))
    else:
        counts = obs.groupby("station_id").size() if not obs.empty else pd.Series(dtype=int)
        picked = [s for s in counts.sort_values(ascending=False).index if s in by_id]
        if max_stations:
            picked = picked[:max_stations]

    end_default = date.today() - timedelta(days=7)
    collectors: dict[str, Any] = {}
    other_rows, caravan_rows, hydro_rows, raw_rows = [], [], [], []
    for sid in picked:
        t0 = time.perf_counter()
        st = by_id[sid]
        gid = gauge_id_for(prefix, sid)
        g = GaugeExport(gid, source, sid, False)
        try:
            # streamflow
            if sid in have:
                q = to_series(obs, sid)
            elif fetch_missing:
                from aquascope.explore import fetch_series

                fetched = fetch_series(source, sid, years=60, prefer_archive=False, variable="discharge")
                q = fetched["series"] if fetched["series"] is not None else pd.Series(dtype=float)
            else:
                raise RuntimeError("no archived discharge (pass fetch_missing=True to fetch from the agency)")
            q = q.dropna()
            q = q[q >= 0]
            q_daily = q.resample("D").mean()
            valid = q_daily.dropna()
            years = (valid.index.max() - valid.index.min()).days / 365.25 if len(valid) > 1 else 0
            if years < min_years:
                raise RuntimeError(f"record too short: {years:.1f} years < {min_years}")

            # area
            area, area_src = station_area_km2(source, st, collectors)
            # BasinATLAS
            sb = None
            hybas_row: dict[str, Any] = {}
            try:
                sb = basins_mod.sub_basin_at(float(st["latitude"]), float(st["longitude"]))
                if sb:
                    attrs = basins_mod.load_attributes([sb["hybas_id"]])
                    if not attrs.empty:
                        hybas_row = {k: (None if isinstance(v, float) and math.isnan(v) else v)
                                     for k, v in attrs.iloc[0].to_dict().items()}
            except Exception as exc:  # noqa: BLE001 - attributes are optional per gauge
                logger.info("BasinATLAS lookup failed for %s: %s", gid, exc)
            if area is None and sb and sb.get("up_area"):
                area, area_src = float(sb["up_area"]), "basinatlas_up_area_of_containing_sub_basin"
            if not area:
                raise RuntimeError("no catchment area available (agency has none, BasinATLAS lookup failed)")

            # mm/day
            q_mm = q_daily * 86.4 / float(area)
            first, last = q_mm.dropna().index.min().date(), q_mm.dropna().index.max().date()
            f_start = start or CARAVAN_START.date()
            f_end = end or min(last, end_default)
            if f_end < f_start:
                raise RuntimeError(f"empty period {f_start}..{f_end}")

            model_used = ""
            if forcing:
                say(f"{gid}: forcing {f_start}..{f_end}")
                if pause and any(x.ok for x in report.gauges):
                    time.sleep(pause)
                frame, model_used = fetch_forcing(float(st["latitude"]), float(st["longitude"]), f_start, f_end,
                                                  models=forcing_models)
            else:
                idx = pd.date_range(f_start, f_end, freq="D", name="date")
                frame = pd.DataFrame(index=idx)
            frame[STREAMFLOW_COL] = q_mm.reindex(frame.index)
            frame = frame.round(2)
            frame.to_csv(ts_dir / f"{gid}.csv", float_format="%.2f", date_format="%Y-%m-%d")
            if write_netcdf:
                try:
                    nc_dir = out / "timeseries" / "netcdf" / prefix
                    nc_dir.mkdir(parents=True, exist_ok=True)
                    frame.to_xarray().to_netcdf(nc_dir / f"{gid}.nc")
                except Exception as exc:  # noqa: BLE001 - xarray/netCDF4 are optional
                    logger.info("netcdf skipped for %s: %s", gid, exc)

            other_rows.append({
                "gauge_id": gid, "gauge_name": st.get("name") or st.get("river") or sid,
                "gauge_lat": float(st["latitude"]), "gauge_lon": float(st["longitude"]),
                "area": round(float(area), 2), "country": st.get("country") or meta.country,
                "area_source": area_src, "source": source, "agency": meta.agency, "station_id": sid,
                "license": meta.license, "streamflow_start": first.isoformat(), "streamflow_end": last.isoformat(),
                "hybas_id": sb["hybas_id"] if sb else None,
                "basinatlas_up_area": sb.get("up_area") if sb else None,
                "forcing_model": model_used,
            })
            if forcing and len(frame) > 30:
                caravan_rows.append({"gauge_id": gid, **climate_indices(frame)})
            if hybas_row:
                hydro_rows.append({"gauge_id": gid, **_hydroatlas_catchment_row(hybas_row)})
                raw_rows.append({"gauge_id": gid, **hybas_row})
            g.ok, g.n_days, g.n_streamflow = True, int(len(frame)), int(frame[STREAMFLOW_COL].notna().sum())
            g.area_km2, g.area_source, g.hybas_id = float(area), area_src, (sb["hybas_id"] if sb else None)
        except Exception as exc:  # noqa: BLE001 - one gauge must not sink the export
            g.error = f"{type(exc).__name__}: {exc}"
            logger.warning("%s skipped: %s", gid, g.error)
        g.seconds = round(time.perf_counter() - t0, 1)
        report.gauges.append(g)
        say(f"{gid}: {'ok' if g.ok else 'skipped'} ({g.seconds}s)")

    if other_rows:
        pd.DataFrame(other_rows).set_index("gauge_id").sort_index().to_csv(attr_dir / f"attributes_other_{prefix}.csv")
    if caravan_rows:
        pd.DataFrame(caravan_rows).set_index("gauge_id").sort_index().sort_index(axis=1).to_csv(
            attr_dir / f"attributes_caravan_{prefix}.csv")
    if hydro_rows:
        pd.DataFrame(hydro_rows).set_index("gauge_id").sort_index().sort_index(axis=1).to_csv(
            attr_dir / f"attributes_hydroatlas_{prefix}.csv")
    if raw_rows:
        pd.DataFrame(raw_rows).set_index("gauge_id").sort_index().to_csv(
            attr_dir / f"attributes_basinatlas_raw_{prefix}.csv")

    lic_dir = out / "licenses"
    lic_dir.mkdir(parents=True, exist_ok=True)
    (lic_dir / f"{prefix}.md").write_text(_license_text(source, forcing), encoding="utf-8")
    prov_path = out / "provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {"subdatasets": {}}
    prov["subdatasets"][prefix] = {
        "source": source, "agency": meta.agency, "license": meta.license, "attribution": meta.attribution,
        "run_at": report.run_at, "aquascope_version": __version__,
        "streamflow": f"Archive bundle obs/discharge/{source}.parquet (daily means, m3/s) converted to mm/d "
                      "with the gauge area"
                      + ("; stations missing from the archive fetched from the agency" if fetch_missing else ""),
        "forcing": {"provider": "Open-Meteo", "models": forcing_models or "era5",
                    "at": "gauge point (not basin average)",
                    "columns": {c: {"open_meteo": om, "unit": u, "note": n} for c, om, u, n in FORCING},
                    "timezone": "local (Open-Meteo timezone=auto), like Caravan's daily aggregation"}
        if forcing else None,
        "climate_indices": "caravan_utils.calculate_climate_indices ported; FAO Penman-Monteith variants only; "
                           f"period {CARAVAN_START.date()}..{CARAVAN_END.date()} intersected with the forcing",
        "hydroatlas": "BasinATLAS v1.0 (CC BY 4.0) row of the level-12 sub-basin containing the gauge; "
                      "attributes_hydroatlas uses the upstream (u) fields under Caravan's column names, "
                      "attributes_basinatlas_raw keeps the full row",
        "area": "agency catchment area where published (usgs drainage_area, uk_ea catchmentArea, hubeau surface_bv), "
                "else BasinATLAS up_area of the containing sub-basin (see area_source)",
        "n_gauges": report.n_ok,
    }
    prov_path.write_text(json.dumps(prov, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "README.md").write_text(_readme_text(), encoding="utf-8")
    return report


def _license_text(source: str, forcing: bool) -> str:
    meta = SOURCES[source]
    lines = [
        f"# Terms for the `{default_prefix(source)}` sub-dataset", "",
        f"- Streamflow: {meta.agency}, licence `{meta.license}`. {meta.attribution}",
        "- Catchment attributes: HydroATLAS v1.0 / BasinATLAS, CC BY 4.0. Linke, S., Lehner, B., Ouellet Dallaire, C., "
        "et al. (2019). Global hydro-environmental sub-basin and river reach characteristics at high spatial "
        "resolution. Scientific Data 6: 283. https://doi.org/10.1038/s41597-019-0300-6",
    ]
    if forcing:
        lines.append("- Forcing: ERA5-Land / ERA5 (Copernicus Climate Change Service) blended and served by "
                     "Open-Meteo, CC BY 4.0. Hersbach, H. et al. (2020), Munoz-Sabater, J. et al. (2021); "
                     "Open-Meteo.com.")
    lines += ["", "Layout after Kratzert, F. et al. (2023). Caravan: a global community dataset for large-sample "
              "hydrology. Scientific Data 10: 61. https://doi.org/10.1038/s41597-023-01975-w"]
    return "\n".join(lines) + "\n"


def _readme_text() -> str:
    return (
        "# Caravan-format export from the AquaScope Archive\n\n"
        "Written by `aquascope caravan export`. Same folder layout as Caravan (Kratzert et al. 2023): "
        "`attributes/<prefix>/attributes_{other,caravan,hydroatlas}_<prefix>.csv` and "
        "`timeseries/csv/<prefix>/<gauge_id>.csv` (date, forcing, `streamflow` in mm/d).\n\n"
        "Differences from Caravan proper, all recorded in `provenance.json`: forcing is ERA5-Land/ERA5 at the gauge "
        "point (Open-Meteo's ERA5-Land + ERA5 blend), not a basin average, and only the daily variables "
        "Open-Meteo serves; the "
        "potential evaporation is FAO-56 ET0 (`potential_evaporation_sum_FAO_PENMAN_MONTEITH`), there is no "
        "ERA5-Land PEV; "
        "radiation is downward shortwave (`surface_solar_radiation_downwards_mean`), not net; HydroATLAS attributes "
        "are BasinATLAS's upstream values for the level-12 sub-basin containing the gauge "
        "(`attributes_basinatlas_raw_*` keeps the full row); the area is the agency's where it publishes one, "
        "else BasinATLAS's (`area_source`). "
        "No basin shapefiles are written.\n"
    )


# ── validation ──────────────────────────────────────────────────────────────

CARAVAN_OTHER_REQUIRED = ("gauge_lat", "gauge_lon", "gauge_name", "country", "area")
CARAVAN_INDEX_REQUIRED = ("p_mean", "pet_mean_FAO_PM", "aridity_FAO_PM", "frac_snow", "moisture_index_FAO_PM",
                          "seasonality_FAO_PM", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur")


def validate_caravan(out_dir: str | Path, prefix: str) -> dict[str, Any]:
    """Structural checks against the Caravan layout; returns ``{"ok": bool, "problems": [...], "n_gauges": int}``."""
    out = Path(out_dir)
    problems: list[str] = []
    attr_dir = out / "attributes" / prefix
    other_p = attr_dir / f"attributes_other_{prefix}.csv"
    if not other_p.exists():
        return {"ok": False, "problems": [f"missing {other_p}"], "n_gauges": 0}
    other = pd.read_csv(other_p, index_col="gauge_id")
    for c in CARAVAN_OTHER_REQUIRED:
        if c not in other.columns:
            problems.append(f"attributes_other lacks {c}")
    if other[[c for c in CARAVAN_OTHER_REQUIRED if c in other.columns]].isna().any().any():
        problems.append("attributes_other has NaN in a required column")
    ids = list(other.index)
    if not all(str(i).startswith(prefix + "_") for i in ids):
        problems.append("gauge_ids do not all carry the prefix")
    for name in ("caravan", "hydroatlas"):
        p = attr_dir / f"attributes_{name}_{prefix}.csv"
        if p.exists():
            df = pd.read_csv(p, index_col="gauge_id")
            extra = set(df.index) - set(ids)
            if extra:
                problems.append(f"attributes_{name} has gauge_ids not in attributes_other: {sorted(extra)[:3]}")
            if name == "caravan":
                for c in CARAVAN_INDEX_REQUIRED:
                    if c not in df.columns:
                        problems.append(f"attributes_caravan lacks {c}")
    ts_dir = out / "timeseries" / "csv" / prefix
    for gid in ids:
        p = ts_dir / f"{gid}.csv"
        if not p.exists():
            problems.append(f"missing timeseries for {gid}")
            continue
        ts = pd.read_csv(p, index_col="date", parse_dates=["date"])
        if STREAMFLOW_COL not in ts.columns:
            problems.append(f"{gid}: no streamflow column")
        elif (ts[STREAMFLOW_COL].dropna() < 0).any():
            problems.append(f"{gid}: negative streamflow (must be NaN)")
        if len(ts) > 1 and pd.infer_freq(ts.index) != "D":
            problems.append(f"{gid}: date index is not daily")
    return {"ok": not problems, "problems": problems, "n_gauges": len(ids)}


__all__ = ["CARAVAN_END", "CARAVAN_START", "CaravanReport", "FORCING", "GaugeExport", "climate_indices",
           "export_caravan", "fetch_forcing", "gauge_id_for", "station_area_km2", "validate_caravan"]
