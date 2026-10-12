"""The nearest real rain gauge: NOAA GHCN-Daily on AWS Open Data (#520).

The station index comes from the Archive's context mirror (``ghcn/prcp_stations.csv.gz``: only stations
that record precipitation, with the years they cover). Until that is published the full GHCN station
list (about 11 MB) is read once instead and a station without precipitation is skipped when its record is
opened. Records are the per-station ``csv.gz`` files on ``noaa-ghcn-pds`` (CORS open, CC0).
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
from datetime import date
from typing import Any

from aquascope.context._common import (
    DEFAULT_REPO_ID,
    check_bbox,
    check_point,
    dataset_info,
    failed,
    get_bytes,
    haversine_km,
    http_client,
    in_bbox,
    layer_result,
    memo,
    mirror_url,
    num,
    read_csv_gz,
)

logger = logging.getLogger(__name__)

__all__ = ["parse_stations_txt", "rain_gauge", "rain_gauges_area", "station_index", "summarise_prcp"]

GHCN_BASE = "https://noaa-ghcn-pds.s3.amazonaws.com"
#: A year counts as complete with this many days of precipitation (missing days would bias the total low).
COMPLETE_DAYS = 330


def parse_stations_txt(text: str) -> list[dict[str, Any]]:
    """The fixed-width ``ghcnd-stations.txt`` as rows of id, lat, lon, elev_m, name."""
    out = []
    for line in text.splitlines():
        if len(line) < 41:
            continue
        try:
            lat, lon = float(line[12:20]), float(line[21:30])
        except ValueError:
            continue
        elev = num(line[31:37].strip())
        out.append({"id": line[0:11].strip(), "lat": lat, "lon": lon,
                    "elev_m": None if elev is None or elev <= -999 else elev, "name": line[41:71].strip(),
                    "first_year": None, "last_year": None})
    return out


def station_index(repo_id: str = DEFAULT_REPO_ID) -> tuple[list[dict[str, Any]], str]:
    """(stations, where from): the mirror's precipitation stations, else the full GHCN list."""
    def build() -> tuple[list[dict[str, Any]], str]:
        if dataset_info("ghcn", repo_id) is not None:
            raw = get_bytes(mirror_url("ghcn/prcp_stations.csv.gz", repo_id))
            if raw:
                rows = []
                for r in read_csv_gz(raw):
                    la, lo = num(r.get("lat")), num(r.get("lon"))
                    if la is None or lo is None:
                        continue
                    fy, ly = num(r.get("first_year")), num(r.get("last_year"))
                    rows.append({"id": r["id"], "lat": la, "lon": lo, "elev_m": num(r.get("elev_m")),
                                 "name": r.get("name") or r["id"], "first_year": int(fy) if fy else None,
                                 "last_year": int(ly) if ly else None})
                return rows, "mirror"
        text = http_client().get_text(f"{GHCN_BASE}/ghcnd-stations.txt")
        return parse_stations_txt(text), "ghcnd-stations.txt"

    return memo(f"ghcn_index:{repo_id}", build)


def summarise_prcp(csv_gz: bytes, *, years: int = 10, today: date | None = None) -> dict[str, Any] | None:
    """The precipitation part of one GHCN-Daily station file; None when it has no precipitation.

    Values are tenths of a millimetre; rows with a quality flag (failed a GHCN check) are dropped.
    """
    text = gzip.decompress(csv_gz).decode("utf-8", errors="replace")
    daily: dict[str, float] = {}
    flagged = 0
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 6 or row[2] != "PRCP":
            continue
        if row[5].strip():
            flagged += 1
            continue
        try:
            daily[row[1]] = float(row[3]) / 10.0
        except ValueError:
            continue
    if not daily:
        return None
    days = sorted(daily)
    per_year: dict[int, list[float]] = {}
    for d, v in daily.items():
        per_year.setdefault(int(d[:4]), []).append(v)
    today = today or date.today()
    complete = {y: round(sum(v), 1) for y, v in per_year.items() if len(v) >= COMPLETE_DAYS and y < today.year}
    recent = dict(sorted(complete.items())[-max(1, int(years)):]) if complete else {}
    wettest_day = max(daily.items(), key=lambda kv: kv[1])
    first, last = days[0], days[-1]
    return {
        "first_date": f"{first[:4]}-{first[4:6]}-{first[6:]}", "last_date": f"{last[:4]}-{last[4:6]}-{last[6:]}",
        "n_days": len(daily), "flagged_days_dropped": flagged, "complete_years": len(complete),
        "mean_annual_mm": round(sum(complete.values()) / len(complete), 1) if complete else None,
        "annual_mm": {str(y): v for y, v in recent.items()},
        "wettest_day": {"date": f"{wettest_day[0][:4]}-{wettest_day[0][4:6]}-{wettest_day[0][6:]}",
                        "mm": wettest_day[1]},
    }


def _record(station_id: str) -> bytes | None:
    return get_bytes(f"{GHCN_BASE}/csv.gz/by_station/{station_id}.csv.gz")


def rain_gauge(lat: float, lon: float, *, max_km: float = 150.0, years: int = 10, tries: int = 3,
               repo_id: str = DEFAULT_REPO_ID) -> dict[str, Any]:
    """The nearest GHCN-Daily station with a precipitation record, and what that record holds.

    ``mean_annual_mm`` is over complete years (at least 330 days with a value); ``annual_mm`` lists the
    last ``years`` complete years. Up to ``tries`` stations are opened before giving up.
    """
    lat, lon = check_point(lat, lon)
    try:
        stations, origin = station_index(repo_id)
        near = sorted(((haversine_km(lat, lon, s["lat"], s["lon"]), s) for s in stations), key=lambda t: t[0])
        near = [(d, s) for d, s in near if d <= max_km][:max(1, int(tries))]
        chosen: dict[str, Any] | None = None
        record: dict[str, Any] | None = None
        skipped = []
        for dist, st in near:
            raw = _record(st["id"])
            summary = summarise_prcp(raw, years=years) if raw else None
            if summary is None:
                skipped.append(st["id"])
                continue
            chosen, record = {**st, "distance_km": round(dist, 1)}, summary
            break
    except Exception as exc:  # noqa: BLE001
        return failed("rain_gauge", ["rain_gauge"], exc)
    if chosen is None or record is None:
        return layer_result("rain_gauge", ["rain_gauge"], ok=True, lat=lat, lon=lon, station=None,
                            index=origin, skipped=skipped,
                            summary=f"No GHCN-Daily rain gauge with a record within {max_km:g} km.")
    span = f"{record['first_date'][:4]} to {record['last_date'][:4]}"
    text = f"Nearest rain gauge: {chosen['name'].title()} ({chosen['id']}), {chosen['distance_km']:.0f} km away, {span}"
    if record["mean_annual_mm"] is not None:
        text += (f"; {record['mean_annual_mm']:,.0f} mm a year on average over {record['complete_years']} "
                 f"complete year{'s' if record['complete_years'] != 1 else ''}")
    return layer_result("rain_gauge", ["rain_gauge"], ok=True, lat=lat, lon=lon, station=chosen, record=record,
                        index=origin, skipped=skipped,
                        url=f"https://www.ncei.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:{chosen['id']}/detail",
                        summary=text + ".")


def rain_gauges_area(west: float, south: float, east: float, north: float, *, limit: int = 20,
                     repo_id: str = DEFAULT_REPO_ID) -> dict[str, Any]:
    """GHCN-Daily precipitation stations inside a box (no records opened)."""
    bbox = check_bbox(west, south, east, north)
    try:
        stations, origin = station_index(repo_id)
    except Exception as exc:  # noqa: BLE001
        return failed("rain_gauge", ["rain_gauge"], exc)
    inside = [s for s in stations if in_bbox(s["lat"], s["lon"], bbox)]
    inside.sort(key=lambda s: -((s.get("last_year") or 0) - (s.get("first_year") or 0)))
    what = "rain gauges" if origin == "mirror" else "GHCN-Daily stations"
    return layer_result("rain_gauge", ["rain_gauge"], ok=True, bbox=list(bbox), n_stations=len(inside),
                        stations=inside[:max(0, int(limit))], index=origin,
                        summary=f"{len(inside)} {what} in this box." if inside else f"No {what} in this box.")
