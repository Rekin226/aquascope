"""
Collector for Japan MLIT — Ministry of Land, Infrastructure, Transport and Tourism.

Data is provided by the Water Information System (水文水質データベース):
    https://www1.river.go.jp/

The site used to answer CGI queries under ``/cgi-bin/``; those paths now answer
404 because the site was rebuilt as a Next.js application.  Its own endpoints
live under ``/water-inf-app/api/`` and answer JSON for the code tables and
station search, and a ZIP of Shift_JIS CSV for the observations.

Those endpoints are undocumented and the site asks automated clients to stay
away: ``robots.txt`` disallows ``/water-inf-app/`` and sets
``Crawl-delay: 3600``, and the FAQ points at the paid 水防災オープンデータ
service for programmatic access.  The collector therefore takes one request per
hour, caches for a day, and never walks the country by default.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timedelta
from typing import Any

from aquascope.collectors.base import BaseCollector, CollectorError
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

BASE_URL = "https://www1.river.go.jp/water-inf-app/api/"

#: The SPA sends these with its own fetches; without them every path under
#: ``/water-inf-app/`` answers 403, including the JSON code tables.
REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Referer": "https://www1.river.go.jp/search/obs",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Origin": "https://www1.river.go.jp",
}

# ---------------------------------------------------------------------------
# Prefecture codes — the four digit codes from ``ken.json``
# ---------------------------------------------------------------------------
PREFECTURE_CODES: dict[str, str] = {
    "Hokkaido": "0101",
    "Aomori": "0201",
    "Iwate": "0301",
    "Miyagi": "0401",
    "Akita": "0501",
    "Yamagata": "0601",
    "Fukushima": "0701",
    "Ibaraki": "0801",
    "Tochigi": "0901",
    "Gunma": "1001",
    "Saitama": "1101",
    "Chiba": "1201",
    "Tokyo": "1301",
    "Kanagawa": "1401",
    "Niigata": "1501",
    "Toyama": "1601",
    "Ishikawa": "1701",
    "Fukui": "1801",
    "Yamanashi": "1901",
    "Nagano": "2001",
    "Gifu": "2101",
    "Shizuoka": "2201",
    "Aichi": "2301",
    "Mie": "2401",
    "Shiga": "2501",
    "Kyoto": "2601",
    "Osaka": "2701",
    "Hyogo": "2801",
    "Nara": "2901",
    "Wakayama": "3001",
    "Tottori": "3101",
    "Shimane": "3201",
    "Okayama": "3301",
    "Hiroshima": "3401",
    "Yamaguchi": "3501",
    "Tokushima": "3601",
    "Kagawa": "3701",
    "Ehime": "3801",
    "Kochi": "3901",
    "Fukuoka": "4001",
    "Saga": "4101",
    "Nagasaki": "4201",
    "Kumamoto": "4301",
    "Oita": "4401",
    "Miyazaki": "4501",
    "Kagoshima": "4601",
    "Okinawa": "4701",
}

# ---------------------------------------------------------------------------
# Major river systems (Japanese name → English transliteration)
# ---------------------------------------------------------------------------
JAPAN_RIVER_SYSTEMS: dict[str, str] = {
    "利根川": "Tone",
    "信濃川": "Shinano",
    "石狩川": "Ishikari",
    "天竜川": "Tenryu",
    "木曽川": "Kiso",
    "淀川": "Yodo",
    "筑後川": "Chikugo",
    "北上川": "Kitakami",
    "最上川": "Mogami",
    "阿賀野川": "Agano",
    "荒川": "Ara",
    "多摩川": "Tama",
    "四万十川": "Shimanto",
    "吉野川": "Yoshino",
    "那珂川": "Naka",
}

# ---------------------------------------------------------------------------
# Japanese parameter name → English equivalent
# ---------------------------------------------------------------------------
PARAMETER_MAP_JA: dict[str, str] = {
    "水位": "water_level",
    "流量": "discharge",
    "雨量": "rainfall",
    "水温": "water_temperature",
    "pH": "pH",
    "DO": "DO",
    "BOD": "BOD",
    "COD": "COD",
    "SS": "SS",
    "大腸菌群数": "coliform",
    "全窒素": "Total Nitrogen",
    "全燐": "Total Phosphorus",
    "透視度": "transparency",
}

# ---------------------------------------------------------------------------
# MLIT parameter → the export key and the 項目 code the station search needs
#
# ``hv``/``qv``/``rv``/``wqv`` are the sub-hourly lists and take ``from=YYYY-MM``;
# the yearly tables (``hd``/``qd``/``rd``) take ``from=YYYY`` and are not used
# here, because a daily mean is not a reading.
# ---------------------------------------------------------------------------
_PARAM_EXPORT: dict[str, tuple[str, str, str]] = {
    "water_level": ("hv", "02", "m"),
    "discharge": ("qv", "02", "m3/s"),
    "rainfall": ("rv", "01", "mm/h"),
    "water_quality": ("wqv", "03", ""),
}

#: ``属性`` codes that mean the value is not a usable reading: 1 欠測, 3 閉局,
#: 8 工事中.  ``2`` (推定値) stays — it is an estimate, not a gap.
_INVALID_ATTRIBUTES = frozenset({"1", "3", "8"})

#: ``データ`` values that stand for a missing reading.
_MISSING_VALUES = frozenset({"", "-", "-9999"})

# ---------------------------------------------------------------------------
# Japanese Environmental Quality Standards (Basic Environment Law)
#
# Each classification maps to threshold values:
#   BOD (mg/L), SS (mg/L), DO (mg/L)
# See: https://www.env.go.jp/en/water/wq/wp.pdf
# ---------------------------------------------------------------------------
QUALITY_STANDARDS_JAPAN: dict[str, dict[str, float]] = {
    "AA": {"BOD": 1.0, "SS": 25.0, "DO": 7.5},
    "A": {"BOD": 2.0, "SS": 25.0, "DO": 7.5},
    "B": {"BOD": 3.0, "SS": 25.0, "DO": 5.0},
    "C": {"BOD": 5.0, "SS": 50.0, "DO": 5.0},
    "D": {"BOD": 8.0, "SS": 100.0, "DO": 2.0},
    "E": {"BOD": 10.0, "SS": -1.0, "DO": 2.0},
}


def parse_datetime(raw: str) -> datetime | None:
    """Parse a CSV ``日時`` cell, where ``24:00`` means midnight of the next day."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        date_part, time_part = text.split(" ", 1)
        hours, minutes = (int(part) for part in time_part.split(":"))
        base = datetime.strptime(date_part, "%Y/%m/%d")
    except (ValueError, TypeError):
        return None
    return base + timedelta(hours=hours, minutes=minutes)


def month_range(start_date: str | None, end_date: str | None) -> list[str]:
    """Expand an ISO date range into the ``YYYY-MM`` anchors the export accepts.

    The export has no range parameter, so every month is asked for by name.  A
    missing *start_date* means the current month, and a missing *end_date* means
    the single month *start_date* names.
    """
    if start_date:
        start = datetime.strptime(start_date[:7], "%Y-%m")
    else:
        start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if end_date:
        end = datetime.strptime(end_date[:7], "%Y-%m")
    else:
        end = start
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}.")
    months: list[str] = []
    cursor = start
    while cursor <= end:
        months.append(cursor.strftime("%Y-%m"))
        cursor = (cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
    return months


def parse_export(body: bytes, *, station: dict, parameter: str, unit: str) -> list[dict]:
    """Parse one ``value/{id}/{key}.zip`` body into raw observation rows.

    The archive holds a Shift_JIS CSV with the columns ``日時,データ,属性`` and,
    for water level and discharge, ``暫定フラグ``.  An empty export is a one line
    CSV rather than an error, and the site answers an HTML page with status 200
    when it refuses a request, so anything that is not a readable archive or not
    a ``日時`` table yields no rows.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile:
        logger.warning("MLIT answered %s with something that is not a ZIP archive", station.get("id"))
        return []

    names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if not names:
        return []
    text = archive.read(names[0]).decode("cp932", errors="replace")
    lines = text.splitlines()
    if not lines or not lines[0].startswith("日時"):
        return []

    rows: list[dict] = []
    for line in lines[1:]:
        cells = line.split(",")
        if len(cells) < 3:
            continue
        when = parse_datetime(cells[0])
        if when is None:
            continue
        raw_value = cells[1].strip()
        if raw_value in _MISSING_VALUES:
            continue
        attribute = cells[2].strip()
        if attribute in _INVALID_ATTRIBUTES:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        rows.append(
            {
                "station_id": station.get("id", ""),
                "station_name": station.get("name_data") or "",
                "river": station.get("suikei_name_data") or "",
                "latitude": station.get("wlatitude"),
                "longitude": station.get("wlongitude"),
                "datetime": when.isoformat(),
                "parameter": parameter,
                "value": value,
                "unit": unit,
                "attribute": attribute,
                "provisional": len(cells) > 3 and cells[3].strip() == "*",
            }
        )
    return rows


class JapanMLITCollector(BaseCollector):
    """
    Collect water data from Japan MLIT Water Information System.

    Supports water-level, discharge, water-quality, and rainfall observations.
    Results are normalised to ``WaterQualitySample`` records with
    ``source = DataSource.JAPAN_MLIT``.

    The site allows one request an hour (``Crawl-delay: 3600``), so a caller that
    asks for several stations and several months is paced accordingly; the cache
    keeps repeated reads of the same month off the network.
    """

    name: str = "japan_mlit"
    BASE_URL: str = BASE_URL

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                rate_limiter=RateLimiter(max_calls=1, period_seconds=3600),
                cache_ttl_seconds=86400,
            )
        )

    # ------------------------------------------------------------------ #
    # fetch_raw
    # ------------------------------------------------------------------ #
    def fetch_raw(
        self,
        station_id: str | None = None,
        prefecture: str | None = None,
        parameter: str = "water_level",
        start_date: str | None = None,
        end_date: str | None = None,
        max_stations: int = 10,
        **kwargs: Any,
    ) -> list[dict]:
        """
        Fetch raw observation data from the MLIT Water Information System.

        Parameters
        ----------
        station_id : str | None
            MLIT observation station code, the 15 digit ``id`` from the station
            search (e.g. ``"303031283301030"``).
        prefecture : str | None
            Prefecture name in English (e.g. ``"Tokyo"``).  Mapped to the four
            digit code from ``ken.json``; Hokkaido is its own first code.
        parameter : str
            One of ``water_level``, ``discharge``, ``water_quality``,
            ``rainfall``.
        start_date : str | None
            First month to read, ISO format (``YYYY-MM-DD`` or ``YYYY-MM``).
            Defaults to the current month.
        end_date : str | None
            Last month to read.  Defaults to *start_date*'s month; the export
            has no range parameter, so the months are asked for one by one.
        max_stations : int
            How many of a prefecture's stations to read, newest first.  The site
            answers a station search without a prefecture with every station in
            the country (about 2,400), which is why one of *station_id* or
            *prefecture* is required.
        **kwargs
            Additional keyword arguments forwarded to the HTTP request.

        Returns
        -------
        list[dict]
            Raw observation rows.  An empty export (a station without that
            observation) returns an empty list, not an error.

        Raises
        ------
        ValueError
            If *parameter* is not supported, if neither *station_id* nor
            *prefecture* is given, or if *end_date* precedes *start_date*.
        CollectorError
            If a request fails.
        """
        if parameter not in _PARAM_EXPORT:
            raise ValueError(f"Unsupported parameter '{parameter}'. Choose from: {list(_PARAM_EXPORT)}")
        if not station_id and not prefecture:
            raise ValueError(
                "japan_mlit needs a station_id or a prefecture: without either, the "
                "station search answers with every station in Japan."
            )

        export_key, komoku_code, unit = _PARAM_EXPORT[parameter]
        months = month_range(start_date, end_date)

        if station_id:
            stations: list[dict] = [{"id": station_id}]
        else:
            code = PREFECTURE_CODES.get(prefecture or "")
            if not code:
                raise ValueError(
                    f"Unknown prefecture {prefecture!r}. Known names: {sorted(PREFECTURE_CODES)}."
                )
            stations = self._find_stations(code, komoku_code)[:max_stations]

        rows: list[dict] = []
        for station in stations:
            for month in months:
                rows.extend(self._fetch_month(station, export_key, month, parameter, unit))
        return rows

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _find_stations(self, ken_code: str, komoku_code: str) -> list[dict]:
        """Search a prefecture's observation stations for one 項目 code."""
        url = f"{BASE_URL}comsite.json"
        try:
            data = self.client.get_json(
                url,
                params={"komoku_code": komoku_code, "ken_code": ken_code},
                headers=REQUEST_HEADERS,
            )
        except Exception as exc:
            logger.warning("MLIT station search failed for %s: %s", url, exc, exc_info=True)
            raise CollectorError(
                f"MLIT station search failed for {url}: {exc}",
                source=self.name,
                url=url,
                cause=exc,
            ) from exc
        if not isinstance(data, list):
            return []
        return [station for station in data if station.get("id")]

    def _fetch_month(
        self,
        station: dict,
        export_key: str,
        month: str,
        parameter: str,
        unit: str,
    ) -> list[dict]:
        """Read one station's export for one month."""
        url = f"{BASE_URL}value/{station['id']}/{export_key}.zip?from={month}"
        try:
            body = self.client.get_bytes(url, headers=REQUEST_HEADERS)
        except Exception as exc:
            logger.warning("MLIT export request failed for %s: %s", url, exc, exc_info=True)
            raise CollectorError(
                f"MLIT export request failed for {url}: {exc}",
                source=self.name,
                url=url,
                cause=exc,
            ) from exc
        return parse_export(body, station=station, parameter=parameter, unit=unit)

    # ------------------------------------------------------------------ #
    # normalise
    # ------------------------------------------------------------------ #
    def normalise(self, raw: list[dict]) -> list[WaterQualitySample]:
        """
        Normalise raw MLIT data into ``WaterQualitySample`` records.

        Parameters
        ----------
        raw : list[dict]
            Raw records as returned by ``fetch_raw``.

        Returns
        -------
        list[WaterQualitySample]
            Unified water-quality sample records.
        """
        if not raw:
            return []

        samples: list[WaterQualitySample] = []
        for row in raw:
            try:
                value = row.get("value")
                if value is None or str(value).strip() in ("", "-", "-9999", "ND", "--"):
                    continue
                value = float(value)

                dt_str = row.get("datetime", row.get("date", ""))
                if not dt_str:
                    logger.debug("Skipping MLIT row without datetime")
                    continue
                sample_dt = datetime.fromisoformat(str(dt_str))

                raw_param = str(row.get("parameter", ""))
                param_en = PARAMETER_MAP_JA.get(raw_param, raw_param)

                location = None
                lat = row.get("latitude")
                lon = row.get("longitude")
                if lat is not None and lon is not None:
                    location = GeoLocation(latitude=float(lat), longitude=float(lon))

                river_jp = row.get("river_system") or row.get("river") or ""
                river_en = JAPAN_RIVER_SYSTEMS.get(river_jp, river_jp)

                samples.append(
                    WaterQualitySample(
                        source=DataSource.JAPAN_MLIT,
                        station_id=str(row.get("station_id", row.get("station_code", "unknown"))),
                        station_name=row.get("station_name") or None,
                        location=location,
                        sample_datetime=sample_dt,
                        parameter=param_en,
                        value=value,
                        unit=str(row.get("unit", "")),
                        basin=river_en or None,
                        river=river_en or None,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping MLIT row: %s", exc)

        return samples
