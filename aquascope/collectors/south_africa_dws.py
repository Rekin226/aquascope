"""Collector for South African DWS Verified Hydrology data.

The DWS portal exposes a deterministic ``HyData.aspx`` GET interface rather
than a documented JSON API. Successful responses contain a whitespace table
inside a ``<pre>`` element. The public ASP.NET page can also return HTTP 200
while its downstream Kisters ScriptServer is unavailable, so response bodies
are validated before any rows are accepted as observations.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterator, Sequence
from datetime import date, datetime, time, timedelta
from html.parser import HTMLParser
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    StreamflowReading,
    WaterLevelReading,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

DWS_VERIFIED_BASE = "https://www.dws.gov.za/Hydrology/Verified"
DWS_SERIES_SUFFIX = "100.00"

_SUPPORTED_VARIABLES = {"discharge", "water_level"}
_STATION_RE = re.compile(r"^[A-Z0-9]{4,12}$")
_DATE_ROW_RE = re.compile(r"^\d{8}(?:\s|$)")
_THOUSANDS_NUMBER_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_APPLICATION_ERROR_MARKERS = (
    "scriptserverodbc",
    "can't connect to scriptserver",
    "client unable to establish connection",
)
_MISSING_VALUES = {"", "-", "--", "NA", "N/A", "NAN", "NULL", "M"}


class _PreTextParser(HTMLParser):
    """Extract text from ``<pre>`` elements without adding a dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() == "pre":
            self._depth += 1
        elif self._depth and tag.lower() == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "pre" and self._depth:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._depth:
            self.parts.append(data)

    @property
    def text(self) -> str:
        return "".join(self.parts)


class SouthAfricaDWSCollector(BaseCollector):
    """Collect verified river discharge and water level from South Africa DWS.

    ``discharge`` requests the DWS daily mean series (``D_AVG_FR``).
    ``water_level`` requests point observations (``COR_LEVEL``).
    """

    name = "south_africa_dws"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=DWS_VERIFIED_BASE,
                timeout=60.0,
                retries=3,
                cache_ttl_seconds=3600,
                rate_limiter=RateLimiter(max_calls=1, period_seconds=1.0),
            )
        )

    def fetch_raw(
        self,
        station_id: str,
        variable: str = "discharge",
        start_date: str | date | datetime | None = None,
        end_date: str | date | datetime | None = None,
        days: int | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Fetch and parse one DWS river series.

        Parameters use the agency station code without the DWS ``100.00``
        series suffix, for example ``C1H001``. A supplied suffix is accepted
        and normalised so it is never duplicated.

        Caching is disabled by default because DWS can return an application
        error document with HTTP 200; callers may opt in once the upstream
        service is known to be healthy.
        """
        del kwargs
        station = self._normalise_station_id(station_id)
        variable = str(variable).strip().lower()
        if variable not in _SUPPORTED_VARIABLES:
            allowed = ", ".join(sorted(_SUPPORTED_VARIABLES))
            raise ValueError(f"DWS variable must be one of: {allowed}; got {variable!r}.")

        start, end = self._resolve_dates(start_date=start_date, end_date=end_date, days=days)
        data_type = "Daily" if variable == "discharge" else "Point"
        chunk_years = 20 if variable == "discharge" else 1

        rows: list[dict[str, Any]] = []
        for chunk_start, chunk_end in self._iter_chunks(start, end, years_per_chunk=chunk_years):
            params = {
                "Station": f"{station}{DWS_SERIES_SUFFIX}",
                "DataType": data_type,
                "StartDT": chunk_start.isoformat(),
                "EndDT": chunk_end.isoformat(),
                "SiteType": "RIV",
            }
            body = self.client.get_text(
                "HyData.aspx",
                params=params,
                headers={
                    "Accept": "text/html, text/plain;q=0.9",
                    "User-Agent": "AquaScope (+https://github.com/Rekin226/aquascope)",
                },
                use_cache=use_cache,
            )
            parsed = self._parse_response(body, station_id=station, variable=variable)
            rows.extend(
                row
                for row in parsed
                if chunk_start <= row["reading_datetime"].date() <= chunk_end
            )

        # Calendar chunks should not overlap. Keep identical duplicates but
        # reject conflicting values rather than silently choosing one.
        unique: dict[datetime, dict[str, Any]] = {}
        for row in rows:
            key = row["reading_datetime"]
            previous = unique.get(key)
            if previous is not None and previous != row:
                raise ValueError(
                    "DWS returned conflicting observations for "
                    f"{station} at {key.isoformat()}."
                )
            unique[key] = row
        return [unique[key] for key in sorted(unique)]

    def normalise(
        self, raw: list[dict[str, Any]]
    ) -> Sequence[StreamflowReading | WaterLevelReading]:
        records: list[StreamflowReading | WaterLevelReading] = []
        for row in raw:
            try:
                when = row["reading_datetime"]
                if isinstance(when, str):
                    when = datetime.fromisoformat(when)
                if not isinstance(when, datetime):
                    raise TypeError("reading_datetime is not a datetime")

                value = float(row["value"])
                if not math.isfinite(value):
                    raise ValueError("value is not finite")

                station_id = str(row["station_id"])
                quality = str(row.get("quality") or "").strip()
                remark = f"DWS quality code: {quality}" if quality else None
                variable = row["variable"]
                if variable == "discharge":
                    records.append(
                        StreamflowReading(
                            source=DataSource.SOUTH_AFRICA_DWS,
                            station_id=station_id,
                            reading_datetime=when,
                            discharge_cms=value,
                            source_type="in_situ",
                            remark=remark,
                        )
                    )
                elif variable == "water_level":
                    records.append(
                        WaterLevelReading(
                            source=DataSource.SOUTH_AFRICA_DWS,
                            station_id=station_id,
                            reading_datetime=when,
                            water_level=value,
                            remark=remark,
                        )
                    )
                else:
                    raise ValueError(f"unsupported raw variable {variable!r}")
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug("Skipping malformed DWS row: %s — %r", exc, row)
        return records

    @staticmethod
    def _normalise_station_id(station_id: str) -> str:
        if not isinstance(station_id, str):
            raise ValueError(
                "DWS station_id must be a 4-12 character alphanumeric agency code, "
                "for example 'C1H001'."
            )
        station = station_id.strip().upper()
        if station.endswith(DWS_SERIES_SUFFIX):
            station = station[: -len(DWS_SERIES_SUFFIX)]
        if not _STATION_RE.fullmatch(station):
            raise ValueError(
                "DWS station_id must be a 4-12 character alphanumeric agency code, "
                "for example 'C1H001'."
            )
        return station

    @classmethod
    def _resolve_dates(
        cls,
        *,
        start_date: str | date | datetime | None,
        end_date: str | date | datetime | None,
        days: int | None,
    ) -> tuple[date, date]:
        end = cls._coerce_date(end_date, "end_date") or date.today()
        start = cls._coerce_date(start_date, "start_date")

        if days is not None:
            if days < 1:
                raise ValueError("days must be at least 1.")
            if start is not None:
                raise ValueError("Give either start_date or days, not both.")
            start = end - timedelta(days=days - 1)
        elif start is None:
            start = end - timedelta(days=29)

        if start > end:
            raise ValueError(f"start_date {start.isoformat()} is after end_date {end.isoformat()}.")
        return start, end

    @staticmethod
    def _coerce_date(value: str | date | datetime | None, name: str) -> date | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value).strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO date in YYYY-MM-DD format; got {value!r}.") from exc

    @staticmethod
    def _iter_chunks(start: date, end: date, *, years_per_chunk: int) -> Iterator[tuple[date, date]]:
        current = start
        while current <= end:
            final_year = min(current.year + years_per_chunk - 1, end.year)
            chunk_end = min(date(final_year, 12, 31), end)
            yield current, chunk_end
            current = chunk_end + timedelta(days=1)

    @classmethod
    def _parse_response(
        cls, body: str, *, station_id: str, variable: str
    ) -> list[dict[str, Any]]:
        table = cls._extract_table(body)
        if not table:
            return []
        if variable == "discharge":
            return cls._parse_daily(table, station_id=station_id)
        return cls._parse_points(table, station_id=station_id)

    @staticmethod
    def _extract_table(body: str) -> str:
        lower = body.lower()
        if any(marker in lower for marker in _APPLICATION_ERROR_MARKERS):
            first_line = next((line.strip() for line in body.splitlines() if line.strip()), "unknown DWS error")
            raise RuntimeError(
                "DWS returned an application-level Kisters ScriptServer failure "
                f"(the HTTP status may still be 200): {first_line}"
            )
        if "no data for this period" in lower:
            return ""

        parser = _PreTextParser()
        parser.feed(body)
        parser.close()
        table = parser.text.strip()
        if table:
            return table

        # Be tolerant of a plain-text response, but fail closed on an HTML page
        # that does not contain the expected hydrological table.
        if re.search(r"(?im)^\s*DATE(?:\s|$)", body):
            return body.strip()
        preview = re.sub(r"\s+", " ", body).strip()[:200]
        raise ValueError(
            "DWS response did not contain the expected <pre> hydrology table. "
            f"Response preview: {preview!r}"
        )

    @classmethod
    def _parse_daily(cls, table: str, *, station_id: str) -> list[dict[str, Any]]:
        header, lines = cls._header_and_data_lines(table)
        if header[:2] != ["DATE", "D_AVG_FR"]:
            raise ValueError(
                "DWS daily table has an unexpected header; expected DATE D_AVG_FR, "
                f"got {' '.join(header)!r}."
            )

        rows: list[dict[str, Any]] = []
        for line in lines:
            tokens = re.split(r"\s+", line.strip())
            if len(tokens) < 2:
                continue
            value = cls._parse_number(tokens[1])
            if value is None:
                continue
            try:
                reading_datetime = datetime.combine(cls._parse_yyyymmdd(tokens[0]), time.min)
            except ValueError as exc:
                logger.debug("Skipping malformed DWS daily row: %s — %r", exc, line)
                continue
            rows.append(
                {
                    "station_id": station_id,
                    "variable": "discharge",
                    "reading_datetime": reading_datetime,
                    "value": value,
                    "quality": tokens[2] if len(tokens) > 2 else None,
                }
            )
        return rows

    @classmethod
    def _parse_points(cls, table: str, *, station_id: str) -> list[dict[str, Any]]:
        header, lines = cls._header_and_data_lines(table)
        if header[:3] != ["DATE", "TIME", "COR_LEVEL"]:
            raise ValueError(
                "DWS point table has an unexpected header; expected DATE TIME COR_LEVEL, "
                f"got {' '.join(header)!r}."
            )

        rows: list[dict[str, Any]] = []
        for line in lines:
            tokens = re.split(r"\s+", line.strip())
            if len(tokens) < 3:
                continue
            value = cls._parse_number(tokens[2])
            if value is None:
                continue
            try:
                reading_datetime = datetime.combine(
                    cls._parse_yyyymmdd(tokens[0]), cls._parse_time(tokens[1])
                )
            except ValueError as exc:
                logger.debug("Skipping malformed DWS point row: %s — %r", exc, line)
                continue
            rows.append(
                {
                    "station_id": station_id,
                    "variable": "water_level",
                    "reading_datetime": reading_datetime,
                    "value": value,
                    "quality": tokens[3] if len(tokens) > 3 else None,
                }
            )
        return rows

    @staticmethod
    def _header_and_data_lines(table: str) -> tuple[list[str], list[str]]:
        lines = [line.strip() for line in table.splitlines() if line.strip()]
        header_index = next(
            (index for index, line in enumerate(lines) if re.match(r"^DATE(?:\s|$)", line.upper())),
            None,
        )
        if header_index is None:
            raise ValueError("DWS hydrology table is missing its DATE header.")
        header = re.split(r"\s+", lines[header_index].upper())
        data_lines = [line for line in lines[header_index + 1 :] if _DATE_ROW_RE.match(line)]
        return header, data_lines

    @staticmethod
    def _parse_yyyymmdd(value: str) -> date:
        try:
            return datetime.strptime(value, "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError(f"Invalid DWS date {value!r}.") from exc

    @staticmethod
    def _parse_time(value: str) -> time:
        cleaned = value.strip()
        for fmt in ("%H:%M:%S", "%H:%M", "%H%M%S", "%H%M"):
            try:
                return datetime.strptime(cleaned, fmt).time()
            except ValueError:
                continue
        raise ValueError(f"Invalid DWS time {value!r}.")

    @staticmethod
    def _parse_number(value: str) -> float | None:
        token = value.strip().upper()
        if token in _MISSING_VALUES:
            return None
        if "," in token:
            if not _THOUSANDS_NUMBER_RE.fullmatch(token):
                return None
            token = token.replace(",", "")
        try:
            parsed = float(token)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
