"""
Collector for USGS (United States Geological Survey) water data.

Uses the USGS Water Data OGC API, version 1 (keyless; a free key raises the rate limit):
    https://api.waterdata.usgs.gov/ogcapi/v1/

Every call goes there. The legacy NWIS Water Services host is being retired (shut down in Q1 2027,
#515), so nothing in the package may call it; a guard test enforces that.

Collections
-----------
- ``daily``               daily-value statistics (mean, min, max); was NWIS ``/dv``
- ``continuous``          continuous (instantaneous) sensor readings; was NWIS ``/iv`` (old alias ``sta``)
- ``field-measurements``  discrete field measurements (old alias ``discrete``)
- ``monitoring-locations`` and ``time-series-metadata`` for site metadata; was NWIS ``/site``
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.station import Station
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    Quality,
    StreamflowReading,
    WaterLevelReading,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimitedError, RateLimiter

logger = logging.getLogger(__name__)

USGS_BASE = "https://api.waterdata.usgs.gov/ogcapi/v1"

# Station names by id: sites per monitoring-locations request, and the most sites asked for that way per call.
_NAME_LOOKUP_CHUNK = 100
_NAME_LOOKUP_MAX_SITES = 2_000

#: Older collection names (the v0 draft and the NWIS endpoints) -> the v1 collection that replaced them.
COLLECTION_ALIASES: dict[str, str] = {
    "sta": "continuous",
    "iv": "continuous",
    "dv": "daily",
    "discrete": "field-measurements",
    "measurements": "field-measurements",
    "gwlevels": "field-measurements",
    "site": "monitoring-locations",
}

# Common USGS parameter codes relevant to water quality
PARAM_LABELS: dict[str, str] = {
    "00010": "Temperature",
    "00060": "Discharge",
    "00065": "Gage height",
    "00095": "Conductivity",
    "00300": "DO",
    "00400": "pH",
    "00410": "Alkalinity",
    "00600": "TN",
    "00665": "TP",
    "00680": "TOC",
    "00940": "Chloride",
    "00945": "Sulfate",
    "71846": "NH3-N",
    "80154": "SS",
}

MILES2_TO_KM2 = 2.589988110336
FT3S_TO_M3S = 0.028316846592
FT_TO_M = 0.3048

# Registry variable -> USGS parameter codes advertised in time-series-metadata.
STATION_VARIABLE_CODES: dict[str, tuple[str, ...]] = {
    "discharge": ("00060",),
    "water_level": ("00065",),
    "water_quality": ("00010", "00095", "00300", "00400"),
}

# NWIS alpha code (FIPS 5-1) -> two digit ANSI numeric code
US_STATE_CODES: dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
    "AQ": "60",
    "FM": "64",
    "GU": "66",
    "MH": "68",
    "MP": "69",
    "PW": "70",
    "PR": "72",
    "VI": "78",
}


def _parse_ogc_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _min_date(current: date | None, value: str | None) -> date | None:
    parsed = _parse_ogc_date(value)
    if parsed is None:
        return current
    return parsed if current is None or parsed < current else current


def _max_date(current: date | None, value: str | None) -> date | None:
    parsed = _parse_ogc_date(value)
    if parsed is None:
        return current
    return parsed if current is None or parsed > current else current


def _map_usgs_quality(props: dict) -> tuple[Quality, str | None]:
    approval = props.get("approval_status")
    qualifier = props.get("qualifier")

    raw_parts = []
    if approval:
        raw_parts.append(str(approval))
    if qualifier:
        if isinstance(qualifier, (list, tuple)):
            raw_parts.extend(str(q) for q in qualifier)
        else:
            raw_parts.append(str(qualifier))
    quality_raw = " | ".join(raw_parts) if raw_parts else None

    if isinstance(qualifier, (list, tuple)):
        qualifiers = [str(q).strip().upper() for q in qualifier]
    elif qualifier:
        qualifiers = [q.strip().upper() for q in str(qualifier).split()]
    else:
        qualifiers = []

    text = " ".join(raw_parts).lower()

    if "ice" in text:
        return Quality.SUSPECT, quality_raw

    if "estimat" in text or "E" in qualifiers:
        return Quality.ESTIMATED, quality_raw

    if str(approval).lower() == "provisional":
        return Quality.PROVISIONAL, quality_raw

    if str(approval).lower() == "approved":
        return Quality.APPROVED, quality_raw

    # Legacy USGS-style qualifiers (no approval_status; A/P alone in `qualifier`)
    if any(q not in {"A", "P"} for q in qualifiers):
        return Quality.SUSPECT, quality_raw

    if "P" in qualifiers:
        return Quality.PROVISIONAL, quality_raw

    if "A" in qualifiers:
        return Quality.APPROVED, quality_raw

    return Quality.UNKNOWN, quality_raw


class USGSCollector(BaseCollector):
    """
    Collect daily-value water data from USGS via the Water Data OGC API (v1).

    Parameters
    ----------
    api_key : str | None
        USGS API key for higher rate limits (get one at
        https://api.waterdata.usgs.gov/signup/). If omitted, the collector
        reads the ``USGS_API_KEY`` environment variable. With neither, the
        attribute is ``"DEMO_KEY"`` and requests go out with no key at all
        (the API answers keyless calls at a lower rate limit).
    """

    name = "usgs"

    #: One pacer for every collector in the process. A keyed USGS quota is 1,000 requests an hour
    #: (https://api.waterdata.usgs.gov/docs/ogcapi/keys); 15 a minute (900 an hour) stays under it.
    #: Callers such as the archive harvest build a fresh collector per station, so a per-instance
    #: limiter paced nothing across stations and the harvest ran into 429s nine minutes in.
    _shared_limiter = RateLimiter(max_calls=15, period_seconds=60)
    #: Drainage areas already looked up, shared for the same reason.
    _shared_area_cache: dict[str, float | None] = {}

    def __init__(
        self,
        api_key: str | None = None,
        client: CachedHTTPClient | None = None,
        *,
        lookup_catchment_area: bool = True,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=USGS_BASE,
                rate_limiter=USGSCollector._shared_limiter,
                # One page of a century-long daily record (up to 50,000 rows) can take 20 s to build.
                timeout=90.0,
            )
        )
        # One extra request per station for the drainage area on StreamflowReading. A caller that only
        # wants the series (the Explorer's fetch_series, the harvest) turns it off.
        self.lookup_catchment_area = lookup_catchment_area
        resolved = api_key or os.environ.get("USGS_API_KEY")
        if not resolved:
            logger.warning(
                "No USGS API key provided (pass api_key=... or set USGS_API_KEY). "
                "Using the keyless USGS Water Data API (DEMO_KEY mode), which is "
                "rate-limited and may fail under load."
            )
            resolved = "DEMO_KEY"
        self.api_key = resolved

    @property
    def keyed(self) -> bool:
        """True when a real USGS API key is set (not the keyless ``DEMO_KEY`` placeholder)."""
        return bool(self.api_key) and self.api_key != "DEMO_KEY"

    def fetch_raw(
        self,
        collection: str = "daily",
        datetime_range: str | None = None,
        days: int | None = None,
        limit: int = 10_000,
        bbox: str | None = None,
        max_items: int | None = 2_000,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch features from a USGS Water Data OGC collection (v1).

        Parameters
        ----------
        collection : str
            ``"daily"`` | ``"continuous"`` | ``"field-measurements"`` (or any
            other v1 collection). The older names ``"sta"``, ``"iv"``,
            ``"dv"`` and ``"discrete"`` are mapped onto their v1 collection.
        datetime_range : str, optional
            Explicit ISO 8601 interval ``"<start>/<end>"`` (USGS does NOT accept
            ISO durations like ``P7D``). If omitted, an interval is built from
            ``days``.
        days : int, optional
            Last N days from now (UTC). Defaults to 30 when ``datetime_range``
            is not supplied.
        limit : int
            Max features per page (the API allows up to 50,000). USGS advises
            one page per query: every further page re-runs the query.
        bbox : str, optional
            Bounding box filter ``"minLon,minLat,maxLon,maxLat"`` (WGS84).
            Without this the API returns data for every US monitoring location,
            which can require hundreds of paginated requests.
        max_items : int, optional
            Hard cap on total records fetched (across all pages). Keeps response
            times predictable. ``None`` means no cap.

        **kwargs
            Filters: ``station_id`` (or ``sites`` / ``monitoring_location_id``),
            ``parameter``, ``statCd`` (the statistic, for example ``"00003"``
            for the daily mean), ``stateCd``, ``countyCd``, ``huc``.
            ``skip_geometry=True`` drops the point geometry from every feature
            and ``properties=[...]`` narrows the fields returned; both make a
            long record a smaller download.
        """
        collection = COLLECTION_ALIASES.get(collection, collection)
        if datetime_range is None:
            window_days = days if days is not None else 30
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=window_days)
            datetime_range = (
                f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
                f"{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )

        sites = kwargs.get("station_id") or kwargs.get("sites") or kwargs.get("monitoring_location_id")
        parameter_cd = kwargs.get("parameter") or kwargs.get("parameterCd") or kwargs.get("parameter_code")
        bbox_val = bbox or kwargs.get("bBox")
        state_cd = kwargs.get("stateCd")
        county_cd = kwargs.get("countyCd")
        huc_val = kwargs.get("huc")
        stat_cd = kwargs.get("statCd") or kwargs.get("stat_cd") or kwargs.get("statistic_id")
        properties = kwargs.get("properties")
        skip_geometry = bool(kwargs.get("skip_geometry") or kwargs.get("skipGeometry"))

        if not self.keyed and not any([sites, bbox_val, state_cd, county_cd, huc_val]):
            raise ValueError(
                "USGS keyless path requires a filter parameter (station_id, bbox, stateCd, countyCd, or huc). "
                "An unfiltered query walks every US monitoring location, which the keyless rate limit cannot "
                "carry: pass a valid USGS_API_KEY via api_key or the USGS_API_KEY environment variable to "
                "request unfiltered data. For a reliable keyless demo source, use OpenMeteoCollector."
            )

        all_features: list[dict] = []
        params: dict[str, Any] = {
            "f": "json",
            "limit": limit,
            "datetime": datetime_range,
        }
        if self.keyed:
            params["api_key"] = self.api_key
        if bbox_val:
            params["bbox"] = bbox_val
        # The OGC collections filter on their own property names (#160): map the
        # legacy NWIS kwargs onto them instead of silently crawling the nation.
        if sites:
            site_list = [str(x).strip() for x in (sites if isinstance(sites, (list, tuple)) else str(sites).split(","))]
            params["monitoring_location_id"] = ",".join(x if "-" in x else f"USGS-{x}" for x in site_list if x)
        if parameter_cd:
            params["parameter_code"] = parameter_cd
        if stat_cd and collection in ("daily", "continuous"):
            params["statistic_id"] = stat_cd  # e.g. 00003, the daily mean only
        if properties:
            wanted = properties.split(",") if isinstance(properties, str) else list(properties)
            params["properties"] = ",".join(str(p).strip() for p in wanted if str(p).strip())
        if skip_geometry:
            params["skipGeometry"] = "true"
        if state_cd:
            state_cd, multiple_states_in_query = self._take_first_value(state_cd)
            if multiple_states_in_query:
                logger.warning(
                    "USGS OGC state_code takes one state per query; stateCd contained a "
                    "comma-separated list. Using the first value (%r) and dropping the rest.",
                    state_cd,
                )
            state_code = self._normalise_state_code(state_cd)
            if state_code is None:
                logger.warning(
                    "Could not map NWIS-style state code %r to a two-digit ANSI code for the "
                    "USGS OGC API; the stateCd filter was dropped from the query.",
                    state_cd,
                )
            else:
                params["state_code"] = state_code
        if county_cd:
            county_cd, multiple_counties_in_query = self._take_first_value(county_cd)
            if multiple_counties_in_query:
                logger.warning(
                    "USGS OGC county_code takes one county per query; countyCd contained a "
                    "comma-separated list. Using the first value (%r) and dropping the rest.",
                    county_cd,
                )
            county_code = self._normalise_county_code(county_cd)
            if county_code is None:
                logger.warning(
                    "Could not map NWIS-style county code %r to a three-digit ANSI code for the "
                    "USGS OGC API; the countyCd filter was dropped from the query.",
                    county_cd,
                )
            else:
                params["county_code"] = county_code
                # A three-digit county code is only unique within its state. A
                # full five-digit FIPS code carries the state prefix, so
                # we attempt to recover it; a bare three-digit code without a
                # state filter matches that county in every state.
                if "state_code" not in params:
                    stripped_county_cd = county_cd.strip()
                    if len(stripped_county_cd) == 5 and stripped_county_cd.isdigit():
                        state_code = self._normalise_state_code(stripped_county_cd[:2])
                        if state_code is not None:
                            params["state_code"] = state_code
                    else:
                        logger.warning(
                            "County code %r is only unique within its state; without a state code "
                            "filter, the USGS OGC query matches county %s in every state, so the "
                            "response will contain data from multiple states.",
                            county_cd,
                            county_code,
                        )
        if huc_val:
            huc_val, multiple_hucs_in_query = self._take_first_value(huc_val)
            if multiple_hucs_in_query:
                logger.warning(
                    "USGS OGC hydrologic_unit_code takes one HUC per query; huc contained a "
                    "comma-separated list. Using the first value %r and dropping the rest.",
                    huc_val,
                )
            params["hydrologic_unit_code"] = huc_val

        url = f"collections/{collection}/items"
        seen_links: set[str] = set()
        while True:
            data = self.client.get_json(url, params=params)
            features = data.get("features", [])
            all_features.extend(features)

            if max_items is not None and len(all_features) >= max_items:
                all_features = all_features[:max_items]
                logger.debug("USGS max_items=%d reached — stopping pagination.", max_items)
                break

            # follow pagination
            next_link = next(
                (lnk["href"] for lnk in data.get("links", []) if lnk.get("rel") == "next"),
                None,
            )
            if not next_link or len(features) == 0 or next_link in seen_links:
                break
            seen_links.add(next_link)
            # next_link is absolute and already carries the cursor. params must be
            # None, not {}: httpx rebuilds the query string from an empty dict and
            # drops the cursor, which re-fetches page one from the disk cache
            # forever (silent infinite loop, seen in the first CI harvest runs).
            url = self._keyed_link(next_link)
            params = None

        return all_features

    def stations(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        variable: str | None = None,
        max_items: int | None = 20_000,
    ) -> list[Station]:
        """USGS monitoring locations with daily-value time series.

        Built from the keyless OGC ``time-series-metadata`` collection (one
        feature per daily series, with geometry, parameter code and period of
        record), joined to ``monitoring-locations`` for names. Without a
        ``bbox`` this walks the whole national network; keep ``max_items``
        unless you mean it.
        """
        codes = STATION_VARIABLE_CODES.get(variable) if variable else None
        if variable and not codes:
            return []
        params: dict[str, Any] = {"f": "json", "limit": 10_000, "computation_identifier": "Mean"}
        if self.keyed:
            params["api_key"] = self.api_key  # a keyed call gets the higher rate limit
        if bbox:
            params["bbox"] = ",".join(str(v) for v in bbox)
        if codes and len(codes) == 1:
            params["parameter_code"] = codes[0]

        series = self._paginate("collections/time-series-metadata/items", params, max_items)

        by_site: dict[str, dict[str, Any]] = {}
        for feat in series:
            props = feat.get("properties", {})
            site = props.get("monitoring_location_id")
            code = props.get("parameter_code")
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None]
            if not site or coords[0] is None or coords[1] is None:
                continue
            var = next((v for v, cs in STATION_VARIABLE_CODES.items() if code in cs), None)
            if var is None or (codes and code not in codes):
                continue
            entry = by_site.setdefault(
                site, {"lon": float(coords[0]), "lat": float(coords[1]), "vars": set(), "begin": None, "end": None}
            )
            entry["vars"].add(var)
            entry["begin"] = _min_date(entry["begin"], props.get("begin"))
            entry["end"] = _max_date(entry["end"], props.get("end"))

        # Names come from monitoring-locations. That collection holds every
        # site type nationwide (wells, springs, ...), so restrict it to
        # streams for the hydrology variables and never let a rate-limited
        # names pass sink the catalog: stations without names beat no stations.
        names: dict[str, str] = {}
        areas: dict[str, float] = {}
        if by_site:
            loc_params: dict[str, Any] = {
                "f": "json", "limit": 10_000, "skipGeometry": "true",
                "properties": "id,monitoring_location_name,drainage_area",
            }
            if "api_key" in params:
                loc_params["api_key"] = params["api_key"]
            if bbox:
                loc_params["bbox"] = params["bbox"]
            if variable in (None, "discharge", "water_level"):
                loc_params["site_type_code"] = "ST"

            def _take(feats: list[dict]) -> None:
                for feat in feats:
                    props = feat.get("properties", {})
                    loc_id = props.get("id") or feat.get("id")
                    if loc_id in by_site:
                        names[loc_id] = props.get("monitoring_location_name")
                        try:
                            if props.get("drainage_area") is not None:
                                areas[loc_id] = float(props["drainage_area"]) * MILES2_TO_KM2
                        except (TypeError, ValueError):
                            pass

            # A capped national call (no bbox, a few sites) would walk the first max_items stream sites of the
            # whole country, which rarely include the ones found; asking for those sites by id is exact.
            walk = bool(bbox) or max_items is None or len(by_site) > _NAME_LOOKUP_MAX_SITES
            if walk:
                try:
                    _take(self._paginate("collections/monitoring-locations/items", loc_params, max_items))
                except RuntimeError as exc:
                    logger.warning("USGS monitoring-locations walk failed (%s); asking for the sites by id.", exc)
            # Sites the walk missed (it failed, was capped, or the site is not a stream) are asked for by id,
            # in small batches and at most _NAME_LOOKUP_MAX_SITES of them. The NWIS site service that used to be
            # the fallback is retired (#515). Stations without names still beat no stations.
            missing = [s for s in by_site if s not in names][:_NAME_LOOKUP_MAX_SITES]
            id_params = {k: v for k, v in loc_params.items() if k not in ("bbox", "site_type_code", "limit")}
            for i in range(0, len(missing), _NAME_LOOKUP_CHUNK):
                chunk = missing[i : i + _NAME_LOOKUP_CHUNK]
                try:
                    page = self.client.get_json(
                        "collections/monitoring-locations/items",
                        params={**id_params, "id": ",".join(chunk), "limit": len(chunk)},
                    )
                except RuntimeError as exc:
                    logger.warning(
                        "USGS monitoring-locations lookup by id failed (%s); returning %d stations without names.",
                        exc, sum(1 for s in by_site if s not in names),
                    )
                    break
                _take(page.get("features", []))

        stations: list[Station] = []
        for site, entry in by_site.items():
            number = site.split("-", 1)[-1]
            stations.append(
                Station(
                    source="usgs",
                    station_id=site,
                    name=names.get(site),
                    latitude=entry["lat"],
                    longitude=entry["lon"],
                    variables=tuple(sorted(entry["vars"])),
                    period_start=entry["begin"],
                    period_end=entry["end"],
                    url=f"https://waterdata.usgs.gov/monitoring-location/{number}/",
                    country="USA",
                    extra={"catchment_area_km2": round(areas[site], 2)} if site in areas else {},
                )
            )
        stations.sort(key=lambda s: s.station_id)
        return stations

    def _keyed_link(self, href: str) -> str:
        """A ``next`` link with the API key on it: USGS builds next links without ``api_key``, and a keyed
        harvest that dropped it after page one would fall back to the keyless rate limit mid-record."""
        if not self.keyed or "api_key=" in href:
            return href
        from urllib.parse import quote

        return f"{href}{'&' if '?' in href else '?'}api_key={quote(self.api_key, safe='')}"

    def _paginate(self, path: str, params: dict[str, Any], max_items: int | None) -> list[dict]:
        """Follow OGC ``next`` links, capping at ``max_items`` features."""
        features: list[dict] = []
        seen: set[str] = set()
        url: str = path
        page_params: dict[str, Any] | None = params
        while True:
            data = self.client.get_json(url, params=page_params)
            page = data.get("features", [])
            features.extend(page)
            if max_items is not None and len(features) >= max_items:
                return features[:max_items]
            next_link = next((lnk["href"] for lnk in data.get("links", []) if lnk.get("rel") == "next"), None)
            if not next_link or not page or next_link in seen:
                return features
            seen.add(next_link)
            url, page_params = self._keyed_link(next_link), None

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample | StreamflowReading | WaterLevelReading]:
        samples: list[WaterQualitySample | StreamflowReading | WaterLevelReading] = []
        for feat in raw:
            try:
                props = feat.get("properties", {})
                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [None, None]) if geom else [None, None]
                quality, quality_raw = _map_usgs_quality(props)

                param_code = props.get("parameter_code", "")
                param_label = PARAM_LABELS.get(param_code, param_code)

                val = props.get("value")
                if val is None:
                    continue

                time_str = props.get("time")
                if time_str is None:
                    continue
                dt = datetime.fromisoformat(str(time_str).replace("Z", "+00:00"))

                loc = None
                if coords[0] is not None:
                    loc = GeoLocation(latitude=coords[1], longitude=coords[0])

                if param_code == "00060":  # Discharge
                    discharge_sig_figs = self._count_sig_figs(val)
                    if not discharge_sig_figs:
                        discharge_sig_figs = 3  # default to 3 significant figures if unable to determine
                    discharge_cms = float(val) * FT3S_TO_M3S
                    rounded_discharge_cms = USGSCollector._round_to_sig_figs(discharge_cms, discharge_sig_figs)

                    catchment_area_km2 = props.get("catchment_area_km2", None)
                    if catchment_area_km2 is None and self.lookup_catchment_area:
                        catchment_area_km2 = self._get_monitoring_location_catchment_area(props.get("monitoring_location_id", ""))

                    samples.append(
                        StreamflowReading(
                            source=DataSource.USGS,
                            station_id=props.get("monitoring_location_id"),
                            station_name=props.get("station_name"),
                            location=loc,
                            reading_datetime=dt,
                            discharge_cms=rounded_discharge_cms,
                            source_type="in_situ",
                            uncertainty_cms=None,
                            catchment_area_km2=catchment_area_km2,
                            unit="m3/s",
                            quality=quality,
                            quality_raw=quality_raw,
                        )
                    )

                elif param_code == "00065":  # Gage height, feet -> metres
                    stage_sig_figs = self._count_sig_figs(val)
                    if not stage_sig_figs:
                        stage_sig_figs = 3  # default to 3 significant figures if unable to determine
                    stage_m = float(val) * FT_TO_M
                    rounded_stage_m = USGSCollector._round_to_sig_figs(stage_m, stage_sig_figs)

                    samples.append(
                        WaterLevelReading(
                            source=DataSource.USGS,
                            station_id=props.get("monitoring_location_id"),
                            station_name=props.get("station_name"),
                            location=loc,
                            reading_datetime=dt,
                            water_level=rounded_stage_m,
                            unit="m",
                            quality=quality,
                            quality_raw=quality_raw,
                        )
                    )

                else:
                    samples.append(
                        WaterQualitySample(
                            source=DataSource.USGS,
                            station_id=props.get("monitoring_location_id", "unknown"),
                            location=loc,
                            sample_datetime=dt,
                            parameter=param_label,
                            value=float(val),
                            unit=props.get("unit_of_measure", ""),
                            quality=quality,
                            quality_raw=quality_raw,
                        )
                    )

            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping USGS feature: %s", exc)

        return samples

    def _get_monitoring_location_catchment_area(self, location_id: str) -> float | None:
        if not location_id:
            return None

        location_id = USGSCollector._normalise_monitoring_location_id(location_id)

        # One lookup per station per process: a long daily record would otherwise
        # re-ask (and, when throttled, re-fail) once per row.
        cache = USGSCollector._shared_area_cache
        if location_id in cache:
            return cache[location_id]

        try:
            loc_params: dict[str, Any] = {"f": "json"}
            if self.keyed:
                loc_params["api_key"] = self.api_key
            feature = self.client.get_json(
                f"collections/monitoring-locations/items/{location_id}",
                params=loc_params,
            )
        except RateLimitedError:
            # Throttled, not missing: do not remember None, a later call may get the area.
            logger.warning(f"Rate limited looking up the drainage area of {location_id}; left out for now.")
            return None
        except RuntimeError:
            logger.warning(
                f"Cannot obtain metadata for station {location_id} - catchment area data is unavailable."
            )
            cache[location_id] = None
            return None

        area = feature.get("properties", {}).get("drainage_area", None)
        if area is None:
            logger.warning(
                f"Metadata for station {location_id} does not contain catchment area data."
            )
            cache[location_id] = None
            return None

        sig_figs = USGSCollector._count_sig_figs(area)
        if not sig_figs:
            sig_figs = 3  # default to 3 significant figures if unable to determine
        area_km2 = float(area) * MILES2_TO_KM2
        rounded_catchment_area = USGSCollector._round_to_sig_figs(area_km2, sig_figs)
        cache[location_id] = rounded_catchment_area

        return rounded_catchment_area

    @staticmethod
    def _normalise_monitoring_location_id(location_id: str) -> str:
        """Ensure an OGC ``monitoring_location_id`` carries its agency prefix."""
        if location_id.startswith("USGS-"):
            return location_id
        return f"USGS-{location_id}"

    @staticmethod
    def _take_first_value(value: str) -> tuple[str, bool]:
        """Return the first element of a comma-separated filter value.

        The OGC API accepts only one state, county or HUC per query; a
        comma-separated list returns an empty response rather than an error.
        Returns ``(first_element, was_list)`` - if the value is a list, the
        caller is warned, and the first element of the list is used
        as a parameter.
        """
        parts = [part.strip() for part in value.split(",")]
        return parts[0], len(parts) > 1

    @staticmethod
    def _normalise_state_code(state_cd: str) -> str | None:
        """Translate an NWIS state code to the two-digit ANSI code the OGC API expects (e.g. "AK" becomes "02").

        Returns ``None`` when ``state_cd`` is neither a recognised abbreviation
        nor a one- or two-digit numeric code, so callers can warn instead of
        sending a filter that silently matches nothing.
        """
        code = state_cd.strip()
        if code.isdigit():
            if len(code) <= 2:
                # If a numeric ANSI code, return the provided code with left padding if needed (e.g. "2" becomes "02").
                return code.zfill(2)
            return None
        # Convert NWIS code to corresponding ANSI code; if a mapping doesn't exist, return None.
        return US_STATE_CODES.get(code.upper())

    @staticmethod
    def _normalise_county_code(county_cd: str) -> str | None:
        """Drop the two-digit state prefix from an NWIS five-digit county code.

        The OGC ``county_code`` queryable is the three-digit ANSI county code
        (e.g. "24033" becomes "033"). Returns ``None`` when ``county_cd`` is
        neither a three- nor five-digit numeric code, so callers can warn
        instead of filtering silently.
        """
        code = county_cd.strip()
        if len(code) == 5 and code.isdigit():
            return code[2:]
        if len(code) == 3 and code.isdigit():
            return code
        return None

    @staticmethod
    def _count_sig_figs(value: str | float) -> int:
        text = str(value).strip()

        if not text or text.lower() in {"nan", "+inf", "inf", "-inf"}:
            return 0

        text = text.lstrip("+-")
        if "." in text:
            if text[-1] == ".":
                return len(text.rstrip("."))
            text = text.replace(".", "")
        else:
            text = text.rstrip("0")

        text = text.lstrip("0")
        if not text:
            return 1

        return len(text)

    @staticmethod
    def _round_to_sig_figs(value: float, sigfigs: int) -> float:
        if value == 0 or sigfigs <= 0:
            return value
        digits = sigfigs - int(math.floor(math.log10(abs(value)))) - 1
        return round(value, digits)
