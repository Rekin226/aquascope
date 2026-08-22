"""
Collector for USGS (United States Geological Survey) water data.

Uses the new OGC-compliant API:
    https://api.waterdata.usgs.gov/ogcapi/v0/

Collections
-----------
- ``daily``       — daily-value statistics (mean, min, max)
- ``sta``         — continuous (instantaneous) sensor readings
- ``discrete``    — discrete field measurements
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
    StreamflowReading,
    WaterLevelReading,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

USGS_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"

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

# 50 states + DC + territories, as the NWIS site service's stateCd expects.
NWIS_STATE_CODES: tuple[str, ...] = (
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "dc", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky",
    "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh",
    "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "pr", "vi", "gu",
    "as", "mp",
)

# Two-digit ANSI (FIPS) codes for the OGC API's ``state_code`` queryable.
# The NWIS keyless path accepts 2-letter abbreviations, so we translate.
STATE_FIPS: dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
    "AS": "60", "GU": "66", "MP": "69", "PR": "72", "VI": "78",
    "FM": "64", "MH": "68", "PW": "70",
}


def _parse_nwis_rdb_sites(text: str) -> list[tuple[str, str]]:
    """Yield ``(site_no, station_nm)`` from an NWIS RDB site listing."""
    out: list[tuple[str, str]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if header is None:
            header = cols
            continue
        if cols and cols[0].endswith("s") and cols[0][:-1].isdigit():  # the RDB dtype row, e.g. "5s\t15s\t50s"
            continue
        row = dict(zip(header, cols))
        site_no, name = row.get("site_no", "").strip(), row.get("station_nm", "").strip()
        if site_no and name:
            out.append((site_no, name))
    return out


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


class USGSCollector(BaseCollector):
    """
    Collect daily-value water data from USGS via OGC API.

    Parameters
    ----------
    api_key : str | None
        USGS API key for higher rate limits (get one at
        https://api.waterdata.usgs.gov/docs/ogcapi/#api-keys). If omitted,
        the collector reads the ``USGS_API_KEY`` environment variable, and
        falls back to the shared ``DEMO_KEY`` (heavily rate-limited) with a
        warning if neither is set.
    """

    name = "usgs"

    def __init__(
        self,
        api_key: str | None = None,
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=USGS_BASE,
                rate_limiter=RateLimiter(max_calls=25, period_seconds=60),
            )
        )
        resolved = api_key or os.environ.get("USGS_API_KEY")
        if not resolved:
            logger.warning(
                "No USGS API key provided (pass api_key=... or set USGS_API_KEY). "
                "Falling back to the shared DEMO_KEY, which is heavily "
                "rate-limited and may fail under load."
            )
            resolved = "DEMO_KEY"
        self.api_key = resolved

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
        Fetch features from a USGS OGC collection.

        Parameters
        ----------
        collection : str
            ``"daily"`` | ``"sta"`` | ``"discrete"``
        datetime_range : str, optional
            Explicit ISO 8601 interval ``"<start>/<end>"`` (USGS does NOT accept
            ISO durations like ``P7D``). If omitted, an interval is built from
            ``days``.
        days : int, optional
            Last N days from now (UTC). Defaults to 30 when ``datetime_range``
            is not supplied.
        limit : int
            Max features per page. Larger values mean fewer round-trips.
        bbox : str, optional
            Bounding box filter ``"minLon,minLat,maxLon,maxLat"`` (WGS84).
            Without this the API returns data for every US monitoring location,
            which can require hundreds of paginated requests.
        max_items : int, optional
            Hard cap on total records fetched (across all pages). Keeps response
            times predictable. ``None`` means no cap.
        """
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

        if self.api_key == "DEMO_KEY" or not self.api_key:
            if collection not in ("daily", "sta"):
                raise ValueError(
                    f"Collection '{collection}' is not supported on the keyless legacy USGS API path. "
                    "Only 'daily' and 'sta' collections are supported without an API key. "
                    "Please provide a valid USGS_API_KEY to use the OGC API path. "
                    "For a reliable keyless demo source, use OpenMeteoCollector."
                )

            if not any([sites, bbox_val, state_cd, county_cd, huc_val]):
                raise ValueError(
                    "USGS keyless path requires a filter parameter (station_id, bbox, stateCd, countyCd, or huc). "
                    "To request unfiltered data, you must provide a valid USGS_API_KEY via api_key or the "
                    "USGS_API_KEY environment variable. For a reliable keyless demo source, use OpenMeteoCollector."
                )

            parts = datetime_range.split("/")
            start_date = parts[0].split("T")[0] if len(parts) == 2 else None
            end_date = parts[1].split("T")[0] if len(parts) == 2 else None

            endpoint = "dv" if collection == "daily" else "iv"
            url = f"https://waterservices.usgs.gov/nwis/{endpoint}/"

            params = {
                "format": "json",
            }
            if sites:
                # Accept "01646500", "USGS-01646500" or another agency's "CA574-09527500":
                # NWIS wants the bare number plus agencyCd for non-USGS sites.
                site_str = ",".join(sites) if isinstance(sites, (list, tuple)) else str(sites)
                numbers, agencies = [], set()
                for part in site_str.split(","):
                    part = part.strip()
                    if "-" in part:
                        agency, number = part.split("-", 1)
                        agencies.add(agency.upper())
                        numbers.append(number)
                    elif part:
                        numbers.append(part)
                params["sites"] = ",".join(numbers)
                if len(agencies) == 1 and next(iter(agencies)) != "USGS":
                    params["agencyCd"] = next(iter(agencies))
            if parameter_cd:
                params["parameterCd"] = parameter_cd
            if bbox_val:
                params["bBox"] = bbox_val
            if state_cd:
                params["stateCd"] = state_cd
            if county_cd:
                params["countyCd"] = county_cd
            if huc_val:
                params["huc"] = huc_val

            if start_date:
                params["startDT"] = start_date
            if end_date:
                params["endDT"] = end_date

            response = self.client.get_json(url, params=params)
            time_series_list = response.get("value", {}).get("timeSeries", [])

            all_features = []
            for ts in time_series_list:
                source_info = ts.get("sourceInfo", {})
                site_codes = source_info.get("siteCode", [])
                site_id = site_codes[0].get("value", "unknown") if site_codes else "unknown"

                geo_loc = source_info.get("geoLocation", {}).get("geogLocation", {})
                latitude = geo_loc.get("latitude")
                longitude = geo_loc.get("longitude")

                var_info = ts.get("variable", {})
                var_codes = var_info.get("variableCode", [])
                param_code = var_codes[0].get("value", "") if var_codes else ""
                unit = var_info.get("unit", {}).get("unitCode", "")
                no_data_val = var_info.get("noDataValue")

                for values_block in ts.get("values", []):
                    for value in values_block.get("value", []):
                        val = value.get("value")
                        dt = value.get("dateTime")
                        if val is None or dt is None:
                            continue

                        try:
                            float_val = float(val)
                            if no_data_val is not None and abs(float_val - no_data_val) < 1e-3:
                                continue
                        except (ValueError, TypeError):
                            continue

                        drainage_area = source_info.get("drainageArea", None)
                        if drainage_area is not None:
                            try:
                                drainage_area = float(drainage_area)
                            except (TypeError, ValueError):
                                drainage_area = None

                        catchment_area_km2 = (
                            drainage_area * MILES2_TO_KM2
                            if drainage_area is not None
                            else None
                        )

                        all_features.append({
                            "geometry": {
                                "coordinates": [longitude, latitude]
                            },
                            "properties": {
                                "monitoring_location_id": site_id,
                                "parameter_code": param_code,
                                "value": val,
                                "time": dt,
                                "unit_of_measure": unit,
                                "catchment_area_km2": catchment_area_km2
                            }
                        })

            if max_items is not None and len(all_features) >= max_items:
                all_features = all_features[:max_items]

            return all_features

        all_features: list[dict] = []
        params: dict[str, Any] = {
            "f": "json",
            "limit": limit,
            "datetime": datetime_range,
            "api_key": self.api_key,
        }
        if bbox:
            params["bbox"] = bbox
        elif kwargs.get("bBox"):
            params["bbox"] = kwargs["bBox"]
        # The OGC collections filter on their own property names (#160): map the
        # legacy NWIS kwargs onto them instead of silently crawling the nation.
        if sites:
            site_list = [str(x).strip() for x in (sites if isinstance(sites, (list, tuple)) else str(sites).split(","))]
            params["monitoring_location_id"] = ",".join(x if "-" in x else f"USGS-{x}" for x in site_list if x)
        if parameter_cd:
            params["parameter_code"] = parameter_cd
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
            url = next_link
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
        if self.api_key and self.api_key != "DEMO_KEY":
            params["api_key"] = self.api_key  # keyed calls are not throttled like the shared demo path
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
            loc_params: dict[str, Any] = {"f": "json", "limit": 10_000}
            if "api_key" in params:
                loc_params["api_key"] = params["api_key"]
            if bbox:
                loc_params["bbox"] = params["bbox"]
            if variable in (None, "discharge", "water_level"):
                loc_params["site_type_code"] = "ST"
            try:
                for feat in self._paginate("collections/monitoring-locations/items", loc_params, max_items):
                    props = feat.get("properties", {})
                    if props.get("id") in by_site:
                        names[props["id"]] = props.get("monitoring_location_name")
                        try:
                            if props.get("drainage_area") is not None:
                                areas[props["id"]] = float(props["drainage_area"]) * MILES2_TO_KM2
                        except (TypeError, ValueError):
                            pass
            except RuntimeError as exc:
                logger.warning("USGS monitoring-locations lookup failed (%s); trying the NWIS site service.", exc)
                try:
                    names = self._nwis_site_names(bbox, wanted=set(by_site))
                except Exception as exc2:  # noqa: BLE001 - names are nice to have, not required
                    logger.warning("NWIS site service lookup failed too (%s); returning stations without names.", exc2)

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

    def _nwis_site_names(
        self, bbox: tuple[float, float, float, float] | None, *, wanted: set[str] | None = None
    ) -> dict[str, str]:
        """Station names from the keyless NWIS site service (RDB), as a fallback.

        One request for a ``bbox``; otherwise one per state/territory (~56
        requests of ~50 KB). Only stream sites with daily values are asked for.
        """
        names: dict[str, str] = {}
        base = "https://waterservices.usgs.gov/nwis/site/"
        common = {"format": "rdb", "siteType": "ST", "siteStatus": "all", "hasDataTypeCd": "dv"}
        if bbox:
            queries = [{**common, "bBox": ",".join(f"{v:.6f}" for v in bbox)}]
        else:
            queries = [{**common, "stateCd": st} for st in NWIS_STATE_CODES]
        for params in queries:
            try:
                text = self.client.get_text(base, params=params)
            except RuntimeError as exc:
                logger.debug("NWIS site query %s failed: %s", params.get("stateCd") or "bbox", exc)
                continue
            for site_no, name in _parse_nwis_rdb_sites(text):
                key = f"USGS-{site_no}"
                if wanted is None or key in wanted:
                    names[key] = name
        return names

    def _paginate(self, path: str, params: dict[str, Any], max_items: int | None) -> list[dict]:
        """Follow OGC ``next`` links, capping at ``max_items`` features."""
        features: list[dict] = []
        url: str = path
        page_params: dict[str, Any] | None = params
        while True:
            data = self.client.get_json(url, params=page_params)
            page = data.get("features", [])
            features.extend(page)
            if max_items is not None and len(features) >= max_items:
                return features[:max_items]
            next_link = next((lnk["href"] for lnk in data.get("links", []) if lnk.get("rel") == "next"), None)
            if not next_link or not page:
                return features
            url, page_params = next_link, None

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample | StreamflowReading | WaterLevelReading]:
        samples: list[WaterQualitySample | StreamflowReading | WaterLevelReading] = []
        for feat in raw:
            try:
                props = feat.get("properties", {})
                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [None, None]) if geom else [None, None]

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
                    if catchment_area_km2 is None:
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
                        )
                    )

            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping USGS feature: %s", exc)

        return samples

    def _get_monitoring_location_catchment_area(self, location_id: str) -> float | None:
        if not location_id:
            return None

        location_id = USGSCollector._normalise_monitoring_location_id(location_id)

        # One lookup per station per collector instance: a long daily record
        # would otherwise re-ask (and, when throttled, re-fail) once per row.
        cache = self.__dict__.setdefault("_area_cache", {})
        if location_id in cache:
            return cache[location_id]

        try:
            feature = self.client.get_json(
                f"collections/monitoring-locations/items/{location_id}",
                params={"f": "json"},
            )
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
        return STATE_FIPS.get(code.upper())

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
