"""
Collector for Hub'Eau (France) — Hydrometrie API.

Real-time river water level and discharge from the French hydrometric
network (DREAL / SCHAPI, Vigicrues), via Hub'Eau:
    https://hubeau.eaufrance.fr/page/api-hydrometrie

Endpoint used: ``observations_tr`` — real-time observations (water level,
discharge), which conveniently also carries each reading's coordinates
inline, so no separate station-metadata lookup is needed. Station *names*
are not included on this endpoint, so ``station_name`` is left unset.

No API key required (open data, unauthenticated).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.station import Station, in_bbox
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    StreamflowReading,
    WaterLevelReading,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

HUBEAU_BASE = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"

# Hub'Eau's two hydrometric "grandeurs" (quantities)
GRANDEUR_LABELS: dict[str, str] = {
    "H": "Water level",
    "Q": "Discharge",
}
GRANDEUR_UNITS: dict[str, str] = {
    "H": "mm",
    "Q": "L/s",
}

_LS_PER_M3S = 1_000  # divide L/s by this to get m³/s (avoids 0.001 float rounding)
_MM_PER_M = 1_000  # divide mm by this to get m (Hub'Eau serves water level in mm)

# Hub'Eau's elaborated ("obs_elab") grandeurs: daily and monthly statistics with
# multi-decade history, unlike observations_tr (real-time, last month only).
# Values keep the observations_tr units (Q in L/s, H in mm).
ELABORATED_GRANDEURS: dict[str, tuple[str, str]] = {
    "QmnJ": ("Q", "daily mean discharge"),
    "QmM": ("Q", "monthly mean discharge"),
    "QIXnJ": ("Q", "daily maximum instantaneous discharge"),
    "QINnJ": ("Q", "daily minimum instantaneous discharge"),
    "QixM": ("Q", "monthly maximum instantaneous discharge"),
    "QINM": ("Q", "monthly minimum instantaneous discharge"),
    "HIXnJ": ("H", "daily maximum instantaneous level"),
    "HIXM": ("H", "monthly maximum instantaneous level"),
}

def _parse_hubeau_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


class HubeauHydrometrieCollector(BaseCollector):
    """
    Collect real-time river level/discharge data from Hub'Eau (France).

    Parameters
    ----------
    api_key : str, optional
        Unused — Hub'Eau is open data with no authentication. Kept for
        interface parity with other collectors.
    """

    name = "hubeau_hydrometrie"

    def __init__(
        self,
        api_key: str = "",
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=HUBEAU_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=1800,  # upstream refreshes every ~2 minutes
            )
        )
        self.api_key = api_key

    def stations(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        variable: str | None = None,
        max_items: int | None = None,
    ) -> list[Station]:
        """Hub'Eau ``referentiel/stations``: every hydrometric station in service.

        Hub'Eau's referentiel does not say which grandeur (H or Q) a station
        reports, so ``variables`` is ``("water_level", "discharge")`` for all;
        the observation endpoints answer that per station. ``bbox`` goes to
        the API and is re-checked client side.
        """
        if variable and variable not in ("water_level", "discharge"):
            return []
        params: dict[str, Any] = {
            "format": "json",
            "size": 10_000,
            "en_service": "true",
            "fields": ",".join(
                [
                    "code_station", "libelle_station", "code_site", "libelle_site", "type_station",
                    "longitude_station", "latitude_station", "libelle_cours_eau", "code_cours_eau",
                    "date_ouverture_station", "date_fermeture_station", "en_service",
                ]
            ),
        }
        if bbox:
            params["bbox"] = ",".join(str(v) for v in bbox)
        stations: list[Station] = []
        url: str | None = "referentiel/stations"
        while url:
            data = self.client.get_json(url, params=params if url == "referentiel/stations" else None)
            for rec in data.get("data", []):
                lat, lon = rec.get("latitude_station"), rec.get("longitude_station")
                code = rec.get("code_station")
                if not code or lat is None or lon is None:
                    continue
                lat, lon = float(lat), float(lon)
                if not in_bbox(lat, lon, bbox):
                    continue
                stations.append(
                    Station(
                        source="hubeau_hydrometrie",
                        station_id=str(code),
                        site_id=str(rec.get("code_site") or str(code)[:8]),
                        name=rec.get("libelle_station") or rec.get("libelle_site"),
                        latitude=lat,
                        longitude=lon,
                        variables=("discharge", "water_level"),
                        period_start=_parse_hubeau_date(rec.get("date_ouverture_station")),
                        period_end=_parse_hubeau_date(rec.get("date_fermeture_station")),
                        url=f"https://www.hydro.eaufrance.fr/sitehydro/{rec.get('code_site')}/fiche"
                        if rec.get("code_site")
                        else None,
                        river=rec.get("libelle_cours_eau"),
                        country="FRA",
                        extra={k: rec[k] for k in ("code_site", "type_station") if rec.get(k) is not None},
                    )
                )
                if max_items is not None and len(stations) >= max_items:
                    url = None
                    break
            else:
                url = data.get("next") or None
        # One referentiel/sites call gives every site's catchment area (surface_bv, km2): the number the
        # archive and the Caravan export need to turn m3/s into mm/day.
        codes = {st.extra.get("code_site") for st in stations if st.extra.get("code_site")}
        if codes:
            try:
                areas = self._get_catchment_areas(codes)
            except Exception as exc:  # noqa: BLE001 - areas are an extra, never a reason to fail the catalog
                logger.info("Hub'Eau catchment areas unavailable (%s)", exc)
                areas = {}
            for st in stations:
                area = areas.get(st.extra.get("code_site"))
                if area:
                    st.extra["catchment_area_km2"] = float(area)
        return stations

    def fetch_raw(
        self,
        code_station: str | None = None,
        grandeur_hydro: str | None = None,
        date_debut_obs: str | None = None,
        date_fin_obs: str | None = None,
        days: int | None = None,
        size: int = 1000,
        max_items: int | None = 5_000,
        elaborated: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch observations from Hub'Eau: real-time ``observations_tr`` (default)
        or the long elaborated series ``obs_elab`` when ``elaborated`` is set.

        Parameters
        ----------
        elaborated : str, optional
            An elaborated grandeur code (see ``ELABORATED_GRANDEURS``), e.g.
            ``"QmnJ"`` for daily mean discharge with multi-decade history.
            When set, the request goes to ``/obs_elab``; ``grandeur_hydro`` is
            ignored and ``date_debut_obs`` / ``date_fin_obs`` (or ``days``) bound
            ``date_obs_elab``.
        code_station : str, optional
            Hydrometric station code (e.g. "K002000101"). Sent to the
            API as ``code_entite``. Without this (or a date/grandeur filter),
            Hub'Eau returns observations across thousands of stations nationwide.
        grandeur_hydro : str, optional
            "H" (water level) or "Q" (discharge). Omit to fetch both.
        date_debut_obs / date_fin_obs : str, optional
            ISO 8601 bounds, e.g. "2026-06-01T00:00:00Z". Built from
            ``days`` if omitted.
        days : int, optional
            Last N days from now (UTC). Ignored if ``date_debut_obs`` is
            given. Hub'Eau's own default lookback applies if neither is set.
        size : int
            Page size (Hub'Eau's hard max is 20000).
        max_items : int, optional
            Hard cap on total records fetched across all pages. ``None``
            means no cap.
        """
        if date_debut_obs is None and days is not None:
            start = datetime.now(timezone.utc) - timedelta(days=days)
            date_debut_obs = start.strftime("%Y-%m-%dT%H:%M:%SZ")

        all_data: list[dict] = []
        params: dict[str, Any] = {"format": "json", "size": min(size, 20_000)}
        if code_station:
            params["code_entite"] = code_station
        if elaborated:
            if elaborated not in ELABORATED_GRANDEURS:
                raise ValueError(f"Unknown elaborated grandeur {elaborated!r}; choose from {list(ELABORATED_GRANDEURS)}")
            params["grandeur_hydro_elab"] = elaborated
            if date_debut_obs:
                params["date_debut_obs_elab"] = date_debut_obs[:10]
            if date_fin_obs:
                params["date_fin_obs_elab"] = date_fin_obs[:10]
            url = "/obs_elab"
        else:
            if grandeur_hydro:
                params["grandeur_hydro"] = grandeur_hydro
            if date_debut_obs:
                params["date_debut_obs"] = date_debut_obs
            if date_fin_obs:
                params["date_fin_obs"] = date_fin_obs
            url = "/observations_tr"
        params.update(kwargs)

        while True:
            resp = self.client.get_json(url, params=params)
            rows = resp.get("data", [])
            all_data.extend(rows)

            if max_items is not None and len(all_data) >= max_items:
                all_data = all_data[:max_items]
                logger.debug("Hub'Eau max_items=%d reached — stopping pagination.", max_items)
                break

            next_link = resp.get("next")
            if not next_link or len(rows) == 0:
                break
            # next_link is absolute and already carries the cursor; switch to direct fetch.
            url = next_link
            # params=None means "use the URL as-is"; we use the exact next_link url provided by the API.
            params = None

        return all_data

    def normalise(
        self,
        raw: list[dict],
    ) -> Sequence[WaterLevelReading | StreamflowReading | WaterQualitySample]:
        # The /referentiel/sites lookup is hoisted out of the row loop. Collect
        # the distinct code_site values carried by discharge (Q) rows and resolve
        # them in a single batched call: Hub'Eau serves the entire hydrometric
        # site referentiel in one page (size=10000), so one call covers every
        # requested site. No referentiel call is made unless a Q row needs it.
        # obs_elab rows carry grandeur_hydro_elab / date_obs_elab / resultat_obs_elab.
        # Fold them into the observations_tr shape so one loop handles both.
        raw = [self._elaborated_to_tr(row) if "grandeur_hydro_elab" in row else row for row in raw]

        discharge_sites = {
            row.get("code_site")
            for row in raw
            if row.get("grandeur_hydro") == "Q"
            and row.get("resultat_obs") is not None
            and row.get("code_site")
        }
        catchment_areas = (
            self._get_catchment_areas(discharge_sites) if discharge_sites else {}
        )

        samples: list[WaterLevelReading | StreamflowReading | WaterQualitySample] = []
        skipped = 0
        for row in raw:
            try:
                grandeur = row.get("grandeur_hydro", "")
                label = GRANDEUR_LABELS.get(grandeur, grandeur)
                unit = GRANDEUR_UNITS.get(grandeur, "")

                val = row.get("resultat_obs")
                if val is None:
                    skipped += 1
                    continue

                # observations_tr includes coordinates directly on each row -
                # no separate referentiel/stations lookup needed.
                lat, lon = row.get("latitude"), row.get("longitude")
                loc = GeoLocation(latitude=lat, longitude=lon) if lat is not None and lon is not None else None

                # Hub'Eau returns date_obs with a trailing "Z". datetime.fromisoformat() only
                # accepts a bare "Z" on Python 3.11+; normalise it explicitly so this works on 3.10 too.
                # Stored tz-naive to match the convention used by other collectors in this codebase.
                dt = datetime.fromisoformat(row["date_obs"].replace("Z", "+00:00")).replace(tzinfo=None)

                station_code = row["code_station"]

                # Map water level (H) to WaterLevelReading and discharge (Q) to
                # StreamflowReading; any other grandeur falls back to WaterQualitySample.
                if label == "Water level":
                    water_level_m = float(val) / _MM_PER_M

                    samples.append(
                        WaterLevelReading(
                            source=DataSource.HUBEAU,
                            station_id=station_code,
                            location=loc,
                            reading_datetime=dt,
                            water_level=water_level_m,
                        )
                    )
                elif label == "Discharge":
                    discharge_cms = float(val) / _LS_PER_M3S

                    # Catchment area comes from the single batched referentiel
                    # call above - no per-row network request.
                    site_code = row.get("code_site")
                    catchment_area = catchment_areas.get(site_code) if site_code else None

                    samples.append(
                        StreamflowReading(
                            source=DataSource.HUBEAU,
                            station_id=station_code,
                            location=loc,
                            reading_datetime=dt,
                            discharge_cms=discharge_cms,
                            source_type="in_situ",
                            catchment_area_km2=catchment_area,
                        )
                    )
                else:
                    samples.append(
                        WaterQualitySample(
                            source=DataSource.HUBEAU,
                            station_id=station_code,
                            location=loc,
                            sample_datetime=dt,
                            parameter=label,
                            value=float(val),
                            unit=unit,
                        )
                    )

            except (ValueError, KeyError, TypeError) as exc:
                skipped += 1
                logger.debug("Skipping Hub'Eau row: %s", exc)

        if skipped:
            logger.warning(
                "Hub'Eau normalise(): skipped %d/%d row(s) (missing/invalid fields)",
                skipped,
                len(raw),
            )
        return samples


    @staticmethod
    def _elaborated_to_tr(row: dict) -> dict:
        """Map an ``obs_elab`` row onto the ``observations_tr`` field names."""
        code = row.get("grandeur_hydro_elab", "")
        base, _ = ELABORATED_GRANDEURS.get(code, (code, ""))
        date_str = str(row.get("date_obs_elab") or "")
        if len(date_str) == 10:  # daily/monthly stats come as bare dates
            date_str = f"{date_str}T00:00:00Z"
        out = dict(row)
        out.setdefault("grandeur_hydro", base)
        out.setdefault("date_obs", date_str)
        out.setdefault("resultat_obs", row.get("resultat_obs_elab"))
        out["elaborated"] = code
        return out

    def _get_catchment_areas(self, site_codes: set[str]) -> dict[str, float | None]:
        """
        Resolve catchment areas (``surface_bv``) for *site_codes* in a single
        ``/referentiel/sites`` call.

        Hub'Eau serves the entire hydrometric site referentiel in one page
        (``size=10000`` holds all ~9.3k sites), so a single call covers every
        requested code. The ``fields`` parameter keeps the payload down to the
        two attributes we need.

        Returns a dict mapping each requested code to its catchment area in
        km², or ``None`` when it is unavailable.
        """
        if not site_codes:
            return {}

        try:
            metadata_response = self.client.get_json(
                "referentiel/sites",
                params={
                    "size": 10_000,
                    "fields": "code_site,surface_bv",
                    "f": "json",
                },
            )
        except (RuntimeError, ValueError):
            logger.warning(
                "Cannot obtain hydrometric site metadata - catchment area data is unavailable."
            )
            return {}

        # For each site, if metadata for the site has been requested, add this data to the catchment areas dict.
        requested = set(site_codes)
        catchment_areas: dict[str, float | None] = {}
        for record in metadata_response.get("data", []):
            code = record.get("code_site")
            if code in requested:
                surface = record.get("surface_bv")
                catchment_areas[code] = None if surface is None else float(surface)

        # If any catchment area is None for a requested site,
        # we emit a warning.
        for code, area in catchment_areas.items():
            if area is None:
                logger.warning(
                    f"Metadata for site {code} does not contain catchment area data."
                )

        # If we cannot find a site code that is part of the request,
        # we set that site's catchment area to None and emit a warning.
        for code in requested - catchment_areas.keys():
            catchment_areas[code] = None
            logger.warning(
                f"No metadata found for site code {code} - catchment area data is unavailable."
            )

        return catchment_areas
