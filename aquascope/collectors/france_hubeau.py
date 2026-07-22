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
from datetime import datetime, timedelta, timezone
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
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

    def fetch_raw(
        self,
        code_station: str | None = None,
        grandeur_hydro: str | None = None,
        date_debut_obs: str | None = None,
        date_fin_obs: str | None = None,
        days: int | None = None,
        size: int = 1000,
        max_items: int | None = 5_000,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch real-time observations from Hub'Eau's ``observations_tr``.

        Parameters
        ----------
        code_station : str, optional
            Hydrometric station code (e.g. "K002000101"). Without this
            (or a date/grandeur filter), Hub'Eau returns observations
            across thousands of stations nationwide.
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
            params["code_station"] = code_station
        if grandeur_hydro:
            params["grandeur_hydro"] = grandeur_hydro
        if date_debut_obs:
            params["date_debut_obs"] = date_debut_obs
        if date_fin_obs:
            params["date_fin_obs"] = date_fin_obs
        params.update(kwargs)

        url = "/observations_tr"
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
            # next_link is absolute and already carries the cursor; switch to direct fetch
            url = next_link
            params = {}

        return all_data

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        samples: list[WaterQualitySample] = []
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
                samples.append(
                    WaterQualitySample(
                        source=DataSource.HUBEAU,
                        station_id=row["code_station"],
                        station_name=row.get("libelle_station"),  # not present on this endpoint; stays None
                        location=loc,
                        sample_datetime=datetime.fromisoformat(row["date_obs"].replace("Z", "+00:00")).replace(
                            tzinfo=None
                        ),
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
