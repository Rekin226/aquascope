"""FAO AQUASTAT data collector.

Fetches country-level water resources data from FAO's AQUASTAT database
through the AQUASTAT dissemination API.

FAO retired the ``AQUASTAT`` domain from the FAOSTAT API (every request to
``https://www.fao.org/faostat/api/v1/en/data/AQUASTAT`` now answers 404), so the
data is read from the BigQuery-backed endpoint that serves the AQUASTAT portal
instead.  It answers with a CSV export rather than JSON.

API endpoint: https://api.data.apps.fao.org/api/v2/bigquery
Data portal:  https://data.apps.fao.org/aquastat/
Query view:   https://data.apps.fao.org/catalog/dataset/945666e6-7803-4621-b8ef-cfd885a84596/

The endpoint requires ``sql_url`` (the query view), ``area`` (English country
names, comma separated), ``type=country``, ``variable`` (variable, group or
subgroup code) and ``year`` (a comma separated list; there is no range syntax).
M49 numeric codes and ISO3 codes are *not* accepted for ``area`` — ``area=EGY``
answers 200 with a header-only CSV — so ISO3 codes are translated locally and an
unknown code raises rather than silently returning nothing.

References
----------
FAO. (2025). AQUASTAT - FAO's Global Information System on Water and Agriculture.
    https://data.apps.fao.org/aquastat/
FAO Statistical Database Terms of Use, Annex 1 (CC BY 4.0):
    https://www.fao.org/contact-us/terms/db-terms-of-use/en/
"""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Sequence
from typing import Any

from aquascope.collectors.base import BaseCollector, CollectorError
from aquascope.schemas.agriculture import AquastatRecord
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

AQUASTAT_BASE = "https://api.data.apps.fao.org/api/v2/bigquery"

#: The published query view the endpoint runs; ``_es`` and ``_fr`` variants exist.
AQUASTAT_SQL_URL = (
    "https://data.apps.fao.org/catalog/dataset/945666e6-7803-4621-b8ef-cfd885a84596"
    "/resource/4a000a1b-24f0-4328-aab6-b9b525892090/download/query_en.sql"
)

#: ``area`` must be a name, and ``World`` is how the API spells "every country".
AQUASTAT_AREA_ALL = "World"

#: The portal is a browser application; it answers a bare request with a 403.
AQUASTAT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; aquascope)",
    "Referer": "https://data.apps.fao.org/aquastat/",
}

# Key AQUASTAT variable IDs, as labelled by the API itself.
AQUASTAT_VARIABLES: dict[int, str] = {
    4263: "Total freshwater withdrawal",
    4253: "Total water withdrawal",
    4254: "Agricultural water withdrawal as % of total water withdrawal",
    4255: "Municipal water withdrawal as % of total withdrawal",
    4192: "Dependency ratio",
}

#: Code ``4312`` ("Area under agricultural water management") is a *subgroup*,
#: not a variable: asking for it as ``variable`` returns no rows.  Group codes
#: (100-104) and subgroup codes (4000-4027) are still accepted as ``variable``,
#: and answer with every variable they contain.
AQUASTAT_SUBGROUPS: dict[int, str] = {
    4312: "Area under agricultural water management",
}

#: ISO3 -> the English area name the API expects.  Covers every country the
#: 2023 release reports; ``AQUASTAT_AREAS`` is the single place to extend.
AQUASTAT_AREAS: dict[str, str] = {
    "AFG": "Afghanistan", "ALB": "Albania", "DZA": "Algeria", "AGO": "Angola",
    "ATG": "Antigua and Barbuda", "ARG": "Argentina", "ARM": "Armenia",
    "AUS": "Australia", "AUT": "Austria", "AZE": "Azerbaijan",
    "BHR": "Bahrain", "BGD": "Bangladesh", "BRB": "Barbados",
    "BLR": "Belarus", "BEL": "Belgium", "BLZ": "Belize", "BEN": "Benin",
    "BTN": "Bhutan", "BOL": "Bolivia (Plurinational State of)",
    "BWA": "Botswana", "BRA": "Brazil", "BRN": "Brunei Darussalam",
    "BGR": "Bulgaria", "BFA": "Burkina Faso", "BDI": "Burundi",
    "CPV": "Cabo Verde", "KHM": "Cambodia", "CMR": "Cameroon",
    "CAN": "Canada", "CAF": "Central African Republic", "TCD": "Chad",
    "CHL": "Chile", "CHN": "China", "COL": "Colombia", "COM": "Comoros",
    "COG": "Congo", "CRI": "Costa Rica", "HRV": "Croatia", "CUB": "Cuba",
    "CYP": "Cyprus", "CZE": "Czechia", "CIV": "Côte d'Ivoire",
    "PRK": "Democratic People's Republic of Korea",
    "COD": "Democratic Republic of the Congo", "DNK": "Denmark",
    "DJI": "Djibouti", "DMA": "Dominica", "DOM": "Dominican Republic",
    "ECU": "Ecuador", "EGY": "Egypt", "SLV": "El Salvador",
    "GNQ": "Equatorial Guinea", "ERI": "Eritrea", "EST": "Estonia",
    "SWZ": "Eswatini", "ETH": "Ethiopia", "FJI": "Fiji", "FIN": "Finland",
    "FRA": "France", "GAB": "Gabon", "GMB": "Gambia", "GEO": "Georgia",
    "DEU": "Germany", "GHA": "Ghana", "GRC": "Greece", "GRD": "Grenada",
    "GTM": "Guatemala", "GIN": "Guinea", "GNB": "Guinea-Bissau",
    "GUY": "Guyana", "HTI": "Haiti", "HND": "Honduras", "HUN": "Hungary",
    "ISL": "Iceland", "IND": "India", "IDN": "Indonesia",
    "IRN": "Iran (Islamic Republic of)", "IRQ": "Iraq", "IRL": "Ireland",
    "ISR": "Israel", "ITA": "Italy", "JAM": "Jamaica", "JPN": "Japan",
    "JOR": "Jordan", "KAZ": "Kazakhstan", "KEN": "Kenya", "KWT": "Kuwait",
    "KGZ": "Kyrgyzstan", "LAO": "Lao People's Democratic Republic",
    "LVA": "Latvia", "LBN": "Lebanon", "LSO": "Lesotho", "LBR": "Liberia",
    "LBY": "Libya", "LTU": "Lithuania", "LUX": "Luxembourg",
    "MDG": "Madagascar", "MWI": "Malawi", "MYS": "Malaysia",
    "MDV": "Maldives", "MLI": "Mali", "MLT": "Malta", "MRT": "Mauritania",
    "MUS": "Mauritius", "MEX": "Mexico", "MCO": "Monaco",
    "MNG": "Mongolia", "MNE": "Montenegro", "MAR": "Morocco",
    "MOZ": "Mozambique", "MMR": "Myanmar", "NAM": "Namibia", "NPL": "Nepal",
    "NLD": "Netherlands (Kingdom of the)", "NZL": "New Zealand",
    "NIC": "Nicaragua", "NER": "Niger", "NGA": "Nigeria",
    "MKD": "North Macedonia", "NOR": "Norway", "OMN": "Oman",
    "PAK": "Pakistan", "PSE": "Palestine", "PAN": "Panama",
    "PNG": "Papua New Guinea", "PRY": "Paraguay", "PER": "Peru",
    "PHL": "Philippines", "POL": "Poland", "PRT": "Portugal",
    "PRI": "Puerto Rico", "QAT": "Qatar", "KOR": "Republic of Korea",
    "MDA": "Republic of Moldova", "ROU": "Romania",
    "RUS": "Russian Federation", "RWA": "Rwanda",
    "KNA": "Saint Kitts and Nevis", "LCA": "Saint Lucia",
    "VCT": "Saint Vincent and the Grenadines",
    "STP": "Sao Tome and Principe", "SAU": "Saudi Arabia",
    "SEN": "Senegal", "SRB": "Serbia", "SYC": "Seychelles",
    "SLE": "Sierra Leone", "SGP": "Singapore", "SVK": "Slovakia",
    "SVN": "Slovenia", "SOM": "Somalia", "ZAF": "South Africa",
    "SSD": "South Sudan", "ESP": "Spain", "LKA": "Sri Lanka",
    "SDN": "Sudan", "SUR": "Suriname", "SWE": "Sweden",
    "CHE": "Switzerland", "SYR": "Syrian Arab Republic", "TJK": "Tajikistan",
    "THA": "Thailand", "TLS": "Timor-Leste", "TGO": "Togo",
    "TTO": "Trinidad and Tobago", "TUN": "Tunisia", "TKM": "Turkmenistan",
    "TUR": "Türkiye", "UGA": "Uganda", "UKR": "Ukraine",
    "ARE": "United Arab Emirates",
    "GBR": "United Kingdom of Great Britain and Northern Ireland",
    "TZA": "United Republic of Tanzania", "USA": "United States of America",
    "URY": "Uruguay", "UZB": "Uzbekistan",
    "VEN": "Venezuela (Bolivarian Republic of)", "VNM": "Viet Nam",
    "YEM": "Yemen", "ZMB": "Zambia", "ZWE": "Zimbabwe",
}


def resolve_area(country_code: str | None) -> str:
    """Translate ``country_code`` into the ``area`` value the API accepts.

    ``"all"`` (or an empty value) becomes ``World``.  ISO3 codes are translated
    through :data:`AQUASTAT_AREAS`; an unknown three-letter code raises rather
    than querying the API with something it silently ignores.  Anything else —
    an English country name, or a comma separated list of them — is passed
    through, so callers can name the area directly when the API renames a
    country.
    """
    if not country_code or str(country_code).strip().lower() in ("all", "*"):
        return AQUASTAT_AREA_ALL

    resolved: list[str] = []
    for part in str(country_code).split(","):
        name = part.strip()
        if not name:
            continue
        upper = name.upper()
        if upper in AQUASTAT_AREAS:
            resolved.append(AQUASTAT_AREAS[upper])
        elif len(name) == 3 and name.isalpha():
            raise ValueError(
                f"Unknown ISO3 country code {name!r} for AQUASTAT. "
                f"Pass the English area name instead, or add the code to AQUASTAT_AREAS."
            )
        else:
            resolved.append(name)

    if not resolved:
        raise ValueError(f"No usable area in {country_code!r} for AQUASTAT.")
    return ",".join(resolved)


def parse_csv(text: str) -> list[dict]:
    """Parse the endpoint's CSV export into a list of row dictionaries."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    return [row for row in reader if any((value or "").strip() for value in row.values())]


class AquastatCollector(BaseCollector):
    """Collect country-level water data from FAO AQUASTAT.

    Parameters
    ----------
    client : CachedHTTPClient | None
        HTTP client instance. A default is created if *None*.

    References
    ----------
    FAO. (2025). AQUASTAT. https://data.apps.fao.org/aquastat/
    """

    name = "aquastat"

    def __init__(self, client: CachedHTTPClient | None = None) -> None:
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=AQUASTAT_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=86400,
            )
        )

    def fetch_raw(
        self,
        country_code: str = "all",
        variable_ids: list[int] | None = None,
        start_year: int = 2000,
        end_year: int = 2023,
        **kwargs: Any,
    ) -> list[dict]:
        """Fetch raw AQUASTAT data as CSV rows.

        Parameters
        ----------
        country_code : str
            ISO3 country code, an English area name, a comma separated list of
            either, or ``'all'`` for every country.
        variable_ids : list[int] | None
            AQUASTAT variable, group or subgroup codes. Defaults to the five key
            variables in :data:`AQUASTAT_VARIABLES`.
        start_year : int
            Start year (default 2000).
        end_year : int
            End year (default 2023).

        Returns
        -------
        list[dict]
            One dictionary per CSV row, keyed by the export's column names.
        """
        if start_year > end_year:
            raise ValueError(f"start_year {start_year} is after end_year {end_year}.")

        area = resolve_area(country_code)
        var_ids = [int(v) for v in (variable_ids or list(AQUASTAT_VARIABLES))]
        params: dict[str, Any] = {
            "sql_url": AQUASTAT_SQL_URL,
            "area": area,
            "type": "country",
            "variable": ",".join(str(v) for v in var_ids),
            # The endpoint has no range syntax; every year is named.
            "year": ",".join(str(year) for year in range(start_year, end_year + 1)),
        }

        try:
            text = self.client.get_text(AQUASTAT_BASE, params=params, headers=AQUASTAT_HEADERS)
        except Exception as exc:  # noqa: BLE001 - re-raised as CollectorError
            raise CollectorError(
                f"AQUASTAT request failed for area={area!r}: {exc}",
                source=self.name,
                url=AQUASTAT_BASE,
                cause=exc,
            ) from exc

        rows = parse_csv(text)
        if not rows and "country" in text.lower():
            logger.warning(
                "AQUASTAT returned no rows for area=%r variable=%r years %d-%d. "
                "Check that the area name matches the portal and that the "
                "variables report data for those years.",
                area,
                params["variable"],
                start_year,
                end_year,
            )
        return rows

    def normalise(self, raw: list[dict]) -> Sequence[AquastatRecord]:
        """Convert raw CSV rows into ``AquastatRecord`` objects.

        Parameters
        ----------
        raw : list[dict]
            Records from ``fetch_raw``.

        Returns
        -------
        Sequence[AquastatRecord]
            Normalised AQUASTAT records.
        """
        records: list[AquastatRecord] = []
        for rec in raw:
            try:
                value = rec.get("Value")
                if value is None or str(value).strip() in ("", "-"):
                    continue
                records.append(
                    AquastatRecord(
                        country=rec.get("Country", "") or "",
                        country_code=rec.get("m49", "") or "",
                        year=int(rec.get("Year", 0)),
                        variable=rec.get("Variable", "") or "",
                        value=float(value),
                        unit=rec.get("Unit", "") or "",
                        source="AQUASTAT",
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping AQUASTAT record: %s", exc)
        return records
