"""
Collector for Brazil — ANA (Agência Nacional de Águas e Saneamento Básico) Hidroweb.

ANA runs the national hydrometeorological network (streamflow, stage and
rainfall) under the Hidroweb / SNIRH umbrella. Two upstream services are used
here, matching the split ANA itself uses between "where are the stations" and
"what did they measure":

- **Station catalog** — the public "Estações Hidrometeorológicas" ArcGIS
  FeatureServer behind the SNIRH portal map. No credentials required:
  https://portal1.snirh.gov.br/server/rest/services/Esta%C3%A7%C3%B5es_Hidrometeorol%C3%B3gicas_SNIRH/FeatureServer/0
- **Time series** — the credentialed ``HidroWebService`` REST API
  (``/EstacoesTelemetricas/...``), which replaced the legacy SOAP
  ``ServiceANA.asmx`` referenced in earlier tickets/tutorials. Access
  requires a free ANA account (CPF/CNPJ + password). ANA's own pages are
  inconsistent about the request address: the HidroWebService migration
  page and API manual both say to email **hidro@ana.gov.br** with subject
  "Solicitação de acesso à API" (this is what's used below); a separate,
  older page for the legacy SOAP service instead lists
  **telemetria@ana.gov.br** / (61) 2109-5391, but only in the context of
  its ``CotaOnline`` data-management options, not general API onboarding.
  If hidro@ana.gov.br doesn't get a response, telemetria@ana.gov.br is
  worth trying as a fallback — but verify against
  https://www.ana.gov.br/hidrowebservice/swagger-ui.html or
  https://www.snirh.gov.br/hidroweb/acesso-api first, since this wasn't
  confirmed against a live request in this environment. Credentials are
  resolved from the ``ANA_HIDROWEB_IDENTIFICADOR`` / ``ANA_HIDROWEB_SENHA``
  environment variables, or passed to the constructor.

  Auth: ``GET /EstacoesTelemetricas/OAUth/v1`` with ``Identificador`` and
  ``Senha`` headers returns a bearer token (``items.tokenautenticacao``),
  valid 60 minutes.

  Data: ``GET /EstacoesTelemetricas/HidroinfoanaSerieTelemetricaAdotada/v1``
  with ``codEstacao`` (+ optional ``dataInicio``/``dataFim``/
  ``tipoFiltroData``) and ``Authorization: Bearer <token>`` returns
  ``items``, one row per reading timestamp, each row bundling adopted stage
  (``Cota_Adotada``, cm), discharge (``Vazao_Adotada``, m3/s) and rainfall
  (``Chuva_Adotada``, mm) together with a 0/1/2 QC status per parameter.

Field names are transcribed from ANA's published API tutorial
("Tutorial de Serviço para Consumo de Dados — API HidroWebService") rather
than a live response, since exercising the credentialed endpoint isn't
possible from this environment. As with ``ireland_opw.py``, values are
parsed defensively (missing/empty fields are skipped per-row rather than
aborting the whole batch) so a header or shape drift on ANA's side degrades
gracefully instead of raising.

- **Historical (conventional-network) series** — ``fetch_raw(mode="historical",
  ...)`` additionally reaches the legacy SOAP-but-GET-able
  ``ServiceANA.asmx/HidroSerieHistorica`` endpoint, which serves ANA's
  *conventional* network: manually read stage/rainfall/discharge going back
  decades, for many more stations than the telemetric network covers (and
  with no credentials needed). ANA's conventional "Vazoes"/"Cotas"/"Chuvas"
  tables store one row per station+month+QA-level with a day-of-month value
  in fields ``Vazao01``..``Vazao31`` (or ``Cota``/``Chuva`` equivalents)
  rather than one row per day — the same wide layout Matthew Heberger
  documented and parsed in ``convert_vazoes_2_csv.py``
  (https://github.com/mheberger/brazil-discharge, MIT License). The
  dedup/quirk-handling in :meth:`_normalise_historical` below is adapted
  from that script: preferring the "Consistido" (QA'd/final) row over
  "Bruto" (provisional) when ANA returns both for the same station+month,
  dropping the occasional duplicate month row ANA lists with a
  day-of-month other than 01, and melting each wide monthly row into one
  reading per day. Credit: Matthew Heberger, MIT License.
"""

from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from datetime import date, datetime, timedelta

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.station import Station, in_bbox
from aquascope.schemas.water_data import (
    ClimateReading,
    DataSource,
    GeoLocation,
    StreamflowReading,
    WaterLevelReading,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

HIDROWEBSERVICE_BASE = "https://www.ana.gov.br/hidrowebservice"
AUTH_PATH = "/EstacoesTelemetricas/OAUth/v1"
SERIES_PATH = "/EstacoesTelemetricas/HidroinfoanaSerieTelemetricaAdotada/v1"

# Public SNIRH station catalog (ArcGIS FeatureServer), no auth required.
STATIONS_URL = (
    "https://portal1.snirh.gov.br/server/rest/services/"
    "Esta%C3%A7%C3%B5es_Hidrometeorol%C3%B3gicas_SNIRH/FeatureServer/0/query"
)
STATIONS_PAGE_SIZE = 1000

# Token is valid for 60 minutes server-side; refresh a little early.
TOKEN_LIFETIME_SECONDS = 55 * 60

# Legacy SOAP-but-GET-able webservice serving ANA's *conventional* (manually
# read, non-telemetric) network — no credentials required. This is where
# decades of pre-telemetric-network daily records live.
LEGACY_SOAP_BASE = "https://telemetriaws1.ana.gov.br/ServiceANA.asmx"
HISTORICAL_SERIES_PATH = "/HidroSerieHistorica"

# tipoDados codes for the legacy HidroSerieHistorica webservice, and the
# corresponding wide-column field prefix (Vazao01..31 etc.) in its response.
_TIPO_DADOS: dict[str, str] = {"water_level": "1", "precipitation": "2", "discharge": "3"}
_HISTORICAL_FIELD_PREFIX: dict[str, str] = {"water_level": "Cota", "precipitation": "Chuva", "discharge": "Vazao"}

# ANA's QA level for conventional readings: 1 = Bruto (provisional), 2 = Consistido (final/reviewed).
_NIVEL_CONSISTENCIA_LABELS: dict[str, str] = {
    "1": "ANA conventional network, Bruto (provisional, not yet QA-reviewed)",
    "2": "ANA conventional network, Consistido (final, QA-reviewed)",
}

# ANA's per-parameter QC flag: 0 = ok, 1 = suspect, 2 = poor.
_QC_LABELS: dict[str, str] = {"1": "ANA QC flag: suspect reading", "2": "ANA QC flag: poor quality reading"}


def _qc_remark(status: str | None) -> str | None:
    if status is None:
        return None
    return _QC_LABELS.get(str(status).strip())


def _clean_float(value: object) -> float | None:
    """Parse a numeric field that ANA serialises as a string, or ``None`` when absent/blank."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_ana_datetime(value: str) -> datetime:
    """Parse ANA's ``"YYYY-MM-DD HH:MM:SS.f"`` timestamps (fractional seconds optional)."""
    text = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised ANA datetime: {value!r}")


def _parse_historical_xml(xml_text: str) -> list[dict]:
    """Parse a ``HidroSerieHistorica`` XML response into a list of field->text row dicts.

    ASP.NET ``.asmx`` DataSet-to-XML serialisation nests one element per row
    under the document root, each with one child element per column — we
    don't hardcode the row/root element names (documented as
    ``DocumentElement``/``SerieHistorica`` but not verified live) and just
    read whatever structure is actually there. A well-formed document with
    zero row elements (ANA's own representation of "no data for this
    station/period") legitimately returns ``[]``. A document that doesn't
    parse as XML at all (HTML error page, truncated response, endpoint
    moved) is a different situation - that's raised as ``ET.ParseError``
    rather than swallowed, so the caller can tell "no data" apart from
    "something broke" instead of both looking like an empty list.
    """
    root = ET.fromstring(xml_text)  # noqa: S314 - trusted gov.br domain, not user-supplied XML

    rows = []
    for elem in root:
        row = {_local_tag(child): child.text for child in elem}
        if row:
            rows.append(row)
    return rows


def _local_tag(elem: ET.Element) -> str:
    """Strip any XML namespace prefix from an element tag (``{ns}Tag`` -> ``Tag``)."""
    tag = elem.tag
    return tag.split("}", 1)[1] if "}" in tag else tag


class BrazilANACollector(BaseCollector):
    """
    Collect telemetric streamflow, stage and rainfall data from ANA Hidroweb.

    Parameters
    ----------
    identificador : str | None
        CPF/CNPJ registered with ANA for API access. Falls back to the
        ``ANA_HIDROWEB_IDENTIFICADOR`` environment variable.
    senha : str | None
        Password for ``identificador``. Falls back to
        ``ANA_HIDROWEB_SENHA``.
    client : CachedHTTPClient, optional
        Injected for testing; a default client is created otherwise.

    Credentials are only needed for :meth:`fetch_raw`'s default telemetric
    mode. :meth:`stations` (public SNIRH catalog) and
    ``fetch_raw(mode="historical", ...)`` (legacy conventional-network
    series) both work without them.
    """

    name = "brazil_ana"

    def __init__(
        self,
        identificador: str | None = None,
        senha: str | None = None,
        client: CachedHTTPClient | None = None,
        legacy_client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=HIDROWEBSERVICE_BASE,
                rate_limiter=RateLimiter(max_calls=30, period_seconds=60),
                cache_ttl_seconds=900,  # telemetric data updates roughly hourly
            )
        )
        # Separate client/limiter for the legacy conventional-network service:
        # older infrastructure, no published rate limit, so keep pauses polite
        # and cache generously (this data is monthly-batch, not real-time).
        self.legacy_client = legacy_client or CachedHTTPClient(
            base_url=LEGACY_SOAP_BASE,
            rate_limiter=RateLimiter(max_calls=12, period_seconds=60),
            cache_ttl_seconds=86400,
        )
        self.identificador = identificador or os.environ.get("ANA_HIDROWEB_IDENTIFICADOR")
        self.senha = senha or os.environ.get("ANA_HIDROWEB_SENHA")
        if not self.identificador or not self.senha:
            logger.warning(
                "No ANA Hidroweb credentials provided (pass identificador=/senha= or set "
                "ANA_HIDROWEB_IDENTIFICADOR / ANA_HIDROWEB_SENHA). Station catalog lookups "
                "still work; fetch_raw() will raise until credentials are supplied."
            )
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._station_meta_cache: dict[str, dict] | None = None

    # ── auth ─────────────────────────────────────────────────────────
    def _get_token(self) -> str:
        if self._token and time.monotonic() < self._token_expiry:
            return self._token

        if not self.identificador or not self.senha:
            raise RuntimeError(
                "ANA Hidroweb requires credentials for time-series data. Pass "
                "identificador=/senha= to BrazilANACollector(), or set the "
                "ANA_HIDROWEB_IDENTIFICADOR / ANA_HIDROWEB_SENHA environment variables. "
                "Request access by emailing hidro@ana.gov.br (subject: 'Solicitação de "
                "acesso à API'); if that doesn't get a response, telemetria@ana.gov.br / "
                "(61) 2109-5391 is ANA's other published telemetry contact — see "
                "https://www.ana.gov.br/hidrowebservice/swagger-ui.html for the current "
                "process."
            )

        data = self.client.get_json(
            AUTH_PATH,
            headers={"Identificador": self.identificador, "Senha": self.senha},
            use_cache=False,
        )
        token = (data.get("items") or {}).get("tokenautenticacao")
        if not token:
            raise RuntimeError(f"ANA Hidroweb authentication failed: {data}")

        self._token = token
        self._token_expiry = time.monotonic() + TOKEN_LIFETIME_SECONDS
        return token

    # ── station catalog (public, no auth) ───────────────────────────
    def stations(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        variable: str | None = None,
        max_items: int | None = None,
        telemetric_only: bool = True,
    ) -> list[Station]:
        """SNIRH station catalog. ``telemetric_only`` restricts to stations with live telemetry
        (the ones :meth:`fetch_raw` can actually query); set it to ``False`` to see the full
        conventional network too.
        """
        where = "EstacaoTelemetrica='Sim'" if telemetric_only else "1=1"
        out_fields = "Codigo,Nome,TipoEstacao,Latitude,Longitude,Rio,UF,Municipio,Bacia,Operando"

        stations: list[Station] = []
        offset = 0
        while True:
            page = self.client.get_json(
                STATIONS_URL,
                params={
                    "where": where,
                    "outFields": out_fields,
                    "f": "json",
                    "returnGeometry": "false",
                    "resultOffset": offset,
                    "resultRecordCount": STATIONS_PAGE_SIZE,
                },
            )
            features = page.get("features", [])
            if not features:
                break

            for feat in features:
                attrs = feat.get("attributes", {})
                lat, lon = attrs.get("Latitude"), attrs.get("Longitude")
                code = attrs.get("Codigo")
                if lat is None or lon is None or code is None:
                    continue
                lat, lon = float(lat), float(lon)
                if not in_bbox(lat, lon, bbox):
                    continue

                tipo = attrs.get("TipoEstacao")
                station_variables = ("discharge", "water_level") if tipo == "Fluviométrica" else ("precipitation",)
                if variable and variable not in station_variables:
                    continue

                stations.append(
                    Station(
                        source="brazil_ana",
                        station_id=str(code),
                        name=attrs.get("Nome"),
                        latitude=lat,
                        longitude=lon,
                        variables=station_variables,
                        river=attrs.get("Rio") or None,
                        country="BRA",
                        url="https://www.snirh.gov.br/hidroweb/apresentacao",
                        extra={
                            "uf": attrs.get("UF"),
                            "municipio": attrs.get("Municipio"),
                            "bacia": attrs.get("Bacia"),
                            "operando": attrs.get("Operando"),
                        },
                    )
                )
                if max_items is not None and len(stations) >= max_items:
                    return stations

            if len(features) < STATIONS_PAGE_SIZE:
                break
            offset += STATIONS_PAGE_SIZE

        return stations

    def _station_meta(self, station_id: str) -> dict | None:
        """Lazily built, cached ``station_id -> {name, lat, lon, river}`` lookup.

        Built once per collector instance from the public catalog so
        :meth:`normalise` can attach coordinates/names without a network call
        per row. Failure to load the catalog (e.g. offline) degrades to no
        enrichment rather than breaking normalisation.
        """
        if self._station_meta_cache is None:
            cache: dict[str, dict] = {}
            try:
                for st in self.stations():
                    cache[st.station_id] = {"name": st.name, "lat": st.latitude, "lon": st.longitude, "river": st.river}
            except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
                logger.debug("Brazil ANA: could not load station catalog for enrichment: %s", exc)
            self._station_meta_cache = cache
        return self._station_meta_cache.get(station_id)

    # ── time series ──────────────────────────────────────────────────
    def fetch_raw(
        self,
        station_ids: Sequence[str] | str | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
        days: int = 30,
        mode: str = "telemetric",
        variables: Sequence[str] = ("discharge",),
        **kwargs,
    ) -> list[dict]:
        """
        Fetch readings for one or more stations.

        Parameters
        ----------
        station_ids : str | Sequence[str]
            One or more ANA station codes (see :meth:`stations`). Required.
        start_date, end_date : str | date, optional
            Window bounds, ``YYYY-MM-DD``. Defaults to the last ``days`` days
            when omitted (``mode="telemetric"`` only — ``"historical"``
            passes bounds straight through to ANA, and omitting them fetches
            each station's full period of record).
        days : int
            Size of the default window when ``start_date`` is not given
            (``mode="telemetric"`` only).
        mode : {"telemetric", "historical"}
            ``"telemetric"`` (default) hits the credentialed HidroWebService
            REST API for near-real-time stage/discharge/rainfall — see
            module docstring. ``"historical"`` hits the legacy, no-auth
            ``HidroSerieHistorica`` webservice for ANA's conventional
            (manually read) network, which reaches decades further back but
            updates far less often.
        variables : Sequence[str]
            Which parameters to request in ``mode="historical"`` — any of
            ``"discharge"``, ``"water_level"``, ``"precipitation"``. Ignored
            in ``mode="telemetric"`` (which always returns all three
            together per ANA's response shape).
        """
        if not station_ids:
            raise ValueError(
                "BrazilANACollector.fetch_raw() requires station_ids (see .stations() for the ANA station catalog)."
            )
        if isinstance(station_ids, str):
            station_ids = [station_ids]

        if mode == "historical":
            return self._fetch_historical_raw(
                station_ids, variables=variables, start_date=start_date, end_date=end_date
            )
        if mode != "telemetric":
            raise ValueError(f"Unknown mode {mode!r}; expected 'telemetric' or 'historical'.")

        if end_date is None:
            end_dt = date.today()
        elif isinstance(end_date, str):
            end_dt = date.fromisoformat(end_date)
        else:
            end_dt = end_date

        if start_date is None:
            start_dt = end_dt - timedelta(days=days)
        elif isinstance(start_date, str):
            start_dt = date.fromisoformat(start_date)
        else:
            start_dt = start_date

        token = self._get_token()
        rows: list[dict] = []
        failures = 0
        for station_id in station_ids:
            try:
                data = self.client.get_json(
                    SERIES_PATH,
                    params={
                        "codEstacao": str(station_id),
                        "dataInicio": start_dt.isoformat(),
                        "dataFim": end_dt.isoformat(),
                        "tipoFiltroData": "DATA_LEITURA",
                    },
                    headers={"Authorization": f"Bearer {token}"},
                )
            except Exception as exc:  # noqa: BLE001 - counted below; one bad station shouldn't abort the batch
                logger.warning("Brazil ANA: failed to fetch station %s: %s", station_id, exc)
                failures += 1
                continue

            if "items" not in data:
                # A genuine HidroWebService response always has an "items" key, even when
                # it's an empty list (no data for the period). A missing key is a shape
                # drift signal (renamed field, error page, etc.), not "no data" - count it
                # as a failure rather than silently treating it as zero readings.
                logger.warning(
                    "Brazil ANA: unexpected response shape for station %s (no 'items' key): %s", station_id, data
                )
                failures += 1
                continue

            items = data.get("items") or []
            for item in items:
                item.setdefault("codigoestacao", str(station_id))
                item["_mode"] = "telemetric"
                rows.append(item)

        if failures and failures == len(station_ids):
            raise RuntimeError(
                f"Brazil ANA: all {failures} station request(s) failed or returned an unexpected "
                "response shape. This most likely means the HidroWebService endpoint or its "
                "response shape has changed, not that the requested station(s) have no data - "
                "see the warnings above for the per-station errors."
            )

        return rows

    def _fetch_historical_raw(
        self,
        station_ids: Sequence[str],
        variables: Sequence[str],
        start_date: str | date | None,
        end_date: str | date | None,
    ) -> list[dict]:
        """Fetch ANA's conventional (non-telemetric) network via the legacy HidroSerieHistorica webservice.

        No credentials required. One request per (station, variable); a
        failure on any one is logged and skipped rather than aborting the
        whole batch (mirrors the telemetric path's per-station isolation) -
        but if *every* request in the batch fails (network error, or a
        response that doesn't parse as XML at all), that's raised rather
        than silently returned as "no data", since it almost always means
        the endpoint or its response shape has changed rather than that
        every requested station/variable genuinely has no records.
        """
        for v in variables:
            if v not in _TIPO_DADOS:
                raise ValueError(f"Unknown variable {v!r}; expected one of {sorted(_TIPO_DADOS)}")

        date_params = {}
        if start_date is not None:
            date_params["dataInicio"] = start_date if isinstance(start_date, str) else start_date.isoformat()
        if end_date is not None:
            date_params["dataFim"] = end_date if isinstance(end_date, str) else end_date.isoformat()

        rows: list[dict] = []
        attempted = 0
        failures = 0
        for station_id in station_ids:
            for variable in variables:
                attempted += 1
                try:
                    xml_text = self.legacy_client.get_text(
                        HISTORICAL_SERIES_PATH,
                        params={
                            "codEstacao": str(station_id),
                            "tipoDados": _TIPO_DADOS[variable],
                            # Empty (not omitted) nivelConsistencia returns both
                            # Bruto and Consistido rows, which is what makes the
                            # Heberger-style dedup below meaningful.
                            "nivelConsistencia": "",
                            **date_params,
                        },
                    )
                    parsed_rows = _parse_historical_xml(xml_text)
                except Exception as exc:  # noqa: BLE001 - counted below; one bad station/variable shouldn't abort the batch
                    logger.warning(
                        "Brazil ANA (historical): failed to fetch/parse station %s / %s: %s",
                        station_id,
                        variable,
                        exc,
                    )
                    failures += 1
                    continue

                for row in parsed_rows:
                    row["_mode"] = "historical"
                    row["_station_id"] = str(station_id)
                    row["_variable"] = variable
                    rows.append(row)

        if failures and failures == attempted:
            raise RuntimeError(
                f"Brazil ANA (historical): all {failures} request(s) failed to fetch or parse. "
                "This most likely means the HidroSerieHistorica endpoint or its response shape "
                "has changed, not that the requested station(s)/variable(s) have no data - see "
                "the warnings above for the per-request errors."
            )

        return rows

    def normalise(self, raw: list[dict]) -> Sequence[StreamflowReading | WaterLevelReading | ClimateReading]:
        telemetric_rows = [r for r in raw if r.get("_mode") != "historical"]
        historical_rows = [r for r in raw if r.get("_mode") == "historical"]

        readings: list[StreamflowReading | WaterLevelReading | ClimateReading] = []
        if telemetric_rows:
            readings.extend(self._normalise_telemetric(telemetric_rows))
        if historical_rows:
            readings.extend(self._normalise_historical(historical_rows))
        return readings

    def _normalise_telemetric(self, raw: list[dict]) -> list[StreamflowReading | WaterLevelReading | ClimateReading]:
        readings: list[StreamflowReading | WaterLevelReading | ClimateReading] = []
        skipped = 0

        for row in raw:
            try:
                station_id = str(row.get("codigoestacao") or "").strip()
                time_str = row.get("Data_Hora_Medicao")
                if not station_id or not time_str:
                    skipped += 1
                    continue
                dt = _parse_ana_datetime(str(time_str))

                meta = self._station_meta(station_id)
                loc = (
                    GeoLocation(latitude=meta["lat"], longitude=meta["lon"])
                    if meta and meta.get("lat") is not None and meta.get("lon") is not None
                    else None
                )
                station_name = meta.get("name") if meta else None

                found_any = False

                discharge_cms = _clean_float(row.get("Vazao_Adotada"))
                if discharge_cms is not None:
                    found_any = True
                    readings.append(
                        StreamflowReading(
                            source=DataSource.BRAZIL_ANA,
                            station_id=station_id,
                            station_name=station_name,
                            location=loc,
                            reading_datetime=dt,
                            discharge_cms=discharge_cms,
                            source_type="in_situ",
                            remark=_qc_remark(row.get("Vazao_Adotada_Status")),
                        )
                    )

                stage_cm = _clean_float(row.get("Cota_Adotada"))
                if stage_cm is not None:
                    found_any = True
                    readings.append(
                        WaterLevelReading(
                            source=DataSource.BRAZIL_ANA,
                            station_id=station_id,
                            station_name=station_name,
                            location=loc,
                            reading_datetime=dt,
                            water_level=stage_cm / 100.0,  # cm -> m
                            unit="m",
                            remark=_qc_remark(row.get("Cota_Adotada_Status")),
                        )
                    )

                rainfall_mm = _clean_float(row.get("Chuva_Adotada"))
                if rainfall_mm is not None:
                    found_any = True
                    readings.append(
                        ClimateReading(
                            source=DataSource.BRAZIL_ANA,
                            station_id=station_id,
                            station_name=station_name,
                            location=loc,
                            sample_datetime=dt,
                            parameter="rainfall_mm",
                            value=rainfall_mm,
                            unit="mm",
                            remark=_qc_remark(row.get("Chuva_Adotada_Status")),
                        )
                    )

                if not found_any:
                    skipped += 1

            except (ValueError, KeyError, TypeError) as exc:
                skipped += 1
                logger.debug("Brazil ANA: skipping row: %s", exc)

        if skipped:
            logger.warning(
                "Brazil ANA normalise(): skipped %d/%d telemetric row(s) (missing/invalid fields)", skipped, len(raw)
            )

        return readings

    def _normalise_historical(self, raw: list[dict]) -> list[StreamflowReading | WaterLevelReading | ClimateReading]:
        """Turn ANA conventional-network rows into readings, one per day.

        Adapted from Matthew Heberger's ``convert_vazoes_2_csv.py``
        (https://github.com/mheberger/brazil-discharge, MIT License), which
        parses this exact wide-per-month layout from ANA's bulk historical
        exports. Ported quirks:

        1. **Provisional vs final duplicates** — ANA returns both "Bruto"
           (``NivelConsistencia="1"``, provisional) and "Consistido"
           (``"2"``, QA-reviewed) rows for the same station+month; Heberger
           sorts by QA code and keeps the first (most-final) row per month.
           We do the same, grouping by (station, variable, year, month) and
           keeping the row with the highest ``NivelConsistencia``.
        2. **Relisted-month bug** — Heberger found some months listed a
           second time under a date whose day-of-month isn't ``01`` (ANA's
           month rows are otherwise always dated the 1st). He drops any row
           where the date's day isn't ``01``; we do the same before grouping.
        3. **Wide-to-long melt** — each surviving row has one column per
           day of the month (``Vazao01``..``Vazao31`` etc.); we emit one
           reading per non-blank day, skipping calendar-invalid days (e.g.
           ``Vazao31`` in a 30-day month) the same way Heberger's
           ``pd.to_datetime(..., errors="coerce")`` + dropna does.

        Unlike Heberger's script, we don't reindex to an evenly-spaced daily
        series with explicit null rows for gaps — that's a good fit for a
        CSV/DataFrame output but not for a stream of typed reading records,
        so gaps are simply absent here rather than represented.
        """
        # Group by (station, variable, year, month) so Bruto/Consistido duplicates
        # for the same month end up together before we pick which one to keep.
        groups: dict[tuple[str, str, int, int], list[dict]] = {}
        skipped = 0

        for row in raw:
            try:
                station_id = row["_station_id"]
                variable = row["_variable"]
                month_str = row.get("Data") or row.get("DataHora")
                if not month_str:
                    skipped += 1
                    continue
                month_dt = _parse_ana_datetime(str(month_str).split("T")[0])
                if month_dt.day != 1:
                    # Heberger's "relisted month" quirk — drop it.
                    skipped += 1
                    continue
                key = (station_id, variable, month_dt.year, month_dt.month)
                groups.setdefault(key, []).append(row)
            except (KeyError, ValueError) as exc:
                skipped += 1
                logger.debug("Brazil ANA (historical): skipping unparseable row: %s", exc)

        readings: list[StreamflowReading | WaterLevelReading | ClimateReading] = []
        for (station_id, variable, year, month), group_rows in groups.items():
            # Prefer NivelConsistencia="2" (Consistido/final) over "1" (Bruto/provisional).
            group_rows.sort(key=lambda r: r.get("NivelConsistencia") or "0", reverse=True)
            row = group_rows[0]
            remark = _NIVEL_CONSISTENCIA_LABELS.get(row.get("NivelConsistencia") or "", None)

            meta = self._station_meta(station_id)
            loc = (
                GeoLocation(latitude=meta["lat"], longitude=meta["lon"])
                if meta and meta.get("lat") is not None and meta.get("lon") is not None
                else None
            )
            station_name = meta.get("name") if meta else None
            prefix = _HISTORICAL_FIELD_PREFIX[variable]

            for day in range(1, 32):
                value = _clean_float(row.get(f"{prefix}{day:02d}"))
                if value is None:
                    continue
                try:
                    dt = datetime(year, month, day)
                except ValueError:
                    continue  # calendar-invalid day (e.g. day 31 in a 30-day month)

                if variable == "discharge":
                    readings.append(
                        StreamflowReading(
                            source=DataSource.BRAZIL_ANA,
                            station_id=station_id,
                            station_name=station_name,
                            location=loc,
                            reading_datetime=dt,
                            discharge_cms=value,
                            source_type="in_situ",
                            remark=remark,
                        )
                    )
                elif variable == "water_level":
                    readings.append(
                        WaterLevelReading(
                            source=DataSource.BRAZIL_ANA,
                            station_id=station_id,
                            station_name=station_name,
                            location=loc,
                            reading_datetime=dt,
                            water_level=value / 100.0,  # cm -> m
                            unit="m",
                            remark=remark,
                        )
                    )
                elif variable == "precipitation":
                    readings.append(
                        ClimateReading(
                            source=DataSource.BRAZIL_ANA,
                            station_id=station_id,
                            station_name=station_name,
                            location=loc,
                            sample_datetime=dt,
                            parameter="rainfall_mm",
                            value=value,
                            unit="mm",
                            remark=remark,
                        )
                    )

        if skipped:
            logger.warning(
                "Brazil ANA normalise(): skipped %d/%d historical row(s) (missing/invalid month or relisted-month quirk)",
                skipped,
                len(raw),
            )

        return readings
