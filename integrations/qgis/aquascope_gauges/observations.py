"""Read daily observations from the AquaScope archive using the standard library."""

import csv
import gzip
import io
import json
import math
from datetime import date
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import urlopen

HARVESTABLE = {
    "usgs": ("discharge", "water_level"),
    "uk_ea": (
        "discharge",
        "water_level",
        "precipitation",
        "groundwater_level",
    ),
    "hubeau_hydrometrie": ("discharge",),
    "taiwan_cwa": ("precipitation",),
    "greece_openhi": ("discharge", "water_level", "precipitation"),
    "poland_imgw": ("discharge", "water_level"),
}

ARCHIVE_UNITS = {
    "discharge": "m3/s",
    "water_level": "m",
    "groundwater_level": "m",
    "precipitation": "mm",
}

DEFAULT_BASE_URL = (
    "https://huggingface.co/datasets/Rekin226/"
    "aquascope-gauges/resolve/main/"
)

TIMEOUT_SECONDS = 30
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


class ObservationError(Exception):
    """Raised when daily observations cannot be retrieved or parsed."""


class ObservationNotFoundError(ObservationError):
    """Raised when a station has no archived daily observations."""


def available_variables(source):
    """Return the variables mirrored for a source."""
    return HARVESTABLE.get(str(source or "").strip(), ())


def _validated_base_url(base_url):
    """Validate and normalize the HTTPS archive base URL."""
    base_url = str(base_url).strip()

    try:
        parsed = urlsplit(base_url)
        # Accessing .port also validates malformed port specifications.
        parsed.port
    except ValueError as exc:
        raise ValueError("Invalid archive base URL.") from exc

    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Archive base URL must be a valid HTTPS URL.")

    return base_url.rstrip("/") + "/"


def observation_url(source, station_id, variable, base_url=DEFAULT_BASE_URL):
    """Build the HTTPS URL for one station's daily observation archive."""
    source = str(source or "").strip()
    station_id = str(station_id or "").strip()
    variable = str(variable or "").strip()

    if source not in HARVESTABLE:
        raise ValueError("Unsupported observation source.")

    if variable not in available_variables(source):
        raise ValueError("Variable is not archived for this source.")

    if not station_id or station_id in {".", ".."}:
        raise ValueError("A valid station ID is required.")

    # Match aquascope.archive.observations.obs_path().
    station_id = station_id.replace("/", "_")

    if "\\" in station_id:
        raise ValueError("Station ID must not contain backslashes.")

    base_url = _validated_base_url(base_url)

    return (
        base_url
        + "obs/"
        + quote(variable, safe="")
        + "/"
        + quote(source, safe="")
        + "/"
        + quote(station_id, safe="")
        + ".csv.gz"
    )


def parse_observations(compressed_data):
    """Decompress and parse CSV.GZ bytes into daily observation records."""
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed_data)) as archive:
            csv_bytes = archive.read(MAX_DOWNLOAD_BYTES + 1)

        if len(csv_bytes) > MAX_DOWNLOAD_BYTES:
            raise ObservationError(
                "Decompressed observation data exceeds the size limit."
            )

        csv_data = csv_bytes.decode("utf-8-sig")

    except ObservationError:
        raise
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise ObservationError("Invalid or unreadable gzip archive.") from exc

    try:
        reader = csv.DictReader(io.StringIO(csv_data))

        if not reader.fieldnames:
            raise ObservationError("Observation CSV is empty.")

        reader.fieldnames = [
            name.strip() if name else name for name in reader.fieldnames
        ]

        if not {"date", "value"}.issubset(reader.fieldnames):
            raise ObservationError(
                "Observation CSV must contain date and value columns."
            )

        records = []

        for row in reader:
            raw_date = (row.get("date") or "").strip()
            raw_value = (row.get("value") or "").strip()

            if not raw_date or not raw_value:
                continue

            try:
                observation_date = date.fromisoformat(raw_date)
                observation_value = float(raw_value)
            except ValueError:
                continue

            if not math.isfinite(observation_value):
                continue

            records.append(
                {
                    "date": observation_date,
                    "value": observation_value,
                }
            )

        return records

    except csv.Error as exc:
        raise ObservationError("Could not parse observation CSV.") from exc


def fetch_observations(
    source,
    station_id,
    variable,
    base_url=DEFAULT_BASE_URL,
    timeout=TIMEOUT_SECONDS,
):
    """Fetch and parse daily observations for one station."""
    url = observation_url(source, station_id, variable, base_url)

    try:
        with urlopen(url, timeout=timeout) as response:
            compressed_data = response.read(MAX_DOWNLOAD_BYTES + 1)

    except HTTPError as exc:
        if exc.code == 404:
            raise ObservationNotFoundError(
                "No archived observations found for this station and variable."
            ) from exc

        raise ObservationError(
            f"Archive server returned HTTP {exc.code}."
        ) from exc

    except (URLError, TimeoutError, OSError) as exc:
        raise ObservationError(
            "Could not connect to the observation archive."
        ) from exc

    if len(compressed_data) > MAX_DOWNLOAD_BYTES:
        raise ObservationError("Observation archive exceeds the download limit.")

    return parse_observations(compressed_data)


@lru_cache(maxsize=1)
def load_observation_manifest(base_url=DEFAULT_BASE_URL):
    """Download and cache the observation manifest for this session."""
    base_url = _validated_base_url(base_url)
    url = base_url + "obs/manifest.json"

    try:
        with urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            data = response.read(MAX_DOWNLOAD_BYTES + 1)

        if len(data) > MAX_DOWNLOAD_BYTES:
            raise ObservationError("Observation manifest exceeds the size limit.")

        manifest = json.loads(data.decode("utf-8"))

    except ObservationError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError) as exc:
        raise ObservationError(
            "Could not download the observation manifest."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ObservationError(
            "The observation manifest contains invalid JSON."
        ) from exc

    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("sources"), dict
    ):
        raise ObservationError("The observation manifest has an invalid structure.")

    return manifest


def available_station_variables(source, station_id, manifest):
    """Return archived variable metadata available for one station."""
    source = str(source or "").strip()
    station_id = str(station_id or "").strip()

    if not source or not station_id or not isinstance(manifest, dict):
        return {}

    sources = manifest.get("sources", {})
    if not isinstance(sources, dict):
        return {}

    available = {}

    for key, entry in sources.items():
        if key.partition("/")[0] != source or not isinstance(entry, dict):
            continue

        stations = entry.get("stations", {})
        if not isinstance(stations, dict):
            continue

        station = stations.get(station_id)
        if not isinstance(station, dict):
            continue

        try:
            count = int(station.get("n", 0))
        except (TypeError, ValueError):
            continue

        if count <= 0 or station.get("empty") or not station.get("file"):
            continue

        variable = key.partition("/")[2]
        if not variable:
            continue

        available[variable] = {
            "first": station.get("first", ""),
            "last": station.get("last", ""),
            "count": count,
            "unit": station.get("unit") or entry.get("unit", ""),
            "attribution": entry.get("attribution", ""),
            "license": entry.get("license", ""),
        }

    return available
