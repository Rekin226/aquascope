"""
Collector for GRDC (Global Runoff Data Centre) river discharge data.

Two source types, tagged via ``source_type`` on :class:`StreamflowReading`:

- ``in_situ``   — curated gauge-station subset published on Zenodo
                  (no request-form gate, unlike the main GRDC portal).
- ``satellite`` — RSEG remote-sensing discharge extension, published on DaRUS.

References
----------
- In-situ subset (Zenodo record 19126732, CC BY-NC 4.0):
  https://zenodo.org/records/19126732
- RSEG (DaRUS, doi:10.18419/darus-3558):
  https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3558
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import DataSource, GeoLocation, StreamflowReading

logger = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api/records/19126732"
DARUS_DATASET_API = "https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi:10.18419/darus-3558"
DARUS_FILE_DOWNLOAD = "https://darus.uni-stuttgart.de/api/access/datafile/{file_id}"


class GRDCCollector(BaseCollector):
    """Collect GRDC in-situ (Zenodo) and RSEG satellite-extension (DaRUS) discharge records."""

    name = "grdc"

    def fetch_raw(self, source_type: str = "in_situ", **kwargs) -> list[dict]:
        """
        Fetch raw GRDC records.

        Parameters
        ----------
        source_type : str
            ``"in_situ"`` (default, Zenodo gauge subset) or
            ``"satellite"`` (RSEG, DaRUS).
        """
        if source_type == "in_situ":
            return self._fetch_zenodo_insitu()
        elif source_type == "satellite":
            return self._fetch_rseg()
        raise ValueError(f"source_type must be 'in_situ' or 'satellite', got {source_type!r}")

    def _fetch_zenodo_insitu(self) -> list[dict]:
        """Download (and locally cache) the Zenodo in-situ ZIP, parse station text files."""
        import hashlib
        import zipfile

        import httpx

        record = self.client.get_json(ZENODO_API)
        files = record.get("files", [])
        if not files:
            logger.warning("GRDC: no files found in Zenodo record")
            return []

        zip_entry = next((f for f in files if f["key"].endswith(".zip")), files[0])
        download_url = zip_entry["links"]["self"]
        checksum = zip_entry.get("checksum", download_url)

        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(checksum.encode()).hexdigest()
        zip_path = cache_dir / f"grdc_insitu_{cache_key}.zip"

        if not zip_path.exists():
            logger.info("GRDC: downloading in-situ archive — one-time download…")
            with httpx.stream("GET", download_url, follow_redirects=True, timeout=600) as resp:
                resp.raise_for_status()
                with zip_path.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=65_536):
                        fh.write(chunk)
        else:
            logger.debug("GRDC: using cached in-situ archive %s", zip_path)

        raw: list[dict] = []
        with zipfile.ZipFile(zip_path) as zf:
            station_files = [n for n in zf.namelist() if n.endswith(".txt") or n.endswith(".Cmd")]
            for name in station_files:
                with zf.open(name) as fh:
                    raw.extend(self._parse_grdc_station_file(fh.read().decode("latin-1"), name))
        return raw

    @staticmethod
    def _parse_grdc_station_file(text: str, filename: str) -> list[dict]:
        """
        Parse a classic GRDC station export (``#``-commented metadata header,
        then semicolon-delimited ``YYYY-MM-DD;HH:MM;value`` data rows).
        """
        lines = text.splitlines()
        meta: dict[str, str] = {}
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("#"):
                if ":" in line:
                    key, _, val = line.lstrip("#").partition(":")
                    meta[key.strip().lower().rstrip(".")] = val.strip()
            elif line.strip().lower().startswith("yyyy-mm-dd"):
                data_start = i + 1
                break

        station_id = meta.get("grdc-no", Path(filename).stem)
        station_name = meta.get("station")
        lat = meta.get("latitude")
        lon = meta.get("longitude")

        rows: list[dict] = []
        for line in lines[data_start:]:
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 3:
                continue
            date, _time, value = parts[0], parts[1], parts[2]
            try:
                value_f = float(value)
            except ValueError:
                continue
            if value_f <= -900:
                # GRDC/repo convention uses large negative sentinels for missing
                # data (e.g. -999, -9998, -9999) — see collectors' established
                # "-9998 sentinel" handling elsewhere in this codebase.
                continue
            rows.append(
                {
                    "station_id": station_id,
                    "station_name": station_name,
                    "latitude": float(lat) if lat else None,
                    "longitude": float(lon) if lon else None,
                    "date": date,
                    "discharge": value_f,
                    "source_type": "in_situ",
                }
            )
        return rows

    def _fetch_rseg(self) -> list[dict]:
        """Fetch the RSEG satellite discharge extension from DaRUS (Dataverse API)."""
        import csv
        import io

        import httpx

        dataset = self.client.get_json(DARUS_DATASET_API)
        files = dataset.get("data", {}).get("latestVersion", {}).get("files", [])
        csv_files = [f for f in files if f["dataFile"]["contentType"] == "text/csv"]
        if not csv_files:
            logger.warning("GRDC/RSEG: no CSV files found in DaRUS dataset")
            return []

        raw: list[dict] = []
        for f in csv_files:
            file_id = f["dataFile"]["id"]
            url = DARUS_FILE_DOWNLOAD.format(file_id=file_id)
            with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
                resp.raise_for_status()
                content = b"".join(resp.iter_bytes()).decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                raw.append({**row, "source_type": "satellite"})
        return raw

    def normalise(self, raw: list[dict]) -> Sequence[StreamflowReading]:
        records: list[StreamflowReading] = []
        for row in raw:
            try:
                loc = None
                if row.get("latitude") is not None and row.get("longitude") is not None:
                    loc = GeoLocation(latitude=row["latitude"], longitude=row["longitude"])
                records.append(
                    StreamflowReading(
                        source=DataSource.GRDC,
                        station_id=str(row["station_id"]),
                        station_name=row.get("station_name"),
                        location=loc,
                        reading_datetime=datetime.fromisoformat(row["date"]),
                        discharge_cms=float(row["discharge"]),
                        source_type=row.get("source_type", "in_situ"),
                        uncertainty_cms=row.get("uncertainty"),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping malformed GRDC record: %s", exc)
        return records
