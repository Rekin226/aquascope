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
        """
        Fetch the RSEG satellite discharge extension from DaRUS.

        RSEG is distributed as a single NetCDF file (RSEG_V01.nc, ~200MB),
        not CSV — see https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3558.

        Per Elmi et al. 2024 (Scientific Data), each station/time record carries
        a "flag" (0 = in-situ, 1-3 = different remote-sensing methods). We only
        emit flag >= 1 rows here as "satellite" — flag == 0 rows are the same
        in-situ data already covered by the Zenodo in-situ source, and including
        them here would double-count records.

        Exact NetCDF variable names are resolved from a candidate list rather
        than hardcoded, since they haven't been confirmed against the live file
        in this environment. If none of the candidates match, a ValueError is
        raised listing the real variable names — update _RSEG_VAR_CANDIDATES
        with the correct name and re-run.
        """
        import hashlib

        import httpx
        import xarray as xr

        dataset = self.client.get_json(DARUS_DATASET_API)
        files = dataset.get("data", {}).get("latestVersion", {}).get("files", [])
        nc_file = next((f for f in files if f["dataFile"]["filename"].endswith(".nc")), None)
        if nc_file is None:
            logger.warning("GRDC/RSEG: no NetCDF file found in DaRUS dataset")
            return []

        file_id = nc_file["dataFile"]["id"]
        checksum = nc_file["dataFile"].get("md5", str(file_id))
        url = DARUS_FILE_DOWNLOAD.format(file_id=file_id)

        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(checksum.encode()).hexdigest()
        nc_path = cache_dir / f"grdc_rseg_{cache_key}.nc"

        if not nc_path.exists():
            logger.info("GRDC/RSEG: downloading RSEG_V01.nc (~200MB) — one-time download…")
            with httpx.stream("GET", url, follow_redirects=True, timeout=1800) as resp:
                resp.raise_for_status()
                with nc_path.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=1_048_576):
                        fh.write(chunk)
        else:
            logger.debug("GRDC/RSEG: using cached NetCDF %s", nc_path)

        with xr.open_dataset(nc_path) as ds:
            var = {
                field: self._resolve_var(ds, field, candidates)
                for field, candidates in self._RSEG_VAR_CANDIDATES.items()
            }
            df = ds[list(var.values())].to_dataframe().reset_index()
            df = df.rename(columns={v: k for k, v in var.items()})

        # Drop in-situ rows (flag == 0) — already covered by the Zenodo source.
        if "flag" in df.columns:
            df = df[df["flag"] >= 1]

        df = df.dropna(subset=["discharge"])
        raw = df.to_dict("records")
        for row in raw:
            row["source_type"] = "satellite"
            row["station_id"] = str(row.get("station_id", row.get("grdc_no", "")))
            # normalise() expects an ISO date string under "date", not a
            # pandas/numpy Timestamp under "time".
            ts = row.pop("time", None)
            if ts is not None:
                row["date"] = ts.isoformat()[:10] if hasattr(ts, "isoformat") else str(ts)[:10]
        return raw

    _RSEG_VAR_CANDIDATES: dict[str, tuple[str, ...]] = {
        "discharge": ("Q", "discharge", "Qmm", "Q_m3s"),
        "uncertainty": ("Q_uncertainty", "uncertainty", "Q_error", "error"),
        "flag": ("flag", "source_flag", "data_flag"),
        "station_id": ("grdc_no", "GRDC_No", "station_id", "id"),
        "latitude": ("lat", "latitude"),
        "longitude": ("lon", "longitude"),
        "time": ("time", "date"),
    }

    @staticmethod
    def _resolve_var(ds, field: str, candidates: tuple[str, ...]) -> str:
        for name in candidates:
            if name in ds.variables:
                return name
        raise ValueError(
            f"GRDC/RSEG: could not resolve a NetCDF variable for '{field}' "
            f"(tried {candidates}). Actual variables in file: {list(ds.variables)}. "
            f"Update GRDCCollector._RSEG_VAR_CANDIDATES['{field}'] with the correct name."
        )

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
