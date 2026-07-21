"""
Collector for the CAMELS-CL (Catchment Attributes and Meteorology for Large
Sample Studies - Chile) dataset, published by CR2 (Centro de Ciencia del
Clima y la Resiliencia).

CAMELS-CL provides daily observed streamflow (compiled from DGA gauge
records) for 516 catchments in Chile, along with catchment attributes
(area, coordinates, etc.) and meteorological forcing data.

References
----------
- Dataset landing page: https://www.cr2.cl/camels-cl/
- Interactive explorer: https://camels.cr2.cl/
- Alvarez-Garreton et al. (2018), HESS: https://doi.org/10.5194/hess-22-5817-2018
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import DataSource, GeoLocation, StreamflowReading

logger = logging.getLogger(__name__)

CAMELS_CL_DOWNLOAD_URL = "https://www.cr2.cl/download/camels-cl-v202201/?wpdmdl=35317"


class CAMELSCLCollector(BaseCollector):
    """Collect CAMELS-CL observed daily streamflow (q_m3s_day.csv) records."""

    name = "camels_cl"

    def fetch_raw(
        self,
        station_ids: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch CAMELS-CL streamflow records, optionally filtered.

        Parameters
        ----------
        station_ids : list[str], optional
            Restrict to specific gauge codes (e.g. ["1001001", "12825002"]).
            Filtering happens before the wide-to-long melt, so this keeps
            memory bounded — unlike loading all ~500 stations at once.
        start, end : str, optional
            ISO date strings ("YYYY-MM-DD") bounding the date range. Both
            are inclusive. If omitted, no date filtering is applied.
        """
        return self._fetch_camels_cl(station_ids=station_ids, start=start, end=end)

    def _fetch_camels_cl(
        self,
        station_ids: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict]:
        """
        Download (and locally cache) the CAMELS-CL v202201 archive, parse the
        daily observed streamflow file (q_m3s_day.csv, m3/s — no unit
        conversion needed for StreamflowReading.discharge_cms), and join
        station coordinates/area from catchment_attributes.csv.

        station_ids/start/end are applied before melting to wide-to-long, to
        avoid materialising the full ~5.9M-row dataset when the caller only
        wants a subset.
        """
        import hashlib
        import io
        import zipfile

        import httpx
        import pandas as pd

        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(CAMELS_CL_DOWNLOAD_URL.encode()).hexdigest()
        zip_path = cache_dir / f"camels_cl_{cache_key}.zip"

        if not zip_path.exists():
            logger.info("CAMELS-CL: downloading CAMELS_CL_v202201.zip (~275MB) — one-time download…")
            with httpx.stream(
                "GET", CAMELS_CL_DOWNLOAD_URL, follow_redirects=True, timeout=600,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as resp:
                resp.raise_for_status()
                with zip_path.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=65_536):
                        fh.write(chunk)
        else:
            logger.debug("CAMELS-CL: using cached archive %s", zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            qobs_name = next(n for n in zf.namelist() if n.endswith("q_m3s_day.csv"))

            # Only load the columns we need: date + requested stations (or
            # all stations if none specified). Read the header first so we
            # can pass usecols on the real read, avoiding ~500 station
            # columns being loaded into memory when only a few are wanted.
            with zf.open(qobs_name) as fh:
                header = pd.read_csv(io.BytesIO(fh.read()), nrows=0).columns.tolist()

            station_cols = [c for c in header if c not in ("date", "year", "month", "day")]
            if station_ids is not None:
                wanted = set(station_ids)
                station_cols = [c for c in station_cols if c in wanted]
                missing = wanted - set(station_cols)
                if missing:
                    logger.warning("CAMELS-CL: station_ids not found in dataset: %s", missing)

            usecols = ["date", *station_cols]
            with zf.open(qobs_name) as fh:
                df = pd.read_csv(io.BytesIO(fh.read()), na_values=["NA"], usecols=usecols)

            attrs_name = next(n for n in zf.namelist() if n.endswith("catchment_attributes.csv"))
            with zf.open(attrs_name) as fh:
                attrs = pd.read_csv(
                    io.BytesIO(fh.read()),
                    na_values=["NA"],
                    dtype={"gauge_id": str},
                    usecols=["gauge_id", "gauge_name", "gauge_lat", "gauge_lon", "area_km2"],
                )

        if start is not None:
            df = df[df["date"] >= start]
        if end is not None:
            df = df[df["date"] <= end]

        long_df = df.melt(id_vars="date", var_name="station_id", value_name="discharge")
        long_df = long_df.dropna(subset=["discharge"])
        long_df["station_id"] = long_df["station_id"].astype(str)

        attrs["gauge_id"] = attrs["gauge_id"].astype(str)
        merged = long_df.merge(attrs, left_on="station_id", right_on="gauge_id", how="left")

        raw = merged.to_dict("records")
        for row in raw:
            row["date"] = str(row["date"])[:10]
        return raw

    def normalise(self, raw: list[dict]) -> Sequence[StreamflowReading]:
        records: list[StreamflowReading] = []
        for row in raw:
            try:
                lat = row.get("gauge_lat")
                lon = row.get("gauge_lon")
                loc = GeoLocation(latitude=float(lat), longitude=float(lon)) if lat and lon else None
                records.append(
                    StreamflowReading(
                        source=DataSource.CAMELS_CL,
                        station_id=str(row["station_id"]),
                        station_name=row.get("gauge_name"),
                        location=loc,
                        reading_datetime=datetime.fromisoformat(row["date"]),
                        discharge_cms=float(row["discharge"]),
                        source_type="in_situ",
                        catchment_area_km2=row.get("area_km2"),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping malformed CAMELS-CL record: %s", exc)
        return records
