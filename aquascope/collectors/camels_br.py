"""
Collector for the CAMELS-BR (Catchment Attributes and Meteorology for Large
Sample Studies - Brazil) dataset, published by Chagas et al. (2020).

CAMELS-BR provides daily observed streamflow time series for 897 catchments
in Brazil, along with 65 catchment attributes (topography, climate, soil, etc.).

References
----------
- Chagas et al. (2020), ESSD: https://doi.org/10.5194/essd-12-2075-2020
- Zenodo repository: https://doi.org/10.5281/zenodo.3709337
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import DataSource, GeoLocation, StreamflowReading

logger = logging.getLogger(__name__)

CAMELS_BR_ATTRS_URL = "https://zenodo.org/api/records/15025488/files/01_CAMELS_BR_attributes.zip/content"
CAMELS_BR_FLOW_URL = (
    "https://zenodo.org/api/records/15025488/files/03_CAMELS_BR_streamflow_selected_catchments.zip/content"
)


def _clean(value):
    """Map NaN to None to prevent validation failures."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


class CAMELSBRCollector(BaseCollector):
    """Collect CAMELS-BR observed daily streamflow records."""

    name = "camels_br"

    def fetch_raw(
        self,
        station_ids: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch CAMELS-BR streamflow records, optionally filtered.

        Parameters
        ----------
        station_ids : list[str], optional
            Restrict to specific gauge codes (e.g. ["10500000", "11400000"]).
        start, end : str, optional
            ISO date strings ("YYYY-MM-DD") bounding the date range. Both
            are inclusive. If omitted, no date filtering is applied.
        """
        # Support single station_id in kwargs for compatibility
        resolved_ids = None
        if station_ids is not None:
            resolved_ids = [str(sid) for sid in station_ids]
        elif "station_id" in kwargs:
            resolved_ids = [str(kwargs["station_id"])]

        return self._fetch_camels_br(station_ids=resolved_ids, start=start, end=end)

    def _fetch_camels_br(
        self,
        station_ids: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict]:
        import hashlib
        import io
        import zipfile

        import httpx
        import pandas as pd

        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download attributes zip if not cached
        attrs_key = hashlib.md5(CAMELS_BR_ATTRS_URL.encode()).hexdigest()
        attrs_zip_path = cache_dir / f"camels_br_attrs_{attrs_key}.zip"
        if not attrs_zip_path.exists():
            logger.info("CAMELS-BR: downloading attributes zip (~273KB) — one-time download…")
            with httpx.stream(
                "GET",
                CAMELS_BR_ATTRS_URL,
                follow_redirects=True,
                timeout=600,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as resp:
                resp.raise_for_status()
                with attrs_zip_path.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=65_536):
                        fh.write(chunk)
        else:
            logger.debug("CAMELS-BR: using cached attributes %s", attrs_zip_path)

        # Download streamflow zip if not cached
        flow_key = hashlib.md5(CAMELS_BR_FLOW_URL.encode()).hexdigest()
        flow_zip_path = cache_dir / f"camels_br_flow_{flow_key}.zip"
        if not flow_zip_path.exists():
            logger.info("CAMELS-BR: downloading streamflow zip (~62MB) — one-time download…")
            with httpx.stream(
                "GET",
                CAMELS_BR_FLOW_URL,
                follow_redirects=True,
                timeout=600,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as resp:
                resp.raise_for_status()
                with flow_zip_path.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=65_536):
                        fh.write(chunk)
        else:
            logger.debug("CAMELS-BR: using cached streamflow %s", flow_zip_path)

        # Parse attributes
        with zipfile.ZipFile(attrs_zip_path) as zf:
            loc_name = next(n for n in zf.namelist() if n.endswith("camels_br_location.txt"))
            with zf.open(loc_name) as fh:
                loc_df = pd.read_csv(
                    io.BytesIO(fh.read()),
                    sep=r"\s+",
                    na_values=["nan", "NaN", "NA"],
                    dtype={"gauge_id": str},
                    usecols=["gauge_id", "gauge_name", "gauge_lat", "gauge_lon", "area_ana"],
                )

            topo_name = next(n for n in zf.namelist() if n.endswith("camels_br_topography.txt"))
            with zf.open(topo_name) as fh:
                topo_df = pd.read_csv(
                    io.BytesIO(fh.read()),
                    sep=r"\s+",
                    na_values=["nan", "NaN", "NA"],
                    dtype={"gauge_id": str},
                    usecols=["gauge_id", "area"],
                )

        loc_df["gauge_id"] = loc_df["gauge_id"].astype(str)
        topo_df["gauge_id"] = topo_df["gauge_id"].astype(str)
        attrs = pd.merge(loc_df, topo_df, on="gauge_id", how="outer")

        # Map 'area' (GSIM) to 'area_km2', falling back to 'area_ana' if missing
        attrs["area_km2"] = attrs["area"].fillna(attrs["area_ana"])
        attrs = attrs.drop(columns=["area", "area_ana"], errors="ignore")

        # Parse streamflow
        dfs = []
        with zipfile.ZipFile(flow_zip_path) as zf:
            flow_files = [n for n in zf.namelist() if n.endswith("_streamflow.txt")]

            if station_ids is not None:
                wanted = set(station_ids)
                flow_files = [f for f in flow_files if f.split("/")[-1].split("_")[0] in wanted]
                found_ids = {f.split("/")[-1].split("_")[0] for f in flow_files}
                missing = wanted - found_ids
                if missing:
                    logger.warning("CAMELS-BR: station_ids not found in dataset: %s", missing)

            for file_path in flow_files:
                gauge_id = file_path.split("/")[-1].split("_")[0]
                with zf.open(file_path) as fh:
                    df_gauge = pd.read_csv(
                        io.BytesIO(fh.read()),
                        sep=r"\s+",
                        na_values=["nan", "NaN", "NA"],
                    )
                    if df_gauge.empty or "year" not in df_gauge.columns:
                        continue

                    # Parse dates
                    df_gauge["date"] = pd.to_datetime(
                        df_gauge["year"].astype(str)
                        + "-"
                        + df_gauge["month"].astype(str).str.zfill(2)
                        + "-"
                        + df_gauge["day"].astype(str).str.zfill(2),
                        format="%Y-%m-%d",
                        errors="coerce",
                    )
                    df_gauge = df_gauge.dropna(subset=["date"])
                    df_gauge = df_gauge.rename(columns={"streamflow_m3s": "discharge"})
                    df_gauge = df_gauge.dropna(subset=["discharge"])

                    df_gauge["station_id"] = gauge_id
                    df_gauge = df_gauge[["date", "station_id", "discharge"]]
                    dfs.append(df_gauge)

        if not dfs:
            return []

        df = pd.concat(dfs, ignore_index=True)
        df["station_id"] = df["station_id"].astype(str)

        # Apply date filtering early
        if start is not None:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["date"] <= pd.to_datetime(end)]

        # Merge with attributes
        merged = df.merge(attrs, left_on="station_id", right_on="gauge_id", how="left")

        raw = merged.to_dict("records")
        for row in raw:
            row["date"] = str(row["date"])[:10]
        return raw

    def normalise(self, raw: list[dict]) -> Sequence[StreamflowReading]:
        records: list[StreamflowReading] = []
        for row in raw:
            try:
                lat = _clean(row.get("gauge_lat"))
                lon = _clean(row.get("gauge_lon"))
                loc = (
                    GeoLocation(latitude=float(lat), longitude=float(lon))
                    if lat is not None and lon is not None
                    else None
                )
                records.append(
                    StreamflowReading(
                        source=DataSource.CAMELS_BR,
                        station_id=str(row["station_id"]),
                        station_name=_clean(row.get("gauge_name")),
                        location=loc,
                        reading_datetime=datetime.fromisoformat(row["date"]),
                        discharge_cms=float(row["discharge"]),
                        source_type="in_situ",
                        catchment_area_km2=_clean(row.get("area_km2")),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping malformed CAMELS-BR record: %s", exc)
        return records
