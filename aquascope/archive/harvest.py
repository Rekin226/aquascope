"""Harvest station catalogs into GeoParquet + GeoJSON + health.json.

Everything here is pure Python plus ``pyarrow`` (the ``archive`` extra). The
GeoParquet output follows the 1.0.0 spec: a WKB ``geometry`` column and a
``geo`` entry in the file metadata, which is what DuckDB, GeoPandas, QGIS and
DuckDB-WASM in a browser all read without help.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from aquascope import __version__
from aquascope.registry import SOURCES, StationCatalog, station_catalogs, station_sources
from aquascope.schemas.station import Station
from aquascope.utils.imports import require

logger = logging.getLogger(__name__)

STATIONS_COLUMNS = [
    "source",
    "station_id",
    "site_id",
    "name",
    "latitude",
    "longitude",
    "variables",
    "period_start",
    "period_end",
    "url",
    "river",
    "country",
    "agency",
    "license",
    "redistributable",
    "extra",
    "geometry",
]


@dataclass
class SourceHealth:
    source: str
    ok: bool
    n_stations: int
    seconds: float
    error: str | None
    license: str
    redistributable: bool
    agency: str


@dataclass
class HarvestReport:
    """What a harvest run did, per source. Serialised as ``health.json``."""

    run_at: str
    aquascope_version: str
    n_stations: int
    sources: list[SourceHealth] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)

    @property
    def n_ok(self) -> int:
        return sum(1 for s in self.sources if s.ok)

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.sources if not s.ok)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_ok"] = self.n_ok
        d["n_failed"] = self.n_failed
        return d


def _point_wkb(lon: float, lat: float) -> bytes:
    """Little-endian WKB for a 2-D point (byte order 1, type 1, x, y)."""
    return struct.pack("<BIdd", 1, 1, lon, lat)


def _bbox(stations: list[Station]) -> list[float] | None:
    if not stations:
        return None
    lons = [s.longitude for s in stations]
    lats = [s.latitude for s in stations]
    return [min(lons), min(lats), max(lons), max(lats)]


def stations_to_table(stations: list[Station]):
    """Build a pyarrow Table with GeoParquet metadata from Station records."""
    pa = require("pyarrow", feature="archive parquet output", group="archive")

    rows = {col: [] for col in STATIONS_COLUMNS}
    for st in stations:
        meta = SOURCES.get(st.source)
        rows["source"].append(st.source)
        rows["station_id"].append(st.station_id)
        rows["site_id"].append(st.site_id)
        rows["name"].append(st.name)
        rows["latitude"].append(st.latitude)
        rows["longitude"].append(st.longitude)
        rows["variables"].append(list(st.variables))
        rows["period_start"].append(st.period_start)
        rows["period_end"].append(st.period_end)
        rows["url"].append(st.url)
        rows["river"].append(st.river)
        rows["country"].append(st.country or (meta.country if meta else None))
        rows["agency"].append(meta.agency if meta else None)
        rows["license"].append(meta.license if meta else "unknown")
        rows["redistributable"].append(bool(meta.redistributable) if meta else False)
        rows["extra"].append(json.dumps(st.extra, ensure_ascii=False, default=str) if st.extra else None)
        rows["geometry"].append(_point_wkb(st.longitude, st.latitude))

    schema = pa.schema(
        [
            ("source", pa.string()),
            ("station_id", pa.string()),
            ("site_id", pa.string()),
            ("name", pa.string()),
            ("latitude", pa.float64()),
            ("longitude", pa.float64()),
            ("variables", pa.list_(pa.string())),
            ("period_start", pa.date32()),
            ("period_end", pa.date32()),
            ("url", pa.string()),
            ("river", pa.string()),
            ("country", pa.string()),
            ("agency", pa.string()),
            ("license", pa.string()),
            ("redistributable", pa.bool_()),
            ("extra", pa.string()),
            ("geometry", pa.binary()),
        ]
    )
    geo = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Point"],
                "crs": None,  # null = OGC:CRS84 (WGS84 lon/lat) per the GeoParquet spec
                "bbox": _bbox(stations),
            }
        },
    }
    metadata = {
        b"geo": json.dumps(geo).encode(),
        b"aquascope": json.dumps({"version": __version__, "kind": "stations"}).encode(),
    }
    table = pa.Table.from_pydict(rows, schema=schema)
    return table.replace_schema_metadata(metadata)


def write_stations_parquet(stations: list[Station], path: Path) -> Path:
    pq = require("pyarrow.parquet", feature="archive parquet output", group="archive")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(stations_to_table(stations), str(path), compression="zstd")
    return path


GEOJSON_PROPERTIES = ("source", "station_id", "site_id", "name", "variables", "period_start", "period_end", "url")


def write_stations_geojson(stations: list[Station], path: Path) -> Path:
    """Slim GeoJSON twin of the parquet for map fallbacks (id, name, variables, period, link).

    Coordinates are rounded to 5 decimals (about 1 m) and the per-source
    extras stay in the parquet, which keeps a 45k-station file under 10 MB.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for st in stations:
        full = st.model_dump(mode="json")
        props = {k: full[k] for k in GEOJSON_PROPERTIES if full.get(k) not in (None, "", [], ())}
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(st.longitude, 5), round(st.latitude, 5)]},
                "properties": props,
            }
        )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _health(catalogs: dict[str, StationCatalog]) -> list[SourceHealth]:
    out: list[SourceHealth] = []
    for key in sorted(catalogs):
        cat = catalogs[key]
        meta = SOURCES[key]
        out.append(
            SourceHealth(
                source=key,
                ok=cat.ok,
                n_stations=len(cat.stations),
                seconds=round(cat.seconds, 2),
                error=cat.error,
                license=meta.license,
                redistributable=meta.redistributable,
                agency=meta.agency,
            )
        )
    return out


def harvest_stations(
    out_dir: str | Path,
    *,
    sources: list[str] | None = None,
    max_items: int | None = None,
    api_key: str | None = None,
    max_workers: int = 4,
    write_geojson: bool = True,
    write_card: bool = True,
) -> HarvestReport:
    """Harvest every station catalog into ``out_dir``.

    Writes ``stations.parquet`` (GeoParquet), ``stations.geojson``,
    ``health.json`` and, unless disabled, the dataset ``README.md``. Sources
    are the registry's station-capable ones (all of them, whatever their
    observation terms: a station *catalog* is factual metadata and always
    links back to the agency). Per-source failures are recorded, never raised.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = sources or station_sources()
    catalogs = station_catalogs(sources=keys, max_items=max_items, api_key=api_key, max_workers=max_workers)

    stations: list[Station] = []
    for key in sorted(catalogs):
        stations.extend(catalogs[key].stations)
    stations.sort(key=lambda s: (s.source, s.station_id))

    report = HarvestReport(
        run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        aquascope_version=__version__,
        n_stations=len(stations),
        sources=_health(catalogs),
    )

    parquet_path = write_stations_parquet(stations, out / "stations.parquet")
    report.files["stations.parquet"] = str(parquet_path.name)
    if write_geojson:
        geojson_path = write_stations_geojson(stations, out / "stations.geojson")
        report.files["stations.geojson"] = geojson_path.name
    (out / "health.json").write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    report.files["health.json"] = "health.json"
    if write_card:
        write_dataset_card(out / "README.md", report)
        report.files["README.md"] = "README.md"

    for s in report.sources:
        if s.ok:
            logger.info("[%s] %d stations in %.1fs", s.source, s.n_stations, s.seconds)
        else:
            logger.warning("[%s] FAILED after %.1fs: %s", s.source, s.seconds, s.error)
    logger.info(
        "Harvested %d stations from %d/%d sources into %s", len(stations), report.n_ok, len(report.sources), out
    )
    return report


def _obs_section(out_dir: Path, repo_id: str) -> str:
    """Describe harvested observations from obs/manifest.json when present."""
    manifest_path = out_dir / "obs" / "manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        from aquascope.archive.observations import _migrate_manifest

        manifest = _migrate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return ""
    lines = [
        "",
        "## Observations (filling up week by week)",
        "",
        "Daily values per station as `obs/<variable>/<source>/<station_id>.csv.gz` (`date,value`; discharge in "
        "m3/s, water and groundwater level in m, precipitation in mm/day), only for sources whose terms allow "
        "mirroring, plus one Parquet bundle per variable and source, `obs/<variable>/<source>.parquet` "
        "(`station_id, date, value`; join `station_id` to `stations.parquet`). `obs/manifest.json` lists every "
        "harvested station with its period, count and unit, and every bundle. Read one station:",
        "",
        "```python",
        f'pd.read_csv("hf://datasets/{repo_id}/obs/discharge/usgs/USGS-01646500.csv.gz")',
        "```",
        "",
        "a whole source in one go:",
        "",
        "```python",
        f'pd.read_parquet("hf://datasets/{repo_id}/obs/discharge/hubeau_hydrometrie.parquet")',
        "```",
        "",
        "or with DuckDB, joined to the catalog:",
        "",
        "```sql",
        "SELECT s.name, o.date, o.value",
        f"FROM 'hf://datasets/{repo_id}/obs/discharge/uk_ea.parquet' o",
        f"JOIN 'hf://datasets/{repo_id}/stations.parquet' s USING (station_id)",
        "WHERE s.name ILIKE '%thames%'",
        "```",
        "",
        "| source | variable | stations harvested | bundle | licence |",
        "| --- | --- | ---: | --- | --- |",
    ]
    bundles = manifest.get("bundles", {})
    for key in sorted(manifest.get("sources", {})):
        entry = manifest["sources"][key]
        n = entry.get("n_stations", sum(1 for v in entry.get("stations", {}).values() if v.get("n")))
        b = bundles.get(key)
        bundle = f"`{b['file']}` ({b['n_rows']:,} rows)" if b else "not yet"
        lines.append(
            f"| `{entry.get('source', key)}` | {entry.get('variable', '')} | {n:,} | {bundle} | "
            f"{entry.get('license', '')} |"
        )
    return "\n".join(lines) + "\n"


def write_dataset_card(path: Path, report: HarvestReport, repo_id: str = "Rekin226/aquascope-gauges") -> Path:
    """Write the Hugging Face dataset card (YAML front matter + per-source table)."""
    rows = []
    for s in report.sources:
        meta = SOURCES[s.source]
        status = f"{s.n_stations:,} stations" if s.ok else f"failed: {s.error}"
        mirror = "yes" if meta.redistributable else "catalog only"
        rows.append(
            f"| `{s.source}` | {meta.label} | {meta.country} | {', '.join(meta.variables)} | "
            f"{meta.license} | {mirror} | {status} |"
        )
    table = "\n".join(rows)
    front = (
        "---\n"
        "license: other\n"
        "license_name: per-source-open-licences\n"
        "license_link: https://github.com/Rekin226/aquascope/blob/main/docs/archive.md\n"
        "pretty_name: AquaScope gauges (world station catalog)\n"
        "tags:\n"
        "- hydrology\n"
        "- streamflow\n"
        "- water-level\n"
        "- groundwater\n"
        "- geoparquet\n"
        "- open-data\n"
        "size_categories:\n"
        "- 1K<n<10K\n"
        "---\n"
    )
    body = f"""# AquaScope gauges

The station catalog of every water-observation source [AquaScope](https://github.com/Rekin226/aquascope)
can reach, harvested on a schedule and published as GeoParquet. Last run
{report.run_at} with aquascope {report.aquascope_version}: **{report.n_stations:,} stations** from
{report.n_ok} of {len(report.sources)} sources.

Files:

- `stations.parquet`: GeoParquet 1.0 (WKB point geometry, WGS84). One row per station: `source`,
  `station_id`, `site_id`, `name`, `latitude`, `longitude`, `variables`, `period_start`, `period_end`, `url`
  (deep link to the agency page), `river`, `country`, `agency`, `license`, `redistributable`, `extra`.
- `stations.geojson`: the same rows as GeoJSON for tools that don't read parquet.
- `health.json`: per-source status of the last run (station count, seconds, error if any).

`site_id` identifies a physical site within a source and defaults to `station_id`.
Group by `(source, site_id)` to associate sub-stations while preserving every station record.

## Query it in place

DuckDB reads the parquet over HTTPS without downloading it:

```sql
INSTALL httpfs; LOAD httpfs;
SELECT source, count(*) FROM
  'https://huggingface.co/datasets/{repo_id}/resolve/main/stations.parquet'
GROUP BY source ORDER BY 2 DESC;
```

Or from Python: `pandas.read_parquet("hf://datasets/{repo_id}/stations.parquet")`.

## Sources and terms

Only sources whose terms allow redistribution will have their observations mirrored (Phase 1 of
[#188](https://github.com/Rekin226/aquascope/issues/188)); every source appears in the catalog with a
link back to the agency. Attribution for each source is in the AquaScope registry
(`aquascope.registry.SOURCES[key].attribution`).

| key | source | country | variables | licence | mirror observations | last run |
| --- | --- | --- | --- | --- | --- | --- |
{table}

{_obs_section(path.parent, repo_id)}
## Catchments (BasinATLAS, `basins/`)

The level-12 sub-basins of [HydroATLAS v1.0 / BasinATLAS](https://www.hydrosheds.org/hydroatlas)
(Linke et al. 2019, **CC BY 4.0**) with their routing and attributes, so any point on land can be
placed in its catchment and described: `basins/lev12.fgb` (simplified polygons, spatially indexed;
point lookups over HTTPS read a few kilobytes), `basins/lev12_topology.parquet` (`hybas_id,
next_down, main_bas, sub_area, up_area, ...` plus a representative point), `basins/lev12_attributes.parquet`
(every BasinATLAS attribute per sub-basin, incl. the upstream-aggregated `*_u*` fields), and
`basins/lev12.pmtiles` / `basins/lev06.pmtiles` for maps. Built by
[basins.yml](https://github.com/Rekin226/aquascope/actions/workflows/basins.yml);
`aquascope basins at LAT LON` and the MCP tool `describe_catchment` read them. Cite: Linke, S., Lehner, B.,
Ouellet Dallaire, C., et al. (2019). Global hydro-environmental sub-basin and river reach characteristics
at high spatial resolution. Scientific Data 6: 283. https://doi.org/10.1038/s41597-019-0300-6

## How it is built

`aquascope harvest stations --out archive --publish {repo_id}` runs weekly from
[GitHub Actions](https://github.com/Rekin226/aquascope/actions/workflows/harvest.yml). Every collector
answers `stations()` from its own agency API; nothing is hand-edited. Add a source to AquaScope and
it appears here on the next run.

## Citation

Cite the AquaScope software (Zenodo concept DOI in the repository README) and the agency of any
source you use, as listed in the table above.
"""
    path.write_text(front + body, encoding="utf-8")
    return path
