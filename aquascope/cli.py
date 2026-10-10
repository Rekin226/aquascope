"""
AquaScope CLI — collect water data, analyse, and get AI methodology recommendations.

Usage
-----
    aquascope collect --source taiwan_moenv --api-key YOUR_KEY
    aquascope recommend --parameters DO,BOD5,COD --goal "trend analysis"
    aquascope eda --file data/raw/water_data.json
    aquascope quality --file data/raw/water_data.json
    aquascope run --method trend_analysis --file data/raw/water_data.json
    aquascope agri plan --crop maize --planting-date 2026-04-01 --eto-file eto.csv --precip-file precip.csv
    aquascope assess 51.415 -0.308 --problem flood_risk
    aquascope list-methods
    aquascope list-sources
    aquascope completion bash
"""

# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any

import argcomplete

# ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
# numexpr announces its thread count at INFO the moment pandas imports it, on every command; nobody asked.
logging.getLogger("numexpr").setLevel(logging.WARNING)
logger = logging.getLogger("aquascope")


def _serializable(value: Any) -> Any:
    """Convert CLI result objects into JSON-compatible values."""
    import numpy as np
    import pandas as pd

    if is_dataclass(value) and not isinstance(value, type):
        return _serializable(asdict(value))
    if isinstance(value, pd.DataFrame):
        return _serializable(value.rename_axis(value.index.name or "index").reset_index().to_dict("records"))
    if isinstance(value, pd.Series):
        name = value.name or "value"
        return _serializable(
            value.rename(name).rename_axis(value.index.name or "index").reset_index().to_dict("records")
        )
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (date, Path)):
        return value.isoformat() if isinstance(value, date) else str(value)
    return value


def _write_output(data: Any, path: str, fmt: str | None = None) -> Path:
    """Write structured CLI output as JSON or CSV, inferring the format from the suffix."""
    import pandas as pd

    output_path = Path(path)
    output_format = fmt or ("csv" if output_path.suffix.lower() == ".csv" else "json")
    serializable = _serializable(data)

    if output_format == "json":
        output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    else:
        records = serializable if isinstance(serializable, list) else [serializable]
        frame = pd.json_normalize(records)
        for column in frame.columns:
            frame[column] = frame[column].map(
                lambda value: json.dumps(value) if isinstance(value, (dict, list)) else value
            )
        frame.to_csv(output_path, index=False)

    return output_path


def _load_dataframe(path: str):
    """Load a JSON or CSV file into a pandas DataFrame."""
    import pandas as pd

    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    if p.suffix == ".csv":
        return pd.read_csv(p)
    elif p.suffix == ".json":
        return pd.read_json(p)
    else:
        logger.error("Unsupported file format: %s (use .json or .csv)", p.suffix)
        sys.exit(1)


def _parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    """Parse a bounding box string in west,south,east,north order."""
    if value is None:
        return None

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("Bounding box must have exactly four comma-separated values: west,south,east,north.")

    west, south, east, north = (float(part) for part in parts)
    return west, south, east, north


def cmd_collect(args: argparse.Namespace) -> None:
    """Run a data collector and save results."""
    from aquascope.collectors.base import CollectorError
    from aquascope.registry import build_collector, source_keys
    from aquascope.utils.storage import save_records

    source = args.source.lower()
    if source not in source_keys():
        logger.error("Unknown source '%s'. Available: %s", source, source_keys())
        sys.exit(1)

    ctor_kwargs = {}
    if source == "openmeteo" and args.mode:
        ctor_kwargs["mode"] = args.mode
    collector = build_collector(source, api_key=args.api_key, **ctor_kwargs)

    kwargs = {}
    if source == "usgs":
        if args.days is not None:
            kwargs["days"] = args.days
        if args.station_id:
            kwargs["station_id"] = args.station_id
        if args.parameter:
            kwargs["parameter"] = args.parameter
        if args.bbox:
            kwargs["bbox"] = args.bbox
        if args.state_code:
            kwargs["stateCd"] = args.state_code
        if args.county_code:
            kwargs["countyCd"] = args.county_code
        if args.huc:
            kwargs["huc"] = args.huc
    if source == "taiwan_cwa":
        if args.station_ids:
            kwargs["station_ids"] = [s.strip() for s in args.station_ids.split(",") if s.strip()]
        if args.start_date:
            kwargs["start"] = args.start_date
        if args.end_date:
            kwargs["end"] = args.end_date
    if source == "uk_ea":
        if args.collection:
            kwargs["collection"] = args.collection
        if args.observed_property:
            kwargs["observed_property"] = args.observed_property
        if args.measure:
            kwargs["measure"] = args.measure
        if args.station:
            kwargs["station"] = args.station
        if args.station_wiski_id:
            kwargs["station_wiski_id"] = args.station_wiski_id
        if args.bbox:
            kwargs["bbox"] = args.bbox
        if args.start_date:
            kwargs["min_date"] = args.start_date
        if args.end_date:
            kwargs["max_date"] = args.end_date
        if args.days is not None:
            kwargs["days"] = args.days
    if source == "sdg6" and args.countries:
        kwargs["country_codes"] = args.countries
    if source == "wqp":
        if args.state:
            kwargs["state_code"] = args.state
    if source == "aquastat":
        kwargs["country_code"] = args.country or "all"
        kwargs["start_year"] = args.start_year
        kwargs["end_year"] = args.end_year
        if args.variables:
            try:
                kwargs["variable_ids"] = [int(item.strip()) for item in args.variables.split(",") if item.strip()]
            except ValueError:
                logger.error("AQUASTAT variable IDs must be integers, e.g. 4263,4253,4312")
                sys.exit(1)
    if source in ("openmeteo", "copernicus"):
        if args.lat is not None:
            kwargs["latitude"] = args.lat
        if args.lon is not None:
            kwargs["longitude"] = args.lon
        if args.start_date:
            kwargs["start_date"] = args.start_date
        if args.end_date:
            kwargs["end_date"] = args.end_date
    if source == "wapor":
        if args.bbox:
            try:
                kwargs["bbox"] = _parse_bbox(args.bbox)
            except ValueError as exc:
                logger.error("%s", exc)
                sys.exit(1)
        if args.variable:
            kwargs["variable"] = args.variable
        if args.start_date:
            kwargs["start_date"] = args.start_date
        if args.end_date:
            kwargs["end_date"] = args.end_date
    if source == "eu_wfd":
        if args.country:
            kwargs["country"] = args.country
        if args.year:
            kwargs["year"] = args.year
        if args.water_body_type:
            kwargs["water_body_type"] = args.water_body_type
    if source == "grdc" and args.mode:
        if args.mode not in ("in_situ", "satellite"):
            logger.error("GRDC --mode must be 'in_situ' or 'satellite'; got '%s'.", args.mode)
            sys.exit(1)
        kwargs["source_type"] = args.mode
    if source in ("camels_cl", "camels_br"):
        if args.station_ids:
            kwargs["station_ids"] = [s.strip() for s in args.station_ids.split(",") if s.strip()]
        if args.start_date:
            kwargs["start"] = args.start_date
        if args.end_date:
            kwargs["end"] = args.end_date
    if source == "brazil_ana":
        if args.station_ids:
            kwargs["station_ids"] = [s.strip() for s in args.station_ids.split(",") if s.strip()]
        if args.days is not None:
            kwargs["days"] = args.days
        if args.start_date:
            kwargs["start_date"] = args.start_date
        if args.end_date:
            kwargs["end_date"] = args.end_date
    if source == "noaa_nwps":
        if not args.bbox and not args.lid:
            logger.error("NOAA NWPS requires either the --bbox or --lid argument.")
            sys.exit(1)
        if args.bbox and args.lid:
            logger.error("NOAA NWPS requires exactly one of --bbox or --lid.")
            sys.exit(1)
        if args.bbox:
            try:
                kwargs["bbox"] = _parse_bbox(args.bbox)
            except ValueError as exc:
                logger.error("%s", exc)
                sys.exit(1)
        if args.lid:
            kwargs["lid"] = args.lid
    if source == "pegelonline":
        if not args.station:
            logger.error("PEGELONLINE requires --station with a station UUID.")
            sys.exit(1)
        kwargs["station_id"] = args.station
        if args.days is not None:
            kwargs["days"] = args.days
        if args.timeseries:
            kwargs["timeseries"] = args.timeseries
        if args.start_date:
            kwargs["start"] = args.start_date
        if args.end_date:
            kwargs["end"] = args.end_date
    if source == "south_africa_dws":
        if not args.station:
            logger.error("South Africa DWS requires --station with a DWS gauge code, e.g. C1H001.")
            sys.exit(1)
        kwargs["station_id"] = args.station
        kwargs["variable"] = args.variable or "discharge"
        if args.days is not None:
            kwargs["days"] = args.days
        if args.start_date:
            kwargs["start_date"] = args.start_date
        if args.end_date:
            kwargs["end_date"] = args.end_date
    if source == "ireland_opw" and args.max_stations:
        kwargs["max_stations"] = args.max_stations
    if source == "bom":
        if not args.station:
            logger.error("BOM requires --station with an AWRC station number, e.g. 410001.")
            sys.exit(1)
        kwargs["station_id"] = args.station
        if args.days is not None:
            kwargs["days"] = args.days
        if args.start_date:
            kwargs["start_date"] = args.start_date
        if args.end_date:
            kwargs["end_date"] = args.end_date
        if args.parameter_type:
            kwargs["parameter_type"] = args.parameter_type
    try:
        records = collector.collect(**kwargs)
    except CollectorError as exc:
        logger.error("[%s] Collection failed: %s", source, exc)
        sys.exit(1)

    if not records:
        logger.warning("No records collected.")
        return

    path = save_records(records, prefix=source, fmt=args.format)
    print(f"✓ Saved {len(records)} records → {path}")


def cmd_recommend(args: argparse.Namespace) -> None:
    """Generate methodology recommendations."""
    from aquascope.ai_engine.recommender import (
        DatasetProfile,
        recommend,
        recommend_with_llm_detailed,
    )

    # Build profile from CLI args or from a data file
    parameters = [p.strip() for p in args.parameters.split(",")] if args.parameters else []
    profile = DatasetProfile(
        parameters=parameters,
        research_goal=args.goal or "",
        keywords=[k.strip() for k in (args.keywords or "").split(",") if k.strip()],
        geographic_scope=args.scope or "Taiwan",
        n_records=args.n_records or 0,
        n_stations=args.n_stations or 0,
        time_span_years=args.years or 0.0,
    )

    # If a data file is provided, infer some profile fields
    if args.from_file:
        path = Path(args.from_file)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                params_from_data = {r.get("parameter", "") for r in data if r.get("parameter")}
                profile.parameters = list(params_from_data | set(profile.parameters))
                profile.n_records = max(profile.n_records, len(data))
                stations = {r.get("station_id", "") for r in data if r.get("station_id")}
                profile.n_stations = max(profile.n_stations, len(stations))
                sources = {r.get("source", "") for r in data if r.get("source")}
                profile.data_sources = list(sources)

    engine_note = ""
    if args.use_llm:
        result = recommend_with_llm_detailed(
            profile,
            top_k=args.top_k,
            model=args.model or "gpt-4o-mini",
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
        )
        recs = result.recommendations
        if result.mode == "llm":
            engine_note = f"  Engine: {result.provider} · {result.model}"
        else:
            # Never degrade silently: say the LLM was skipped and why.
            print(f"⚠️  LLM unavailable — showing rule-based results. {result.error}")
            engine_note = "  Engine: rule-based (LLM fallback)"
    else:
        recs = recommend(profile, top_k=args.top_k)

    if not recs:
        if args.output:
            _write_output([], args.output, args.format)
        print("No matching methodologies found. Try broader parameters or keywords.")
        return

    print(f"\n{'=' * 70}")
    print(f"  AquaScope — Top {len(recs)} Research Methodology Recommendations")
    if engine_note:
        print(engine_note)
    print(f"{'=' * 70}\n")
    for i, rec in enumerate(recs, 1):
        m = rec.methodology
        print(f"  {i}. {m.name}  (score: {rec.score})")
        print(f"     Category   : {m.category}")
        print(f"     Scale      : {m.typical_scale}")
        print(f"     Complexity : {m.complexity}")
        print(f"     Rationale  : {rec.rationale}")
        if m.references:
            print(f"     Reference  : {m.references[0]}")
        print()

    if args.output:
        output_path = _write_output(recs, args.output, args.format)
        print(f"  ✓ Recommendations saved → {output_path}\n")


def cmd_eda(args: argparse.Namespace) -> None:
    """Run Exploratory Data Analysis on a data file."""
    from aquascope.analysis.eda import generate_eda_report, print_eda_report

    df = _load_dataframe(args.file)
    report = generate_eda_report(df)
    print(print_eda_report(report))

    if args.recommend:
        from aquascope.ai_engine.recommender import recommend
        from aquascope.analysis.eda import profile_dataset

        profile = profile_dataset(df)
        recs = recommend(profile, top_k=args.top_k)
        print(f"\n{'=' * 70}")
        print("  AI-Recommended Methodologies Based on EDA Profile")
        print(f"{'=' * 70}\n")
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec.methodology.name}  (score: {rec.score})")
            print(f"     {rec.rationale}\n")


def cmd_quality(args: argparse.Namespace) -> None:
    """Run data quality assessment."""
    from aquascope.analysis.quality import assess_quality, preprocess, print_quality_report

    df = _load_dataframe(args.file)
    report = assess_quality(df)
    print(print_quality_report(report))

    if args.fix:
        print(f"\n  Applying recommended fixes: {report.recommended_steps}")
        cleaned = preprocess(df, steps=report.recommended_steps)
        out_path = Path(args.file).with_stem(Path(args.file).stem + "_cleaned")
        if out_path.suffix == ".json":
            cleaned.to_json(out_path, orient="records", indent=2)
        else:
            cleaned.to_csv(out_path, index=False)
        print(f"  ✓ Cleaned data saved → {out_path}  ({len(df)} → {len(cleaned)} rows)")


def cmd_run_pipeline(args: argparse.Namespace) -> None:
    """Execute a methodology pipeline on data."""
    from aquascope.pipelines.model_builder import list_available_pipelines, run_pipeline

    if args.method not in list_available_pipelines():
        print(f"Unknown method '{args.method}'. Available pipelines:")
        for m in list_available_pipelines():
            print(f"  - {m}")
        sys.exit(1)

    df = _load_dataframe(args.file)
    config = json.loads(args.config) if args.config else None

    result = run_pipeline(args.method, df, config=config)

    print(f"\n{'=' * 70}")
    print(f"  AquaScope — Pipeline Result: {result.method_name}")
    print(f"{'=' * 70}\n")
    print(f"  {result.summary}\n")

    if result.metrics:
        print("  Metrics:")
        for k, v in result.metrics.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for kk, vv in v.items():
                    print(f"      {kk}: {vv}")
            else:
                print(f"    {k}: {v}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(
                {
                    "method_id": result.method_id,
                    "method_name": result.method_name,
                    "summary": result.summary,
                    "metrics": result.metrics,
                    "details": result.details,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\n  ✓ Full results saved → {out_path}")


def cmd_list_methods(args: argparse.Namespace) -> None:
    """List all available methodologies and pipelines."""
    from aquascope.ai_engine.knowledge_base import get_all_methodologies
    from aquascope.pipelines.model_builder import list_available_pipelines

    pipelines = set(list_available_pipelines())
    methods = get_all_methodologies()

    print(f"\n{'=' * 70}")
    print(f"  AquaScope — {len(methods)} Research Methodologies")
    print(f"{'=' * 70}\n")

    by_category: dict[str, list] = {}
    for m in methods:
        by_category.setdefault(m.category, []).append(m)

    for cat, items in sorted(by_category.items()):
        print(f"  [{cat}]")
        for m in items:
            runnable = " ✓ pipeline" if m.id in pipelines else ""
            print(f"    • {m.name} ({m.complexity}){runnable}")
        print()

    print(f"  Runnable pipelines: {len(pipelines)} / {len(methods)} methodologies")
    print("  Use 'aquascope run --method <id> --file <data>' to execute.\n")


def cmd_list_sources(args: argparse.Namespace) -> None:
    """List every registered data source (driven by aquascope.registry)."""
    from aquascope.registry import SOURCES

    print(f"\n{'=' * 70}")
    print(f"  AquaScope — {len(SOURCES)} Data Sources")
    print(f"{'=' * 70}\n")

    for key in sorted(SOURCES):
        meta = SOURCES[key]
        flags = []
        if meta.supports_station_lookup:
            flags.append("station catalog")
        if meta.supports_bbox:
            flags.append("bbox")
        if meta.requires_api_key:
            flags.append("API key")
        print(f"  {key}  ({meta.label})")
        print(f"    Region    : {meta.region}")
        print(f"    Agency    : {meta.agency or '—'}")
        print(f"    Data      : {meta.description}")
        print(f"    Variables : {', '.join(meta.variables) or '—'}")
        print(f"    License   : {meta.license}{' (redistributable)' if meta.redistributable else ''}")
        if flags:
            print(f"    Supports  : {', '.join(flags)}")
        print(f"    URL       : {meta.homepage or '—'}")
        print()


def cmd_stations(args: argparse.Namespace) -> None:
    """Search station catalogs across sources and save the result."""
    from aquascope.registry import station_catalogs, station_sources

    bbox = _parse_bbox(args.bbox) if args.bbox else None
    sources = args.source or None
    if sources is None and args.variable is None:
        logger.info("Searching every station-capable source: %s", station_sources())

    catalogs = station_catalogs(
        bbox=bbox,
        variable=args.variable,
        sources=sources,
        max_items=args.max_items,
        api_key=args.api_key,
    )
    if not catalogs:
        logger.error("No station-capable source matches. Sources with a catalog: %s", station_sources(args.variable))
        sys.exit(1)

    stations = [s for key in sorted(catalogs) for s in catalogs[key].stations]
    for key in sorted(catalogs):
        cat = catalogs[key]
        status = f"{len(cat.stations)} stations" if cat.ok else f"FAILED: {cat.error}"
        logger.info("[%s] %s (%.1fs)", key, status, cat.seconds)

    if not stations:
        logger.warning("No stations found.")
        if any(not c.ok for c in catalogs.values()):
            sys.exit(1)
        return

    fmt = args.format
    out_path = Path(args.output) if args.output else Path("data") / f"stations_{'_'.join(sorted(catalogs))}.{fmt}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [st.model_dump(mode="json") for st in stations]

    if fmt == "geojson":
        features = []
        for props in rows:
            lon, lat = props.pop("longitude"), props.pop("latitude")
            features.append(
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props}
            )
        out_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif fmt == "json":
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        import csv

        fields = [
            "source",
            "station_id",
            "name",
            "latitude",
            "longitude",
            "variables",
            "period_start",
            "period_end",
            "url",
            "river",
            "country",
            "extra",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["variables"] = "|".join(row.get("variables") or ())
                row["extra"] = json.dumps(row.get("extra") or {}, ensure_ascii=False)
                writer.writerow({k: row.get(k) for k in fields})
    logger.info("Saved %d stations to %s", len(stations), out_path)


def cmd_harvest(args: argparse.Namespace) -> None:
    """Harvest station catalogs into GeoParquet (+ GeoJSON, health.json) and optionally publish."""
    from aquascope.archive import harvest_stations, publish_folder

    if args.what == "obs":
        _cmd_harvest_obs(args)
        return
    if args.what == "bundles":
        _cmd_harvest_bundles(args)
        return
    if args.what == "signatures":
        _cmd_harvest_signatures(args)
        return

    report = harvest_stations(
        args.out,
        sources=args.source or None,
        max_items=args.max_items,
        api_key=args.api_key,
        max_workers=args.workers,
        write_geojson=not args.no_geojson,
        write_signatures=not args.no_signatures,
    )
    for s in report.sources:
        status = f"{s.n_stations:>6} stations" if s.ok else f"FAILED: {s.error}"
        print(f"  {s.source:<24} {status}  ({s.seconds:.1f}s)")
    print(f"\n  {report.n_stations} stations, {report.n_ok}/{len(report.sources)} sources OK -> {args.out}")

    if args.publish:
        if report.n_ok == 0:
            logger.error("Every source failed; not publishing an empty catalog.")
            sys.exit(1)
        url = publish_folder(args.out, args.publish, commit_message=f"harvest stations {report.run_at}")
        print(f"  published: {url}")

    if report.n_ok == 0:
        sys.exit(1)


def _cmd_harvest_obs(args: argparse.Namespace) -> None:
    """`aquascope harvest obs`: budgeted, incremental per-station daily series (#188 Phase 1)."""
    from aquascope.archive import publish_folder
    from aquascope.archive.observations import HARVESTABLE, harvest_observations, sync_from_hub

    if args.sync_from:
        sync_from_hub(args.out, args.sync_from)
    sources = args.source or None
    if sources:
        bad = [s for s in sources if s not in HARVESTABLE]
        if bad:
            logger.error("Not harvestable yet: %s. Choose from %s", bad, list(HARVESTABLE))
            sys.exit(2)
        if args.variable:
            bad = [s for s in sources if args.variable not in HARVESTABLE[s]]
            if bad:
                logger.error(
                    "%s is not harvested for %s (they mirror %s)",
                    args.variable,
                    bad,
                    {s: list(HARVESTABLE[s]) for s in bad},
                )
                sys.exit(2)
    report = harvest_observations(
        args.out,
        sources=sources,
        variable=args.variable,
        years=args.years,
        max_stations=args.max_stations,
        refresh_days=args.refresh_days,
        only_stations=args.station or None,
        max_seconds=args.max_seconds,
    )
    for h in report.sources:
        print(
            f"  {h.source:<20} {h.variable:<14} harvested {h.harvested:>4}  empty {h.empty:>4}  "
            f"failed {h.failed:>3}  of {h.attempted:>4} picked  ({h.seconds:.0f}s)"
            + (f"  stopped: {h.stopped}" if h.stopped else "")
        )
        for err in h.errors[:3]:
            print(f"      {err}")
    total = sum(h.harvested for h in report.sources)
    print(f"\n  {total} station files written under {args.out}/obs")

    if args.publish:
        url = publish_folder(args.out, args.publish, commit_message=f"harvest obs {report.run_at}")
        print(f"  published: {url}")


def _cmd_harvest_bundles(args: argparse.Namespace) -> None:
    """`aquascope harvest bundles`: roll obs/<variable>/<source>/*.csv.gz into one Parquet per pair (Phase 2)."""
    from aquascope.archive import publish_folder
    from aquascope.archive.bundles import build_bundles

    infos = build_bundles(args.out, variables=args.variable_list or None, sources=args.source or None)
    if not infos:
        print(f"  no observation files under {args.out}/obs; nothing to bundle")
        return
    for b in infos:
        print(
            f"  {b.file:<44} {b.n_stations:>6} stations {b.n_rows:>10,} rows  {b.bytes / 1e6:6.1f} MB  "
            f"{b.first} to {b.last}  ({b.seconds:.0f}s)"
        )
    print(f"\n  {len(infos)} bundles written")
    if args.publish:
        url = publish_folder(args.out, args.publish, commit_message="harvest bundles")
        print(f"  published: {url}")


def _cmd_harvest_signatures(args: argparse.Namespace) -> None:
    """`aquascope harvest signatures`: build signatures.parquet from the mirrored discharge files in --out."""
    from aquascope.archive import publish_folder
    from aquascope.archive.signatures import build_signatures_file

    summary = build_signatures_file(args.out, sources=args.source or None)
    if not summary.get("file"):
        print(f"  {summary.get('note', 'nothing to do')}")
        return
    print(
        f"  {summary['n_stations']:,} stations -> {summary['file']}: flows for {summary['n_with_flows']:,}, "
        f"flood trend for {summary['n_with_trend']:,} ({summary['n_rising']:,} rising, "
        f"{summary['n_falling']:,} falling), Q100 for {summary['n_with_q100']:,}"
    )
    if args.publish:
        url = publish_folder(args.out, args.publish, commit_message="harvest signatures")
        print(f"  published: {url}")


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a water question; the analyst calls aquascope tools and writes a cited answer."""
    from aquascope.ai_engine.analyst import ask

    def on_event(msg: str) -> None:
        if not args.quiet:
            print(f"  · {msg}", file=sys.stderr)

    try:
        result = ask(
            args.question,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            max_steps=args.max_steps,
            on_event=on_event,
        )
    except (RuntimeError, ValueError, ImportError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    md = result.to_markdown()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"\n  Report saved to {args.out}")
        print(result.answer)
    else:
        print(md)
    if args.study and result.study:
        Path(args.study).parent.mkdir(parents=True, exist_ok=True)
        Path(args.study).write_text(result.study, encoding="utf-8")
        print(f"  Study saved to {args.study}; re-run it with `aquascope run {args.study}`")
    unmet = [c for c in result.checks if not c.get("passed")]
    if unmet and not args.quiet:
        print("\n  Checks this answer did not meet:", file=sys.stderr)
        for c in unmet:
            print(f"   · {c.get('detail') or c.get('name')}", file=sys.stderr)


def cmd_run(args: argparse.Namespace) -> None:
    """`aquascope run`: a study file if one is named, otherwise a methodology pipeline."""
    if args.study:
        cmd_run_study(args)
        return
    if not (args.method and args.file):
        logger.error("Give a study file (aquascope run study.yaml) or --method and --file for a pipeline.")
        sys.exit(1)
    cmd_run_pipeline(args)


def cmd_run_study(args: argparse.Namespace) -> None:
    """Run a study file: the steps behind an answer, again, with no model in the loop."""
    from aquascope.study import load, run_study, write_outputs

    try:
        study = load(args.study)
    except (OSError, ValueError) as exc:
        logger.error("Could not read %s: %s", args.study, exc)
        sys.exit(1)

    def on_event(event: dict) -> None:
        if not args.quiet:
            print(f"  · {_format_event(event)}", file=sys.stderr)

    if args.dry_run:
        print(f"{len(study.steps)} step(s) in {args.study}:")
        for i, step in enumerate(study.steps, 1):
            label = f"{step.id}: " if step.id else ""
            print(f"  {i}. {label}{step.tool}({', '.join(f'{k}={v!r}' for k, v in step.arguments.items())})")
            for g in step.expects:
                print(f"       gate {g.get('check')} {g.get('path') or g.get('paths') or ''} {g.get('value', '')}")
        return
    run = run_study(study, on_event=on_event)
    if args.out:
        paths = write_outputs(run, args.out)
        print(f"\n  Report saved to {paths['report.md']}")
        for g in run.gates:
            print(f"  gate {g['step']} {g['check']}: {'passed' if g['passed'] else 'FAILED'}, {g.get('detail', '')}")
        if run.stop_reason:
            print(f"  stopped at {run.stopped_at}: {run.stop_reason}")
    else:
        print(run.to_markdown())
    if not run.ok:
        sys.exit(1)


def _format_event(event: dict) -> str:
    """One line for a runner or team event ({role, step, event, detail})."""
    if not isinstance(event, dict):
        return str(event)
    step = f" {event['step']}" if event.get("step") else ""
    return f"{event.get('role', '')}{step}: {event.get('event', '')} {event.get('detail', '')}".strip()


def cmd_playbooks(args: argparse.Namespace) -> None:
    """`aquascope playbooks [list | show ID]`: the method chains `aquascope solve` follows (#307)."""
    from aquascope import playbooks as pbk

    if getattr(args, "playbooks_cmd", None) == "show":
        try:
            pb = pbk.load(args.id)
        except pbk.PlaybookError as exc:
            logger.error("%s", exc)
            sys.exit(1)
        print(f"{pb.id}: {pb.title}")
        print(f"  problem: {pb.problem}")
        if pb.description:
            print(f"  {pb.description}")
        if pb.intake:
            print("  intake:")
            for f in pb.intake:
                kind = f.type + (" " + " | ".join(str(o) for o in f.options) if f.options else "")
                print(f"    {f.name} ({kind}; default {f.default!r}): {f.label or ''}")
        print("  branches (first match wins):")
        for br in pb.branches:
            cond = " and ".join(f"{c.path} {c.op} {c.value!r}" for c in br.when) or "otherwise"
            print(f"    {br.id}: when {cond}")
            for s in br.steps:
                gates = ", ".join(str(g.get("check")) for g in s.expects)
                extra = f"  [gates: {gates}]" if gates else ""
                opt = " (optional)" if s.optional else ""
                print(f"      {s.id} {s.tool}{opt}{extra}")
        if pb.declines:
            print("  declines:")
            for d in pb.declines:
                print(f"    - {d.say}")
        if pb.caveats:
            print("  caveats:")
            for c in pb.caveats:
                print(f"    - {c if isinstance(c, str) else c.say}")
        if pb.citations:
            print("  citations:")
            for c in pb.citations:
                print(f"    - {c}")
        return
    rows = pbk.list_playbooks()
    for r in rows:
        if r.get("error"):
            print(f"  {r['id']:<22} (broken: {r['error']})")
            continue
        print(f"  {r['id']:<22} {r['title']}  (branches: {', '.join(r['branches'])})")
    print(
        f"\n  {len(rows)} playbook(s); `aquascope playbooks show ID` prints one, "
        '`aquascope solve "PROBLEM" --lat LAT --lon LON` runs one.'
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    """Map + QA an arbitrary CSV/Excel export into a clean series with a report."""
    from aquascope.ingest import ingest, write_outputs

    client = model = None
    if args.llm:
        try:
            from aquascope.ai_engine.analyst import resolve_llm
            from aquascope.ai_engine.llm_transport import make_client

            cfg = resolve_llm(args.provider, args.model, args.api_key)
            client, model = make_client(cfg["api_key"], cfg["base_url"], provider=cfg["provider"]), cfg["model"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM mapping unavailable (%s); using heuristics", exc)
    try:
        result = ingest(
            args.file,
            variable=args.variable,
            date_column=args.date_column,
            value_column=args.value_column,
            unit=args.unit,
            station=args.station,
            sheet=args.sheet,
            llm_client=client,
            llm_model=model,
            description=args.describe or "",
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    m, q = result["mapping"], result["qa"]
    print(
        f"  mapping  : {m['datetime_column']} + {m['value_column']} -> {m['variable']} [{m['unit']}] "
        f"(x{m['to_si_factor']}, {m['method']}, confidence {m['confidence']:.0%})"
    )
    print(
        f"  values   : {q['n_values']:,} kept of {q['n_rows_in']:,} rows; {q['start']} -> {q['end']}; "
        f"coverage {q['coverage_pct']}%"
    )
    print(
        f"  dropped  : {q['n_duplicates_dropped']} duplicates, {q['n_sentinels_dropped']} sentinels; "
        f"flagged {q['n_negative']} negative, {q['n_spikes_flagged']} spikes"
    )
    for w in q["warnings"]:
        print(f"  warning  : {w}")
    stem = args.out or str(Path(args.file).with_suffix("")) + "_clean"
    paths = write_outputs(result, stem)
    print(f"  written  : {paths['csv']}, {paths['qa_md']}")


def cmd_mcp(args: argparse.Namespace) -> None:
    """Serve aquascope's tools over the Model Context Protocol (stdio by default)."""
    try:
        from aquascope.mcp_server import main as mcp_main
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    mcp_main(transport=args.transport)


def _print_river_status(res: dict) -> None:
    """`aquascope layers status`: one month of the world river status map (#544)."""
    rng = res.get("valid_range") or {}
    if res.get("live_error"):
        print(f"  Could not list the bucket ({res['live_error']}); showing what it held on {res.get('checked')}.")
    if res.get("available"):
        print(f"  World river status, {res['month']}: {res['url']}")
    else:
        print(f"  {res.get('error', 'no map')}")
    if rng:
        gaps = f"; no map for {', '.join(res['missing'])}" if res.get("missing") else ""
        print(f"  {rng['months']} months, {rng['first']} to {rng['latest']}{gaps}.")
    for c in res.get("legend") or []:
        print(f"  {c['hex']}  {c['label']:<18} {c['range']}")
    if res.get("method"):
        print(f"\n  {res['method']}")
        print(f"  {res['attribution']} ({res['licence']})")


def cmd_layers(args: argparse.Namespace) -> None:
    """`aquascope layers`: the dated map layers and their valid dates, or the frames of a time-lapse (#522)."""
    from aquascope.map_time import dated_layers, layer_frames

    if args.layers_cmd == "status":
        from aquascope.map_layers import river_status_month

        res = river_status_month(args.month, live=not args.offline)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        _print_river_status(res)
        if not res.get("available"):
            sys.exit(1)
        return
    if args.layers_cmd == "frames":
        res = layer_frames(args.layer, args.start, args.end, step=args.step, max_frames=args.max_frames)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        if res.get("error"):
            print(f"  {res['error']}")
            sys.exit(1)
        print(f"  {res['label']}, every {res['step']}, {len(res['frames'])} frames"
              + (" (capped)" if res.get("truncated") else "")
              + (f", {res['skipped']} dates it cannot show" if res.get("skipped") else ""))
        if res.get("note"):
            print(f"  {res['note']}")
        for f in res["frames"]:
            print(f"  {f['date']}  {f['tiles']}")
        print(f"\n  {res['attribution']} ({res['licence']})")
        return
    res = dated_layers(live=args.live)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if res.get("live_error"):
        print(f"  Could not read GIBS ({res['live_error']}); showing the ranges recorded on {res['checked']}.")
    for lay in res["layers"]:
        until = lay.get("latest") or lay.get("until") or "now"
        gaps = f", {len(lay['gaps'])} gaps" if lay.get("gaps") else ""
        print(f"  {lay['id']:<8} {lay['label']:<28} {lay['cadence']:<6} {lay['since']} to {until}{gaps}")
    print(f"\n  Steps: {', '.join(res['steps'])}. {res['note']}")


def cmd_map(args: argparse.Namespace) -> None:
    """`aquascope map "<request>"`: a plain-English request about the Explorer's map, read into map actions (#561)."""
    from aquascope import map_commands as mc

    text = " ".join(args.request)
    if args.llm:
        try:
            res = mc.model_command(text, provider=args.provider, model=args.model, api_key=args.api_key, today=args.today)
        except (RuntimeError, ValueError) as exc:
            print(f"  {exc}")
            sys.exit(1)
        res["by"] = res.pop("model")
    else:
        res = mc.parse_command(text, today=args.today)
        res["by"] = "rules"
    if args.resolve and res.get("actions"):
        resolved = mc.resolve_actions(res["actions"])
        res.update(actions=resolved["actions"], said=resolved["said"], notes=resolved["notes"],
                   credit=resolved["credit"])
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if res.get("control"):
        print(f"  {res['control'].replace('_', ' ')}: the Explorer's action log does this")
        return
    if not res.get("actions"):
        unknown = ", ".join(res.get("unknown") or [])
        print("  Not understood by the rules" + (f" (unknown words: {unknown})" if unknown else "")
              + ". Try --llm with your own key, or rephrase.")
        for e in res.get("errors") or []:
            print(f"  {e}")
        sys.exit(1)
    print(f"  Understood by {res['by']}:")
    for said, action in zip(res["said"], res["actions"]):
        print(f"  - {said:<44} {json.dumps(action, ensure_ascii=False)}")
    for note in res.get("notes") or []:
        print(f"  note: {note}")
    if res.get("credit"):
        print(f"\n  {res['credit']}")


def cmd_basins(args: argparse.Namespace) -> None:
    """`aquascope basins`: catchments from BasinATLAS in the Archive (at LAT LON | upstream HYBAS_ID | build GDB)."""
    from aquascope.archive import basins

    if args.basins_cmd == "build":
        report = basins.build_basins(args.gdb, args.out, max_features=args.max_features, write_fgb=args.fgb)
        for name, size in report.files.items():
            print(f"  {name:<32} {size / 1e6:8.1f} MB")
        print(f"\n  {report.n_basins:,} sub-basins in {report.seconds:.0f}s -> {args.out}/basins")
        return
    if args.basins_cmd == "assign":
        from aquascope.archive.catalog import load_stations
        from aquascope.archive.similar import assign_station_catchments

        catalog = load_stations()
        table = assign_station_catchments(catalog, args.fgb, args.attributes, args.out)
        print(f"  {len(table):,} of {len(catalog):,} stations assigned to a sub-basin -> {args.out}")
        return
    if args.basins_cmd == "similar":
        from aquascope.archive.similar import similar_for_point, similar_for_station

        if args.station:
            src, _, sid = args.station.partition("/")
            res = similar_for_station(src, sid, k=args.k, method=args.method, sources=args.source or None)
        else:
            res = similar_for_point(args.lat, args.lon, k=args.k, method=args.method, sources=args.source or None)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
            return
        if res.get("error"):
            print(f"  {res['error']}")
            sys.exit(1)
        print(
            f"  {res['k']} of {res['n_candidates']} gauged basins, method {res['method']}, "
            f"features {', '.join(res['features_used'])}"
        )
        for i, st in enumerate(res["stations"], 1):
            dist = f"{st['distance_km']:,.0f} km" if st.get("distance_km") is not None else ""
            print(
                f"  {i:>2}. {st['source']:<20} {st['station_id']:<40} {(st.get('name') or '')[:38]:<38} "
                f"area {st['up_area_km2']:>9,.0f} km2  score {st['score']:.3f} {dist}"
            )
        return
    if args.basins_cmd == "regionalize":
        from aquascope.archive.regionalize import regionalize_point

        res = regionalize_point(args.lat, args.lon, k=args.k, method=args.method)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
            return
        if res.get("error"):
            print(f"  {res['error']}")
            sys.exit(1)
        est = res.get("estimates", {})
        print(
            f"  {len(est)} signatures from {res.get('n_donors_available', 0):,} donors, method {res['method']}"
            + (f", k={res['similarity']['k']}" if "similarity" in res else "")
        )
        skill = (res.get("skill") or {}).get("by_signature", {})
        for name, e in est.items():
            sk = skill.get(name) or {}
            tail = f"  LOO NSE {sk['nse']:.2f}, median error {sk['median_ape'] * 100:.0f} %" if sk else ""
            print(f"  {e['label']:<48} {e['value']:>10.3f} {e['unit']:<7} [{e['low']:.3f}, {e['high']:.3f}]{tail}")
        return
    if args.basins_cmd == "signatures":
        from aquascope.archive.bundles import read_bundle
        from aquascope.archive.regionalize import compute_station_signatures
        from aquascope.archive.similar import load_station_catchments

        root = Path(args.archive) / "obs" / "discharge"
        bundles = {p.stem: read_bundle(p) for p in sorted(root.glob("*.parquet"))} if root.exists() else {}
        cat = load_station_catchments(path=args.catchments) if args.catchments else load_station_catchments()
        table = compute_station_signatures(bundles, cat, min_years=args.min_years)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(out, index=False)
        print(f"  {len(table):,} stations with signatures ({len(bundles)} discharge bundles) -> {out}")
        return
    if args.basins_cmd == "loo":
        from aquascope.archive.catalog import load_stations
        from aquascope.archive.regionalize import load_station_signatures, loo_skill
        from aquascope.archive.similar import load_station_catchments

        sig = load_station_signatures(path=args.signatures) if args.signatures else load_station_signatures()
        cat = load_station_catchments(path=args.catchments) if args.catchments else load_station_catchments()
        skill = loo_skill(sig, cat, load_stations(), k=args.k, max_stations=args.max_stations or None)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(skill, indent=1), encoding="utf-8")
        print(f"  leave-one-out over {skill['n_stations']:,} stations -> {out}")
        for m, per in skill["methods"].items():
            for name, sk in per.items():
                print(f"  {m:<11} {name:<22} n={sk['n']:>6}  NSE {sk['nse']:>6.2f}  median APE {sk['median_ape']:.2f}")
        return
    if args.basins_cmd == "upstream":
        topo = basins.Topology(basins.load_topology())
        ids = topo.upstream_ids(int(args.hybas_id), limit=args.limit)
        print("\n".join(str(i) for i in ids))
        print(f"\n  {len(ids)} sub-basins upstream of (and including) {args.hybas_id}", file=sys.stderr)
        return
    try:
        res = basins.describe_catchment(args.lat, args.lon, upstream=not args.local)
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if res.get("error"):
        print(f"  {res['error']}")
        sys.exit(1)
    sb = res["sub_basin"]
    print(
        f"  Sub-basin {sb['hybas_id']} (Pfafstetter {sb.get('pfaf_id')}), {sb.get('sub_area', 0):,.1f} km², "
        f"upstream area {sb.get('up_area', 0):,.1f} km²"
    )
    print(f"  {res['upstream']['note']}")
    attrs = res.get("attributes", {})
    for key, v in attrs.items():
        if isinstance(v, dict):
            print(f"  {v['label']:<48} {v['value']:>12,.2f} {v['unit']}")
    print(f"\n  {res['attribution']}")


def _format_assessment(res: dict, *, radius_km: float) -> str:
    """The assess_site result as a short table: defensible first, one line per method with its reason."""
    from aquascope.methods import DEFENSIBLE, MARGINAL, NOT_DEFENSIBLE

    ctx = res.get("context") or {}
    catch = res.get("catchment") or {}
    stations = res.get("stations") or []
    rows = res.get("sufficiency") or []
    head = [
        f"{res['point']['lat']:.4f}, {res['point']['lon']:.4f}",
        f"{len(stations)} gauge{'' if len(stations) == 1 else 's'} within {radius_km:g} km",
    ]
    area = ctx.get("area_km2")
    if area:
        head.append(f"catchment {area:,.0f} km²" + (" (caller)" if catch.get("source") == "caller" else ""))
    elif catch.get("error"):
        head.append("catchment unknown")
    if ctx.get("donors") is not None:
        head.append(f"{ctx['donors']} donors")
    lines = ["  " + "  ·  ".join(head)]
    years = ctx.get("years_by_variable") or {}
    if years:
        for var, yr in years.items():
            st = next((s for s in stations if var in (s.get("variables") or []) and s.get("years") == yr), None)
            tail = f", {st.get('name') or st['station_id']} ({st['source']}/{st['station_id']})" if st else ""
            lines.append(f"  {var.replace('_', ' ')}: {yr:g} yr{tail}")
    else:
        lines.append("  no gauge record within reach: ungauged")
    width = max((len(r["label"]) for r in rows), default=20)
    for status, title in ((DEFENSIBLE, "defensible"), (MARGINAL, "marginal"), (NOT_DEFENSIBLE, "not defensible")):
        block = [r for r in rows if r["status"] == status]
        if not block:
            continue
        lines.append(f"\n  {title}")
        for r in block:
            lines.append(f"    {r['label']:<{width}}  {r['reason']}")
    if res.get("notes"):
        lines.append("\n  notes")
        lines.extend(f"    - {n}" for n in res["notes"])
    return "\n".join(lines)


def cmd_assess(args: argparse.Namespace) -> None:
    """`aquascope assess LAT LON`: what can be answered at a place, from the catalog and BasinATLAS, no agency call."""
    from aquascope.explore import assess_site

    res = assess_site(
        args.lat, args.lon, radius_km=args.radius_km, problem=args.problem, return_period=args.return_period
    )
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    print(_format_assessment(res, radius_km=args.radius_km))


# ── river (GEOGLOWS v2 reaches) ──────────────────────────────────────────────


def _river_target(args: argparse.Namespace) -> tuple[int | None, dict | None]:
    """The river_id the user gave, or the one ``--at LAT LON`` snaps to (with the snap, to report it)."""
    from aquascope import rivers

    if getattr(args, "at", None):
        snap = rivers.snap_to_river(args.at[0], args.at[1], max_distance_m=args.max_distance)
        print(f"  {snap['message']}")
        if not snap["snapped"]:
            sys.exit(1)
        return int(snap["river_id"]), snap
    if args.river_id is None:
        print("  Give a RIVER_ID, or --at LAT LON to snap to the nearest reach.")
        sys.exit(2)
    return int(args.river_id), None


# ── evidence (the model-skill ladder, #518) ────────────────────────────────


def _print_skill(res: dict) -> None:
    where = f"{res.get('source')}/{res.get('station_id')}" if res.get("source") else f"{res.get('lat')}, {res.get('lon')}"
    print(f"  Model skill at {where}, gauge record {res.get('obs_start')} to {res.get('obs_end')}")
    if res.get("error"):
        print(f"  {res['error']}")
        return
    print(f"  {'model':<12} {'grade':>5} {'KGE':>6} {'r':>6} {'alpha':>6} {'beta':>6} {'NSE':>6} {'PBIAS':>7} "
          f"{'Q2 err':>7} {'Q10 err':>8} {'Q100 err':>9}")

    def f(x, d=2, pct=False):
        if x is None:
            return "-"
        return f"{x:+.0f} %" if pct else f"{x:.{d}f}"

    for r in res.get("models") or []:
        if r.get("kge") is None:
            print(f"  {r.get('label', r.get('model')):<12} {'-':>5}  {r.get('why') or ''}")
            continue
        print(f"  {r['label']:<12} {r.get('grade') or '-':>5} {f(r.get('kge')):>6} {f(r.get('r')):>6} "
              f"{f(r.get('alpha')):>6} {f(r.get('beta')):>6} {f(r.get('nse')):>6} {f(r.get('pbias'), 1):>7} "
              f"{f(r.get('q2_error_pct'), pct=True):>7} {f(r.get('q10_error_pct'), pct=True):>8} "
              f"{f(r.get('q100_error_pct'), pct=True):>9}")
    print(f"  {res.get('sentence')}")
    for n in res.get("notes") or []:
        print(f"  {n}")
    print("  Grades: A KGE >= 0.75, B >= 0.5, C > -0.41 (the mean-flow benchmark), else D; one letter lower when "
          "the 100-year flow is off by more than 50 %.")


def cmd_evidence(args: argparse.Namespace) -> None:
    """`aquascope evidence skill|near|build|publish`: every global model scored at a gauge (#518)."""
    if args.evidence_cmd == "skill":
        from aquascope import evidence

        series = None
        if args.csv:
            import pandas as pd

            df = pd.read_csv(args.csv)
            series = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(float),
                               index=pd.to_datetime(df.iloc[:, 0]))
        at = args.at or (None, None)
        res = evidence.model_skill(args.source, args.station_id, series=series, lat=at[0], lon=at[1],
                                   area_km2=args.area, models=args.models, years=args.years or None)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
            return
        _print_skill(res)
        return
    if args.evidence_cmd == "near":
        from aquascope import evidence

        res = evidence.lean_on(args.lat, args.lon, radius_km=args.radius_km)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
            return
        print(f"  {res['sentence']}")
        return
    from aquascope.archive import skill

    if args.evidence_cmd == "build":
        argv = ["build", "--archive", args.archive, "--out", args.out, "--max-gauges", str(args.max_gauges),
                "--nwm-years", str(args.nwm_years), "--nwm-max-columns", str(args.nwm_max_columns),
                "--grrr-max-chunks", str(args.grrr_max_chunks), "--workers", str(args.workers)]
        if args.smoke:
            argv.append("--smoke")
        sys.exit(skill.main(argv))
    sys.exit(skill.main(["publish", "--out", args.out, "--repo", args.repo]))


def cmd_bulletin(args: argparse.Namespace) -> None:
    """`aquascope bulletin [YYYY-MM]`: the month's state of the rivers, HydroSOS style (#523)."""
    from aquascope import bulletin

    try:
        res = bulletin.status_bulletin(args.month, args.sources, archive=args.archive, rebuild=args.rebuild,
                                       top_up=args.top_up, workers=args.workers)
    except ValueError as exc:
        print(f"  {exc}")
        sys.exit(2)
    if args.out:
        written = bulletin.write_bulletin(res, args.out, figure=not args.no_map)
        if not args.json:
            for what, path in written.items():
                print(f"  {what:<8} -> {path}")
    if args.json:
        print(json.dumps(res if args.gauges else {k: v for k, v in res.items() if k != "gauges"}, indent=2,
                         ensure_ascii=False, default=str))
        return
    where = "published" if res.get("origin") == "published" else "built from the Archive's discharge records"
    print(f"  {res['title']}, {res['label']} ({where})")
    print(f"  {res['summary']}")
    cov = res["coverage"]
    if cov["classed"]:
        print()
        print(f"  {'Country':<16} {'Gauges':>6}  {'Much below':>10} {'Below':>6} {'Normal':>6} {'Above':>6} "
              f"{'Much above':>10}  Median pct")
        order = ("much_below", "below", "normal", "above", "much_above")
        for c in res["countries"]:
            k = [c["counts"][x] for x in order]
            print(f"  {c['name'][:16]:<16} {c['n']:>6}  {k[0]:>10} {k[1]:>6} {k[2]:>6} {k[3]:>6} {k[4]:>10}  "
                  f"{c['median_percentile']:.0f} ({c['median_label']})")
    left = {k: v for k, v in cov["excluded"].items() if v}
    if left:
        print()
        print("  Left out: " + "; ".join(f"{v:,} with {cov['excluded_text'][k]}" for k, v in left.items()) + ".")
    if not args.out:
        print("  Write the HTML, Markdown, map and status table with --out DIR.")


def cmd_warnings(args: argparse.Namespace) -> None:
    """`aquascope warnings [--bbox W S E N]`: Floods ahead, the reaches expected to reach their 2-year flow (#546)."""
    from aquascope.archive import warnings

    try:
        res = warnings.flood_warnings(args.bbox, min_rp=args.min_rp, limit=args.limit, local=args.local)
    except ValueError as exc:
        print(f"  {exc}")
        sys.exit(2)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return
    print(f"  {res['sentence']}")
    if not res.get("available"):
        return
    counts = ", ".join(f"{v:,} at {k}-year" for k, v in res["counts"].items() if v)
    if counts:
        print(f"  {counts}")
    if res["reaches"]:
        print()
        print(f"  {'River id':>10}  {'Lat':>7} {'Lon':>8}  {'Class':>6}  {'Peak m3/s':>10}  {'2-yr m3/s':>10}  "
              f"{'Peak day':<10}  Members")
        for r in res["reaches"]:
            print(f"  {r['river_id']:>10}  {r['lat']:>7.2f} {r['lon']:>8.2f}  {str(r['rp']) + '-yr':>6}  "
                  f"{r['peak_cms'] or 0:>10,.0f}  {r['q2'] or 0:>10,.0f}  {r['peak_date']:<10}  {r['share']:.0%}")
        if res.get("truncated"):
            print(f"  ... the first {len(res['reaches'])} shown; --limit N for more")
    print()
    print(f"  {res.get('not') or ''}")


def cmd_now(args: argparse.Namespace) -> None:
    """`aquascope now LAT LON | --station SOURCE/ID | --river-id ID`: today against normal and the next 15 days."""
    from aquascope import nownext

    coords = list(args.coords or [])
    if coords and len(coords) != 2:
        print("  Give LAT LON, --station SOURCE/ID or --river-id ID.")
        sys.exit(2)
    lat, lon = (coords[0], coords[1]) if coords else (None, None)
    if lat is None and not args.station and args.river_id is None:
        print("  Give LAT LON, --station SOURCE/ID or --river-id ID.")
        sys.exit(2)
    try:
        res = nownext.now(lat, lon, station=args.station, river_id=args.river_id, days=args.days, date=args.date,
                          with_forecast=not args.status_only, correct=not args.raw, history=not args.quick)
    except ValueError as exc:
        print(f"  {exc}")
        sys.exit(2)
    fc = res.get("forecast") or {}
    if args.csv and fc:
        _now_csv(fc, args.csv)
        print(f"  forecast -> {args.csv}")
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return
    st = res.get("status") or {}
    if res.get("station"):
        s = res["station"]
        print(f"  {s.get('name') or s['station_id']} ({s['source']}/{s['station_id']}), {s.get('variable')}")
    if st:
        print(f"  {st.get('sentence') or st.get('error')}")
        if st.get("top_up"):
            print(f"    {st['top_up']}")
    if not fc:
        return
    if fc.get("snap"):
        print(f"  {fc['snap'].get('message')}")
    if not st and (fc.get("status") or {}).get("sentence"):
        print(f"  {fc['status']['sentence']}")
    rows = _now_table(fc)
    if rows:
        corrected = any(r[2] is not None for r in rows)
        head = f"  {'day':<10}  {'GEOGLOWS mean [25-75 %]':>28}" + (f"  {'corrected':>10}" if corrected else "")
        print(head + f"  {'GloFAS mean':>11}   (m3/s, modelled)")
        for day, g, c, gl in rows:
            band = f"{g[0]:,.4g} [{g[1]:,.4g}-{g[2]:,.4g}]" if g and g[0] is not None else "-"
            line = f"  {day:<10}  {band:>28}"
            if corrected:
                line += f"  {c:>10,.4g}" if c is not None else f"  {'-':>10}"
            line += f"  {gl:>11,.4g}" if gl is not None else f"  {'-':>11}"
            print(line)
    thr = fc.get("gauge_thresholds") if (fc.get("correction") or {}).get("forecast") else fc.get("thresholds")
    if thr and thr.get("q"):
        pairs = ", ".join(f"{t:g}-yr {q:,.4g}" for t, q in zip(thr["return_periods"], thr["q"]) if q is not None)
        print(f"  Thresholds from {thr.get('source')}, {thr.get('method')}: {pairs}")
    print(f"  {fc.get('sentence')}")
    corr = fc.get("correction") or {}
    if corr.get("skill_line"):
        print(f"  {corr['skill_line']}")
        if corr.get("skill_detail"):
            print(f"    {corr['skill_detail']}")
    elif corr.get("error"):
        print(f"  No correction: {corr['error']}")
    if (fc.get("reach_check") or {}).get("note"):
        print(f"  {fc['reach_check']['note']}")
    for g in ("geoglows", "glofas"):
        if (fc.get(g) or {}).get("error"):
            print(f"  {g}: {fc[g]['error']}")
    print("  Data: GEOGLOWS v2 (CC BY 4.0); GloFAS v4 via Open-Meteo (CC BY 4.0).")


def cmd_watch(args: argparse.Namespace) -> None:
    """`aquascope watch ID... --since DATE`: what changed at watched gauges, reaches and areas."""
    from aquascope import watch

    thresholds: dict[str, str] = {}
    for spec in args.threshold or []:
        key, sep, value = spec.rpartition("=")
        if not sep or not key:
            print(f"  --threshold takes ID=VALUE or ID=10y, not {spec!r}")
            sys.exit(2)
        thresholds[key] = value
    items = []
    for ident in args.ids:
        try:
            item = watch.parse_item(ident)
        except ValueError as exc:
            print(f"  {exc}")
            sys.exit(2)
        if item["id"] in thresholds or ident in thresholds:
            item["threshold"] = thresholds.get(item["id"], thresholds.get(ident))
        items.append(item)
    try:
        res = watch.watch_digest(items, args.since, forecast="off" if args.no_forecast else args.forecast,
                                 floods=not args.no_floods, refresh=not args.no_refresh)
    except ValueError as exc:
        print(f"  {exc}")
        sys.exit(2)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return
    print(f"  {res['summary']}")
    for item in res["items"]:
        mark = "*" if item.get("alerts") else ("+" if item.get("changed") else " ")
        print(f"  {mark} {item.get('name')} ({item['id']})")
        print(f"      {item.get('line')}")
        for note in item.get("notes") or []:
            print(f"      ({note})")
    for note in res.get("notes") or []:
        print(f"  {note}")
    for err in res.get("errors") or []:
        print(f"  Skipped {err['item']}: {err['error']}")
    print("  Forecasts are model output (GEOGLOWS v2, CC BY 4.0). Flood events: Groundsource (CC BY 4.0).")


def _now_table(fc: dict) -> list[tuple]:
    g, gl = fc.get("geoglows") or {}, fc.get("glofas") or {}
    c = (fc.get("correction") or {}).get("forecast") or {}
    days = sorted(set(g.get("date") or []) | set(gl.get("date") or []))

    def at(part: dict, key: str, day: str):
        dates = part.get("date") or []
        vals = part.get(key) or []
        return vals[dates.index(day)] if day in dates and dates.index(day) < len(vals) else None

    return [(d, (at(g, "mean", d), at(g, "p25", d), at(g, "p75", d)), at(c, "mean", d), at(gl, "mean", d))
            for d in days]


def _now_csv(fc: dict, path: str) -> None:
    from aquascope.nownext import STAT_KEYS

    c = (fc.get("correction") or {}).get("forecast") or {}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("model,date," + ",".join(STAT_KEYS) + ",corrected\n")
        for model in ("geoglows", "glofas"):
            part = fc.get(model) or {}
            for i, day in enumerate(part.get("date") or []):
                vals = [part.get(k)[i] if isinstance(part.get(k), list) else None for k in STAT_KEYS]
                corrected = (c.get("mean") or [None] * (i + 1))[i] if model == "geoglows" and c else None
                fh.write(",".join([model, day, *("" if v is None else f"{v:g}" for v in [*vals, corrected])]) + "\n")


def cmd_river(args: argparse.Namespace) -> None:
    """`aquascope river snap|record|area|trace|dams|upstream|downstream`: GEOGLOWS v2 river reaches, keyless
    (the modelled record is CC BY); the dams come from the Archive's Global Dam Watch mirror (CC BY 4.0)."""
    from aquascope import rivers

    if args.river_cmd == "snap":
        res = rivers.snap_to_river(args.lat, args.lon, max_distance_m=args.max_distance,
                                   prefer="nearest" if args.nearest else "main", area_km2=args.area)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        print(f"  {res['message']}")
        if res["snapped"]:
            print(f"  river_id {res['river_id']} at {res['snap_lat']:.5f}, {res['snap_lon']:.5f}")
        return
    rid, snap = _river_target(args)
    # Where the snapped reach is: it lets the area and the trace read the right processing unit first.
    near = {"lat": snap["snap_lat"], "lon": snap["snap_lon"]} if snap else {}
    if args.river_cmd == "record":
        store: dict = {}
        res = rivers.reach_record(rid, years=args.years, store=store)
        if args.csv and store.get("series") is not None:
            series = store["series"]
            with open(args.csv, "w", encoding="utf-8") as fh:
                fh.write("date,discharge_m3_per_s_modelled\n")
                fh.writelines(f"{t.date().isoformat()},{v:g}\n" for t, v in series.items())
            print(f"  {len(series):,} simulated days -> {args.csv}")
        if args.json:
            res.pop("series", None)
            print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
            return
        if res.get("error"):
            print(f"  {res['error']}")
            sys.exit(1)
        st = res["stats"]
        print(f"  River reach {rid}: MODELLED daily discharge, {res['start']} to {res['end']} ({res['years']:g} years)")
        print(f"  mean {st['mean']:,.3g} m3/s, max {st['max']:,.4g} m3/s; Q95 {res['fdc']['q95']:,.3g}, "
              f"Q50 {res['fdc']['q50']:,.3g}, Q10 {res['fdc']['q10']:,.3g}")
        ffa = res.get("ffa") or {}
        gev, lp3 = (ffa.get("fits") or {}).get("gev_lmoments") or {}, (ffa.get("fits") or {}).get("lp3") or {}
        if gev.get("q"):
            print(f"  Return periods from {ffa['n_years']} annual maxima (m3/s):")
            for k, t in enumerate(ffa["return_periods"]):
                ci = (lp3.get("ci") or [[None, None]] * len(ffa["return_periods"]))[k]
                band = f" (LP3 {lp3['q'][k]:,.4g}, 90 % CI {ci[0]:,.4g} to {ci[1]:,.4g})" if lp3.get("q") else ""
                print(f"    {t:>5g}-yr  GEV {gev['q'][k]:,.4g}{band}")
        print(f"  {res['notes'][0]}")
        print(f"  {res['attribution']}")
        return
    if args.river_cmd == "area":
        res = rivers.upstream_area(rid, **near)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        print(f"  River reach {rid}: {res['upstream_area_km2']:,.1f} km2 drain to it, "
              f"{res['n_reaches_upstream']:,} reaches upstream (GEOGLOWS unit {res['vpu']}).")
        print(f"  {res['note']}")
        return
    if args.river_cmd == "dams":
        res = rivers.upstream_dams(rid, with_flow=not args.no_flow, **near)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
            return
        print(f"  River reach {rid}: {res['summary']}")
        for d in res.get("dams") or []:
            cap = f"{d['capacity_mcm']:,.1f} million m3" if d.get("capacity_mcm") else "storage unknown"
            print(f"    {d.get('name') or 'unnamed':<28} {cap:<24} {d.get('purpose') or ''}")
        if res.get("note"):
            print(f"  {res['note']}")
        print(f"  {res['source']['attribution']}")
        return
    if args.river_cmd == "trace":
        res = rivers.trace_downstream(rid, gauge_km=args.gauge_km, dam_km=args.dam_km, **near)
        if args.geojson:
            feature = {"type": "Feature", "geometry": res.get("geometry"),
                       "properties": {"river_id": rid, "length_km": res.get("length_km"),
                                      "licence": res.get("geometry_licence")}}
            with open(args.geojson, "w", encoding="utf-8") as fh:
                json.dump({"type": "FeatureCollection", "features": [feature]}, fh)
            print(f"  path -> {args.geojson} (TDX-Hydro geometry, CC BY-SA 4.0)")
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
            return
        print(f"  {res['message']}")
        if res.get("upstream_area_km2") is not None:
            print(f"  {res['upstream_area_km2']:,.0f} km2 drain to the first reach.")
        for g in res.get("gauges") or []:
            print(f"    km {g['along_km']:>7,.1f}  {g['source']}/{g['station_id']}  {g.get('name') or ''}")
        countries = res.get("countries_info") or {}
        if countries.get("summary"):
            print(f"  {countries['summary']}")
        dams_info = res.get("dams_info") or {}
        if dams_info.get("summary"):
            print(f"  {dams_info['summary']}")
        for d in res.get("dams") or []:
            cap = f"  {d['capacity_mcm']:,.1f} million m3" if d.get("capacity_mcm") else ""
            use = f"  {d['purpose']}" if d.get("purpose") else ""
            print(f"    km {d['along_km']:>7,.1f}  {d.get('name') or 'unnamed'}{cap}{use}")
        up = res.get("upstream_dams") or {}
        if up.get("summary"):
            print(f"  Upstream: {up['summary']}")
        for note in res.get("notes") or []:
            print(f"  {note}")
        return
    if args.river_cmd in ("upstream", "downstream"):
        fn = rivers.upstream_ids if args.river_cmd == "upstream" else rivers.downstream_ids
        res = fn(rid, max_n=args.max, **near)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        print(f"  {res['message']}")
        ids = res["ids"]
        shown = ids if len(ids) <= 12 else [*ids[:6], "...", *ids[-6:]]
        print("  " + " ".join(str(x) for x in shown))
        return


# ── context (place context layers, #520) ────────────────────────────────────

_CONTEXT_LABELS = {
    "flood_history": "Flood history", "surface_water": "Surface water", "flood_hazard": "Flood hazard",
    "dams": "Dams", "rain_gauge": "Rain gauge", "actual_et": "Actual ET", "soil": "Soil",
}


def _format_context(res: dict[str, Any]) -> str:
    if res.get("bbox"):
        w, s, e, n = res["bbox"]
        lines = [f"Context of the box {w:g}, {s:g} to {e:g}, {n:g} (west, south to east, north)"]
    else:
        lines = [f"Context of {res['lat']:.4f}, {res['lon']:.4f}"]
    for name, layer in res["layers"].items():
        lines.append(f"  {_CONTEXT_LABELS.get(name, name):<14}  {layer.get('summary') or ''}")
    lines.append("")
    lines.append("Data: " + "; ".join(res.get("attribution") or []))
    return "\n".join(lines)


def cmd_context(args: argparse.Namespace) -> None:
    """`aquascope context LAT LON` (or `--bbox`): flood history, surface water, flood hazard, dams, rain gauge,
    actual ET and soil at a place, from open global data, each line with its source (thin face)."""
    from aquascope import context

    if args.floods_past or args.month or args.start or args.end:
        cmd_floods_past(args)
        return
    layers = args.layers or None
    try:
        if args.bbox:
            res = context.area_context(*_parse_bbox(args.bbox), layers=layers)
        elif args.lat is not None and args.lon is not None:
            res = context.place_context(args.lat, args.lon, layers=layers)
        else:
            logger.error("give LAT LON, or --bbox=west,south,east,north")
            sys.exit(2)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return
    print(_format_context(res))


def _format_floods_past(res: dict[str, Any]) -> str:
    lines = [res.get("summary") or ""]
    if res.get("available") is False:
        return "\n".join(lines)
    news, radar = res.get("news") or {}, res.get("radar") or {}
    for label, part, key in (("Most news", news, "news"), ("Most radar", radar, "radar")):
        spots = part.get("hotspots") or []
        if spots and res.get("bbox") is None:
            lines.append(f"{label}:")
            for h in spots:
                lines.append(f"  {h['lat']:7.2f}, {h['lon']:8.2f}   news {h['news']:>7,}   radar {h['radar']:>12,}")
    if news.get("events"):
        lines.append("News events (latest first):")
        for e in news["events"]:
            dates = e["start"] if not e.get("end") or e["end"] == e["start"] else f"{e['start']} to {e['end']}"
            area = f"  {e['area_km2']:,.0f} km2" if e.get("area_km2") else ""
            lines.append(f"  {dates}{area}")
        more = (news.get("events_found") or 0) - len(news["events"])
        if more > 0:
            lines.append(f"  and {more:,} more")
    if res.get("bbox") is not None and radar.get("by_month"):
        lines.append("Radar detections by month: " + ", ".join(f"{m} {n:,}" for m, n in radar["by_month"].items()))
    lines.append("")
    lines.append("Data: " + (res.get("attribution") or ""))
    return "\n".join(lines)


def cmd_floods_past(args: argparse.Namespace) -> None:
    """`aquascope context --floods-past [--month YYYY-MM | --from --to] [--bbox]`: flood events in the news and
    floods seen by radar, per month, worldwide or in a box (#547, thin face over aquascope.context.floods_past)."""
    from aquascope.context.floods_past import flood_events_month

    try:
        res = flood_events_month(args.month, start=args.start, end=args.end,
                                 bbox=_parse_bbox(args.bbox), limit=args.limit)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return
    print(_format_floods_past(res))


# ── export (engineering tools, #519) ─────────────────────────────────────────


#: The ``--to`` choices, kept literal so the parser stays import-light (a test pins it to engineering.TOOLS).
EXPORT_TOOLS = ("hec-hms", "hec-ras", "hec-ssp", "dss", "swmm", "modflow6", "fews", "raven")


def _export_series_from_file(path: str, column: str | None):
    """A CSV's first datetime-like column as the index and ``column`` (or the first numeric one) as values."""
    import pandas as pd

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path}: need a date column and a value column")
    when = pd.to_datetime(df.iloc[:, 0], errors="coerce", utc=False)
    if column:
        if column not in df.columns:
            raise ValueError(f"{path}: no column {column!r}; columns are {list(df.columns)}")
        values = df[column]
    else:
        numeric = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError(f"{path}: no numeric value column")
        values = df[numeric[0]]
    s = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(), index=pd.DatetimeIndex(when))
    return s[s.index.notna()]


def cmd_export(args: argparse.Namespace) -> None:
    """`aquascope export --to TOOL`: a gauge's record (or a CSV) as inputs for an engineering tool (thin face)."""
    from aquascope.io import engineering as eng

    opts = {"regional_skew": args.regional_skew, "regional_skew_mse": args.regional_skew_mse,
            "year_start_month": args.year_start_month, "subbasin_id": args.subbasin_id,
            "cellid": tuple(args.cell) if args.cell else (1, 1, 1), "cond": args.cond, "rbot": args.rbot,
            "period": args.period, "engine": args.engine, "dss_binary": False if args.no_dss else None}
    try:
        if args.station:
            source, _, station_id = args.station.partition("/")
            if not station_id:
                raise ValueError("give --station as source/station_id, for example usgs/01646500")
            res = eng.export_station(source, station_id, args.to, years=args.years, variable=args.variable,
                                     **opts)
        elif args.file:
            s = _export_series_from_file(args.file, args.column)
            res = eng.export_series(s, args.to, variable=args.variable or "discharge", unit=args.unit,
                                    location=args.location or Path(args.file).stem, name=args.location, **opts)
        else:
            raise ValueError("give --station source/station_id or --file a CSV")
    except (ValueError, KeyError) as exc:
        logger.error("%s", exc)
        sys.exit(2)
    if "error" in res:
        logger.error("%s", res["error"])
        sys.exit(1)
    files = eng.files_from(res)
    if args.zip:
        out = Path(args.out_dir if args.out_dir.endswith(".zip") else f"{args.out_dir}.zip")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(eng.zip_bytes(files))
        written = [str(out)]
    else:
        written = eng.write_files(files, args.out_dir)
    if args.json:
        print(json.dumps({"written": written, "notes": res["notes"], "tool": res["tool"]}, indent=2))
        return
    print(f"{res['label']}: {len(files)} file(s)")
    for w in written:
        print(f"  {w}")
    for n in res["notes"]:
        print(f"- {n}")


# ── area-study (Study this area) ─────────────────────────────────────────────


def cmd_area_study(args: argparse.Namespace) -> None:
    """`aquascope area-study`: a multi-gauge flood study over a box or a list of stations (thin face)."""
    from aquascope import area_study

    if not args.bbox and not args.station:
        logger.error("give --bbox=west,south,east,north or one or more --station source/station_id")
        sys.exit(2)
    res = area_study.study_area(
        args.station or None, bbox=_parse_bbox(args.bbox) if not args.station else None, question=args.question,
        max_sites=args.max_sites, max_live=args.max_live,
    )
    if args.output:
        out = Path(args.output)
        if out.suffix == ".xlsx":
            out.write_bytes(area_study.to_xlsx(res))
        elif out.suffix == ".csv":
            out.write_text(area_study.to_csv(res), encoding="utf-8")
        elif out.suffix == ".geojson":
            out.write_text(json.dumps(res["geojson"], ensure_ascii=False), encoding="utf-8")
        else:
            out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    print(res["headline"])
    print()
    for r in res["sites"]:
        if r["status"] != "studied":
            print(f"  {r['source']}/{r['station_id']}: {r['status']} ({r['note']})")
            continue
        q = f"Q100 {r['q100']:g} {r['unit']}" if r.get("q100") is not None else "no Q100"
        p = f", p = {r['trend_p']:.3f}" if r.get("trend_p") is not None else ""
        print(f"  {r['source']}/{r['station_id']}: {r['record_years']} yr, {q}, trend {r['trend']}{p}")
    for note in res["notes"]:
        print(f"- {note}")


def cmd_gym(args: argparse.Namespace) -> None:
    """`aquascope gym basins|run|leaderboard`: HydroGym, the calibration environment over real basins (Phase 0),
    `aquascope gym tasks|bench|leaderboard FILES`: the playbook benchmark (Phase 1), `aquascope gym plans
    list|show|validate|score|rescore`: the plan-quality benchmark (Phase 2), and `aquascope gym reports
    list|show|score|bench`: report-quality scoring over a finished study bundle (#382)."""
    from aquascope import gym as hg

    def say(msg: str) -> None:
        if not getattr(args, "quiet", False):
            print(f"  · {msg}", file=sys.stderr)

    if args.gym_cmd == "plans":
        from aquascope.gym import plans as gp

        if args.plans_cmd == "list":
            rows = gp.list_references(args.plans)
            if args.json:
                print(json.dumps(rows, indent=2, default=str))
                return
            print(f"  {len(rows)} reference plans in {args.plans or gp.PLANS_DIR}")
            for r in rows:
                what = (
                    "decline"
                    if r["decline"]
                    else (f"{r['steps']} step(s)" + (f" + {r['optional']} optional" if r["optional"] else ""))
                )
                tags = f" [{', '.join(r['tags'])}]" if r["tags"] else ""
                print(f"  {r['id']:<42} {r['playbook']:<24} {what:<24} {str(r['site'])[:44]}{tags}")
            return
        if args.plans_cmd == "show":
            ref = gp.load_reference(args.id, args.plans)
            if args.json or not ref.path:
                print(json.dumps(ref.to_dict(), indent=2, default=str))
                return
            print(Path(ref.path).read_text(encoding="utf-8"))
            return
        if args.plans_cmd == "validate":
            refs = gp.load_references(args.plans)
            bad = 0
            for ref in refs:
                errors = gp.validate_reference(ref, args.plans)
                if errors:
                    bad += 1
                    print(f"  {ref.id}: " + "; ".join(errors))
            print(f"  {len(refs)} reference plans, {len(refs) - bad} valid, {bad} with errors")
            if bad:
                sys.exit(1)
            return
        if args.plans_cmd == "rescore":
            for path in args.results:
                rows = gp.load_plan_results([path], latest=False)
                rows = gp.rescore_plans(rows, plans_dir=args.plans)
                target = Path(args.out) if args.out and len(args.results) == 1 else Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    "".join(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n" for r in rows),
                    encoding="utf-8",
                )
                print(f"  {len(rows)} rows re-scored -> {target}")
            return
        if args.plans_cmd == "score":
            ref = gp.load_reference(args.id, args.plans)
            if args.candidate:
                with open(args.candidate, encoding="utf-8") as fh:
                    cand = gp.candidate_from(json.load(fh))
            else:
                cand, _detail = gp.tree_candidate(ref, args.plans)
            scored = gp.score_plan(ref, cand)
            if args.json:
                print(json.dumps({"case": ref.id, **scored, "candidate": cand.to_dict()}, indent=2, default=str))
                return
            pct = lambda x: "-" if x is None else f"{100 * x:.0f} %"  # noqa: E731
            print(
                f"  {ref.id} ({'the tree' if not args.candidate else args.candidate}): score {scored['score']:.2f}; "
                f"tools {pct(scored['coverage_tools'])}, methods {pct(scored['coverage_methods'])}, gates "
                f"{pct(scored['coverage_gates'])}, extraneous {pct(scored['extraneous'])}, forbidden "
                f"{scored['forbidden_used']}, decline correct {scored['decline_correct']}"
            )
            for line in scored["explain"]:
                print(f"    {line}")
            return

    if args.gym_cmd == "reports":
        from aquascope.gym import reports as gru

        if args.reports_cmd == "list":
            rows = gru.list_references(args.reports)
            if args.json:
                print(json.dumps(rows, indent=2, default=str))
                return
            print(f"  {len(rows)} report references in {args.reports or gru.REPORTS_DIR}")
            for r in rows:
                tags = f" [{', '.join(r['tags'])}]" if r["tags"] else ""
                print(
                    f"  {r['id']:<28} study {r['study']:<24} must_say {r['must_say']:>2} must_not_say "
                    f"{r['must_not_say']:>2}{tags}"
                )
            return
        if args.reports_cmd == "show":
            ref = gru.load_reference(args.id, args.reports)
            if args.json or not ref.path:
                print(json.dumps(ref.to_dict(), indent=2, default=str))
                return
            print(Path(ref.path).read_text(encoding="utf-8"))
            return
        if args.reports_cmd == "score":
            study_dir = Path(args.study_dir)
            ref = (
                gru.load_reference(args.reference, args.reports)
                if args.reference
                else next((r for r in gru.load_references(args.reports) if r.study == study_dir.name), None)
            )
            scored = gru.score_study_dir(study_dir, reference=ref)
            if args.json:
                print(json.dumps(scored.to_dict(), indent=2, default=str))
                return
            print(f"  {scored.study_id}: mean {scored.mean}")
            for dim in gru.DIMENSIONS:
                print(f"    {dim}: {scored.dimensions.get(dim)}")
            if scored.reference:
                print(f"  reference {scored.reference['reference_id']}: score {scored.reference['score']}")
                for line in scored.reference["must_say"]["failed"]:
                    print(f"    must_say failed: {line}")
                for line in scored.reference["must_not_say"]["violated"]:
                    print(f"    must_not_say violated: {line}")
            if scored.error:
                print(f"  error: {scored.error}")
            return
        if args.reports_cmd == "bench":
            refs = gru.load_references(args.reports)

            def say(msg: str) -> None:
                if not args.quiet:
                    print(f"  · {msg}", file=sys.stderr)

            results = gru.run_report_bench(args.dir, refs, study_ids=args.study or None, out=args.out, on_event=say)
            if args.json:
                print(json.dumps(gru.summarize_reports(results), indent=2, default=str))
                return
            print(gru.report_leaderboard(results))
            if args.out:
                print(f"  -> {args.out}")
            return

    if args.gym_cmd == "tasks":
        from aquascope.gym import tasks as gt
        from aquascope.playbooks import list_playbooks

        pbs = args.playbook or [p["id"] for p in list_playbooks() if "error" not in p]
        probes = None if args.probes == "all" else int(args.probes)
        n_probes = sum(len(gt.decline_probes(p)) for p in pbs) if probes is None else probes
        per_site = max(1, len(pbs) + n_probes)
        sites = gt.suggest_sites(
            -(-args.n // per_site),
            seed=args.seed,
            sources=args.source or None,
            ungauged_share=args.ungauged_share,
            on_land=None if args.no_check_land else gt.on_land_basinatlas,
        )
        skipped: list[dict] = []
        tasks = gt.tasks_from_playbooks(sites, pbs, probes=probes, on_event=say, skipped=skipped)[: args.n]
        gt.write_tasks(tasks, args.out)
        hard = sum(1 for t in tasks if t.unsolvable)
        test = sum(1 for t in tasks if t.split == "test")
        n_sites = len({gt.site_key(t.site) for t in tasks})
        print(f"  {len(tasks)} tasks from {n_sites} sites ({hard} unsolvable, {test} held out as test) -> {args.out}")
        if skipped:
            sites_lost = sum(1 for e in skipped if "playbook" not in e)
            print(
                f"  skipped: {sites_lost} of {len(sites)} sites (reconnaissance unavailable), "
                f"{len(skipped) - sites_lost} tasks (no key):"
            )
            for entry in skipped:
                what = f" {entry['playbook']}" if entry.get("playbook") else ""
                print(f"    {gt.site_key(entry['site'])}{what}: {entry['error'][:100]}")
        counts: dict[str, int] = {}
        for t in tasks:
            key = f"{t.playbook}/{'declined' if t.unsolvable else t.expected.get('branch')}"
            counts[key] = counts.get(key, 0) + 1
        for key, n in sorted(counts.items()):
            print(f"    {key:<36} {n}")
        return

    if args.gym_cmd == "bench":
        from aquascope.gym import bench as gb

        if args.agent in ("methodologist", "file") or args.plans or not args.tasks:
            # Phase 2: plan quality against the reference plans.
            from aquascope.gym import plans as gp

            if args.agent not in gp.AGENTS:
                sys.exit(f"  --agent {args.agent} plays Phase 1 tasks; pass --tasks, or one of {gp.AGENTS} for Phase 2")
            plan_results = gp.run_plan_bench(
                args.plans,
                args.agent,
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                candidates_dir=args.candidates,
                limit=args.limit,
                case_ids=args.case or None,
                repeats=args.repeats,
                out=args.out,
                resume=args.resume,
                timeout=args.timeout or None,
                on_event=say,
            )
            if args.json:
                print(json.dumps(gp.summarize_plans(plan_results), indent=2, default=str))
                return
            print(gp.plan_leaderboard(plan_results, title=f"aquascope gym bench: {args.agent}"))
            if args.out:
                print(f"  -> {args.out}")
            return
        results = gb.run_bench(
            args.tasks,
            args.agent,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            limit=args.limit,
            unsolvable=args.unsolvable,
            task_ids=args.task or None,
            timeout=args.timeout or None,
            out=args.out,
            max_steps=args.max_steps,
            context_chars=args.context_chars,
            on_event=say,
            spread=args.spread,
            resume=args.resume,
        )
        if args.json:
            print(json.dumps(gb.summarize(results), indent=2, default=str))
            return
        print(gb.leaderboard(results, title=f"aquascope gym bench: {args.agent}"))
        if args.out:
            print(f"  -> {args.out}")
        return

    if args.gym_cmd == "leaderboard" and args.results:
        from aquascope.gym import bench as gb
        from aquascope.gym import plans as gp
        from aquascope.gym import reports as gru

        results = gb.load_results(args.results)
        plan_results = gp.load_plan_results(args.results)
        report_results = gru.load_report_results(args.results)
        if args.json:
            summary = {}
            if results:
                summary["tasks"] = gb.summarize(results)
            if plan_results:
                summary["plans"] = gp.summarize_plans(plan_results)
            if report_results:
                summary["reports"] = gru.summarize_reports(report_results)
            print(json.dumps(summary if len(summary) > 1 else next(iter(summary.values()), {}), indent=2, default=str))
            return
        parts = []
        if results:
            parts.append(gb.leaderboard(results, title=args.title))
        if plan_results:
            parts.append(gp.plan_leaderboard(plan_results, title=(args.title if not results else None)))
        if report_results:
            parts.append(
                gru.report_leaderboard(report_results, title=(args.title if not (results or plan_results) else None))
            )
        text = "\n".join(parts)
        print(text)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"  -> {args.out}")
        return

    if args.gym_cmd == "basins":
        rows = hg.suggest_basins(
            args.n,
            sources=args.source or None,
            min_years=args.min_years,
            max_snow_pct=None if args.allow_snow else 20.0,
        )
        if args.json:
            print(json.dumps(rows, indent=2, default=str))
            return
        if not rows:
            print("  no candidate basins yet (the archive publishes basins/station_signatures.parquet weekly)")
            return
        print(f"  {len(rows)} basins with long archived discharge and a catchment area (use SOURCE/ID with `gym run`)")
        for r in rows:
            snow = f"snow {r['snow_cover_pct']:.0f} %" if r.get("snow_cover_pct") is not None else ""
            print(
                f"  {r['source']}/{r['station_id']:<42} {r['area_km2']:>9,.0f} km2  {r['n_years']:>4.0f} yr  "
                f"q {r['q_mean_mm']:.2f} mm/d  RR {r['runoff_ratio'] if r['runoff_ratio'] is not None else float('nan'):.2f}  {snow}"
            )
        return

    def _basins():
        if args.synthetic or not args.basin:
            return [hg.synthetic_basin(i) for i in range(args.n_synthetic)]
        out = []
        for spec in args.basin:
            src, _, sid = spec.partition("/")
            out.append(hg.load_basin(src, sid))
        return out

    if args.gym_cmd == "leaderboard":
        basins = _basins()
        table = hg.run_leaderboard(
            basins, args.agent or None, objective=args.objective, max_steps=args.steps, seeds=tuple(range(args.seeds))
        )
        if args.json:
            print(table.to_json(orient="records", indent=2))
            return
        cols = ["agent", "basin", "seed", "steps", "simulator_calls", "best_reward", "val_nse", "val_kge", "seconds"]
        print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if args.out:
            table.to_csv(args.out, index=False)
            print(f"\n  -> {args.out}")
        return
    # run: one agent on one (or the first) basin
    basins = _basins()
    env = hg.CalibrationEnv(basins, objective=args.objective, max_steps=args.steps)
    env.reset(seed=args.seed)
    fn = hg.BASELINES[args.agent[0] if args.agent else "differential_evolution"]
    res = fn(env, {"seed": args.seed} if (args.agent or ["differential_evolution"])[0] != "nelder_mead" else {})
    if args.json:
        print(json.dumps({**res, "history": hg.episode_table(env).to_dict("records")}, indent=2, default=str))
        return
    print(env.render())
    print(
        f"  {res['agent']}: {res['steps']} steps, {res.get('simulator_calls', res['steps'])} simulator calls, "
        f"{res['seconds']} s"
    )
    val = res.get("validation") or {}
    print(f"  validation: NSE {val.get('nse')}, KGE {val.get('kge')}, PBIAS {val.get('pbias')}")


def cmd_caravan(args: argparse.Namespace) -> None:
    """`aquascope caravan export|validate`: Caravan-format sub-datasets from the Archive."""
    from aquascope.archive import caravan

    if args.caravan_cmd == "validate":
        res = caravan.validate_caravan(args.out, args.prefix)
        print(f"  {res['n_gauges']} gauges, {'OK' if res['ok'] else str(len(res['problems'])) + ' problems'}")
        for pr in res["problems"][:30]:
            print(f"    - {pr}")
        if not res["ok"]:
            sys.exit(1)
        return

    def say(msg: str) -> None:
        if not args.quiet:
            print(f"  · {msg}", file=sys.stderr)

    try:
        report = caravan.export_caravan(
            args.source,
            args.out,
            station_ids=args.station or None,
            max_stations=args.max_stations,
            min_years=args.min_years,
            start=args.start,
            end=args.end,
            prefix=args.prefix,
            forcing=not args.no_forcing,
            forcing_models=None if args.era5 else "best_match",
            fetch_missing=args.fetch_missing,
            write_netcdf=args.netcdf,
            pause=args.pause,
            on_event=say,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    for g in report.gauges:
        status = (
            f"{g.n_days:>6} days, {g.n_streamflow:>6} with flow, area {g.area_km2:,.0f} km2 ({g.area_source})"
            if g.ok
            else f"skipped: {g.error}"
        )
        print(f"  {g.gauge_id:<48} {status}")
    print(f"\n  {report.n_ok}/{len(report.gauges)} gauges written under {report.out_dir} (prefix {report.prefix})")
    res = (
        caravan.validate_caravan(args.out, report.prefix)
        if report.n_ok
        else {"ok": False, "problems": ["nothing written"]}
    )
    print(f"  validation: {'OK' if res['ok'] else '; '.join(res['problems'][:5])}")
    if report.n_ok == 0:
        sys.exit(1)


def cmd_completion(args: argparse.Namespace) -> None:
    """Print the shell activation line for tab-completion."""
    from argcomplete.shell_integration import shellcode

    print(shellcode(["aquascope"], shell=args.shell))


def cmd_solve(args: argparse.Namespace) -> None:
    """`aquascope solve`: a problem at a point through the plan-first team (with --lat/--lon), or the legacy
    challenge agent over a data file."""
    if args.lat is not None or args.lon is not None or args.playbook:
        cmd_solve_team(args)
        return
    from aquascope.ai_engine.agent import HydroAgent

    agent = HydroAgent(default_model=args.model)

    data = None
    if args.file:
        data = _load_dataframe(args.file)
        if "datetime" in data.columns:
            data["datetime"] = __import__("pandas").to_datetime(data["datetime"])
            data = data.set_index("datetime").sort_index()
        elif "sample_datetime" in data.columns:
            data["sample_datetime"] = __import__("pandas").to_datetime(data["sample_datetime"])
            data = data.rename(columns={"sample_datetime": "datetime"}).set_index("datetime").sort_index()

    result = agent.solve(args.query, data=data)
    explanation = agent.explain(result)
    print(explanation)


def _parse_intake(pairs: list[str] | None) -> dict:
    out: dict = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--intake expects KEY=VALUE, got {pair!r}")
        out[key.strip()] = value.strip()
    return out


def cmd_solve_team(args: argparse.Namespace) -> None:
    """The plan-first Analyst (#308): recon, plan, your review, execution with gates, report."""
    from aquascope.ai_engine.team import solve

    if args.lat is None or args.lon is None:
        logger.error("solve needs both --lat and --lon (or neither, for the legacy challenge agent).")
        sys.exit(1)
    try:
        intake = _parse_intake(args.intake)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    def on_event(event: dict) -> None:
        if not args.quiet:
            print(f"  · {_format_event(event)}", file=sys.stderr)

    def review(study):
        plan = study.plan or {}
        print(f"\nPlan: playbook {plan.get('playbook')}, branch {plan.get('branch')}, {len(study.steps)} step(s)")
        if plan.get("rationale"):
            print(f"  {plan['rationale']}")
        for n in (plan.get("recon_notes") or []) + (plan.get("notes") or []):
            print(f"  note: {n}")
        for i, step in enumerate(study.steps, 1):
            print(f"  {i}. {step.tool}({', '.join(f'{k}={v!r}' for k, v in step.arguments.items())})")
            if step.rationale:
                print(f"     {step.rationale}")
            for g in step.expects:
                where = g.get("path") or ", ".join(g.get("paths") or [])
                value = f" {g['value']}" if g.get("value") is not None else ""
                print(f"     gate {g.get('check')}{value} on {where}")
            if isinstance(step.fallback, dict) and step.fallback.get("step"):
                print(f"     fallback: {step.fallback['step'].get('tool')}")
        if plan.get("caveats"):
            print(f"  {len(plan['caveats'])} caveat(s) will be printed verbatim in the report.")
        if args.yes:
            return study
        if not sys.stdin.isatty():
            print("  Not a terminal: pass --yes to run the plan.", file=sys.stderr)
            return None
        try:
            answer = input("Run this plan? [y/N] ")
        except EOFError:
            return None
        return study if answer.strip().lower() in ("y", "yes") else None

    try:
        result = solve(
            args.query,
            lat=args.lat,
            lon=args.lon,
            playbook=args.playbook,
            intake=intake,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            review=review,
            on_event=on_event,
        )
    except (RuntimeError, ValueError, ImportError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    md = result.to_markdown()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"\n  Report saved to {args.out}")
        if result.answer:
            print(result.answer)
    else:
        print(md)
    if args.study and result.study.steps:
        Path(args.study).parent.mkdir(parents=True, exist_ok=True)
        Path(args.study).write_text(result.study_yaml, encoding="utf-8")
        print(f"  Study saved to {args.study}; re-run it with `aquascope run {args.study}`")
    if result.declined:
        print(f"\n  Declined: {result.declined_reason}", file=sys.stderr)
    elif result.not_established and not args.quiet:
        print("\n  What this answer does not establish:", file=sys.stderr)
        for line in result.not_established:
            print(f"   · {line}", file=sys.stderr)


_LATLON = r"^\s*(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)\s*$"


def _place_matches(text: str, limit: int = 5) -> list[dict]:
    """Where a study goes, from what a person types: ``lat, lon`` (one match, no catalog read), a station
    id (``USGS-01013500``), or words of a gauge's name or river ("Fish River Fort Kent"), searched in the
    Archive's station catalog. Rows carry ``name``, ``latitude``, ``longitude`` and, from the catalog,
    ``source`` and ``station_id``."""
    import re

    m = re.match(_LATLON, text or "")
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"{lat}, {lon} is not a latitude, longitude")
        return [{"name": f"{lat:g}, {lon:g}", "latitude": lat, "longitude": lon}]
    import logging

    from aquascope.archive.catalog import load_stations, search_stations

    quiet = [logging.getLogger(n) for n in ("httpx", "aquascope.archive.catalog")]
    levels = [lg.level for lg in quiet]
    for lg in quiet:  # the daily catalog download logs a signed URL a screen wide
        lg.setLevel(logging.WARNING)
    try:
        rows = load_stations()
    finally:
        for lg, level in zip(quiet, levels):
            lg.setLevel(level)
    wanted = (text or "").strip().lower()
    exact = [r for r in rows if str(r.get("station_id", "")).lower() == wanted]
    return exact[:limit] or search_stations(rows, query=text, limit=limit, group_sites=True)


def _place_line(row: dict) -> str:
    where = f"{row['latitude']:.4f}, {row['longitude']:.4f}"
    if row.get("station_id"):
        return f"{row.get('name') or row['station_id']} ({row.get('source')} {row['station_id']}) at {where}"
    return where


def _studio_key_offer() -> str | None:
    """The provider whose key is in the environment, in the CLI's usual scan order, or None."""
    import os

    from aquascope.ai_engine.providers import ENV_SCAN_ORDER, env_var

    for name in ENV_SCAN_ORDER:
        env = env_var(name)
        if env and os.environ.get(env):
            return name
    return None


def _studio_start(args: argparse.Namespace) -> bool:
    """``aquascope studio`` with nothing else: ask where, what, and whether to use a key found in the
    environment, then fill ``args`` as if they had been typed. False when the person leaves."""

    def ask(prompt: str) -> str | None:
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return None

    print("AquaScope Studio: a crew that runs a complete study at a place and hands you the bundle.")
    print("Nothing runs until you approve the plan.\n")
    while args.lat is None:
        text = ask('Where? A gauge name or id ("Fish River Fort Kent", USGS-01013500) or lat, lon: ')
        if text is None:
            return False
        if not text.strip():
            continue
        try:
            rows = _place_matches(text)
        except Exception as exc:  # noqa: BLE001 - a failed catalog read: coordinates still work
            print(f"  cannot search the station catalog ({exc}); type lat, lon instead.", file=sys.stderr)
            continue
        if not rows:
            print("  no gauge matches that; try other words, a station id, or lat, lon.", file=sys.stderr)
            continue
        pick = rows[0]
        if len(rows) > 1:
            for i, row in enumerate(rows, 1):
                print(f"  {i}. {_place_line(row)}")
            choice = (ask(f"Which one? [1-{len(rows)}, Enter for 1, or search again]: ") or "").strip()
            if choice and not choice.isdigit():
                continue
            if choice:
                if not 1 <= int(choice) <= len(rows):
                    continue
                pick = rows[int(choice) - 1]
        print(f"  → {_place_line(pick)}")
        args.lat, args.lon = pick["latitude"], pick["longitude"]
    while not args.query:
        text = ask('\nWhat do you want to know? (e.g. "Is flooding here getting worse?"): ')
        if text is None:
            return False
        args.query = text.strip() or None
    if not any((args.provider, args.model, args.api_key, args.base_url)):
        if not _studio_key_step(args):
            return False
    print()
    return True


_OTHER = "Other (type it)"


def _choose(q: dict) -> str | None:
    """One Studio question as a pick list: arrow keys and Enter with ``questionary`` (the ``studio`` extra),
    numbers otherwise; the last row takes the answer in your own words. None when the person leaves."""
    options = [str(o) for o in q.get("options") or []]
    default = str(q["default"]) if q.get("default") is not None and str(q["default"]) in options else None
    try:
        import questionary
    except ImportError:
        questionary = None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        questionary = None      # a pick list needs a terminal on both ends; numbers work anywhere
    print()
    if q.get("retry"):
        print(f"  {q['retry']}")
    if questionary is not None:
        if q.get("why"):
            questionary.print(f"  {q['why']}", style="italic fg:ansibrightblack")
        picked = questionary.select(str(q.get("text") or ""), choices=[*options, _OTHER], default=default,
                                    qmark="?").ask()
        if picked == _OTHER:
            picked = questionary.text("Your answer:", qmark="›").ask()
        return picked
    print(q.get("text") or "")
    if q.get("why"):
        print(f"  ({q['why']})")
    for i, o in enumerate([*options, _OTHER], 1):
        print(f"  {i}. {o}" + ("  (default)" if o == default else ""))
    while True:
        try:
            answer = input(f"Pick 1-{len(options) + 1} (Enter for the default): ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return None
        if not answer:
            return default or "just go"
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer.isdigit() and int(answer) == len(options) + 1:
            try:
                return input("Your answer: ")
            except (EOFError, KeyboardInterrupt):
                return None
        return answer


_NO_KEY, _PASTE, _FREE = "No, run keyless", "Yes, paste a key", "Get a free key first (Groq)"


def _studio_key_step(args: argparse.Namespace) -> bool:
    """Whether a model joins the crew: a key already set (or saved) is offered; otherwise the person is asked
    if they have one, pastes it (hidden), and it is checked with one tiny request before the study starts.
    Fills ``args.provider``, ``args.api_key`` and a $1 ``args.max_usd``. False when the person leaves."""
    import getpass

    from aquascope.ai_engine import keys

    why = "With a key the model writes the brief, the plan and the prose; the numbers come from the tools either way."
    cap = args.max_usd if args.max_usd is not None else 1.0
    found = _studio_key_offer()
    if found:
        answer = _choose({"text": f"A {found} key is set. Use it?", "why": f"{why} Spend ceiling ${cap:g}.",
                          "options": ["Yes, use it", "No, run keyless"], "default": "Yes, use it"})
        if answer is None:
            return False
        if answer.strip().lower() in ("yes, use it", "y", "yes"):
            args.provider, args.max_usd = found, cap
        return True
    answer = _choose({"text": "Do you have an AI model key? (optional)", "why": why,
                      "options": [_NO_KEY, _PASTE, _FREE], "default": _NO_KEY})
    if answer is None:
        return False
    if answer == _FREE:
        print("  Make one at https://console.groq.com/keys (free tier, no card), then paste it here.")
    elif answer != _PASTE and answer.strip().lower() not in ("y", "yes"):
        return True
    while True:
        try:
            key = getpass.getpass("  Paste your key (it stays hidden; Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return False
        if not key:
            print("  Running keyless.")
            return True
        provider = keys.guess_provider(key)
        if provider is None:
            from aquascope.ai_engine.providers import PROVIDERS

            ids = [p for p in PROVIDERS if PROVIDERS[p].env]
            picked = _choose({"text": "Which service is this key for?",
                              "options": [PROVIDERS[p].label for p in ids], "default": PROVIDERS[ids[0]].label})
            if picked is None:
                return False
            provider = next((p for p in ids if PROVIDERS[p].label == picked), None)
            if provider is None:
                continue
        print("  Checking the key with one short request...")
        works, said = keys.check_key(provider, key, model=args.model)
        if works:
            print(f"  OK: {said}. Spend ceiling ${cap:g} (--max-usd changes it).")
            args.provider, args.api_key, args.max_usd = provider, key, cap
            keep = _choose({"text": "Remember this key on this computer?",
                            "why": f"Saved to {keys.keys_path()}, readable only by you; next time the Studio "
                                   "offers it instead of asking.",
                            "options": ["No, just this time", "Yes, remember it"], "default": "No, just this time"})
            if keep == "Yes, remember it":
                keys.save_key(provider, key)
                print("  Saved.")
            return True
        again = _choose({"text": f"That key did not work: {said}.", "options": ["Paste it again", _NO_KEY],
                         "default": "Paste it again"})
        if again is None:
            return False
        if again != "Paste it again":
            return True


def _studio_missing_extras() -> list[str]:
    """The ``studio`` extra's modules this install lacks. Without them the bundle is the Markdown and HTML
    report and the tables only: no Word report, workbook, figures, notebook or bundle.zip."""
    import importlib.util

    return [mod for mod in ("docx", "openpyxl", "matplotlib") if importlib.util.find_spec(mod) is None]


def _parse_edits(text: str) -> dict:
    """``s3.return_period=200, s2.k=8`` -> ``{"s3": {"arguments": {"return_period": 200}}, "s2": {...}}``."""
    import re

    out: dict = {}
    for item in re.split(r"[,;]\s*|\s{2,}", text or ""):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        if not sep or "." not in key:
            raise ValueError(f"an edit is STEP.ARG=VALUE, got {item!r}")
        sid, _, arg = key.strip().partition(".")
        try:
            parsed = json.loads(value.strip())
        except json.JSONDecodeError:
            parsed = value.strip()
        out.setdefault(sid, {"arguments": {}})["arguments"][arg.strip()] = parsed
    return out


def _finished_study(target: str | None) -> Path | None:
    """The workspace.json of a finished study ``target`` names (its bundle folder or the file), else None."""
    if not target:
        return None
    path = Path(target).expanduser()
    if path.is_dir():
        path = path / "workspace.json"
    if path.suffix != ".json" or not path.is_file():
        return None
    try:
        status = json.loads(path.read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None
    return path if status == "done" else None


#: The Desk's options, so `aquascope studio <bundle>` and `aquascope desk <bundle>` take the same ones.
_DESK_OPTIONS = ("set", "years", "exclude_years", "estimator", "sign", "comment", "section", "resolve", "by",
                 "note")


def cmd_studio(args: argparse.Namespace) -> None:
    """`aquascope studio`: the crew, from the brief you agree and the plan you approve to the bundle. Given a
    finished study (its bundle folder or workspace.json) it opens the Study Desk instead."""
    from aquascope.studio import Studio

    finished = _finished_study(args.query)
    if finished is not None:
        args.workspace = str(finished)
        cmd_desk(args)
        return
    if args.query and Path(args.query).expanduser().exists():
        logger.error("%s is not a finished study; pick an unfinished one up with --resume WORKSPACE.JSON",
                     args.query)
        sys.exit(1)
    if any(getattr(args, k, None) for k in _DESK_OPTIONS if k != "by"):
        logger.error("those options revise a finished study: give its bundle folder, e.g. "
                     "aquascope studio ./studio-<id>/ --exclude-years 2008")
        sys.exit(1)
    if getattr(args, "return_period", None) is not None:
        args.intake = [*list(args.intake or []), f"return_period={args.return_period:g}"]

    workspace = None
    if args.resume:
        try:
            workspace = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.error("cannot read %s: %s", args.resume, exc)
            sys.exit(1)
    else:
        if args.at and (args.lat is None or args.lon is None):
            try:
                rows = _place_matches(args.at)
            except Exception as exc:  # noqa: BLE001 - a bad place or a failed catalog read
                logger.error("cannot place %r: %s", args.at, exc)
                sys.exit(1)
            if not rows:
                logger.error("no gauge matches %r; try a station id or lat, lon.", args.at)
                sys.exit(1)
            print(f"  Study at {_place_line(rows[0])}", file=sys.stderr)
            args.lat, args.lon = rows[0]["latitude"], rows[0]["longitude"]
        if (args.lat is None or args.lon is None or not args.query) and sys.stdin.isatty() and not args.yes:
            if not _studio_start(args):
                return
        if args.lat is None or args.lon is None:
            logger.error("studio needs a place: --at 'GAUGE, ID or lat,lon', or --lat and --lon "
                         "(or run `aquascope studio` in a terminal and it asks).")
            sys.exit(1)
    try:
        intake = _parse_intake(args.intake)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    data: dict = {}
    for path in args.data or []:
        try:
            from aquascope.ingest import read_table

            data[f"upload:{Path(path).name}"] = read_table(path)
        except (OSError, ValueError) as exc:
            logger.error("cannot read %s: %s", path, exc)
            sys.exit(1)

    for name in ("httpx", "aquascope.archive", "aquascope.collectors"):
        logging.getLogger(name).setLevel(logging.WARNING)   # the crew's timeline says what is happening
    if not any((args.provider, args.model, args.api_key, args.base_url)):
        from aquascope.ai_engine.keys import load_saved_keys

        load_saved_keys()      # a key the person asked the Studio to remember; the shell's own wins

    from aquascope.studio.progress import Narrator

    narrator = Narrator()
    verbose = bool(getattr(args, "verbose", False))
    if not verbose:
        # The short log speaks for the crew: library notes (fit parameters, a missing optional API key) stay out
        # of it unless something is wrong.
        logging.getLogger("aquascope").setLevel(logging.WARNING)
        for name in ("aquascope.collectors", "aquascope.archive", "numexpr"):
            logging.getLogger(name).setLevel(logging.ERROR)

    def on_event(event: dict) -> None:
        if args.quiet:
            return
        if verbose:
            print(f"  · {_format_event(event)}", file=sys.stderr)
            return
        for line in narrator.feed(event):
            print(f"  {line}", file=sys.stderr)

    try:
        studio = Studio(
            args.lat,
            args.lon,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            data=data,
            on_event=on_event,
            workspace=workspace,
            intake=intake,
            max_usd=args.max_usd,
        )
    except (RuntimeError, ValueError, ImportError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    ws = studio.workspace
    if getattr(args, "style", None):
        from aquascope.studio.document import load_style

        try:
            ws.house_style = load_style(args.style).to_dict()
        except (OSError, ValueError) as exc:
            logger.error("cannot read the house style %s: %s", args.style, exc)
            sys.exit(1)
    out_dir = Path(args.out or f"./studio-{ws.id}")
    interactive = sys.stdin.isatty() and not args.yes

    def checkpoint() -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "workspace.json").write_text(ws.to_json(), encoding="utf-8")

    def ask(prompt: str) -> str | None:
        try:
            return input(prompt)
        except EOFError:
            return None

    reply = None
    if ws.status == "intake":
        if not args.query and not ws.messages:
            logger.error("studio needs the problem in plain language.")
            sys.exit(1)
        reply = studio.say(args.query) if args.query else studio.say("just go")
    elif ws.status == "waiting":
        reply = studio._request_reply()
    if reply is not None and reply.kind in ("questions", "data_request"):
        while reply.kind in ("questions", "data_request"):
            qs = reply.payload.get("questions") or []     # a question, or a choice the crew offers (a gauge)
            if interactive and len(qs) == 1 and qs[0].get("options"):
                answer = _choose(qs[0])
                if answer is None:
                    checkpoint()
                    return
                reply = studio.say(answer)
                continue
            print(reply.text)
            if reply.kind == "data_request":
                if args.continue_without or not interactive:
                    print("  Continuing without it (--continue-without or no terminal).", file=sys.stderr)
                    reply = studio.say("continue without")
                    continue
                answer = ask("Path to a table (CSV, Excel, JSON), or 'continue' to go on without: ")
                if answer is None:
                    checkpoint()
                    return
                path = Path(answer.strip())
                if path.exists():
                    try:
                        from aquascope.ingest import read_table

                        reply = studio.add_table(path.name, read_table(str(path)))
                    except (OSError, ValueError) as exc:
                        print(f"  cannot read {path}: {exc}", file=sys.stderr)
                else:
                    reply = studio.say(answer)
                continue
            if not interactive:
                print("  Proceeding on the defaults (--yes or no terminal).", file=sys.stderr)
                reply = studio.say("just go")
                continue
            answer = ask("> ")
            if answer is None:
                checkpoint()
                return
            reply = studio.say(answer)
    if ws.status == "review" and reply is None:
        reply = studio._plan_reply()
    elif ws.status == "done":
        reply = studio._report_reply()
    if reply is None or reply.kind == "declined":
        print(reply.text if reply else f"Status {ws.status}; nothing to do.", file=sys.stderr)
        checkpoint()
        sys.exit(1 if reply is not None else 0)
    if reply.kind == "plan":
        # Asked to approve, the whole plan is shown; run with --yes, its first line says what will run (the
        # progress log shows each analysis as it happens) unless --verbose asks for everything.
        print(reply.text if interactive or getattr(args, "verbose", False) else reply.text.splitlines()[0])
        edits = None
        if interactive:
            run, change, later = "Run it", "Change a step first", "Not now (save it for later)"
            while True:
                answer = _choose({"text": "Run this plan?", "options": [run, change, later], "default": run})
                low = (answer or "").strip().lower()
                if answer is None or answer == later or low in ("n", "no", "not now", "later"):
                    checkpoint()
                    print(f"\n  Saved. Pick it up any time: aquascope studio --resume {out_dir / 'workspace.json'}",
                          file=sys.stderr)
                    return
                if answer == run or low in ("y", "yes", "run", "go", "just go"):
                    break
                if answer == change or low == "e":
                    line = ask("Overrides, STEP.ARG=VALUE separated by commas, e.g. s3.return_period=200 "
                               "(blank keeps the plan): ") or ""
                    try:
                        edits = _parse_edits(line) or None
                    except ValueError as exc:
                        print(f"  {exc}", file=sys.stderr)
                        continue
                    break
                # anything else is a change to the brief in your own words: the crew plans again
                reply = studio.say(answer)
                print(reply.text)
                if reply.kind != "plan":
                    checkpoint()
                    return
        elif not args.yes:
            print("  Not a terminal: pass --yes to run the plan.", file=sys.stderr)
            checkpoint()
            return
        reply = studio.approve(edits=edits)
        while reply.kind == "plan" and reply.payload.get("errors") and interactive:
            print(reply.text)
            line = ask("Overrides again (blank runs the plan as it is): ") or ""
            try:
                reply = studio.approve(edits=_parse_edits(line) or None)
            except ValueError as exc:
                print(f"  {exc}", file=sys.stderr)
    if reply.kind == "report":
        for line in narrator.flush():
            if not args.quiet and not verbose:
                print(f"  {line}", file=sys.stderr)
        paths = studio.export(out_dir)
        if verbose:
            print()
            print(reply.text)
            decision = reply.payload.get("decision") or {}
            if decision.get("grade"):
                print(f"\n  Grade: {str(decision['grade']).replace('_', ' ')}", file=sys.stderr)
            for line in decision.get("conditions") or []:
                print(f"   · holds if: {line}", file=sys.stderr)
            for line in decision.get("what_would_change_it") or []:
                print(f"   · would change it: {line}", file=sys.stderr)
            for f in (reply.payload.get("findings") or [])[:12]:
                print(f"   · [{str(f.get('grade') or '').replace('_', ' ')}] {f.get('claim')}", file=sys.stderr)
            for line in reply.payload.get("not_established") or []:
                print(f"   · not established: {line}", file=sys.stderr)
            print(f"\n  Bundle written to {out_dir}: {', '.join(sorted(paths))}")
        else:
            print()
            try:
                from aquascope.studio.document import terminal_summary

                lines = terminal_summary(ws, str(out_dir))
            except Exception as exc:  # noqa: BLE001 - the summary is a nicety; the answer must still print
                logger.debug("terminal summary unavailable: %s", exc)
                lines = [reply.text, "", f"Bundle written to {out_dir}"]
            for line in lines:
                print(line)
        if _studio_missing_extras():
            print(
                "  This install writes the Markdown and HTML report and the tables only. For the Word report, "
                "the Excel workbook, the figures, the notebook and bundle.zip: pip install \"aquascope[studio]\"",
                file=sys.stderr,
            )
        if interactive:
            while True:
                text = ask("Follow-up (or 'done'): ")
                if text is None or text.strip().lower() in ("", "done", "quit", "exit"):
                    break
                more = studio.follow_up(text)
                print(more.text)
                if more.kind == "report":
                    studio.export(out_dir)
                    print(f"  Bundle updated in {out_dir}")
    elif reply.kind != "plan":
        print(reply.text)
    checkpoint()


def cmd_desk(args: argparse.Namespace) -> None:
    """`aquascope desk WORKSPACE.json`: revise, review and sign a finished study (aquascope.studio.desk)."""
    from aquascope.studio import desk
    from aquascope.studio.coordinator import Studio
    from aquascope.studio.workspace import Workspace

    path = Path(args.workspace)
    if path.is_dir():
        path = path / "workspace.json"
    try:
        ws = Workspace.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("cannot read %s: %s", path, exc)
        sys.exit(1)
    if not getattr(args, "verbose", False):
        logging.getLogger("aquascope").setLevel(logging.WARNING)
        for name in ("aquascope.collectors", "aquascope.archive", "numexpr", "httpx"):
            logging.getLogger(name).setLevel(logging.ERROR)
    out_dir = Path(args.out) if args.out else path.parent
    from aquascope.studio.progress import Narrator

    narrator = Narrator()

    def on_event(event: dict) -> None:
        lines = [f"· {_format_event(event)}"] if getattr(args, "verbose", False) else narrator.feed(event)
        for line in lines:
            print(f"  {line}", file=sys.stderr)

    studio = Studio(workspace=ws, on_event=on_event)

    changes: dict[str, Any] = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        changes[key.strip()] = value.strip()
    if args.return_period is not None:
        changes["return_period"] = args.return_period
    if args.years is not None:
        changes["years"] = args.years if args.years > 0 else None
    if args.exclude_years is not None:
        changes["exclude_years"] = args.exclude_years
    if args.estimator:
        changes["estimator"] = args.estimator
    did = False
    if changes:
        print("  Revising: " + ", ".join(f"{k} = {v}" for k, v in changes.items()), file=sys.stderr)
        reply = studio.revise(changes, by=args.by, note=args.note)
        if reply.payload.get("errors"):
            logger.error("%s", reply.text)
            sys.exit(1)
        rev = reply.payload.get("revision") or {}
        print(f"  Revision {rev.get('rev')}: {rev.get('description')}")
        did = True
    for item in args.comment or []:
        c = desk.comment(ws, item, section=args.section or "", author=args.by or "")
        print(f"  Comment {c['id']} recorded.")
        did = True
    for item in args.resolve or []:
        cid, _, response = item.partition("=")
        try:
            desk.resolve(ws, cid.strip(), response.strip(), by=args.by or "")
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)
        print(f"  Comment {cid.strip()} resolved.")
        did = True
    for item in args.sign or []:
        role, _, name = item.partition("=")
        role = role.strip().lower()
        role = role if role.endswith("_by") else f"{role}_by"
        reply = studio.sign(role, name.strip())
        if reply.payload.get("errors"):
            logger.error("%s", reply.text)
            sys.exit(1)
        print(f"  {desk.ROLE_WORDS.get(role, role)} by {name.strip()}; status "
              f"{(ws.house_style or {}).get('status')}.")
        did = True
    if (args.comment or args.resolve) and not (changes or args.sign):
        studio._build_deliverables()

    if did:
        studio.export(out_dir)
        (out_dir / "workspace.json").write_text(ws.to_json(), encoding="utf-8")
        print(f"  Documents rewritten in {out_dir}")

    print()
    from aquascope.studio.document.text import num, value

    levers = desk.levers(ws)
    if levers:
        print("Levers (change with --set ID=VALUE, or the shortcuts in --help):")
        for lv in levers:
            current = lv["value"]
            if lv["id"] == "estimator":
                current = (lv.get("labels") or {}).get(current, current)
            if isinstance(current, list):
                current = ", ".join(str(x) for x in current) or "none"
            print(f"  {lv['id']:<15} {lv['label']}: {current if current not in (None, '') else 'default'}")
        cand = next((lv.get("candidates") for lv in levers if lv["id"] == "exclude_years"), None)
        if cand:
            print("  largest floods: " + ", ".join(f"{c['year']} ({num(c['value'])})" for c in cand[:6]))
    rows = desk.sensitivity(ws)
    if len(rows) > 1:
        unit = next(((r.get("result") or {}).get("unit") for r in (ws.run or {}).get("results") or []
                     if isinstance(r.get("result"), dict) and (r["result"]).get("ffa")), None)
        print("\nSensitivity of the design value:")
        for r in rows:
            ch = "" if r is rows[0] or r.get("change_pct") is None else f"{r['change_pct']:+.0f} %"
            print(f"  {r['case']:<52} {value(r['value'], unit):>12} {ch:>6}")
    revs = (ws.desk or {}).get("revisions") or []
    if revs:
        print("\nRevisions:")
        for r in revs:
            print(f"  {r['rev']:<3} {str(r.get('at'))[:10]}  {r.get('description')}" +
                  (f" ({r['by']})" if r.get("by") else ""))
    open_comments = [c for c in (ws.desk or {}).get("comments") or [] if c.get("status") != "resolved"]
    if open_comments:
        print("\nOpen review comments:")
        for c in open_comments:
            print(f"  {c['id']}: {c['text']}")


def cmd_studio_showcase(args: argparse.Namespace) -> None:
    """`aquascope studio-showcase record | list`: the recorded studies the Explorer replays keyless."""
    from aquascope.studio import showcase

    if getattr(args, "showcase_cmd", None) == "list":
        print(showcase.diagnose(args.out))
        return
    only = [s for s in (args.only or "").split(",") if s.strip()] or None
    say = (lambda m: None) if args.quiet else (lambda m: print(m, flush=True))
    written = showcase.record(
        out_dir=args.out,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        max_usd=args.max_usd,
        fresh_for_days=args.refresh_after,
        only=only,
        on_event=say,
    )
    ok = sum(1 for m in written if m.get("status") in ("done", "declined") and not m.get("error"))
    print(f"recorded {ok}/{len(written)} this run, {sum(float(m.get('usd') or 0) for m in written):.2f} USD")
    print(showcase.diagnose(args.out))
    if written and ok == 0:
        sys.exit(1)


def cmd_update(args: argparse.Namespace) -> None:
    """`aquascope update [--check] [--yes]`: upgrade to the newest release, the way this copy was installed."""
    import logging
    import shlex
    import subprocess

    from aquascope import updates

    logging.getLogger("httpx").setLevel(logging.WARNING)
    info = updates.check()
    shown = shlex.join(info["command"]) if info["command"] else None
    print(f"  aquascope {info['installed']} ({info['kind']} install at {info['location']})")
    if info["kind"] == "editable":
        print("  This is a development checkout: update it with `git pull` in the clone (pip would not touch it).")
        if info["latest"] and info["newer"]:
            print(f"  The newest release on PyPI is {info['latest']}.")
        return
    if info["latest"] is None:
        print(f"  {info['error']}", file=sys.stderr)
        sys.exit(1)
    if not info["newer"]:
        print(f"  Up to date: {info['latest']} is the newest release.")
        return
    print(f"  {info['installed']} -> {info['latest']} available. Upgrade command: {shown}")
    if info["error"]:
        print(f"  {info['error']}", file=sys.stderr)
        sys.exit(1)
    if args.check:
        return
    if not args.yes:
        if not sys.stdin.isatty():
            print("  Not a terminal: rerun with --yes to upgrade, or run the command above yourself.", file=sys.stderr)
            sys.exit(1)
        if input("  Upgrade now? [y/N] ").strip().lower() not in ("y", "yes"):
            print("  Not upgraded.")
            return
    code = subprocess.run(info["command"], check=False).returncode
    if code:
        print(f"  The upgrade command exited with {code}.", file=sys.stderr)
        sys.exit(code)
    print("  Done. `aquascope --version` shows the version now installed.")


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f} %"


def _secs(x: float | None) -> str:
    return "-" if x is None else f"{x:,.0f} s"


def _print_scorecard(c: dict) -> None:
    head = " · ".join(str(v) for v in (
        c["study"], c["playbook"] + (f" ({c['branch']})" if c.get("branch") else "") if c.get("playbook") else None,
        c.get("model") or "keyless", f"aquascope {c['aquascope_version']}" if c.get("aquascope_version") else None,
        c.get("date")) if v)
    print(f"  {head}")
    if c.get("question"):
        print(f"  question  {c['question']}")
    o, r, k, rep, cost = c["outcome"], c["run"], c["critic"], c["report"], c["cost"]
    print(f"  outcome   {o['status']}, grade {o.get('grade') or '-'}" + (f", declined: {o['declined']}"
                                                                          if o.get("declined") else ""))
    if o.get("headline"):
        print(f"            {o['headline']}")
    g = r["gates"]
    print(f"  run       {r['ok']} of {r['planned']} steps ok, {r['failed']} failed, {r['skipped']} skipped · gates "
          f"{g['passed']} passed, {g['failed']} failed, {g['skipped']} skipped · {r['fallbacks']} fallback(s), "
          f"{r['replans']} replan(s)")
    print(f"  critic    {k['passed']} of {k['total']} checks passed · {k['issues']} issue(s) · "
          f"{k['not_established']} not established")
    dims = ", ".join(f"{d.replace('_', ' ')} {('-' if v is None else f'{v:.2f}')}"
                     for d, v in (rep.get("dimensions") or {}).items())
    mean = "-" if rep.get("mean") is None else f"{rep['mean']:.2f}"
    print(f"  report    mean {mean} · {dims or rep.get('error', '')}"
          + (f" · reference {rep['reference']:.2f}" if rep.get("reference") is not None else "")
          + (f" · {rep['numbers_without_evidence']} number(s) without evidence"
             if rep.get("numbers_without_evidence") else ""))
    if c.get("plan"):
        pl = c["plan"]
        print(f"  plan      vs {pl['case']}: {pl['score']:.2f} (tools {_pct(pl['tools'])}, methods "
              f"{_pct(pl['methods'])}, gates {_pct(pl['gates'])}, extraneous {_pct(pl['extraneous'])}, "
              f"forbidden {pl['forbidden']})")
        for line in pl["explain"][:6]:
            print(f"              {line}")
    slow = cost.get("slowest_phase")
    step = cost.get("slowest_step")
    where = "; ".join(x for x in (
        slow and f"slowest phase {slow['phase']} {_secs(slow['seconds'])}",
        step and f"slowest step {step['id']} {step['tool']} {_secs(step['seconds'])}") if x)
    model = (f"{cost['calls']} model call(s), {cost['prompt_tokens'] + cost['completion_tokens']:,} tokens"
             + (f", {cost['usd']:.2f} USD" if cost.get("usd") is not None else "")) if cost["calls"] else "keyless"
    print(f"  cost      {_secs(cost['seconds'])}" + (f" ({where})" if where else "") + f" · {model}")


def _print_trace(t: dict, *, events: bool) -> None:
    print(f"  {t['study']} · {_secs(t['seconds'])}" + (f" · {t['question']}" if t.get("question") else ""))
    print("  phases    " + " · ".join(
        f"{p['phase']} {_secs(p['seconds'])}" + (" (includes waiting for you)" if p["phase"] == "review" else "")
        for p in t["phases"]))
    print("  steps")
    for s in t["steps"]:
        g = s["gates"]
        state = "skipped" if s["skipped"] else ("ok" if s["ok"] else "failed")
        extra = " (fallback ran)" if s["fallback_used"] else ""
        print(f"    {s['id']:<4} {str(s['tool']):<24} {_secs(s['seconds']):>7}  {state}{extra} · gates "
              f"{g['passed']} passed, {g['failed']} failed, {g['skipped']} skipped")
        for n in s["not_passed"]:
            detail = n["detail"][:140] + ("…" if len(n["detail"]) > 140 else "")
            print(f"           {n['state']} {n['check']}: {detail}")
        if s.get("error"):
            print(f"           error: {str(s['error'])[:140]}")
    if t["model_calls"]:
        print("  model")
        for role, row in t["model_calls"].items():
            tok = int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)
            usd = f", {row['cost_usd']:.3f} USD" if row.get("cost_usd") is not None else ""
            print(f"    {role:<14} {row.get('calls', 0)} call(s), {tok:,} tokens{usd}")
    else:
        print("  model     keyless (no model calls)")
    if events:
        print("  events")
        for e in t["events"]:
            at = "" if e["t"] is None else f"+{e['t']:.0f}s"
            step = f" {e['step']}" if e.get("step") else ""
            print(f"    {at:>6} {e['role']}{step} {e['event']}: {e['detail'][:150]}")


def _print_stats(result: dict) -> None:
    key = result["by"] if result["by"] != "none" else "group"
    print(f"  {result['studies']} studies, by {result['by']}")
    print()
    print(f"| {key} | n | grades | report | plan | gates passed / failed / skipped | critic | median time | "
          f"tokens | USD | USD per study |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in result["rows"]:
        grades = ", ".join(f"{g} {n}" for g, n in sorted(r["grades"].items()))
        fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
        print(f"| {r[key]} | {r['n']} | {grades} | {fmt(r['report_mean'])} | {fmt(r['plan_mean'])} | "
              f"{_pct(r['gates_passed'])} / {_pct(r['gates_failed'])} / {_pct(r['gates_skipped'])} | "
              f"{_pct(r['critic_passed'])} | {_secs(r['median_seconds'])} | {r['tokens']:,} | "
              f"{fmt(r['usd'])} | {fmt(r['usd_per_study'])} |")
    probs = result["gate_problems"]
    for kind in ("failed", "skipped"):
        if probs[kind]:
            print(f"\n  most often {kind}: " + ", ".join(
                f"{p['check']} ({p['studies']} {'study' if p['studies'] == 1 else 'studies'})" for p in probs[kind]))


def cmd_eval(args: argparse.Namespace) -> None:
    """`aquascope eval score | trace | stats`: evaluate finished studies from their bundles, no model."""
    from aquascope import evaluation as ev

    if args.eval_cmd == "score":
        cards = [ev.score_path(p, plan_case=args.case) for p in args.study]
        if args.json:
            print(json.dumps(cards if len(cards) > 1 else cards[0], indent=2, default=str))
            return
        for i, c in enumerate(cards):
            if i:
                print()
            _print_scorecard(c)
        return
    if args.eval_cmd == "trace":
        ws, meta, _d = ev.load_study(args.study)
        t = ev.trace(ws, meta)
        if args.json:
            print(json.dumps(t, indent=2, default=str))
            return
        _print_trace(t, events=args.events)
        return
    if args.eval_cmd == "stats":
        dirs = ev.find_studies(args.paths)
        if not dirs:
            print(f"  no studies (no workspace.json) under {', '.join(args.paths)}", file=sys.stderr)
            sys.exit(1)
        cards = []
        for d in dirs:
            try:
                cards.append(ev.score_path(d, plan_case=args.case))
            except Exception as exc:  # one broken bundle should not hide the others
                print(f"  skipped {d}: {exc}", file=sys.stderr)
        result = ev.stats(cards, by=args.by, top=args.top)
        if args.csv:
            Path(args.csv).write_text(ev.stats_rows_csv(result), encoding="utf-8")
            print(f"  -> {args.csv}")
        if args.json:
            print(json.dumps({**result, "cards": cards if args.cards else None}, indent=2, default=str))
            return
        _print_stats(result)
        return
    print("usage: aquascope eval {score,trace,stats} ... (see aquascope eval --help)", file=sys.stderr)
    sys.exit(2)


def cmd_forecast(args: argparse.Namespace) -> None:
    """Run a predictive model on a time-series data file."""
    import pandas as pd

    from aquascope.models import get_model_map

    model_map = get_model_map()
    if args.model not in model_map:
        print(f"Unknown model '{args.model}'. Available: {list(model_map.keys())}")
        sys.exit(1)

    df = _load_dataframe(args.file)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    if "value" not in df.columns:
        # Use first numeric column
        numeric_cols = df.select_dtypes("number").columns
        if numeric_cols.empty:
            print("No numeric column found in data")
            sys.exit(1)
        df = df.rename(columns={numeric_cols[0]: "value"})

    df = df[["value"]].sort_index().dropna()

    model = model_map[args.model]()
    model.fit(df)
    forecast = model.predict(horizon=args.days)
    metrics = model.evaluate(df)

    print(f"\n{'=' * 70}")
    print(f"  AquaScope — Forecast ({args.model}, {args.days} days)")
    print(f"{'=' * 70}\n")
    print(forecast.to_string())
    print("\n  Metrics on training data:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")
    print()


def cmd_plot(args: argparse.Namespace) -> None:
    """Visualise data or analysis results."""
    import pandas as pd

    from aquascope.viz import (
        plot_boxplot,
        plot_fdc,
        plot_forecast,
        plot_heatmap,
        plot_timeseries,
    )

    df = pd.read_csv(args.file, index_col=0, parse_dates=True)

    plot_fn_map = {
        "timeseries": lambda: plot_timeseries(df, title=args.title or "Time Series", save_path=args.output),
        "forecast": lambda: plot_forecast(forecast=df, title=args.title or "Forecast", save_path=args.output),
        "boxplot": lambda: plot_boxplot(df, title=args.title or "Box Plot", save_path=args.output),
        "heatmap": lambda: plot_heatmap(df, title=args.title or "Correlation Heatmap", save_path=args.output),
        "fdc": lambda: plot_fdc(df.iloc[:, 0], title=args.title or "Flow Duration Curve", save_path=args.output),
    }

    fn = plot_fn_map.get(args.type)
    if fn:
        fn()
        if args.output:
            print(f"  ✓ Plot saved to {args.output}")
        else:
            print("  ✓ Plot displayed")
    else:
        print(f"  ✗ Unknown plot type: {args.type}")


def cmd_alerts(args: argparse.Namespace) -> None:
    """Check water-quality data against regulatory thresholds."""
    from aquascope.alerts.checker import check_dataframe

    df = _load_dataframe(args.source)
    standards = args.standards if args.standards else None

    report = check_dataframe(
        df,
        value_col=args.value_col,
        param_col=args.param_col,
        standards=standards,
    )

    print(f"\n{'=' * 70}")
    print("  AquaScope — Threshold Alert Report")
    print(f"{'=' * 70}\n")
    print(f"  Total samples checked : {report.total_samples}")
    print(f"  Samples with alerts   : {report.samples_with_alerts}")
    print(f"  Standards used        : {', '.join(report.standards_used)}")
    print(f"  Parameters checked    : {', '.join(report.parameters_checked)}")
    print()
    print("  Alerts by severity:")
    for sev in ("critical", "warning", "info"):
        count = report.summary.get(sev, 0)
        print(f"    {sev:>8s} : {count}")
    print()

    if report.alerts:
        print("  Top alerts:")
        shown = sorted(report.alerts, key=lambda a: a.exceedance_ratio, reverse=True)[:20]
        for a in shown:
            print(f"    [{a.severity.upper():>8s}] {a.message}")
        print()

    if args.output:
        out_path = Path(args.output)
        out_data = {
            "total_samples": report.total_samples,
            "samples_with_alerts": report.samples_with_alerts,
            "standards_used": report.standards_used,
            "parameters_checked": report.parameters_checked,
            "summary": report.summary,
            "alerts": [
                {
                    "parameter": a.parameter,
                    "value": a.value,
                    "limit": a.threshold.limit,
                    "standard": a.threshold.standard,
                    "severity": a.severity,
                    "exceedance_ratio": a.exceedance_ratio,
                    "timestamp": a.timestamp.isoformat() if a.timestamp else None,
                    "station_id": a.station_id,
                    "message": a.message,
                }
                for a in report.alerts
            ],
        }
        out_path.write_text(json.dumps(out_data, indent=2, default=str), encoding="utf-8")
        print(f"  ✓ Report saved → {out_path}\n")


def cmd_hydro(args: argparse.Namespace) -> None:
    """Run hydrological analysis."""
    import pandas as pd

    df = pd.read_csv(args.file, index_col=0, parse_dates=True)
    q = df.iloc[:, 0]  # first column as discharge
    output_data: Any = None

    if args.analysis == "fdc":
        from aquascope.hydrology import flow_duration_curve

        result = flow_duration_curve(q)
        output_data = pd.DataFrame({"exceedance": result.exceedance, "discharge": result.discharge})
        print("\n  Flow Duration Curve Percentiles:")
        for pct, val in sorted(result.percentiles.items()):
            print(f"    Q{pct:g} = {val:.3f}")

    elif args.analysis == "baseflow":
        from aquascope.hydrology import eckhardt, lyne_hollick

        method = args.method or "lyne_hollick"
        if method == "eckhardt":
            result = eckhardt(q)
        else:
            result = lyne_hollick(q)
        output_data = result.df.assign(method=result.method, bfi=result.bfi)
        print(f"\n  Baseflow Separation ({result.method}):")
        print(f"    BFI = {result.bfi:.3f}")

    elif args.analysis == "recession":
        from aquascope.hydrology import recession_analysis

        result = recession_analysis(q)
        output_data = result
        print("\n  Recession Analysis:")
        print(f"    Segments found: {len(result.segments)}")
        print(f"    Recession constant: {result.recession_constant:.2f} days")
        print(f"    Half-life: {result.half_life_days:.2f} days")
        print(f"    R²: {result.r_squared:.4f}")

    elif args.analysis == "flood-freq":
        from aquascope.hydrology import fit_gev

        result = fit_gev(q)
        output_data = [
            {
                "return_period": return_period,
                "discharge": discharge,
                "ci_lower": result.confidence_intervals.get(return_period, (None, None))[0],
                "ci_upper": result.confidence_intervals.get(return_period, (None, None))[1],
            }
            for return_period, discharge in sorted(result.return_periods.items())
        ]
        print("\n  Flood Frequency Analysis (GEV):")
        for rp, val in sorted(result.return_periods.items()):
            ci = result.confidence_intervals.get(rp)
            ci_str = f"  [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else ""
            print(f"    {rp:>5d}-yr: {val:.1f}{ci_str}")

    elif args.analysis == "low-flow":
        from aquascope.hydrology import low_flow_stat

        n_day = args.n_day or 7
        return_period = args.return_period or 10
        val = low_flow_stat(q, n_day=n_day, return_period=return_period)
        output_data = {"n_day": n_day, "return_period": return_period, "discharge": val}
        print(f"\n  {n_day}Q{return_period} = {val:.3f}")

    if args.output:
        output_path = _write_output(output_data, args.output, args.format)
        print(f"  ✓ Results saved → {output_path}")
    print()


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the interactive Streamlit dashboard."""
    from aquascope.dashboard import launch

    logger.info("Launching AquaScope dashboard on %s:%d …", args.host, args.port)
    launch(port=args.port, host=args.host)


def cmd_agri(args: argparse.Namespace) -> None:
    """Dispatch agriculture workflows."""
    if args.agri_command == "plan":
        cmd_agri_plan(args)
    elif args.agri_command == "benchmark":
        cmd_agri_benchmark(args)
    elif args.agri_command == "productivity":
        cmd_agri_productivity(args)


def cmd_groundwater(args: argparse.Namespace) -> None:
    """Run groundwater analysis."""
    import numpy as np
    import pandas as pd

    analysis = args.analysis

    if analysis == "theis":
        from aquascope.groundwater.aquifer import theis_drawdown

        T = args.transmissivity or 500.0  # noqa: N806
        S = args.storativity or 0.001  # noqa: N806
        Q = args.pumping_rate or 1000.0  # noqa: N806
        r = args.distance or 100.0
        t = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10, 24, 48, 72])
        s = theis_drawdown(T, S, Q, r, t)
        print(f"\nTheis Drawdown (T={T}, S={S}, Q={Q}, r={r})")
        print(f"{'Time (days)':>12}  {'Drawdown (m)':>12}")
        for ti, si in zip(t, s):
            print(f"{ti:12.2f}  {si:12.4f}")
        return

    if analysis == "recharge-wtf":
        from aquascope.groundwater.recharge import water_table_fluctuation

        df = _load_dataframe(args.file)
        col = df.columns[0] if len(df.columns) == 1 else "water_level"
        levels = pd.Series(df[col].values, index=pd.to_datetime(df.index))
        result = water_table_fluctuation(levels, specific_yield=args.specific_yield)
        print(f"\nWTF Recharge Estimation (Sy={args.specific_yield})")
        print(f"  Recharge: {result.value_mm_per_year:.1f} mm/year")
        print(f"  Method: {result.method}")
        return

    df = _load_dataframe(args.file)
    col = df.columns[0] if len(df.columns) == 1 else "water_level"
    levels = pd.Series(df[col].values, index=pd.to_datetime(df.index))

    if analysis == "trend":
        from aquascope.groundwater.wells import trend_detection
        from aquascope.utils.formatting import format_p_value

        result = trend_detection(levels)
        print("\nWell Trend Analysis (Mann-Kendall)")
        print(f"  Trend: {result.trend}")
        print(f"  Slope: {result.slope:.6f} per time-step")
        print(f"  p-value: {format_p_value(result.p_value)}")
    elif analysis == "recession":
        from aquascope.groundwater.wells import recession_analysis

        result = recession_analysis(levels)
        print("\nRecession Analysis")
        print(f"  Events found: {result.n_events}")
        if result.time_constant is not None:
            print(f"  Mean time constant: {result.time_constant:.2f} days")
    elif analysis == "seasonal":
        from aquascope.groundwater.wells import seasonal_decomposition

        result = seasonal_decomposition(levels)
        print("\nSeasonal Decomposition")
        print(f"  Period: {result.period}")
        print(f"  Trend range: {result.trend.min():.3f} to {result.trend.max():.3f}")
    elif analysis == "hydrograph":
        from aquascope.groundwater.wells import well_hydrograph

        result = well_hydrograph(levels)
        print("\nWell Hydrograph Summary")
        print(f"  Mean level: {result.mean:.3f}")
        print(f"  Min: {result.min:.3f}, Max: {result.max:.3f}")
        print(f"  Std: {result.std:.3f}")


def cmd_climate(args: argparse.Namespace) -> None:
    """Run climate analysis."""
    import pandas as pd

    analysis = args.analysis

    if analysis == "downscale":
        if not args.obs_file or not args.gcm_hist_file or not args.gcm_future_file:
            logger.error("Downscaling requires --obs-file, --gcm-hist-file, and --gcm-future-file")
            sys.exit(1)
        obs_df = _load_dataframe(args.obs_file)
        hist_df = _load_dataframe(args.gcm_hist_file)
        fut_df = _load_dataframe(args.gcm_future_file)
        obs = pd.Series(obs_df.iloc[:, 0].values, index=pd.to_datetime(obs_df.index))
        hist = pd.Series(hist_df.iloc[:, 0].values, index=pd.to_datetime(hist_df.index))
        fut = pd.Series(fut_df.iloc[:, 0].values, index=pd.to_datetime(fut_df.index))
        from aquascope.api import climate_downscale

        result = climate_downscale(obs, hist, fut, method=args.method)
        print(f"\nDownscaled ({args.method}): mean={result.mean():.2f}, std={result.std():.2f}")
        if args.output:
            result.to_csv(args.output)
            print(f"Saved to {args.output}")

    elif analysis == "indices":
        if not args.file:
            logger.error("Climate indices require --file")
            sys.exit(1)
        df = _load_dataframe(args.file)
        series = pd.Series(df.iloc[:, 0].values, index=pd.to_datetime(df.index))
        from aquascope.api import climate_indices

        result = climate_indices(precip=series, index=args.index)
        print(f"\nClimate Index: {args.index}")
        print(f"  Result: {result}")

    elif analysis == "drought":
        if not args.file:
            logger.error("Drought analysis requires --file")
            sys.exit(1)
        df = _load_dataframe(args.file)
        series = pd.Series(df.iloc[:, 0].values, index=pd.to_datetime(df.index))
        from aquascope.climate.scenarios import drought_frequency

        result = drought_frequency(series)
        print("\nDrought Frequency Analysis")
        print(f"  Events: {result.n_events}")
        print(f"  Mean duration: {result.mean_duration:.1f} time-steps")
        print(f"  Max duration: {result.max_duration}")
        print(f"  Total deficit: {result.total_deficit:.1f}")

    elif analysis == "scenario":
        logger.info("Scenario comparison requires programmatic access — see aquascope.climate.scenarios")
        print("Use the Python API for scenario comparison:")
        print("  from aquascope.climate.scenarios import scenario_comparison")
        print("  result = scenario_comparison(scenarios_dict, baseline)")


def cmd_agri_plan(args: argparse.Namespace) -> None:
    """Plan irrigation demand from files or live Open-Meteo inputs."""
    from aquascope.agri import default_season_end_date, fetch_openmeteo_plan_inputs, plan_irrigation
    from aquascope.agri.planner import series_from_dataframe
    from aquascope.agri.water_balance import SoilProperties

    planting_date = date.fromisoformat(args.planting_date)

    eto_series = None
    precip_series = None

    if args.eto_file:
        eto_series = series_from_dataframe(
            _load_dataframe(args.eto_file),
            value_columns=("eto_mm", "value", "et0_fao_evapotranspiration"),
            parameter=args.eto_parameter,
        )

    if args.precip_file:
        precip_series = series_from_dataframe(
            _load_dataframe(args.precip_file),
            value_columns=("precipitation_sum", "value"),
            parameter=args.precip_parameter,
        )

    if eto_series is None or precip_series is None:
        if args.lat is None or args.lon is None:
            logger.error("Latitude and longitude are required when ET or precipitation files are not provided.")
            sys.exit(1)

        start_date = args.start_date or args.planting_date
        if args.end_date:
            end_date = args.end_date
        else:
            try:
                end_date = default_season_end_date(args.crop, planting_date).isoformat()
            except ValueError as exc:
                logger.error("%s", exc)
                sys.exit(1)

        fetched_eto, fetched_precip = fetch_openmeteo_plan_inputs(args.lat, args.lon, start_date, end_date)
        eto_series = eto_series if eto_series is not None else fetched_eto
        precip_series = precip_series if precip_series is not None else fetched_precip

    soil = SoilProperties(
        field_capacity=args.soil_fc,
        wilting_point=args.soil_wp,
        root_depth=args.root_depth,
    )
    plan = plan_irrigation(
        crop=args.crop,
        planting_date=planting_date,
        eto_series=eto_series,
        precip_series=precip_series,
        soil=soil,
        efficiency=args.efficiency,
        depletion_fraction=args.depletion_fraction,
        initial_depletion=args.initial_depletion,
    )

    print(f"\n{'=' * 70}")
    print("  AquaScope — Irrigation Plan")
    print(f"{'=' * 70}\n")
    print(f"  Crop                     : {plan.crop}")
    print(f"  Planting date            : {plan.planting_date.isoformat()}")
    print(f"  Season end               : {plan.season_end_date.isoformat()}")
    print(f"  Irrigation efficiency    : {plan.efficiency:.2f}")
    print(f"  Total ET0                : {plan.total_eto_mm:.2f} mm")
    print(f"  Total precipitation      : {plan.total_precipitation_mm:.2f} mm")
    print(f"  Effective rainfall       : {plan.total_effective_rain_mm:.2f} mm")
    print(f"  Total ETc                : {plan.total_etc_mm:.2f} mm")
    print(f"  Net irrigation demand    : {plan.total_net_irrigation_mm:.2f} mm")
    print(f"  Gross irrigation demand  : {plan.total_gross_irrigation_mm:.2f} mm")
    print(f"  Applied irrigation       : {plan.total_applied_irrigation_mm:.2f} mm")
    print(f"  Irrigation trigger days  : {plan.irrigation_trigger_days}")

    if args.output:
        output_path = _write_output(plan.to_dict(), args.output, args.format)
        print(f"\n  ✓ Full irrigation plan saved → {output_path}")


def cmd_agri_benchmark(args: argparse.Namespace) -> None:
    """Benchmark agricultural water metrics using AQUASTAT data."""
    from aquascope.agri import benchmark_aquastat

    countries = None
    if args.countries:
        countries = [country.strip() for country in args.countries.split(",") if country.strip()]

    result = benchmark_aquastat(
        _load_dataframe(args.aquastat_file),
        args.metric,
        year=args.year,
        countries=countries,
        latest_only=not args.all_years,
        top_n=args.top,
    )

    print(f"\n{'=' * 70}")
    print("  AquaScope — Agriculture Benchmark")
    print(f"{'=' * 70}\n")
    print(f"  Metric      : {result.metric_name}")
    print(f"  Unit        : {result.output_unit}")
    print(f"  Summary     : {result.summary}")
    print()
    print(result.table.to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"\n  ✓ Benchmark results saved → {out_path}")


def cmd_agri_productivity(args: argparse.Namespace) -> None:
    """Estimate water productivity from WaPOR outputs."""
    from aquascope.agri import estimate_wapor_productivity

    aquastat_countries = None
    if args.aquastat_countries:
        aquastat_countries = [country.strip() for country in args.aquastat_countries.split(",") if country.strip()]

    aquastat_metrics = None
    if args.aquastat_metrics:
        aquastat_metrics = [metric.strip() for metric in args.aquastat_metrics.split(",") if metric.strip()]

    result = estimate_wapor_productivity(
        metric_id=args.metric,
        aeti_df=_load_dataframe(args.aeti_file) if args.aeti_file else None,
        npp_df=_load_dataframe(args.npp_file) if args.npp_file else None,
        ret_df=_load_dataframe(args.ret_file) if args.ret_file else None,
        aquastat_df=_load_dataframe(args.aquastat_file) if args.aquastat_file else None,
        aquastat_metrics=aquastat_metrics,
        aquastat_year=args.aquastat_year,
        aquastat_countries=aquastat_countries,
        aquastat_top_n=args.aquastat_top,
    )

    print(f"\n{'=' * 70}")
    print("  AquaScope — WaPOR Productivity")
    print(f"{'=' * 70}\n")
    print(f"  Metric          : {result.metric_name}")
    print(f"  Unit            : {result.output_unit}")
    print(f"  Aggregate value : {result.aggregate_value:.4f}")
    print(f"  Summary         : {result.summary}")
    print()
    print(result.table.to_string(index=False))

    if result.aquastat_context:
        print(f"\n{'-' * 70}")
        print("  AQUASTAT Context")
        print(f"{'-' * 70}\n")
        for context in result.aquastat_context:
            print(f"  Metric  : {context.metric_name}")
            print(f"  Unit    : {context.output_unit}")
            print(f"  Summary : {context.summary}")
            print()
            print(context.table.to_string(index=False))
            print()

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"\n  ✓ Productivity results saved → {out_path}")


def _add_desk_args(p: argparse.ArgumentParser, *, studio: bool = False) -> None:
    """The Study Desk's options (aquascope.studio.desk), on `aquascope studio` and its `aquascope desk` alias."""
    p.add_argument("--set", action="append", default=[], metavar="LEVER=VALUE",
                   help="Revise a finished study: change a lever (return_period, years, exclude_years, estimator)")
    p.add_argument("--return-period", type=float, default=None,
                   help="The design return period: for a new study its intake, for a finished one a revision")
    p.add_argument("--years", type=int, default=None, help="Revise: use only the last N years (0: the full record)")
    p.add_argument("--exclude-years", default=None, metavar="YEARS",
                   help="Revise: leave these years' floods out of the fit, e.g. 2008,1936 (empty string clears)")
    p.add_argument("--estimator", default=None, choices=["gev_lmoments", "lp3", "gev_bootstrap"],
                   help="Revise: the distribution the answer quotes (every fit stays in the tables)")
    p.add_argument("--sign", action="append", default=[], metavar="ROLE=NAME",
                   help="Sign as prepared, checked or approved, e.g. --sign checked=\"A. Hydrologist\"")
    p.add_argument("--comment", action="append", default=[], metavar="TEXT", help="Add a review comment")
    p.add_argument("--section", default=None, help="The section a --comment is about")
    p.add_argument("--resolve", action="append", default=[], metavar="ID=RESPONSE",
                   help="Close a review comment with the response, e.g. --resolve c1=\"Done in rev C\"")
    p.add_argument("--by", default=None, help="Who is making the change, comment or signature (for the record)")
    p.add_argument("--note", default=None, help="A revision's description, instead of the generated one")
    if not studio:
        p.add_argument("--out", "-o", default=None, help="Where the documents go (default: beside the workspace)")
        p.add_argument("--verbose", "-v", action="store_true", help="Show every event of a rerun")


def main() -> None:
    from aquascope import __version__
    from aquascope.registry import source_keys
    from aquascope.schemas.station import VARIABLES

    parser = argparse.ArgumentParser(description="AquaScope — Water data collection, analysis & AI research recomm...")
    parser.add_argument("--version", action="version", version=f"aquascope {__version__}")
    sub = parser.add_subparsers(dest="command")

    # — collect ——————————————————————————————
    p_collect = sub.add_parser("collect", help="Collect water data from an API source")
    p_collect.add_argument(
        "--source",
        required=True,
        choices=source_keys(),
        help="Data source to collect from",
    )
    p_collect.add_argument("--api-key", default=None, help="API key (if required)")
    p_collect.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days (USGS/UKEA/PEGELONLINE/BOM/South Africa DWS; PEGELONLINE max: 31)",
    )
    p_collect.add_argument(
        "--parameter-type",
        default=None,
        help='BOM parameter type, e.g. "Water Course Discharge", "Water Course Level" (BOM). '
        'Defaults to "Water Course Discharge".',
    )
    p_collect.add_argument("--max-stations", type=int, default=None, help="Cap stations to fetch (Ireland OPW)")
    p_collect.add_argument("--country", default="all", help="ISO3 country code or 'all' (AQUASTAT)")
    p_collect.add_argument("--countries", default=None, help="ISO3 country codes, comma-separated (SDG6)")
    p_collect.add_argument("--state", default=None, help="US state code e.g. US:06 (WQP)")
    p_collect.add_argument("--collection", default=None, choices=["15min", "daily"], help="Collection period (UKEA)")
    p_collect.add_argument("--station-wiski-id", default=None, help="Station Wiski ID (UKEA)")
    p_collect.add_argument("--observed-property", default=None, help="Observed property (UKEA)")
    p_collect.add_argument("--measure", default=None, help="Measure identifier (UKEA)")
    p_collect.add_argument("--variables", default=None, help="Comma-separated variable IDs (AQUASTAT)")
    p_collect.add_argument(
        "--bbox",
        default=None,
        help="Bounding box west,south,east,north (WaPOR), or min_lon, min_lat, max_lon, max_lat (USGS/UKEA)",
    )
    p_collect.add_argument(
        "--mode", default=None, help="Collector mode (openmeteo: weather/forecast/flood; grdc: in_situ/satellite)"
    )
    p_collect.add_argument(
        "--variable",
        default=None,
        help="Variable code (WaPOR), or discharge/water_level (South Africa DWS)",
    )
    p_collect.add_argument("--lid", default=None, help="A unique 5-character alphanumeric code e.g. ANAW1 (NOAA_NWPS)")
    p_collect.add_argument("--lat", type=float, default=None, help="Latitude (openmeteo/copernicus)")
    p_collect.add_argument("--lon", type=float, default=None, help="Longitude (openmeteo/copernicus)")
    p_collect.add_argument(
        "--start-date",
        default=None,
        help="Start date YYYY-MM-DD (openmeteo/copernicus/UKEA/BOM/South Africa DWS)",
    )
    p_collect.add_argument(
        "--end-date",
        default=None,
        help="End date YYYY-MM-DD (openmeteo/copernicus/UKEA/BOM/South Africa DWS)",
    )
    p_collect.add_argument("--start-year", type=int, default=2000, help="Start year (AQUASTAT)")
    p_collect.add_argument("--end-year", type=int, default=2023, help="End year (AQUASTAT)")
    p_collect.add_argument("--format", default="json", choices=["json", "csv", "geojson"], help="Output format")
    p_collect.add_argument("--year", type=int, default=None, help="Year filter (EU WFD)")
    p_collect.add_argument(
        "--station-ids", default=None, help="Comma-separated gauge codes to filter (camels_cl, camels_br, brazil_ana)"
    )
    p_collect.add_argument(
        "--station",
        default=None,
        help="Station UUID/SUID (PEGELONLINE/UKEA), AWRC number (BOM), or DWS gauge code",
    )
    p_collect.add_argument("--station-id", default=None, help="USGS monitoring station identifier")
    p_collect.add_argument("--parameter", default=None, help="USGS parameter code, e.g. 00060 for discharge")
    p_collect.add_argument("--state-code", default=None, help="USGS state code filter, e.g. MD")
    p_collect.add_argument("--county-code", default=None, help="USGS county code filter")
    p_collect.add_argument("--huc", default=None, help="USGS hydrologic unit code filter")
    p_collect.add_argument(
        "--timeseries",
        default=None,
        choices=["W", "Q"],
        help="PEGELONLINE timeseries: W for water level or Q for discharge (default: both)",
    )
    p_collect.add_argument(
        "--water-body-type",
        default=None,
        choices=["river", "lake", "groundwater"],
        help="Water body type (EU WFD)",
    )

    # ── recommend ────────────────────────────────────────────────────
    p_rec = sub.add_parser("recommend", help="Get AI methodology recommendations")
    p_rec.add_argument("--parameters", default="", help="Comma-separated water quality parameters")
    p_rec.add_argument("--goal", default="", help="Research goal (free text)")
    p_rec.add_argument("--keywords", default="", help="Comma-separated keywords")
    p_rec.add_argument("--scope", default="Taiwan", help="Geographic scope")
    p_rec.add_argument("--n-records", type=int, default=0, help="Number of data records")
    p_rec.add_argument("--n-stations", type=int, default=0, help="Number of monitoring stations")
    p_rec.add_argument("--years", type=float, default=0.0, help="Time span in years")
    p_rec.add_argument("--from-file", default=None, help="Path to a collected JSON data file")
    p_rec.add_argument("--top-k", type=int, default=5, help="Number of recommendations")
    p_rec.add_argument("--use-llm", action="store_true", help="Use LLM for enhanced recommendations")
    p_rec.add_argument("--model", default=None, help="LLM model name (default: gpt-4o-mini)")
    p_rec.add_argument("--llm-api-key", default=None, help="OpenAI-compatible API key")
    p_rec.add_argument("--llm-base-url", default=None, help="Custom LLM base URL (e.g. Ollama)")
    p_rec.add_argument("-o", "--output", default=None, help="Write recommendations to a JSON or CSV file")
    p_rec.add_argument("--format", choices=["json", "csv"], default=None, help="Output format (inferred from suffix)")

    # ── eda ──────────────────────────────────────────────────────────
    p_eda = sub.add_parser("eda", help="Run exploratory data analysis on a data file")
    p_eda.add_argument("--file", required=True, help="Path to JSON or CSV data file")
    p_eda.add_argument("--recommend", action="store_true", help="Also run AI recommendations based on EDA profile")
    p_eda.add_argument("--top-k", type=int, default=5, help="Number of recommendations")

    # ── quality ──────────────────────────────────────────────────────
    p_quality = sub.add_parser("quality", help="Assess data quality and optionally fix issues")
    p_quality.add_argument("--file", required=True, help="Path to JSON or CSV data file")
    p_quality.add_argument("--fix", action="store_true", help="Apply recommended preprocessing and save cleaned file")

    # ── run ───────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Run a study file (the steps behind an answer, reproducibly), or a methodology pipeline on data",
    )
    p_run.add_argument(
        "study", nargs="?", default=None, help="A study.yaml from `aquascope ask --study` or written by hand (#54)"
    )
    p_run.add_argument("--method", default=None, help="Pipeline method ID (use list-methods to see available)")
    p_run.add_argument("--file", default=None, help="Path to JSON or CSV data file")
    p_run.add_argument("--config", default=None, help="Pipeline config as JSON string")
    p_run.add_argument("--output", default=None, help="Path to save results JSON")
    p_run.add_argument("--out", "-o", default=None, help="Study: write report.md, manifest.json and results.json here")
    p_run.add_argument("--dry-run", action="store_true", help="Study: list the steps without running them")
    p_run.add_argument("--quiet", "-q", action="store_true", help="Study: do not print steps as they run")

    # ── completion  ────────────────────────────────────────────────────
    p_completion = sub.add_parser("completion", help="Print shell tab-completion activation script")
    p_completion.add_argument("shell", choices=["bash", "zsh", "fish"], help="Shell to generate completion for")

    # ── list-methods ─────────────────────────────────────────────────
    sub.add_parser("list-methods", help="List all available research methodologies and pipelines")

    # ── list-sources ─────────────────────────────────────────────────
    sub.add_parser("list-sources", help="List all available data sources")

    # ── stations ─────────────────────────────────────────────────────
    p_stations = sub.add_parser("stations", help="Search station catalogs across sources")
    p_stations.add_argument(
        "--source",
        action="append",
        choices=source_keys(),
        help="Source to search (repeatable). Default: every source with a station catalog",
    )
    p_stations.add_argument(
        "--bbox",
        default=None,
        help="Bounding box west,south,east,north (WGS84). Write --bbox=-77,38,-76,39 when it starts with a minus",
    )
    p_stations.add_argument(
        "--variable",
        default=None,
        choices=list(VARIABLES),
        help="Only stations measuring this variable",
    )
    p_stations.add_argument("--max-items", type=int, default=None, help="Cap per source")
    p_stations.add_argument("--api-key", default=None, help="API key for sources that take one")
    p_stations.add_argument("--format", choices=["json", "csv", "geojson"], default="geojson")
    p_stations.add_argument("--output", "-o", default=None, help="Output path (default: data/stations_<sources>.<ext>)")

    # ── harvest ──────────────────────────────────────────────────────
    p_harvest = sub.add_parser("harvest", help="Harvest catalogs into GeoParquet for the open archive (#188)")
    p_harvest.add_argument(
        "what",
        choices=["stations", "obs", "bundles", "signatures"],
        help="stations: the catalog; obs: daily series per station; bundles: one Parquet per "
        "variable and source rolled up from obs/; signatures: signatures.parquet from the mirrored discharge",
    )
    p_harvest.add_argument("--out", default="archive", help="Output folder (default: ./archive)")
    p_harvest.add_argument("--source", action="append", choices=source_keys(), help="Restrict to a source (repeatable)")
    p_harvest.add_argument("--max-items", type=int, default=None, help="stations: cap per source (for smoke tests)")
    p_harvest.add_argument(
        "--variable",
        default=None,
        dest="variable",
        help="obs: harvest only this variable (default: every harvestable variable per source)",
    )
    p_harvest.add_argument(
        "--variables",
        action="append",
        dest="variable_list",
        metavar="VAR",
        help="bundles: restrict to these variables (repeatable)",
    )
    p_harvest.add_argument(
        "--years",
        type=int,
        default=None,
        help="obs: cap the record asked for, in years (default: the full record, from the catalog's first date)",
    )
    p_harvest.add_argument("--max-stations", type=int, default=100, help="obs: stations per source per run")
    p_harvest.add_argument("--refresh-days", type=int, default=30, help="obs: re-harvest a station older than this")
    p_harvest.add_argument(
        "--max-seconds", type=float, default=None,
        help="obs: time budget per source and variable; stations not reached wait for the next run",
    )
    p_harvest.add_argument("--station", action="append", help="obs: only these station ids (repeatable)")
    p_harvest.add_argument(
        "--sync-from",
        default=None,
        metavar="REPO_ID",
        help="obs: download the existing obs/ tree from this dataset first (incremental runs)",
    )
    p_harvest.add_argument("--api-key", default=None)
    p_harvest.add_argument("--workers", type=int, default=4)
    p_harvest.add_argument("--no-geojson", action="store_true", help="Skip stations.geojson")
    p_harvest.add_argument(
        "--no-signatures", action="store_true", help="stations: skip rebuilding signatures.parquet from obs/"
    )
    p_harvest.add_argument(
        "--publish",
        default=None,
        metavar="REPO_ID",
        help="Upload the folder to this Hugging Face dataset (needs HF_TOKEN)",
    )

    # ── ask ──────────────────────────────────────────────────────────
    from aquascope.ai_engine.providers import provider_ids  # light: no pandas behind it

    p_ask = sub.add_parser("ask", help="Ask a water question in plain language; get a cited answer from real data")
    p_ask.add_argument("question")
    p_ask.add_argument("--provider", choices=provider_ids(), default=None)
    p_ask.add_argument("--model", default=None)
    p_ask.add_argument("--api-key", default=None)
    p_ask.add_argument("--base-url", default=None, help="Any OpenAI-compatible endpoint (or Anthropic's)")
    p_ask.add_argument("--max-steps", type=int, default=8, help="Tool-call rounds allowed (default 8)")
    p_ask.add_argument("--out", "-o", default=None, help="Save the Markdown report here")
    p_ask.add_argument("--quiet", "-q", action="store_true", help="Do not print tool calls as they happen")
    p_ask.add_argument(
        "--study", default=None, help="Write the steps behind the answer here, to re-run with `aquascope run`"
    )

    # ── ingest ───────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Map + QA any CSV/Excel export into a clean daily series with a report")
    p_ingest.add_argument("file")
    p_ingest.add_argument("--variable", default=None, choices=list(VARIABLES))
    p_ingest.add_argument("--date-column", default=None)
    p_ingest.add_argument("--value-column", default=None)
    p_ingest.add_argument("--unit", default=None, help="Unit of the value column (cfs, m3/s, l/s, mm, cm, ft, in)")
    p_ingest.add_argument("--station", default=None, help="Keep only this station id when the file holds several")
    p_ingest.add_argument("--sheet", default=None, help="Excel sheet name or index")
    p_ingest.add_argument("--describe", default=None, help="A sentence about the file (helps the LLM mapping)")
    p_ingest.add_argument("--llm", action="store_true", help="Let a configured LLM propose the column mapping")
    p_ingest.add_argument("--provider", choices=provider_ids(), default=None)
    p_ingest.add_argument("--model", default=None)
    p_ingest.add_argument("--api-key", default=None)
    p_ingest.add_argument("--out", "-o", default=None, help="Output stem (default: <file>_clean)")

    # ── mcp ──────────────────────────────────────────────────────────
    # ── layers (#522) ────────────────────────────────────────────────
    p_layers = sub.add_parser("layers", help="The dated map layers and their valid dates, or a time-lapse's frames")
    layers_sub = p_layers.add_subparsers(dest="layers_cmd", required=True)
    p_llist = layers_sub.add_parser("list", help="List the dated layers, their cadence and first and last day")
    p_llist.add_argument("--live", action="store_true", help="Read the exact intervals (gaps included) from GIBS")
    p_llist.add_argument("--json", action="store_true")
    p_lframes = layers_sub.add_parser("frames", help="The dates and tile URLs of a time-lapse of one layer")
    p_lframes.add_argument("layer", help="daily, precip, soil, snow, lst or storage")
    p_lframes.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_lframes.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_lframes.add_argument("--step", choices=["day", "week", "month"], default="day")
    p_lframes.add_argument("--max-frames", type=int, default=60)
    p_lframes.add_argument("--json", action="store_true")
    p_lstatus = layers_sub.add_parser("status", help="The world river status map for a month (GEOGLOWS HydroSOS, "
                                      "1990 on): its URL, legend, licence and the months that exist")
    p_lstatus.add_argument("month", nargs="?", default=None, help="YYYY-MM (default: the newest month)")
    p_lstatus.add_argument("--offline", action="store_true", help="Do not list the bucket; use the recorded range")
    p_lstatus.add_argument("--json", action="store_true")
    # ── map (#561) ───────────────────────────────────────────────────
    p_map = sub.add_parser("map", help="Read a plain-English request about the Explorer's map into map actions "
                           "(the keyless phrase grammar, or --llm with your own key)")
    p_map.add_argument("request", nargs="+", help='For example: "trace the Nile to the sea"')
    p_map.add_argument("--resolve", action="store_true", help="Look place names up in the gazetteer (Photon, OSM)")
    p_map.add_argument("--llm", action="store_true", help="Ask your own model instead of the rules")
    p_map.add_argument("--provider", choices=provider_ids(), default=None)
    p_map.add_argument("--model", default=None)
    p_map.add_argument("--api-key", default=None)
    p_map.add_argument("--today", default=None, help="Read dates as if today were YYYY-MM-DD")
    p_map.add_argument("--json", action="store_true")
    # ── basins ───────────────────────────────────────────────────────
    p_bul = sub.add_parser("bulletin", help="The month's state of the rivers: every Archive gauge against normal, "
                           "HydroSOS classes")
    p_bul.add_argument("month", nargs="?", default=None, help="YYYY-MM (default: the latest published, else the last full month)")
    p_bul.add_argument("--sources", nargs="+", default=None, help="Only these sources (usgs, uk_ea, ...)")
    p_bul.add_argument("--archive", default=None, help="A local copy of the Archive dataset instead of the Hub")
    p_bul.add_argument("--rebuild", action="store_true", help="Build it even when a bulletin is published")
    p_bul.add_argument("--top-up", type=int, default=0, help="Ask the agencies for up to N gauges' missing days")
    p_bul.add_argument("--workers", type=int, default=4)
    p_bul.add_argument("--out", default=None, help="Write bulletins/<month>/ (HTML, Markdown, map, status) here")
    p_bul.add_argument("--no-map", action="store_true", help="Leave the map out (no matplotlib needed)")
    p_bul.add_argument("--gauges", action="store_true", help="With --json, include every classed gauge")
    p_bul.add_argument("--json", action="store_true")
    p_warn = sub.add_parser("warnings", help="Floods ahead: river reaches the GEOGLOWS forecast expects to reach "
                            "their 2-year flow in the next 15 days (model output, not an official warning)")
    p_warn.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p_warn.add_argument("--min-rp", type=int, default=2, help="Only reaches at or above this return period (years)")
    p_warn.add_argument("--limit", type=int, default=20, help="Reaches listed (highest class first)")
    p_warn.add_argument("--local", default=None, help="Read an issue written by python -m aquascope.archive.warnings")
    p_warn.add_argument("--json", action="store_true")
    p_now = sub.add_parser("now", help="Today against normal and the next 15 days (GEOGLOWS, GloFAS), corrected to a gauge")
    p_now.add_argument("coords", nargs="*", type=float, metavar="LAT LON", help="A point, snapped to its river reach")
    p_now.add_argument("--station", default=None, metavar="SOURCE/ID", help="A gauge: its status, and the forecast "
                       "corrected to its record")
    p_now.add_argument("--river-id", type=int, default=None, help="A GEOGLOWS river reach")
    p_now.add_argument("--days", type=int, default=15, help="Forecast days (15)")
    p_now.add_argument("--date", default=None, help="The status on another day (YYYY-MM-DD)")
    p_now.add_argument("--raw", action="store_true", help="Do not correct the forecast to the gauge")
    p_now.add_argument("--status-only", action="store_true", help="Only today against normal, no forecast")
    p_now.add_argument("--quick", action="store_true",
                       help="Only the two forecasts: no thresholds or correction (skips the simulated record)")
    p_now.add_argument("--csv", default=None, help="Write the forecast to this CSV")
    p_now.add_argument("--json", action="store_true")
    p_watch = sub.add_parser("watch", help="What changed at watched gauges, reaches and areas since a date")
    p_watch.add_argument("ids", nargs="+", metavar="ID",
                         help="source/station_id, river:<reach id> or area:west,south,east,north")
    p_watch.add_argument("--since", default=None, help="Last look, YYYY-MM-DD (default: a week ago)")
    p_watch.add_argument("--threshold", action="append", metavar="ID=VALUE",
                         help="Per item: a value (usgs/USGS-01646500=300) or a return period (=10y); repeatable. "
                         "Without one, forecasts are checked against the 2-year flow")
    p_watch.add_argument("--forecast", choices=["auto", "archive", "live", "off"], default="auto",
                         help="Where the forecast comes from (auto: the forecast archive, else GEOGLOWS now)")
    p_watch.add_argument("--no-forecast", action="store_true", help="Skip the forecast")
    p_watch.add_argument("--no-floods", action="store_true", help="Skip flood events in the news")
    p_watch.add_argument("--no-refresh", action="store_true", help="Do not ask the agency for its newest days")
    p_watch.add_argument("--json", action="store_true")
    p_river = sub.add_parser("river", help="River reaches (GEOGLOWS v2): snap a point, the modelled record, the trace")
    river_sub = p_river.add_subparsers(dest="river_cmd", required=True)
    p_rsnap = river_sub.add_parser("snap", help="The river reach a point stands for (the main channel within the "
                                   "tolerance), or 'no stream within N m'")
    p_rsnap.add_argument("lat", type=float)
    p_rsnap.add_argument("lon", type=float)
    p_rsnap.add_argument("--max-distance", type=float, default=1000.0, help="Snap tolerance in metres (1000)")
    p_rsnap.add_argument("--nearest", action="store_true",
                         help="Take the nearest line rather than the main channel within the tolerance")
    p_rsnap.add_argument("--area", type=float, default=None,
                         help="A gauge's catchment area in km2: take the reach whose upstream area matches it")
    p_rsnap.add_argument("--json", action="store_true")
    for name, helptext in (("record", "86 years of simulated daily flow for a reach, analysed like a gauge"),
                           ("area", "The area draining to a reach"),
                           ("trace", "Follow a reach to its outlet: length, path, gauges, dams and countries"),
                           ("dams", "Dams upstream of a reach (Global Dam Watch) and the degree of regulation"),
                           ("upstream", "The river_ids of every reach that drains to a reach"),
                           ("downstream", "The river_ids from a reach to its outlet")):
        p_r = river_sub.add_parser(name, help=helptext)
        p_r.add_argument("river_id", nargs="?", type=int, default=None)
        p_r.add_argument("--at", nargs=2, type=float, metavar=("LAT", "LON"), help="Snap this point first")
        p_r.add_argument("--max-distance", type=float, default=1000.0, help="Snap tolerance in metres (1000)")
        p_r.add_argument("--json", action="store_true")
        if name == "record":
            p_r.add_argument("--years", type=int, default=None, help="Only the last N years")
            p_r.add_argument("--csv", default=None, help="Write the daily simulated series to this CSV")
        if name == "trace":
            p_r.add_argument("--gauge-km", type=float, default=2.0, help="List gauges this close to the path (2)")
            p_r.add_argument("--dam-km", type=float, default=2.0, help="List dams this close to the path (2)")
            p_r.add_argument("--geojson", default=None, help="Write the path to this GeoJSON file")
        if name in ("upstream", "downstream"):
            p_r.add_argument("--max", type=int, default=20_000 if name == "upstream" else 5_000,
                             help="At most this many ids (upstream keeps the largest drainage areas)")
        if name == "dams":
            p_r.add_argument("--no-flow", action="store_true",
                             help="Skip the mean-flow request (one ~10 s GEOGLOWS call) and the degree of regulation")

    p_ev = sub.add_parser("evidence", help="Model skill at a gauge: GEOGLOWS, GloFAS, NWM and GRRR graded A to D")
    ev_sub = p_ev.add_subparsers(dest="evidence_cmd", required=True)
    p_evs = ev_sub.add_parser("skill", help="Score every global model against one gauge's record")
    p_evs.add_argument("source", nargs="?", default=None, help="Station source (usgs, uk_ea, ...)")
    p_evs.add_argument("station_id", nargs="?", default=None)
    p_evs.add_argument("--at", nargs=2, type=float, metavar=("LAT", "LON"), help="The gauge position (with --csv)")
    p_evs.add_argument("--csv", default=None, help="Your own daily discharge record: date,value in m3/s")
    p_evs.add_argument("--area", type=float, default=None, help="The gauge's catchment area in km2")
    p_evs.add_argument("--models", nargs="+", choices=["geoglows", "glofas", "nwm", "grrr"], default=None)
    p_evs.add_argument("--years", type=int, default=30, help="Score the last N years of the record (30; 0 = all)")
    p_evs.add_argument("--json", action="store_true")
    p_evn = ev_sub.add_parser("near", help="Which model to lean on near a site, from the published skill")
    p_evn.add_argument("lat", type=float)
    p_evn.add_argument("lon", type=float)
    p_evn.add_argument("--radius-km", type=float, default=150.0)
    p_evn.add_argument("--json", action="store_true")
    p_evb = ev_sub.add_parser("build", help="The monthly skill table for the Archive gauges (CI)")
    p_evb.add_argument("--archive", required=True, help="A local copy of the dataset")
    p_evb.add_argument("--out", required=True)
    p_evb.add_argument("--max-gauges", type=int, default=4000)
    p_evb.add_argument("--nwm-years", type=int, default=10)
    p_evb.add_argument("--nwm-max-columns", type=int, default=24)
    p_evb.add_argument("--grrr-max-chunks", type=int, default=2000)
    p_evb.add_argument("--workers", type=int, default=8)
    p_evb.add_argument("--smoke", action="store_true", help="Six gauges, never published")
    p_evp = ev_sub.add_parser("publish", help="Upload the skill table to the Archive (skill/ only)")
    p_evp.add_argument("--out", required=True)
    p_evp.add_argument("--repo", default="Rekin226/aquascope-gauges")

    p_basins = sub.add_parser("basins", help="Catchments from BasinATLAS (HydroATLAS, CC BY 4.0) in the Archive")
    basins_sub = p_basins.add_subparsers(dest="basins_cmd", required=True)
    p_bat = basins_sub.add_parser("at", help="Describe the catchment upstream of a point")
    p_bat.add_argument("lat", type=float)
    p_bat.add_argument("lon", type=float)
    p_bat.add_argument("--local", action="store_true", help="Only the level-12 sub-basin containing the point")
    p_bat.add_argument("--json", action="store_true")
    p_bsim = basins_sub.add_parser(
        "similar", help="Gauged basins whose catchments most resemble a point's or a station's"
    )
    p_bsim.add_argument("lat", type=float, nargs="?", default=None)
    p_bsim.add_argument("lon", type=float, nargs="?", default=None)
    p_bsim.add_argument(
        "--station", default=None, metavar="SOURCE/ID", help="Use a station's own catchment as the target"
    )
    p_bsim.add_argument("--k", type=int, default=10)
    p_bsim.add_argument("--method", choices=["similarity", "proximity", "combined"], default="combined")
    p_bsim.add_argument("--source", action="append", help="Restrict donors to these sources (repeatable)")
    p_bsim.add_argument("--json", action="store_true")
    p_bassign = basins_sub.add_parser("assign", help="Build basins/station_catchments.parquet (harvest workflow step)")
    p_bassign.add_argument("--fgb", required=True, help="Local lev12.fgb")
    p_bassign.add_argument("--attributes", required=True, help="Local lev12_attributes.parquet")
    p_bassign.add_argument("--out", default="archive/basins/station_catchments.parquet")
    p_breg = basins_sub.add_parser("regionalize", help="Estimate the flow signatures of an ungauged point from donors")
    p_breg.add_argument("lat", type=float)
    p_breg.add_argument("lon", type=float)
    p_breg.add_argument("--k", type=int, default=10)
    p_breg.add_argument("--method", choices=["similarity", "regression", "both"], default="similarity")
    p_breg.add_argument("--json", action="store_true")
    p_bsig = basins_sub.add_parser(
        "signatures", help="Build basins/station_signatures.parquet from the discharge bundles"
    )
    p_bsig.add_argument("--archive", default="archive", help="Local archive folder holding obs/discharge/*.parquet")
    p_bsig.add_argument("--catchments", default=None, help="Local station_catchments.parquet (default: from the Hub)")
    p_bsig.add_argument("--out", default="archive/basins/station_signatures.parquet")
    p_bsig.add_argument("--min-years", type=float, default=10.0)
    p_bloo = basins_sub.add_parser(
        "loo", help="Leave-one-out regionalisation skill -> basins/regionalization_skill.json"
    )
    p_bloo.add_argument("--signatures", default=None, help="Local station_signatures.parquet (default: from the Hub)")
    p_bloo.add_argument("--catchments", default=None, help="Local station_catchments.parquet (default: from the Hub)")
    p_bloo.add_argument("--out", default="archive/basins/regionalization_skill.json")
    p_bloo.add_argument("--k", type=int, default=10)
    p_bloo.add_argument("--max-stations", type=int, default=3000, help="Even stride sample of donors (0 = all)")
    p_bup = basins_sub.add_parser("upstream", help="List the level-12 sub-basins upstream of a HYBAS_ID")
    p_bup.add_argument("hybas_id", type=int)
    p_bup.add_argument("--limit", type=int, default=200_000)
    p_bbuild = basins_sub.add_parser("build", help="Build the basins/ files from the BasinATLAS FileGDB")
    p_bbuild.add_argument("gdb", help="Path to BasinATLAS_v10.gdb")
    p_bbuild.add_argument("--out", default="archive")
    p_bbuild.add_argument("--max-features", type=int, default=None)
    p_bbuild.add_argument("--fgb", action="store_true", help="Also write lev12.fgb from Python (needs memory)")

    # ── gym (HydroGym) ───────────────────────────────────────────────
    from aquascope.methods import METHODS as _METHODS

    p_assess = sub.add_parser(
        "assess", help="What can be answered at a place: gauges in reach, catchment, which methods the record supports"
    )
    p_assess.add_argument("lat", type=float)
    p_assess.add_argument("lon", type=float, help="Longitude (a negative value is fine as a positional)")
    p_assess.add_argument(
        "--problem",
        choices=sorted({p for m in _METHODS.values() for p in m.problems}),
        default=None,
        help="Only the methods for this problem kind",
    )
    p_assess.add_argument("--radius-km", type=float, default=50.0, help="How far a gauge may be to count (default 50)")
    p_assess.add_argument("--return-period", type=float, default=None, help="The T (years) the question asks for")
    p_assess.add_argument("--json", action="store_true")

    p_ctx = sub.add_parser(
        "context",
        help="Flood history, surface water, flood hazard, dams, rain gauge, actual ET and soil at a place",
    )
    p_ctx.add_argument("lat", type=float, nargs="?", default=None)
    p_ctx.add_argument("lon", type=float, nargs="?", default=None,
                       help="Longitude (a negative value is fine as a positional)")
    p_ctx.add_argument("--bbox", default=None,
                       help="west,south,east,north instead of a point (write --bbox=-77,38,-76,39 when it starts "
                            "with a minus)")
    p_ctx.add_argument("--layers", default=None,
                       help="Comma-separated: flood_history, surface_water, flood_hazard, dams, rain_gauge, "
                            "actual_et, soil (default all)")
    p_ctx.add_argument("--floods-past", action="store_true",
                       help="Floods past (#547): flood events in the news and floods seen by radar by month, "
                            "worldwide or in --bbox; the latest 12 months on record unless --month or --from/--to")
    p_ctx.add_argument("--month", default=None, help="With --floods-past: one month, YYYY-MM")
    p_ctx.add_argument("--from", dest="start", default=None, help="With --floods-past: first month, YYYY-MM")
    p_ctx.add_argument("--to", dest="end", default=None,
                       help="With --floods-past: last month, YYYY-MM (at most 60 months)")
    p_ctx.add_argument("--limit", type=int, default=20,
                       help="With --floods-past: news events listed for a small box (default 20)")
    p_ctx.add_argument("--json", action="store_true")

    p_area = sub.add_parser(
        "area-study", help="Study the gauges of an area together: per-site floods, trend field, regional growth curve"
    )
    p_area.add_argument("--bbox", default=None,
                        help="west,south,east,north (write --bbox=-77,38,-76,39 when it starts with a minus)")
    p_area.add_argument("--station", action="append", default=None, help="source/station_id (repeatable)")
    p_area.add_argument("--question", default=None, help="A title for the study")
    p_area.add_argument("--max-sites", type=int, default=60, help="Gauges studied at most (default 60)")
    p_area.add_argument("--max-live", type=int, default=25,
                        help="Gauges fetched live from an agency at most; the rest come from the Archive (default 25)")
    p_area.add_argument("--output", "-o", default=None, help="Write .xlsx, .csv, .geojson or .json")
    p_area.add_argument("--json", action="store_true")

    p_gym = sub.add_parser("gym", help="HydroGym: a gym-style calibration environment over real basins (#175)")
    gym_sub = p_gym.add_subparsers(dest="gym_cmd", required=True)
    p_gb = gym_sub.add_parser("basins", help="Suggest gauged basins from the Archive that make good tasks")
    p_gb.add_argument("--n", type=int, default=10)
    p_gb.add_argument("--source", action="append", help="Restrict to these sources (repeatable)")
    p_gb.add_argument("--min-years", type=float, default=15.0)
    p_gb.add_argument("--allow-snow", action="store_true", help="Keep snowy catchments (GR4J has no snow routine)")
    p_gb.add_argument("--json", action="store_true")
    for name, help_ in (
        ("run", "Play one baseline agent on a basin"),
        ("leaderboard", "Play the baselines on one or more basins, one row per run"),
    ):
        p_g = gym_sub.add_parser(name, help=help_)
        p_g.add_argument("--basin", action="append", metavar="SOURCE/ID", help="Archive station (repeatable)")
        p_g.add_argument("--synthetic", action="store_true", help="Use synthetic GR4J basins (no network)")
        p_g.add_argument("--n-synthetic", type=int, default=1)
        p_g.add_argument(
            "--agent",
            action="append",
            choices=["random_search", "nelder_mead", "differential_evolution"],
            help="Baseline agent(s); default: differential_evolution for run, all three for leaderboard",
        )
        p_g.add_argument("--objective", choices=["nse", "kge", "log_nse"], default="nse")
        p_g.add_argument("--steps", type=int, default=30, help="Step budget per episode")
        p_g.add_argument("--seed", type=int, default=0)
        p_g.add_argument("--seeds", type=int, default=1, help="leaderboard: number of seeds per agent and basin")
        p_g.add_argument("--out", default=None, help="leaderboard: write the table (CSV; Markdown for bench results)")
        p_g.add_argument("--json", action="store_true")
    p_g_lb = gym_sub.choices["leaderboard"]
    p_g_lb.add_argument(
        "results",
        nargs="*",
        metavar="RESULTS.jsonl",
        help="Bench result files (Phase 1): render their leaderboard instead of playing the baselines",
    )
    p_g_lb.add_argument("--title", default=None, help="Heading of the Markdown leaderboard")
    p_gt = gym_sub.add_parser("tasks", help="Generate benchmark tasks from the playbooks on catalog sites (Phase 1)")
    p_gt.add_argument("--n", type=int, default=60, help="Number of tasks (default 60)")
    p_gt.add_argument("--seed", type=int, default=0)
    p_gt.add_argument("--source", action="append", help="Restrict sites to these sources (repeatable)")
    p_gt.add_argument("--playbook", action="append", help="Only these playbooks (repeatable; default all)")
    p_gt.add_argument(
        "--probes",
        default="1",
        help="Decline probes per site: an integer or 'all' (default 1, rotating over the rules)",
    )
    p_gt.add_argument("--ungauged-share", type=float, default=0.25, help="Share of sites that are bare points")
    p_gt.add_argument(
        "--no-check-land",
        action="store_true",
        help="Do not ask BasinATLAS whether a bare point is on land (offline; the gauge proxy still applies)",
    )
    p_gt.add_argument("--out", default="tasks.jsonl")
    p_gt.add_argument("--quiet", action="store_true")
    p_gp = gym_sub.add_parser("plans", help="The reference plans of the plan-quality benchmark (Phase 2)")
    gp_sub = p_gp.add_subparsers(dest="plans_cmd", required=True)
    for name, help_ in (
        ("list", "List the cases"),
        ("show", "Print one case"),
        ("validate", "Check every case against the catalogue, the registry and its recon"),
        ("score", "Score a plan (the tree's, or a JSON file) against one case"),
        ("rescore", "Score stored result rows again from the plans they carry (no model run)"),
    ):
        p_gpc = gp_sub.add_parser(name, help=help_)
        if name in ("show", "score"):
            p_gpc.add_argument("id", help="The case id")
        if name == "score":
            p_gpc.add_argument("--candidate", default=None, help="A plan JSON (a study, a workspace or a decline)")
        if name == "rescore":
            p_gpc.add_argument("results", nargs="+", metavar="RESULTS.jsonl", help="Result files, rewritten in place")
            p_gpc.add_argument("--out", default=None, help="Write the re-scored rows here instead (one file only)")
        p_gpc.add_argument("--plans", default=None, help="A folder of reference plans (default: the package's)")
        p_gpc.add_argument("--json", action="store_true")
    p_gr = gym_sub.add_parser(
        "reports", help="Score the report a finished study bundle carries, not just its plan (#382)"
    )
    gr_sub = p_gr.add_subparsers(dest="reports_cmd", required=True)
    p_gr_list = gr_sub.add_parser("list", help="List the report-quality reference cases")
    p_gr_show = gr_sub.add_parser("show", help="Print one reference case")
    p_gr_show.add_argument("id", help="The case id")
    p_gr_score = gr_sub.add_parser("score", help="Score one recorded study's workspace.json")
    p_gr_score.add_argument("study_dir", help="A recorded study's directory (workspace.json sits inside it)")
    p_gr_score.add_argument(
        "--reference",
        default=None,
        help="A reference case id to score against (default: the reference, if any, whose "
        "study matches the directory name)",
    )
    p_gr_bench = gr_sub.add_parser("bench", help="Score every recorded study under a directory")
    p_gr_bench.add_argument(
        "--dir",
        default=None,
        help="A directory of recorded studies (default: the Explorer's showcase, explorer/showcase/studies)",
    )
    p_gr_bench.add_argument("--study", action="append", help="Only these study ids (repeatable)")
    p_gr_bench.add_argument("--out", default=None, help="Append results as JSONL")
    p_gr_bench.add_argument("--quiet", action="store_true")
    for p_grc in (p_gr_list, p_gr_show, p_gr_score, p_gr_bench):
        p_grc.add_argument("--reports", default=None, help="A folder of report references (default: the package's)")
        p_grc.add_argument("--json", action="store_true")
    p_gbench = gym_sub.add_parser(
        "bench", help="Play an agent on the tasks (Phase 1) or on the reference plans (Phase 2) and score it"
    )
    p_gbench.add_argument("--tasks", default=None, help="Phase 1: tasks.jsonl from `gym tasks`")
    p_gbench.add_argument(
        "--agent",
        choices=["tree", "team", "ask", "methodologist", "file"],
        default="tree",
        help="Phase 1: tree, team, ask; Phase 2: tree, methodologist, file",
    )
    p_gbench.add_argument(
        "--plans",
        default=None,
        help="Phase 2: a folder of reference plans (default: the package's); with --agent "
        "methodologist or file, or without --tasks, the bench scores plan quality",
    )
    p_gbench.add_argument("--case", action="append", help="Phase 2: play these case ids only (repeatable)")
    p_gbench.add_argument(
        "--repeats", type=int, default=1, help="Phase 2: play every case this many times on a model (its spread)"
    )
    p_gbench.add_argument(
        "--candidates", default=None, help="Phase 2, --agent file: a folder of <case id>.json plans produced elsewhere"
    )
    p_gbench.add_argument(
        "--provider",
        default=None,
        help="LLM provider (anthropic, openai, groq, huggingface, ollama, ...); none: keyless team",
    )
    p_gbench.add_argument("--model", default=None)
    p_gbench.add_argument("--api-key", default=None)
    p_gbench.add_argument("--base-url", default=None)
    p_gbench.add_argument("--limit", type=int, default=None, help="Play the first N tasks")
    p_gbench.add_argument(
        "--unsolvable", type=int, default=None, help="With --limit: at most this many unsolvable tasks among the N"
    )
    p_gbench.add_argument("--task", action="append", help="Play these task ids only (repeatable)")
    p_gbench.add_argument(
        "--spread",
        action="store_true",
        help="With --limit: take tasks round robin over the sites rather than the first N",
    )
    p_gbench.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks --out already holds a finished row for (errors and timeouts are replayed)",
    )
    p_gbench.add_argument("--timeout", type=float, default=900.0, help="Seconds per task (0: none)")
    p_gbench.add_argument("--max-steps", type=int, default=8, help="ask: tool-call steps")
    p_gbench.add_argument("--context-chars", type=int, default=40_000, help="ask: conversation budget in characters")
    p_gbench.add_argument("--out", default=None, help="Append results as JSONL")
    p_gbench.add_argument("--json", action="store_true", help="Print the summary rows as JSON")
    p_gbench.add_argument("--quiet", action="store_true")

    # ── caravan ──────────────────────────────────────────────────────
    p_car = sub.add_parser(
        "caravan", help="Caravan-format sub-datasets (forcing + mm/day streamflow + attributes) from the Archive"
    )
    car_sub = p_car.add_subparsers(dest="caravan_cmd", required=True)
    p_cex = car_sub.add_parser("export", help="Export one source's discharge stations in the Caravan layout")
    p_cex.add_argument("--source", required=True, choices=["usgs", "uk_ea", "hubeau_hydrometrie"])
    p_cex.add_argument("--out", required=True, help="Output folder (Caravan tree is written inside it)")
    p_cex.add_argument("--station", action="append", help="Only these station ids (repeatable)")
    p_cex.add_argument("--max-stations", type=int, default=None, help="Cap (longest archived records first)")
    p_cex.add_argument("--min-years", type=float, default=10.0, help="Minimum streamflow record length (default 10)")
    p_cex.add_argument("--start", type=date.fromisoformat, default=None, help="Forcing start (default 1981-01-01)")
    p_cex.add_argument("--end", type=date.fromisoformat, default=None, help="Forcing end (default last observation)")
    p_cex.add_argument("--prefix", default=None, help="Sub-dataset prefix (default aquascope_<source>)")
    p_cex.add_argument("--no-forcing", action="store_true", help="Streamflow and attributes only, no Open-Meteo calls")
    p_cex.add_argument(
        "--era5", action="store_true", help="Use plain ERA5 (25 km) instead of Open-Meteo's ERA5-Land + ERA5 blend"
    )
    p_cex.add_argument("--fetch-missing", action="store_true", help="Fetch stations the archive lacks from the agency")
    p_cex.add_argument("--netcdf", action="store_true", help="Also write timeseries/netcdf (needs xarray + netCDF4)")
    p_cex.add_argument("--pause", type=float, default=3.0, help="Seconds between Open-Meteo calls (default 3)")
    p_cex.add_argument("--quiet", action="store_true")
    p_cval = car_sub.add_parser("validate", help="Check a folder against the Caravan layout")
    p_cval.add_argument("out")
    p_cval.add_argument("--prefix", required=True)

    p_mcp = sub.add_parser("mcp", help="Serve find_stations / get_timeseries / analyze_station over MCP (#113)")
    p_mcp.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")

    # ── playbooks ─────────────────────────────────────────────────────
    p_playbooks = sub.add_parser("playbooks", help="The problem playbooks `aquascope solve` follows (#307)")
    playbooks_sub = p_playbooks.add_subparsers(dest="playbooks_cmd")
    playbooks_sub.add_parser("list", help="List the playbooks")
    p_pb_show = playbooks_sub.add_parser("show", help="Print one playbook: intake, branches, gates, declines")
    p_pb_show.add_argument("id")

    # ── solve ─────────────────────────────────────────────────────────
    p_solve = sub.add_parser(
        "solve",
        help="Solve a problem at a point: recon, plan, your review, execution with gates, report (#308); "
        "without --lat/--lon, the legacy challenge agent",
    )
    p_solve.add_argument(
        "query",
        help="The problem in plain language (e.g. 'Design flow for a road crossing, 100-year return period')",
    )
    p_solve.add_argument("--lat", type=float, default=None, help="Latitude of the site (with --lon: the team)")
    p_solve.add_argument("--lon", type=float, default=None, help="Longitude of the site")
    p_solve.add_argument("--playbook", default=None, help="Playbook id (see `aquascope playbooks`); else keyword rules")
    p_solve.add_argument(
        "--intake",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="An intake field, e.g. --intake return_period=200 (repeatable)",
    )
    p_solve.add_argument("--yes", "-y", action="store_true", help="Run the plan without asking")
    p_solve.add_argument(
        "--provider",
        choices=provider_ids(),
        default=None,
        help="Use a model for the rationale, fallbacks and prose (keyless otherwise)",
    )
    p_solve.add_argument(
        "--model", default=None, help="Model name (with --lat/--lon: the LLM; otherwise the legacy forecast model)"
    )
    p_solve.add_argument("--api-key", default=None)
    p_solve.add_argument("--base-url", default=None, help="Any OpenAI-compatible endpoint (or Anthropic's)")
    p_solve.add_argument("--out", "-o", default=None, help="Save the Markdown report here")
    p_solve.add_argument("--study", default=None, help="Write the executed study here, to re-run with `aquascope run`")
    p_solve.add_argument("--quiet", "-q", action="store_true", help="Do not print the timeline as it happens")
    p_solve.add_argument("--file", default=None, help="Legacy agent: a data file (JSON/CSV) instead of fetching")

    # ── studio ────────────────────────────────────────────────────────
    p_studio = sub.add_parser(
        "studio",
        help="A complete study at a place by a crew of roles: the brief you agree, the plan you approve, the "
        "run with gates, the report and the bundle (keyless by default)",
    )
    p_studio.add_argument(
        "query", nargs="?", default=None,
        help="The problem in plain language (run `aquascope studio` alone in a terminal and it asks), or a "
             "finished study's bundle folder to revise, review and sign it on the Study Desk",
    )
    p_studio.add_argument(
        "--at", default=None, metavar="PLACE",
        help="Where: a gauge name or river words, a station id (USGS-01013500), or 'lat,lon'",
    )
    p_studio.add_argument("--lat", type=float, default=None, help="Latitude of the site")
    p_studio.add_argument("--lon", type=float, default=None, help="Longitude of the site")
    p_studio.add_argument(
        "--data",
        action="append",
        default=[],
        metavar="FILE",
        help="A table of your own (CSV, Excel, JSON) the crew may use (repeatable)",
    )
    p_studio.add_argument(
        "--intake",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="An intake field, e.g. --intake return_period=200 (repeatable)",
    )
    p_studio.add_argument(
        "--provider",
        choices=provider_ids(),
        default=None,
        help="Use a model for the brief, the methodology and the prose (keyless otherwise)",
    )
    p_studio.add_argument("--model", default=None, help="Model name")
    p_studio.add_argument("--api-key", default=None)
    p_studio.add_argument("--base-url", default=None, help="Any OpenAI-compatible endpoint (or Anthropic's)")
    p_studio.add_argument(
        "--max-usd",
        type=float,
        default=None,
        metavar="USD",
        help="A spend ceiling for the model calls: past it the roles run keyless and the "
        "footer says so (models in the price table only)",
    )
    p_studio.add_argument("--out", "-o", default=None, help="The bundle's directory (default ./studio-<id>/)")
    p_studio.add_argument("--yes", "-y", action="store_true", help="Answer the defaults, approve the plan, export")
    p_studio.add_argument(
        "--continue-without",
        action="store_true",
        help="When the crew asks for data you do not have, go on at the lower grade it names",
    )
    p_studio.add_argument("--resume", default=None, metavar="WORKSPACE.JSON", help="Resume a saved workspace")
    p_studio.add_argument("--quiet", "-q", action="store_true", help="Do not print the timeline as it happens")
    p_studio.add_argument("--verbose", "-v", action="store_true",
                          help="Print every event the crew emits (every gate, figure and file), not the short log")
    p_studio.add_argument("--style", default=None, metavar="STYLE.yaml",
                          help="A house style for the documents: organisation, project, client, prepared_by, "
                               "checked_by, logo, accent (YAML or JSON)")
    _add_desk_args(p_studio, studio=True)

    # ── desk ────────────────────────────────────────────────────────────
    p_desk = sub.add_parser(
        "desk",
        help="Revise, review and sign a finished study (the same as `aquascope studio <bundle folder>`)",
    )
    p_desk.add_argument("workspace", help="The study's bundle folder or its workspace.json")
    _add_desk_args(p_desk)

    # ── studio-showcase ───────────────────────────────────────────────
    p_show = sub.add_parser(
        "studio-showcase",
        help="Record the crew's worked studies once with a model (a maintainer's command) and list them; the "
        "Explorer replays them with no key",
    )
    show_sub = p_show.add_subparsers(dest="showcase_cmd")
    p_show_rec = show_sub.add_parser("record", help="Run the cases that are not fresh and write the recordings")
    p_show_rec.add_argument("--out", default="explorer/showcase/studies", help="The recordings' directory")
    p_show_rec.add_argument("--only", default=None, help="Comma-separated case ids to (re)record whatever their age")
    p_show_rec.add_argument("--max-usd", type=float, default=15.0, help="Stop the run at this estimated spend")
    p_show_rec.add_argument("--provider", default="anthropic")
    p_show_rec.add_argument("--model", default="claude-sonnet-5-5")
    p_show_rec.add_argument("--api-key", default=None)
    p_show_rec.add_argument(
        "--refresh-after",
        type=float,
        default=30.0,
        metavar="DAYS",
        help="Re-record a case only when its recording is older than this (0: every case)",
    )
    p_show_rec.add_argument("--quiet", "-q", action="store_true", help="Do not print the timeline as it happens")
    p_show_list = show_sub.add_parser("list", help="The recordings on disk, as a table")
    p_show_list.add_argument("--out", default="explorer/showcase/studies", help="The recordings' directory")

    # ── update ────────────────────────────────────────────────────────
    p_update = sub.add_parser(
        "update", help="Upgrade aquascope to the newest release, the same way it was installed (uv, pipx, pip)")
    p_update.add_argument("--check", action="store_true", help="Only say whether a newer release exists")
    p_update.add_argument("--yes", "-y", action="store_true", help="Upgrade without asking")
    # ── eval ──────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "eval",
        help="Evaluate finished studies from their bundles: a scorecard, a trace, stats across runs (no model)",
    )
    eval_sub = p_eval.add_subparsers(dest="eval_cmd")
    p_eval_score = eval_sub.add_parser(
        "score", help="One study's scorecard: outcome, gates, Critic, report quality, plan accuracy, time and cost")
    p_eval_score.add_argument("study", nargs="+", help="A study bundle directory or its workspace.json (repeatable)")
    p_eval_score.add_argument("--case", default=None,
                              help="Score the plan against this HydroGym reference case (aquascope gym plans list)")
    p_eval_score.add_argument("--json", action="store_true")
    p_eval_trace = eval_sub.add_parser(
        "trace", help="One study as a timeline: phases, steps with their gates, model calls per role")
    p_eval_trace.add_argument("study", help="A study bundle directory or its workspace.json")
    p_eval_trace.add_argument("--events", action="store_true", help="Also print every event, with its time")
    p_eval_trace.add_argument("--json", action="store_true")
    p_eval_stats = eval_sub.add_parser(
        "stats", help="Many studies at once: grades, report scores, gate failure and skip rates, time and cost")
    p_eval_stats.add_argument("paths", nargs="+", help="Directories to search for study bundles (recursively)")
    p_eval_stats.add_argument("--by", default="playbook", choices=["playbook", "model", "date", "grade", "none"])
    p_eval_stats.add_argument("--case", default=None, help="Score every plan against this reference case")
    p_eval_stats.add_argument("--top", type=int, default=3, help="How many failing and skipped checks to name")
    p_eval_stats.add_argument("--csv", default=None, metavar="FILE", help="Also write the rows as CSV")
    p_eval_stats.add_argument("--json", action="store_true")
    p_eval_stats.add_argument("--cards", action="store_true", help="With --json, include every study's scorecard")

    # ── forecast ──────────────────────────────────────────────────────
    p_forecast = sub.add_parser("forecast", help="Run a predictive model on time-series data")
    p_forecast.add_argument("--model", required=True, help="Model ID (prophet, arima, random_forest, xgboost, lstm)")
    p_forecast.add_argument("--file", required=True, help="Path to time-series data file (JSON/CSV)")
    p_forecast.add_argument("--days", type=int, default=30, help="Forecast horizon in days")

    # ── plot ──────────────────────────────────────────────────────────
    p_plot = sub.add_parser("plot", help="Visualise data or analysis results")
    p_plot.add_argument(
        "--type", required=True, choices=["timeseries", "forecast", "boxplot", "heatmap", "fdc"], help="Plot type"
    )
    p_plot.add_argument("--file", required=True, help="Path to data file (CSV with DatetimeIndex)")
    p_plot.add_argument("--output", default=None, help="Save plot to file (PNG/SVG/PDF)")
    p_plot.add_argument("--title", default=None, help="Custom plot title")

    # ── dashboard ────────────────────────────────────────────────────
    p_dash = sub.add_parser("dashboard", help="Launch the interactive Streamlit dashboard")
    p_dash.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")
    p_dash.add_argument("--host", default="localhost", help="Host address (default: localhost)")

    # ── agri ─────────────────────────────────────────────────────────
    p_agri = sub.add_parser("agri", help="Run agricultural water planning workflows")
    agri_sub = p_agri.add_subparsers(dest="agri_command")
    agri_sub.required = True

    p_agri_plan = agri_sub.add_parser("plan", help="Create an irrigation plan from files or coordinates")
    p_agri_plan.add_argument("--crop", required=True, help="Crop name (e.g. maize, wheat_winter, rice_paddy)")
    p_agri_plan.add_argument("--planting-date", required=True, help="Planting date YYYY-MM-DD")
    p_agri_plan.add_argument("--eto-file", default=None, help="Path to ET0 data file (WaPOR/Open-Meteo/CSV/JSON)")
    p_agri_plan.add_argument("--precip-file", default=None, help="Path to precipitation data file (CSV/JSON)")
    p_agri_plan.add_argument(
        "--eto-parameter",
        default="et0_fao_evapotranspiration",
        help="Parameter name to extract when the ET0 file is in long-form collector format",
    )
    p_agri_plan.add_argument(
        "--precip-parameter",
        default="precipitation_sum",
        help="Parameter name to extract when the precipitation file is in long-form collector format",
    )
    p_agri_plan.add_argument("--lat", type=float, default=None, help="Latitude for Open-Meteo fallback inputs")
    p_agri_plan.add_argument("--lon", type=float, default=None, help="Longitude for Open-Meteo fallback inputs")
    p_agri_plan.add_argument(
        "--start-date", default=None, help="Input start date YYYY-MM-DD (defaults to planting date)"
    )
    p_agri_plan.add_argument("--end-date", default=None, help="Input end date YYYY-MM-DD")
    p_agri_plan.add_argument("--soil-fc", type=float, default=0.30, help="Soil field capacity as m3/m3")
    p_agri_plan.add_argument("--soil-wp", type=float, default=0.15, help="Soil wilting point as m3/m3")
    p_agri_plan.add_argument("--root-depth", type=float, default=1.0, help="Effective root depth in metres")
    p_agri_plan.add_argument("--efficiency", type=float, default=0.7, help="Irrigation efficiency (0-1)")
    p_agri_plan.add_argument("--depletion-fraction", type=float, default=0.5, help="RAW depletion fraction")
    p_agri_plan.add_argument("--initial-depletion", type=float, default=0.0, help="Initial root-zone depletion in mm")
    p_agri_plan.add_argument("-o", "--output", default=None, help="Write the irrigation plan to a JSON or CSV file")
    p_agri_plan.add_argument(
        "--format", choices=["json", "csv"], default=None, help="Output format (inferred from suffix)"
    )

    p_agri_benchmark = agri_sub.add_parser("benchmark", help="Benchmark AQUASTAT country-scale water metrics")
    p_agri_benchmark.add_argument("--aquastat-file", required=True, help="Path to AQUASTAT CSV or JSON data")
    p_agri_benchmark.add_argument(
        "--metric",
        required=True,
        choices=[
            "agricultural_withdrawal_per_irrigated_area",
            "agricultural_withdrawal_share_pct",
            "withdrawal_pressure_on_renewable_resources_pct",
        ],
        help="Benchmark metric to compute",
    )
    p_agri_benchmark.add_argument("--year", type=int, default=None, help="Specific year to benchmark")
    p_agri_benchmark.add_argument("--countries", default=None, help="Comma-separated country names or ISO3 codes")
    p_agri_benchmark.add_argument(
        "--all-years",
        action="store_true",
        help="Keep all country-year rows instead of using the latest year per country",
    )
    p_agri_benchmark.add_argument("--top", type=int, default=20, help="Maximum number of rows to print or save")
    p_agri_benchmark.add_argument("--output", default=None, help="Path to save benchmark results as JSON")

    p_agri_productivity = agri_sub.add_parser("productivity", help="Estimate WaPOR-based water productivity metrics")
    p_agri_productivity.add_argument(
        "--metric",
        required=True,
        choices=[
            "biomass_water_productivity",
            "relative_evapotranspiration_pct",
            "biomass_per_reference_et",
        ],
        help="Productivity or ET performance metric to compute",
    )
    p_agri_productivity.add_argument("--aeti-file", default=None, help="Path to WaPOR AETI CSV or JSON data")
    p_agri_productivity.add_argument("--npp-file", default=None, help="Path to WaPOR NPP CSV or JSON data")
    p_agri_productivity.add_argument("--ret-file", default=None, help="Path to WaPOR RET CSV or JSON data")
    p_agri_productivity.add_argument(
        "--aquastat-file", default=None, help="Optional AQUASTAT CSV or JSON data for country benchmark context"
    )
    p_agri_productivity.add_argument(
        "--aquastat-year", type=int, default=None, help="Optional year filter for AQUASTAT context"
    )
    p_agri_productivity.add_argument(
        "--aquastat-countries",
        default=None,
        help="Optional comma-separated country names or ISO3 codes for AQUASTAT context",
    )
    p_agri_productivity.add_argument(
        "--aquastat-metrics",
        default=None,
        help="Optional comma-separated AQUASTAT benchmark IDs for context; defaults to withdrawal share and withdrawal per irrigated area when available",
    )
    p_agri_productivity.add_argument(
        "--aquastat-top", type=int, default=10, help="Maximum number of rows per AQUASTAT context table"
    )
    p_agri_productivity.add_argument("--output", default=None, help="Path to save productivity results as JSON")

    # ── alerts ─────────────────────────────────────────────────────────
    p_alerts = sub.add_parser("alerts", help="Check water-quality data against regulatory thresholds")
    p_alerts.add_argument("--source", required=True, help="Path to CSV or JSON data file")
    p_alerts.add_argument("--standards", nargs="+", default=None, help="Standards to check (WHO EPA EU_WFD)")
    p_alerts.add_argument("--output", default=None, help="Path to save alert report as JSON")
    p_alerts.add_argument("--value-col", default="value", help="Column containing measured values")
    p_alerts.add_argument("--param-col", default="parameter", help="Column containing parameter names")

    # ── groundwater ──────────────────────────────────────────────────
    p_gw = sub.add_parser("groundwater", help="Run groundwater analysis (trend, recession, recharge, Theis)")
    p_gw.add_argument(
        "--analysis",
        required=True,
        choices=["trend", "recession", "seasonal", "hydrograph", "recharge-wtf", "theis"],
        help="Analysis type",
    )
    p_gw.add_argument("--file", required=True, help="Path to well level data (CSV with DatetimeIndex)")
    p_gw.add_argument(
        "--specific-yield", type=float, default=0.15, help="Specific yield for WTF recharge (default: 0.15)"
    )
    p_gw.add_argument("--transmissivity", type=float, default=None, help="Transmissivity m²/day (Theis)")
    p_gw.add_argument("--storativity", type=float, default=None, help="Storativity (Theis)")
    p_gw.add_argument("--pumping-rate", type=float, default=None, help="Pumping rate m³/day (Theis)")
    p_gw.add_argument("--distance", type=float, default=None, help="Distance from well in metres (Theis)")
    p_gw.add_argument("--output", default=None, help="Save results to JSON")

    # ── climate ──────────────────────────────────────────────────────
    p_climate = sub.add_parser("climate", help="Climate projections and indices")
    p_climate.add_argument(
        "--analysis", required=True, choices=["downscale", "indices", "drought", "scenario"], help="Analysis type"
    )
    p_climate.add_argument("--obs-file", default=None, help="Path to observed data (CSV)")
    p_climate.add_argument("--gcm-hist-file", default=None, help="Path to GCM historical data (CSV)")
    p_climate.add_argument("--gcm-future-file", default=None, help="Path to GCM future data (CSV)")
    p_climate.add_argument(
        "--method", default="quantile_mapping", help="Downscaling method (delta, quantile_mapping, qdm)"
    )
    p_climate.add_argument("--index", default="cdd", help="Climate index (cdd, cwd, pci, heat_wave, aridity)")
    p_climate.add_argument("--file", default=None, help="Path to data file (CSV)")
    p_climate.add_argument("--output", default=None, help="Save results to JSON")

    # ── hydro ─────────────────────────────────────────────────────────
    p_export = sub.add_parser(
        "export", help="Inputs for HEC-HMS, HEC-RAS, HEC-SSP, HEC-DSS, SWMM, MODFLOW 6, Delft-FEWS or Raven (#519)"
    )
    p_export.add_argument("--to", required=True, choices=[*EXPORT_TOOLS, "all"], help="The tool to write inputs for")
    p_export.add_argument("--station", default=None, help="A gauge as source/station_id, e.g. usgs/01646500")
    p_export.add_argument("--file", default=None, help="Or a CSV: a date column, then the values")
    p_export.add_argument("--column", default=None, help="The value column of --file (default: first numeric)")
    p_export.add_argument("--variable", default=None,
                          help="discharge (default), water_level, groundwater_level or precipitation")
    p_export.add_argument("--unit", default=None, help="Unit of --file values (default m3/s, m or mm)")
    p_export.add_argument("--location", default=None, help="Identifier for --file in the outputs")
    p_export.add_argument("--years", type=int, default=None, help="Only the last N years of the gauge record")
    p_export.add_argument("-o", "--out-dir", default="aquascope-export", help="Folder to write (default aquascope-export)")
    p_export.add_argument("--zip", action="store_true", help="Write one zip (OUT_DIR.zip) instead of a folder")
    p_export.add_argument("--no-dss", action="store_true", help="Write the DSS CSV even where hecdss loads")
    p_export.add_argument("--regional-skew", type=float, default=None, help="HEC-SSP: weight the skew with this")
    p_export.add_argument("--regional-skew-mse", type=float, default=None, help="HEC-SSP: MSE of the regional skew")
    p_export.add_argument("--year-start-month", type=int, default=10, help="HEC-SSP: first month of the water year")
    p_export.add_argument("--subbasin-id", type=int, default=1, help="Raven: the subbasin at the gauge")
    p_export.add_argument("--cell", type=int, nargs=3, metavar=("LAYER", "ROW", "COL"), default=None,
                          help="MODFLOW 6: the river or well cell (default 1 1 1)")
    p_export.add_argument("--cond", type=float, default=None, help="MODFLOW 6 RIV: riverbed conductance")
    p_export.add_argument("--rbot", type=float, default=None, help="MODFLOW 6 RIV: riverbed bottom elevation")
    p_export.add_argument("--period", default=None, help="MODFLOW 6: resample first, e.g. MS for monthly")
    p_export.add_argument("--engine", choices=["text", "flopy"], default="text",
                          help="MODFLOW 6: write as text (default) or through FloPy")
    p_export.add_argument("--json", action="store_true", help="Print the written paths and notes as JSON")

    p_hydro = sub.add_parser("hydro", help="Run hydrological analysis (FDC, baseflow, recession, flood-freq)")
    p_hydro.add_argument(
        "--analysis",
        required=True,
        choices=["fdc", "baseflow", "recession", "flood-freq", "low-flow"],
        help="Analysis type",
    )
    p_hydro.add_argument("--file", required=True, help="Path to discharge data (CSV with DatetimeIndex)")
    p_hydro.add_argument("--method", default=None, help="Sub-method (e.g. lyne_hollick, eckhardt for baseflow)")
    p_hydro.add_argument("-o", "--output", default=None, help="Write results to a JSON or CSV file")
    p_hydro.add_argument("--format", choices=["json", "csv"], default=None, help="Output format (inferred from suffix)")
    p_hydro.add_argument("--n-day", type=int, default=None, help="N-day window for low-flow (default: 7)")
    p_hydro.add_argument("--return-period", type=int, default=None, help="Return period for low-flow (default: 10)")

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    commands = {
        "collect": cmd_collect,
        "recommend": cmd_recommend,
        "eda": cmd_eda,
        "quality": cmd_quality,
        "list-methods": cmd_list_methods,
        "list-sources": cmd_list_sources,
        "stations": cmd_stations,
        "harvest": cmd_harvest,
        "mcp": cmd_mcp,
        "basins": cmd_basins,
        "layers": cmd_layers,
        "map": cmd_map,
        "river": cmd_river,
        "evidence": cmd_evidence,
        "now": cmd_now,
        "bulletin": cmd_bulletin,
        "warnings": cmd_warnings,
        "watch": cmd_watch,
        "assess": cmd_assess,
        "context": cmd_context,
        "area-study": cmd_area_study,
        "gym": cmd_gym,
        "caravan": cmd_caravan,
        "ask": cmd_ask,
        "run": cmd_run,
        "ingest": cmd_ingest,
        "solve": cmd_solve,
        "studio": cmd_studio,
        "studio-showcase": cmd_studio_showcase,
        "desk": cmd_desk,
        "update": cmd_update,
        "eval": cmd_eval,
        "playbooks": cmd_playbooks,
        "forecast": cmd_forecast,
        "plot": cmd_plot,
        "hydro": cmd_hydro,
        "export": cmd_export,
        "alerts": cmd_alerts,
        "dashboard": cmd_dashboard,
        "agri": cmd_agri,
        "groundwater": cmd_groundwater,
        "climate": cmd_climate,
        "completion": cmd_completion,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
