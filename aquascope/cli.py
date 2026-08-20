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
from datetime import date
from pathlib import Path

import argcomplete

# ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("aquascope")


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
    records = collector.collect(**kwargs)
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
            data = json.loads(path.read_text())
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
            )
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
        logger.error(
            "No station-capable source matches. Sources with a catalog: %s", station_sources(args.variable)
        )
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
            features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props})
        out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2))
    elif fmt == "json":
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        import csv

        fields = ["source", "station_id", "name", "latitude", "longitude", "variables", "period_start", "period_end",
                  "url", "river", "country", "extra"]
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

    report = harvest_stations(
        args.out,
        sources=args.source or None,
        max_items=args.max_items,
        api_key=args.api_key,
        max_workers=args.workers,
        write_geojson=not args.no_geojson,
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
                logger.error("%s is not harvested for %s (they mirror %s)", args.variable, bad,
                             {s: list(HARVESTABLE[s]) for s in bad})
                sys.exit(2)
    report = harvest_observations(
        args.out,
        sources=sources,
        variable=args.variable,
        years=args.years,
        max_stations=args.max_stations,
        refresh_days=args.refresh_days,
        only_stations=args.station or None,
    )
    for h in report.sources:
        print(
            f"  {h.source:<20} {h.variable:<14} harvested {h.harvested:>4}  empty {h.empty:>4}  "
            f"failed {h.failed:>3}  of {h.attempted:>4} picked  ({h.seconds:.0f}s)"
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


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a water question; the analyst calls aquascope tools and writes a cited answer."""
    from aquascope.ai_engine.analyst import ask

    def on_event(msg: str) -> None:
        if not args.quiet:
            print(f"  · {msg}", file=sys.stderr)

    try:
        result = ask(
            args.question, provider=args.provider, model=args.model, api_key=args.api_key, base_url=args.base_url,
            max_steps=args.max_steps, on_event=on_event,
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

    def on_event(msg: str) -> None:
        if not args.quiet:
            print(f"  · {msg}", file=sys.stderr)

    if args.dry_run:
        print(f"{len(study.steps)} step(s) in {args.study}:")
        for i, step in enumerate(study.steps, 1):
            print(f"  {i}. {step.tool}({', '.join(f'{k}={v!r}' for k, v in step.arguments.items())})")
        return
    run = run_study(study, on_event=on_event)
    if args.out:
        paths = write_outputs(run, args.out)
        print(f"\n  Report saved to {paths['report.md']}")
    else:
        print(run.to_markdown())
    if not run.ok:
        sys.exit(1)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Map + QA an arbitrary CSV/Excel export into a clean series with a report."""
    from aquascope.ingest import ingest, write_outputs

    client = model = None
    if args.llm:
        try:
            from aquascope.ai_engine.analyst import resolve_llm
            from aquascope.ai_engine.llm_transport import make_client

            cfg = resolve_llm(args.provider, args.model, args.api_key)
            client, model = make_client(cfg["api_key"], cfg["base_url"]), cfg["model"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM mapping unavailable (%s); using heuristics", exc)
    try:
        result = ingest(
            args.file, variable=args.variable, date_column=args.date_column, value_column=args.value_column,
            unit=args.unit, station=args.station, sheet=args.sheet, llm_client=client, llm_model=model,
            description=args.describe or "",
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    m, q = result["mapping"], result["qa"]
    print(f"  mapping  : {m['datetime_column']} + {m['value_column']} -> {m['variable']} [{m['unit']}] "
          f"(x{m['to_si_factor']}, {m['method']}, confidence {m['confidence']:.0%})")
    print(f"  values   : {q['n_values']:,} kept of {q['n_rows_in']:,} rows; {q['start']} -> {q['end']}; "
          f"coverage {q['coverage_pct']}%")
    print(f"  dropped  : {q['n_duplicates_dropped']} duplicates, {q['n_sentinels_dropped']} sentinels; "
          f"flagged {q['n_negative']} negative, {q['n_spikes_flagged']} spikes")
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
        print(f"  {res['k']} of {res['n_candidates']} gauged basins, method {res['method']}, "
              f"features {', '.join(res['features_used'])}")
        for i, st in enumerate(res["stations"], 1):
            dist = f"{st['distance_km']:,.0f} km" if st.get("distance_km") is not None else ""
            print(f"  {i:>2}. {st['source']:<20} {st['station_id']:<40} {(st.get('name') or '')[:38]:<38} "
                  f"area {st['up_area_km2']:>9,.0f} km2  score {st['score']:.3f} {dist}")
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
        print(f"  {len(est)} signatures from {res.get('n_donors_available', 0):,} donors, method {res['method']}"
              + (f", k={res['similarity']['k']}" if "similarity" in res else ""))
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
    print(f"  Sub-basin {sb['hybas_id']} (Pfafstetter {sb.get('pfaf_id')}), {sb.get('sub_area', 0):,.1f} km², "
          f"upstream area {sb.get('up_area', 0):,.1f} km²")
    print(f"  {res['upstream']['note']}")
    attrs = res.get("attributes", {})
    for key, v in attrs.items():
        if isinstance(v, dict):
            print(f"  {v['label']:<48} {v['value']:>12,.2f} {v['unit']}")
    print(f"\n  {res['attribution']}")


def cmd_gym(args: argparse.Namespace) -> None:
    """`aquascope gym basins|run|leaderboard`: HydroGym, the calibration environment over real basins."""
    from aquascope import gym as hg

    if args.gym_cmd == "basins":
        rows = hg.suggest_basins(args.n, sources=args.source or None, min_years=args.min_years,
                                 max_snow_pct=None if args.allow_snow else 20.0)
        if args.json:
            print(json.dumps(rows, indent=2, default=str))
            return
        if not rows:
            print("  no candidate basins yet (the archive publishes basins/station_signatures.parquet weekly)")
            return
        print(f"  {len(rows)} basins with long archived discharge and a catchment area (use SOURCE/ID with `gym run`)")
        for r in rows:
            snow = f"snow {r['snow_cover_pct']:.0f} %" if r.get("snow_cover_pct") is not None else ""
            print(f"  {r['source']}/{r['station_id']:<42} {r['area_km2']:>9,.0f} km2  {r['n_years']:>4.0f} yr  "
                  f"q {r['q_mean_mm']:.2f} mm/d  RR {r['runoff_ratio'] if r['runoff_ratio'] is not None else float('nan'):.2f}  {snow}")
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
        table = hg.run_leaderboard(basins, args.agent or None, objective=args.objective, max_steps=args.steps,
                                   seeds=tuple(range(args.seeds)))
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
    print(f"  {res['agent']}: {res['steps']} steps, {res.get('simulator_calls', res['steps'])} simulator calls, "
          f"{res['seconds']} s")
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
            args.source, args.out, station_ids=args.station or None, max_stations=args.max_stations,
            min_years=args.min_years, start=args.start, end=args.end, prefix=args.prefix,
            forcing=not args.no_forcing, forcing_models=None if args.era5 else "best_match",
            fetch_missing=args.fetch_missing, write_netcdf=args.netcdf, pause=args.pause, on_event=say,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    for g in report.gauges:
        status = f"{g.n_days:>6} days, {g.n_streamflow:>6} with flow, area {g.area_km2:,.0f} km2 ({g.area_source})" \
            if g.ok else f"skipped: {g.error}"
        print(f"  {g.gauge_id:<48} {status}")
    print(f"\n  {report.n_ok}/{len(report.gauges)} gauges written under {report.out_dir} (prefix {report.prefix})")
    res = caravan.validate_caravan(args.out, report.prefix) if report.n_ok else {"ok": False, "problems": ["nothing written"]}
    print(f"  validation: {'OK' if res['ok'] else '; '.join(res['problems'][:5])}")
    if report.n_ok == 0:
        sys.exit(1)


def cmd_completion(args: argparse.Namespace) -> None:
    """Print the shell activation line for tab-completion."""
    from argcomplete.shell_integration import shellcode
    print(shellcode(["aquascope"], shell=args.shell))


def cmd_solve(args: argparse.Namespace) -> None:
    """Solve a water challenge using NL description (agent mode)."""
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
        out_path.write_text(json.dumps(out_data, indent=2, default=str))
        print(f"  ✓ Report saved → {out_path}\n")


def cmd_hydro(args: argparse.Namespace) -> None:
    """Run hydrological analysis."""
    import pandas as pd

    df = pd.read_csv(args.file, index_col=0, parse_dates=True)
    q = df.iloc[:, 0]  # first column as discharge

    if args.analysis == "fdc":
        from aquascope.hydrology import flow_duration_curve

        result = flow_duration_curve(q)
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
        print(f"\n  Baseflow Separation ({result.method}):")
        print(f"    BFI = {result.bfi:.3f}")
        if args.output:
            result.df.to_csv(args.output)
            print(f"    Saved to {args.output}")

    elif args.analysis == "recession":
        from aquascope.hydrology import recession_analysis

        result = recession_analysis(q)
        print("\n  Recession Analysis:")
        print(f"    Segments found: {len(result.segments)}")
        print(f"    Recession constant: {result.recession_constant:.2f} days")
        print(f"    Half-life: {result.half_life_days:.2f} days")
        print(f"    R²: {result.r_squared:.4f}")

    elif args.analysis == "flood-freq":
        from aquascope.hydrology import fit_gev

        result = fit_gev(q)
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
        print(f"\n  {n_day}Q{return_period} = {val:.3f}")

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

        result = trend_detection(levels)
        print("\nWell Trend Analysis (Mann-Kendall)")
        print(f"  Trend: {result.trend}")
        print(f"  Slope: {result.slope:.6f} per time-step")
        print(f"  p-value: {result.p_value:.4f}")
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
        out_path = Path(args.output)
        out_path.write_text(json.dumps(plan.to_dict(), indent=2, default=str))
        print(f"\n  ✓ Full irrigation plan saved → {out_path}")


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
        out_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
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
        out_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        print(f"\n  ✓ Productivity results saved → {out_path}")


def main() -> None:
    from aquascope.registry import source_keys
    from aquascope.schemas.station import VARIABLES

    parser = argparse.ArgumentParser(
        description="AquaScope — Water data collection, analysis & AI research recomm..."
    )
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
        "--days", type=int, default=None, help="Number of days (USGS/UKEA/PEGELONLINE/BOM; PEGELONLINE max: 31)"
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
      help="Bounding box west,south,east,north (WaPOR), or min_lon, min_lat, max_lon, max_lat (USGS/UKEA)"
    )
    p_collect.add_argument(
        "--mode", default=None, help="Collector mode (openmeteo: weather/forecast/flood; grdc: in_situ/satellite)"
    )
    p_collect.add_argument("--variable", default=None, help="Variable code for the selected collector (WaPOR)")
    p_collect.add_argument("--lid", default=None, help="A unique 5-character alphanumeric code e.g. ANAW1 (NOAA_NWPS)")
    p_collect.add_argument("--lat", type=float, default=None, help="Latitude (openmeteo/copernicus)")
    p_collect.add_argument("--lon", type=float, default=None, help="Longitude (openmeteo/copernicus)")
    p_collect.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD (openmeteo/copernicus/UKEA/BOM)")
    p_collect.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (openmeteo/copernicus/UKEA/BOM)")
    p_collect.add_argument("--start-year", type=int, default=2000, help="Start year (AQUASTAT)")
    p_collect.add_argument("--end-year", type=int, default=2023, help="End year (AQUASTAT)")
    p_collect.add_argument("--format", default="json", choices=["json", "csv", "geojson"], help="Output format")
    p_collect.add_argument("--year", type=int, default=None, help="Year filter (EU WFD)")
    p_collect.add_argument(
        "--station-ids", default=None, help="Comma-separated gauge codes to filter (camels_cl, camels_br)"
    )
    p_collect.add_argument(
        "--station", default=None, help="Station UUID/SUID (PEGELONLINE/UKEA), or AWRC station number (BOM)"
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
    p_run.add_argument("study", nargs="?", default=None,
                       help="A study.yaml from `aquascope ask --study` or written by hand (#54)")
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
        "--bbox", default=None,
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
    p_harvest.add_argument("what", choices=["stations", "obs", "bundles"],
                           help="stations: the catalog; obs: daily series per station; bundles: one Parquet per "
                                "variable and source rolled up from obs/")
    p_harvest.add_argument("--out", default="archive", help="Output folder (default: ./archive)")
    p_harvest.add_argument("--source", action="append", choices=source_keys(), help="Restrict to a source (repeatable)")
    p_harvest.add_argument("--max-items", type=int, default=None, help="stations: cap per source (for smoke tests)")
    p_harvest.add_argument("--variable", default=None, dest="variable",
                           help="obs: harvest only this variable (default: every harvestable variable per source)")
    p_harvest.add_argument("--variables", action="append", dest="variable_list", metavar="VAR",
                           help="bundles: restrict to these variables (repeatable)")
    p_harvest.add_argument("--years", type=int, default=40, help="obs: how far back to ask (default 40)")
    p_harvest.add_argument("--max-stations", type=int, default=100, help="obs: stations per source per run")
    p_harvest.add_argument("--refresh-days", type=int, default=30, help="obs: re-harvest a station older than this")
    p_harvest.add_argument("--station", action="append", help="obs: only these station ids (repeatable)")
    p_harvest.add_argument("--sync-from", default=None, metavar="REPO_ID",
                           help="obs: download the existing obs/ tree from this dataset first (incremental runs)")
    p_harvest.add_argument("--api-key", default=None)
    p_harvest.add_argument("--workers", type=int, default=4)
    p_harvest.add_argument("--no-geojson", action="store_true", help="Skip stations.geojson")
    p_harvest.add_argument("--publish", default=None, metavar="REPO_ID",
                           help="Upload the folder to this Hugging Face dataset (needs HF_TOKEN)")

    # ── ask ──────────────────────────────────────────────────────────
    p_ask = sub.add_parser("ask", help="Ask a water question in plain language; get a cited answer from real data")
    p_ask.add_argument("question")
    p_ask.add_argument("--provider", choices=["openai", "groq", "huggingface", "mistral", "openrouter", "ollama"],
                       default=None)
    p_ask.add_argument("--model", default=None)
    p_ask.add_argument("--api-key", default=None)
    p_ask.add_argument("--base-url", default=None, help="Any OpenAI-compatible endpoint")
    p_ask.add_argument("--max-steps", type=int, default=8, help="Tool-call rounds allowed (default 8)")
    p_ask.add_argument("--out", "-o", default=None, help="Save the Markdown report here")
    p_ask.add_argument("--quiet", "-q", action="store_true", help="Do not print tool calls as they happen")
    p_ask.add_argument("--study", default=None,
                       help="Write the steps behind the answer here, to re-run with `aquascope run`")


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
    p_ingest.add_argument("--provider", choices=["openai", "groq", "huggingface", "mistral", "openrouter", "ollama"],
                          default=None)
    p_ingest.add_argument("--model", default=None)
    p_ingest.add_argument("--api-key", default=None)
    p_ingest.add_argument("--out", "-o", default=None, help="Output stem (default: <file>_clean)")

    # ── mcp ──────────────────────────────────────────────────────────
    # ── basins ───────────────────────────────────────────────────────
    p_basins = sub.add_parser("basins", help="Catchments from BasinATLAS (HydroATLAS, CC BY 4.0) in the Archive")
    basins_sub = p_basins.add_subparsers(dest="basins_cmd", required=True)
    p_bat = basins_sub.add_parser("at", help="Describe the catchment upstream of a point")
    p_bat.add_argument("lat", type=float)
    p_bat.add_argument("lon", type=float)
    p_bat.add_argument("--local", action="store_true", help="Only the level-12 sub-basin containing the point")
    p_bat.add_argument("--json", action="store_true")
    p_bsim = basins_sub.add_parser("similar", help="Gauged basins whose catchments most resemble a point's or a station's")
    p_bsim.add_argument("lat", type=float, nargs="?", default=None)
    p_bsim.add_argument("lon", type=float, nargs="?", default=None)
    p_bsim.add_argument("--station", default=None, metavar="SOURCE/ID", help="Use a station's own catchment as the target")
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
    p_bsig = basins_sub.add_parser("signatures", help="Build basins/station_signatures.parquet from the discharge bundles")
    p_bsig.add_argument("--archive", default="archive", help="Local archive folder holding obs/discharge/*.parquet")
    p_bsig.add_argument("--catchments", default=None, help="Local station_catchments.parquet (default: from the Hub)")
    p_bsig.add_argument("--out", default="archive/basins/station_signatures.parquet")
    p_bsig.add_argument("--min-years", type=float, default=10.0)
    p_bloo = basins_sub.add_parser("loo", help="Leave-one-out regionalisation skill -> basins/regionalization_skill.json")
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
    p_gym = sub.add_parser("gym", help="HydroGym: a gym-style calibration environment over real basins (#175)")
    gym_sub = p_gym.add_subparsers(dest="gym_cmd", required=True)
    p_gb = gym_sub.add_parser("basins", help="Suggest gauged basins from the Archive that make good tasks")
    p_gb.add_argument("--n", type=int, default=10)
    p_gb.add_argument("--source", action="append", help="Restrict to these sources (repeatable)")
    p_gb.add_argument("--min-years", type=float, default=15.0)
    p_gb.add_argument("--allow-snow", action="store_true", help="Keep snowy catchments (GR4J has no snow routine)")
    p_gb.add_argument("--json", action="store_true")
    for name, help_ in (("run", "Play one baseline agent on a basin"),
                        ("leaderboard", "Play the baselines on one or more basins, one row per run")):
        p_g = gym_sub.add_parser(name, help=help_)
        p_g.add_argument("--basin", action="append", metavar="SOURCE/ID", help="Archive station (repeatable)")
        p_g.add_argument("--synthetic", action="store_true", help="Use synthetic GR4J basins (no network)")
        p_g.add_argument("--n-synthetic", type=int, default=1)
        p_g.add_argument("--agent", action="append", choices=["random_search", "nelder_mead", "differential_evolution"],
                         help="Baseline agent(s); default: differential_evolution for run, all three for leaderboard")
        p_g.add_argument("--objective", choices=["nse", "kge", "log_nse"], default="nse")
        p_g.add_argument("--steps", type=int, default=30, help="Step budget per episode")
        p_g.add_argument("--seed", type=int, default=0)
        p_g.add_argument("--seeds", type=int, default=1, help="leaderboard: number of seeds per agent and basin")
        p_g.add_argument("--out", default=None, help="leaderboard: write the table as CSV")
        p_g.add_argument("--json", action="store_true")

    # ── caravan ──────────────────────────────────────────────────────
    p_car = sub.add_parser("caravan", help="Caravan-format sub-datasets (forcing + mm/day streamflow + attributes) from the Archive")
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
    p_cex.add_argument("--era5", action="store_true",
                       help="Use plain ERA5 (25 km) instead of Open-Meteo's ERA5-Land + ERA5 blend")
    p_cex.add_argument("--fetch-missing", action="store_true", help="Fetch stations the archive lacks from the agency")
    p_cex.add_argument("--netcdf", action="store_true", help="Also write timeseries/netcdf (needs xarray + netCDF4)")
    p_cex.add_argument("--pause", type=float, default=3.0, help="Seconds between Open-Meteo calls (default 3)")
    p_cex.add_argument("--quiet", action="store_true")
    p_cval = car_sub.add_parser("validate", help="Check a folder against the Caravan layout")
    p_cval.add_argument("out")
    p_cval.add_argument("--prefix", required=True)

    p_mcp = sub.add_parser("mcp", help="Serve find_stations / get_timeseries / analyze_station over MCP (#113)")
    p_mcp.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")

    # ── solve ─────────────────────────────────────────────────────────
    p_solve = sub.add_parser("solve", help="Solve a water challenge from a natural-language description")
    p_solve.add_argument(
        "query",
        help="Natural-language challenge description (e.g. 'Forecast flooding at lat 13.5, lon 2.1')",
    )
    p_solve.add_argument("--model", default=None, help="Override model (e.g. prophet, arima, random_forest)")
    p_solve.add_argument("--file", default=None, help="Optional data file (JSON/CSV) to use instead of fetching")

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
    p_agri_plan.add_argument("--output", default=None, help="Path to save the irrigation plan as JSON")

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
    p_hydro = sub.add_parser("hydro", help="Run hydrological analysis (FDC, baseflow, recession, flood-freq)")
    p_hydro.add_argument(
        "--analysis",
        required=True,
        choices=["fdc", "baseflow", "recession", "flood-freq", "low-flow"],
        help="Analysis type",
    )
    p_hydro.add_argument("--file", required=True, help="Path to discharge data (CSV with DatetimeIndex)")
    p_hydro.add_argument("--method", default=None, help="Sub-method (e.g. lyne_hollick, eckhardt for baseflow)")
    p_hydro.add_argument("--output", default=None, help="Save results to CSV")
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
        "gym": cmd_gym,
        "caravan": cmd_caravan,
        "ask": cmd_ask,
        "run": cmd_run,
        "ingest": cmd_ingest,
        "solve": cmd_solve,
        "forecast": cmd_forecast,
        "plot": cmd_plot,
        "hydro": cmd_hydro,
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
