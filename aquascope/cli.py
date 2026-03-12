"""
AquaScope CLI — collect water data, analyse, and get AI methodology recommendations.

Usage
-----
    aquascope collect --source taiwan_moenv --api-key YOUR_KEY
    aquascope recommend --parameters DO,BOD5,COD --goal "trend analysis"
    aquascope eda --file data/raw/water_data.json
    aquascope quality --file data/raw/water_data.json
    aquascope run --method trend_analysis --file data/raw/water_data.json
    aquascope list-methods
    aquascope list-sources
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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


def cmd_collect(args: argparse.Namespace) -> None:
    """Run a data collector and save results."""
    from aquascope.collectors import (
        GEMStatCollector,
        SDG6Collector,
        TaiwanCivilIoTCollector,
        TaiwanMOENVCollector,
        TaiwanWRAReservoirCollector,
        TaiwanWRAWaterLevelCollector,
        USGSCollector,
        WQPCollector,
    )
    from aquascope.utils.storage import save_records

    source = args.source.lower()
    collector_map = {
        "taiwan_moenv": lambda: TaiwanMOENVCollector(api_key=args.api_key or ""),
        "taiwan_wra_level": lambda: TaiwanWRAWaterLevelCollector(),
        "taiwan_wra_reservoir": lambda: TaiwanWRAReservoirCollector(),
        "usgs": lambda: USGSCollector(api_key=args.api_key or "DEMO_KEY"),
        "sdg6": lambda: SDG6Collector(),
        "gemstat": lambda: GEMStatCollector(),
        "taiwan_civil_iot": lambda: TaiwanCivilIoTCollector(),
        "wqp": lambda: WQPCollector(),
    }

    if source not in collector_map:
        logger.error("Unknown source '%s'. Available: %s", source, list(collector_map.keys()))
        sys.exit(1)

    collector = collector_map[source]()

    kwargs = {}
    if source == "usgs" and args.days:
        kwargs["datetime_range"] = f"P{args.days}D"
    if source == "sdg6" and args.countries:
        kwargs["country_codes"] = args.countries
    if source == "wqp":
        if args.state:
            kwargs["state_code"] = args.state

    records = collector.collect(**kwargs)
    if not records:
        logger.warning("No records collected.")
        return

    path = save_records(records, prefix=source, fmt=args.format)
    print(f"✓ Saved {len(records)} records → {path}")


def cmd_recommend(args: argparse.Namespace) -> None:
    """Generate methodology recommendations."""
    from aquascope.ai_engine.recommender import DatasetProfile, recommend, recommend_with_llm

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

    if args.use_llm:
        recs = recommend_with_llm(
            profile,
            top_k=args.top_k,
            model=args.model or "gpt-4o-mini",
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
        )
    else:
        recs = recommend(profile, top_k=args.top_k)

    if not recs:
        print("No matching methodologies found. Try broader parameters or keywords.")
        return

    print(f"\n{'='*70}")
    print(f"  AquaScope — Top {len(recs)} Research Methodology Recommendations")
    print(f"{'='*70}\n")
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
        print(f"\n{'='*70}")
        print("  AI-Recommended Methodologies Based on EDA Profile")
        print(f"{'='*70}\n")
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

    print(f"\n{'='*70}")
    print(f"  AquaScope — Pipeline Result: {result.method_name}")
    print(f"{'='*70}\n")
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
        out_path.write_text(json.dumps({
            "method_id": result.method_id,
            "method_name": result.method_name,
            "summary": result.summary,
            "metrics": result.metrics,
            "details": result.details,
        }, indent=2, default=str))
        print(f"\n  ✓ Full results saved → {out_path}")


def cmd_list_methods(args: argparse.Namespace) -> None:
    """List all available methodologies and pipelines."""
    from aquascope.ai_engine.knowledge_base import get_all_methodologies
    from aquascope.pipelines.model_builder import list_available_pipelines

    pipelines = set(list_available_pipelines())
    methods = get_all_methodologies()

    print(f"\n{'='*70}")
    print(f"  AquaScope — {len(methods)} Research Methodologies")
    print(f"{'='*70}\n")

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
    """List all available data sources."""
    from aquascope.schemas.water_data import DataSource

    print(f"\n{'='*70}")
    print(f"  AquaScope — {len(DataSource)} Data Sources")
    print(f"{'='*70}\n")

    source_info = {
        "taiwan_moenv": ("Taiwan MOENV", "Taiwan", "River/tap water quality, RPI", "https://data.moenv.gov.tw"),
        "taiwan_wra": ("Taiwan WRA", "Taiwan", "Water levels, reservoir status", "https://opendata.wra.gov.tw"),
        "taiwan_civil_iot": ("Taiwan Civil IoT", "Taiwan", "Real-time sensor data (water level, flow, rain)", "https://sta.ci.taiwan.gov.tw"),
        "usgs": ("USGS", "USA", "Streamflow, water quality, gage height", "https://api.waterdata.usgs.gov"),
        "sdg6": ("UN SDG 6", "Global", "SDG 6 indicators (6.1.1 – 6.6.1)", "https://sdg6data.org"),
        "gemstat": ("GEMStat", "Global", "Freshwater quality (170+ countries)", "https://gemstat.org"),
        "wqp": ("Water Quality Portal", "USA", "Integrated WQ from USGS+EPA+400 agencies", "https://waterqualitydata.us"),
    }

    for src in DataSource:
        info = source_info.get(src.value, (src.value, "—", "—", "—"))
        print(f"  {info[0]}")
        print(f"    Region : {info[1]}")
        print(f"    Data   : {info[2]}")
        print(f"    URL    : {info[3]}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aquascope",
        description="AquaScope — Water data collection, analysis & AI research recommender",
    )
    sub = parser.add_subparsers(dest="command")

    # ── collect ──────────────────────────────────────────────────────
    p_collect = sub.add_parser("collect", help="Collect water data from an API source")
    p_collect.add_argument(
        "--source", required=True,
        choices=[
            "taiwan_moenv", "taiwan_wra_level", "taiwan_wra_reservoir",
            "usgs", "sdg6", "gemstat", "taiwan_civil_iot", "wqp",
        ],
        help="Data source to collect from",
    )
    p_collect.add_argument("--api-key", default=None, help="API key (if required)")
    p_collect.add_argument("--days", type=int, default=30, help="Number of days (USGS)")
    p_collect.add_argument("--countries", default=None, help="ISO3 country codes, comma-separated (SDG6)")
    p_collect.add_argument("--state", default=None, help="US state code e.g. US:06 (WQP)")
    p_collect.add_argument("--format", default="json", choices=["json", "csv"], help="Output format")

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
    p_run = sub.add_parser("run", help="Execute a methodology pipeline on data")
    p_run.add_argument("--method", required=True, help="Pipeline method ID (use list-methods to see available)")
    p_run.add_argument("--file", required=True, help="Path to JSON or CSV data file")
    p_run.add_argument("--config", default=None, help="Pipeline config as JSON string")
    p_run.add_argument("--output", default=None, help="Path to save results JSON")

    # ── list-methods ─────────────────────────────────────────────────
    sub.add_parser("list-methods", help="List all available research methodologies and pipelines")

    # ── list-sources ─────────────────────────────────────────────────
    sub.add_parser("list-sources", help="List all available data sources")

    args = parser.parse_args()
    commands = {
        "collect": cmd_collect,
        "recommend": cmd_recommend,
        "eda": cmd_eda,
        "quality": cmd_quality,
        "run": cmd_run_pipeline,
        "list-methods": cmd_list_methods,
        "list-sources": cmd_list_sources,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
