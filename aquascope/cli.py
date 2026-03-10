"""
AquaScope CLI — collect water data and get AI methodology recommendations.

Usage
-----
    python -m aquascope collect --source taiwan_moenv --api-key YOUR_KEY
    python -m aquascope collect --source usgs --days 30
    python -m aquascope recommend --parameters DO,BOD5,COD --goal "trend analysis"
    python -m aquascope recommend --from-file data/raw/water_data_20260310.json
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


def cmd_collect(args: argparse.Namespace) -> None:
    """Run a data collector and save results."""
    from aquascope.collectors import (
        SDG6Collector,
        TaiwanMOENVCollector,
        TaiwanWRAReservoirCollector,
        TaiwanWRAWaterLevelCollector,
        USGSCollector,
    )
    from aquascope.utils.storage import save_records

    source = args.source.lower()
    collector_map = {
        "taiwan_moenv": lambda: TaiwanMOENVCollector(api_key=args.api_key or ""),
        "taiwan_wra_level": lambda: TaiwanWRAWaterLevelCollector(),
        "taiwan_wra_reservoir": lambda: TaiwanWRAReservoirCollector(),
        "usgs": lambda: USGSCollector(api_key=args.api_key or "DEMO_KEY"),
        "sdg6": lambda: SDG6Collector(),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aquascope",
        description="AquaScope — Water data collection & AI research recommender",
    )
    sub = parser.add_subparsers(dest="command")

    # ── collect ──────────────────────────────────────────────────────
    p_collect = sub.add_parser("collect", help="Collect water data from an API source")
    p_collect.add_argument(
        "--source", required=True,
        choices=["taiwan_moenv", "taiwan_wra_level", "taiwan_wra_reservoir", "usgs", "sdg6"],
        help="Data source to collect from",
    )
    p_collect.add_argument("--api-key", default=None, help="API key (if required)")
    p_collect.add_argument("--days", type=int, default=30, help="Number of days (USGS)")
    p_collect.add_argument("--countries", default=None, help="ISO3 country codes, comma-separated (SDG6)")
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

    args = parser.parse_args()
    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "recommend":
        cmd_recommend(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
