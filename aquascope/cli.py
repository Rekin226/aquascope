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
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
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
    from aquascope.collectors import (
        AquastatCollector,
        CopernicusCollector,
        EUWFDCollector,
        GEMStatCollector,
        JapanMLITCollector,
        KoreaWAMISCollector,
        OpenMeteoCollector,
        SDG6Collector,
        TaiwanCivilIoTCollector,
        TaiwanDataGovCollector,
        TaiwanMOENVCollector,
        TaiwanWRAFhyCollector,
        TaiwanWRAIoTCollector,
        TaiwanWRAReservoirCollector,
        TaiwanWRAWaterLevelCollector,
        USGSCollector,
        WaPORCollector,
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
        "aquastat": lambda: AquastatCollector(),
        "taiwan_civil_iot": lambda: TaiwanCivilIoTCollector(),
        "taiwan_wra_fhy": lambda: TaiwanWRAFhyCollector(),
        "taiwan_wra_iot": lambda: TaiwanWRAIoTCollector(),
        "taiwan_datagov": lambda: TaiwanDataGovCollector(),
        "wqp": lambda: WQPCollector(),
        "openmeteo": lambda: OpenMeteoCollector(mode=args.mode or "weather"),
        "copernicus": lambda: CopernicusCollector(),
        "wapor": lambda: WaPORCollector(),
        "eu_wfd": lambda: EUWFDCollector(),
        "japan_mlit": lambda: JapanMLITCollector(),
        "korea_wamis": lambda: KoreaWAMISCollector(),
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
        "taiwan_wra_fhy": ("Taiwan WRA Fhy", "Taiwan", "Real-time water level, rainfall, flow (防災資訊網)", "https://fhy.wra.gov.tw/WraApi"),
        "taiwan_wra_iot": ("Taiwan WRA IoT", "Taiwan", "Real-time groundwater level, rainfall accumulation", "https://iot.wra.gov.tw"),
        "taiwan_datagov": ("Taiwan Data.gov.tw", "Taiwan", "Real-time river & groundwater level (open gov data)", "https://data.gov.tw"),
        "usgs": ("USGS", "USA", "Streamflow, water quality, gage height", "https://api.waterdata.usgs.gov"),
        "sdg6": ("UN SDG 6", "Global", "SDG 6 indicators (6.1.1 – 6.6.1)", "https://sdg6data.org"),
        "gemstat": ("GEMStat", "Global", "Freshwater quality (170+ countries)", "https://gemstat.org"),
        "aquastat": ("FAO AQUASTAT", "Global", "Country-level water withdrawal and irrigation", "https://www.fao.org/aquastat"),
        "wqp": ("Water Quality Portal", "USA", "Integrated WQ from USGS+EPA+400 agencies", "https://waterqualitydata.us"),
        "openmeteo": ("Open-Meteo", "Global", "ERA5 reanalysis, weather forecasts, GloFAS discharge", "https://open-meteo.com"),
        "copernicus": ("Copernicus CDS", "Global", "GloFAS river discharge forecasts", "https://cds.climate.copernicus.eu"),
        "wapor": ("FAO WaPOR", "Global", "Satellite ET, biomass, and water productivity", "https://www.fao.org/in-action/remote-sensing-for-water-productivity"),
    }

    for src in DataSource:
        info = source_info.get(src.value, (src.value, "—", "—", "—"))
        print(f"  {info[0]}")
        print(f"    Region : {info[1]}")
        print(f"    Data   : {info[2]}")
        print(f"    URL    : {info[3]}")
        print()


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

    print(f"\n{'='*70}")
    print(f"  AquaScope — Forecast ({args.model}, {args.days} days)")
    print(f"{'='*70}\n")
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

    print(f"\n{'='*70}")
    print("  AquaScope — Threshold Alert Report")
    print(f"{'='*70}\n")
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

    print(f"\n{'='*70}")
    print("  AquaScope — Irrigation Plan")
    print(f"{'='*70}\n")
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

    print(f"\n{'='*70}")
    print("  AquaScope — Agriculture Benchmark")
    print(f"{'='*70}\n")
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

    print(f"\n{'='*70}")
    print("  AquaScope — WaPOR Productivity")
    print(f"{'='*70}\n")
    print(f"  Metric          : {result.metric_name}")
    print(f"  Unit            : {result.output_unit}")
    print(f"  Aggregate value : {result.aggregate_value:.4f}")
    print(f"  Summary         : {result.summary}")
    print()
    print(result.table.to_string(index=False))

    if result.aquastat_context:
        print(f"\n{'-'*70}")
        print("  AQUASTAT Context")
        print(f"{'-'*70}\n")
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
            "taiwan_wra_fhy", "taiwan_wra_iot", "taiwan_datagov",
            "usgs", "sdg6", "gemstat", "aquastat", "taiwan_civil_iot", "wqp",
            "openmeteo", "copernicus", "wapor", "eu_wfd",
        ],
        help="Data source to collect from",
    )
    p_collect.add_argument("--api-key", default=None, help="API key (if required)")
    p_collect.add_argument("--days", type=int, default=30, help="Number of days (USGS)")
    p_collect.add_argument("--country", default="all", help="ISO3 country code or 'all' (AQUASTAT)")
    p_collect.add_argument("--countries", default=None, help="ISO3 country codes, comma-separated (SDG6)")
    p_collect.add_argument("--state", default=None, help="US state code e.g. US:06 (WQP)")
    p_collect.add_argument("--variables", default=None, help="Comma-separated variable IDs (AQUASTAT)")
    p_collect.add_argument("--mode", default=None, help="Collector mode (openmeteo: weather/forecast/flood)")
    p_collect.add_argument("--bbox", default=None, help="Bounding box west,south,east,north (WaPOR)")
    p_collect.add_argument("--variable", default=None, help="Variable code for the selected collector (WaPOR)")
    p_collect.add_argument("--lat", type=float, default=None, help="Latitude (openmeteo/copernicus)")
    p_collect.add_argument("--lon", type=float, default=None, help="Longitude (openmeteo/copernicus)")
    p_collect.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD (openmeteo/copernicus)")
    p_collect.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (openmeteo/copernicus)")
    p_collect.add_argument("--start-year", type=int, default=2000, help="Start year (AQUASTAT)")
    p_collect.add_argument("--end-year", type=int, default=2023, help="End year (AQUASTAT)")
    p_collect.add_argument("--format", default="json", choices=["json", "csv"], help="Output format")
    p_collect.add_argument("--year", type=int, default=None, help="Year filter (EU WFD)")
    p_collect.add_argument(
        "--water-body-type", default=None,
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
    p_run = sub.add_parser("run", help="Execute a methodology pipeline on data")
    p_run.add_argument("--method", required=True, help="Pipeline method ID (use list-methods to see available)")
    p_run.add_argument("--file", required=True, help="Path to JSON or CSV data file")
    p_run.add_argument("--config", default=None, help="Pipeline config as JSON string")
    p_run.add_argument("--output", default=None, help="Path to save results JSON")

    # ── list-methods ─────────────────────────────────────────────────
    sub.add_parser("list-methods", help="List all available research methodologies and pipelines")

    # ── list-sources ─────────────────────────────────────────────────
    sub.add_parser("list-sources", help="List all available data sources")

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
    p_plot.add_argument("--type", required=True, choices=["timeseries", "forecast", "boxplot", "heatmap", "fdc"],
                        help="Plot type")
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
    p_agri_plan.add_argument("--start-date", default=None, help="Input start date YYYY-MM-DD (defaults to planting date)")
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
    p_agri_benchmark.add_argument("--all-years", action="store_true", help="Keep all country-year rows instead of using the latest year per country")
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
    p_agri_productivity.add_argument("--aquastat-file", default=None, help="Optional AQUASTAT CSV or JSON data for country benchmark context")
    p_agri_productivity.add_argument("--aquastat-year", type=int, default=None, help="Optional year filter for AQUASTAT context")
    p_agri_productivity.add_argument("--aquastat-countries", default=None, help="Optional comma-separated country names or ISO3 codes for AQUASTAT context")
    p_agri_productivity.add_argument(
        "--aquastat-metrics",
        default=None,
        help="Optional comma-separated AQUASTAT benchmark IDs for context; defaults to withdrawal share and withdrawal per irrigated area when available",
    )
    p_agri_productivity.add_argument("--aquastat-top", type=int, default=10, help="Maximum number of rows per AQUASTAT context table")
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
    p_gw.add_argument("--analysis", required=True,
                       choices=["trend", "recession", "seasonal", "hydrograph", "recharge-wtf", "theis"],
                       help="Analysis type")
    p_gw.add_argument("--file", required=True, help="Path to well level data (CSV with DatetimeIndex)")
    p_gw.add_argument("--specific-yield", type=float, default=0.15, help="Specific yield for WTF recharge (default: 0.15)")
    p_gw.add_argument("--transmissivity", type=float, default=None, help="Transmissivity m²/day (Theis)")
    p_gw.add_argument("--storativity", type=float, default=None, help="Storativity (Theis)")
    p_gw.add_argument("--pumping-rate", type=float, default=None, help="Pumping rate m³/day (Theis)")
    p_gw.add_argument("--distance", type=float, default=None, help="Distance from well in metres (Theis)")
    p_gw.add_argument("--output", default=None, help="Save results to JSON")

    # ── climate ──────────────────────────────────────────────────────
    p_climate = sub.add_parser("climate", help="Climate projections and indices")
    p_climate.add_argument("--analysis", required=True,
                           choices=["downscale", "indices", "drought", "scenario"],
                           help="Analysis type")
    p_climate.add_argument("--obs-file", default=None, help="Path to observed data (CSV)")
    p_climate.add_argument("--gcm-hist-file", default=None, help="Path to GCM historical data (CSV)")
    p_climate.add_argument("--gcm-future-file", default=None, help="Path to GCM future data (CSV)")
    p_climate.add_argument("--method", default="quantile_mapping",
                           help="Downscaling method (delta, quantile_mapping, qdm)")
    p_climate.add_argument("--index", default="cdd",
                           help="Climate index (cdd, cwd, pci, heat_wave, aridity)")
    p_climate.add_argument("--file", default=None, help="Path to data file (CSV)")
    p_climate.add_argument("--output", default=None, help="Save results to JSON")

    # ── hydro ─────────────────────────────────────────────────────────
    p_hydro = sub.add_parser("hydro", help="Run hydrological analysis (FDC, baseflow, recession, flood-freq)")
    p_hydro.add_argument("--analysis", required=True,
                         choices=["fdc", "baseflow", "recession", "flood-freq", "low-flow"],
                         help="Analysis type")
    p_hydro.add_argument("--file", required=True, help="Path to discharge data (CSV with DatetimeIndex)")
    p_hydro.add_argument("--method", default=None, help="Sub-method (e.g. lyne_hollick, eckhardt for baseflow)")
    p_hydro.add_argument("--output", default=None, help="Save results to CSV")
    p_hydro.add_argument("--n-day", type=int, default=None, help="N-day window for low-flow (default: 7)")
    p_hydro.add_argument("--return-period", type=int, default=None, help="Return period for low-flow (default: 10)")

    args = parser.parse_args()
    commands = {
        "collect": cmd_collect,
        "recommend": cmd_recommend,
        "eda": cmd_eda,
        "quality": cmd_quality,
        "run": cmd_run_pipeline,
        "list-methods": cmd_list_methods,
        "list-sources": cmd_list_sources,
        "solve": cmd_solve,
        "forecast": cmd_forecast,
        "plot": cmd_plot,
        "hydro": cmd_hydro,
        "alerts": cmd_alerts,
        "dashboard": cmd_dashboard,
        "agri": cmd_agri,
        "groundwater": cmd_groundwater,
        "climate": cmd_climate,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
