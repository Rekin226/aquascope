"""Bound each source/variable refresh so healthy work can always be published.

Run ``python -m aquascope.archive.refresh --out archive`` from the scheduled
workflow. A subprocess deadline also covers a collector stuck inside a retry
loop; completed station files and the manifest are checkpointed atomically.
No workers write the shared manifest concurrently.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aquascope.archive.observations import HARVESTABLE, harvest_observations


def scheduled_budget(source: str, variable: str, budget: int) -> tuple[int, float]:
    """Preserve the production per-source budgets, with USGS processed last."""
    if source == "greece_openhi":
        return 65, 300
    if source == "taiwan_cwa":
        return 15, 900
    limits = {
        ("uk_ea", "discharge"): (budget, 1200),
        ("hubeau_hydrometrie", "discharge"): (budget, 1200),
        ("uk_ea", "groundwater_level"): (max(1, budget // 2), 600),
        ("uk_ea", "precipitation"): (max(1, budget // 3), 400),
        ("uk_ea", "water_level"): (max(1, budget // 3), 400),
        ("poland_imgw", "discharge"): (budget, 1500),
        ("poland_imgw", "water_level"): (max(1, budget // 2), 300),
        ("usgs", "discharge"): (budget, 2100),
        ("usgs", "water_level"): (max(1, budget // 5), 480),
    }
    return limits.get((source, variable), (budget, 480))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _checkpoint(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def refresh(out: Path, *, budget: int = 150, timeout: float = 480,
            pairs: list[tuple[str, str]] | None = None, scheduled_budgets: bool = False) -> dict[str, Any]:
    """Return explicit per-source outcomes, including timeouts and partial results."""
    if budget < 1 or timeout <= 0:
        raise ValueError("budget and timeout must be positive")
    pairs = pairs if pairs is not None else [(s, v) for s, vs in HARVESTABLE.items() for v in vs]
    if scheduled_budgets:
        pairs = sorted(pairs, key=lambda pair: pair[0] == "usgs")
    report: dict[str, Any] = {"run_at": _now(), "completed_at": None, "status": "running", "sources": []}
    path = out / "obs" / "refresh_status.json"
    _checkpoint(path, report)
    for source, variable in pairs:
        stations, seconds = scheduled_budget(source, variable, budget) if scheduled_budgets else (budget, timeout)
        row: dict[str, Any] = {"source": source, "variable": variable, "started_at": _now(), "status": "running"}
        row.update({"station_budget": stations, "time_budget_seconds": seconds})
        report["sources"].append(row)
        _checkpoint(path, report)
        command = [sys.executable, "-m", "aquascope.archive.refresh", "--out", str(out), "--budget", str(stations),
                   "--timeout", str(seconds), "--worker", source, variable]
        try:
            done = subprocess.run(command, timeout=seconds + 15, check=False)
            row["status"] = "ok" if done.returncode == 0 else "partial" if done.returncode == 2 else "failed"
            row["exit_code"] = done.returncode
        except subprocess.TimeoutExpired:
            row["status"] = "timeout"
        row["completed_at"] = _now()
        _checkpoint(path, report)
    report["status"] = "ok" if all(r["status"] == "ok" for r in report["sources"]) else "partial"
    report["completed_at"] = _now()
    _checkpoint(path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("archive"))
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--timeout", type=float, default=480)
    parser.add_argument("--scheduled-budgets", action="store_true",
                        help="Use production station/time budgets and process USGS last")
    parser.add_argument("--worker", nargs=2, metavar=("SOURCE", "VARIABLE"))
    args = parser.parse_args()
    if args.worker:
        source, variable = args.worker
        report = harvest_observations(args.out, sources=[source], variable=variable,
                                      max_stations=args.budget, max_seconds=args.timeout)
        partial = any(h.failed or h.budget_exhausted or h.stopped for h in report.sources)
        raise SystemExit(2 if partial else 0)
    report = refresh(args.out, budget=args.budget, timeout=args.timeout, scheduled_budgets=args.scheduled_budgets)
    print(json.dumps(report, indent=2))
    # Partial is a publishable outcome, explicitly recorded rather than hidden.


if __name__ == "__main__":
    main()
