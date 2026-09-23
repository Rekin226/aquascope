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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _checkpoint(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def refresh(out: Path, *, budget: int = 150, timeout: float = 480,
            pairs: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """Return explicit per-source outcomes, including timeouts and partial results."""
    if budget < 1 or timeout <= 0:
        raise ValueError("budget and timeout must be positive")
    pairs = pairs if pairs is not None else [(s, v) for s, vs in HARVESTABLE.items() for v in vs]
    report: dict[str, Any] = {"run_at": _now(), "completed_at": None, "status": "running", "sources": []}
    path = out / "obs" / "refresh_status.json"
    _checkpoint(path, report)
    for source, variable in pairs:
        row: dict[str, Any] = {"source": source, "variable": variable, "started_at": _now(), "status": "running"}
        report["sources"].append(row)
        _checkpoint(path, report)
        command = [sys.executable, "-m", "aquascope.archive.refresh", "--out", str(out), "--budget", str(budget),
                   "--timeout", str(timeout), "--worker", source, variable]
        try:
            done = subprocess.run(command, timeout=timeout + 15, check=False)
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
    parser.add_argument("--worker", nargs=2, metavar=("SOURCE", "VARIABLE"))
    args = parser.parse_args()
    if args.worker:
        source, variable = args.worker
        report = harvest_observations(args.out, sources=[source], variable=variable,
                                      max_stations=args.budget, max_seconds_per_source=args.timeout)
        partial = any(h.failed or h.budget_exhausted for h in report.sources)
        raise SystemExit(2 if partial else 0)
    report = refresh(args.out, budget=args.budget, timeout=args.timeout)
    print(json.dumps(report, indent=2))
    # Partial is a publishable outcome, explicitly recorded rather than hidden.


if __name__ == "__main__":
    main()
