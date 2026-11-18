"""Score the issued forecasts against what the gauge then recorded (#785).

The Archive keeps every day's forecast (``forecasts/issued/<date>.parquet``) and every day's gauge snapshot
(``forecasts/status/<date>.parquet``, :mod:`aquascope.archive.forecasts`). Read side by side, they are a
verification record. This module pairs them on gauge and valid date and scores each model, raw and corrected to
the gauge separately, per lead day and month of the valid date:

* ``median_abs_log10_error``: the median of ``|log10(forecast) - log10(observed)|`` with the ensemble mean as
  the forecast (the number the river card, the watch digest and Floods ahead show). ``factor`` is
  ``10 ** median_abs_log10_error`` ("within a factor of 2.6"). Positive flows only.
* ``crps_log10``: the mean continuous ranked probability score of the stored quantiles in log10 flow, from the
  quantile (pinball) scores at 0.25, 0.5 and 0.75 (:func:`crps_from_quantiles`). Scale free, so a small brook and a
  big river weigh the same; for a point forecast it reduces to the absolute log10 error.
* ``hit_rate`` and ``false_alarm_ratio``: the forecast mean and the observed flow against the gauge's own 2-year
  flow (``gauge_q2``, from the gauge's annual maxima, carried on every issued row), with the four counts.

Every row carries its pair count, gauges and date range, and ``enough_pairs`` is false below ``min_pairs``
(:data:`MIN_PAIRS`): the number exists, but a card should say "not enough pairs yet". Numbers come from the
published files by code only, no key and no model call. ``docs/forecast-verification.md`` says what the scores
mean and where they stop.

Writes ``<out>/verify/forecast_skill.parquet`` and ``<out>/verify/summary.json``:

    python -m aquascope.archive.verify --out build [--archive DIR] [--days N] [--since D] [--until D]
    aquascope evidence verify --out build --days 14

Needs the ``archive`` extra (pyarrow).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_PAIRS",
    "crps_from_quantiles",
    "observations",
    "pair",
    "read_archive",
    "run",
    "score",
    "summary",
]

FOLDER = "verify"
TABLE = "forecast_skill.parquet"
SUMMARY = "summary.json"
DEFAULT_REPO = "Rekin226/aquascope-gauges"
#: Below this many pairs a score is published but flagged (``enough_pairs`` false).
MIN_PAIRS = 30
#: The forecast that is scored as a single number: the ensemble mean, as the cards and Floods ahead use it.
POINT = "mean"
#: The stored quantiles the CRPS is built from, as (level, column).
QUANTILES: tuple[tuple[float, str], ...] = ((0.25, "p25"), (0.5, "median"), (0.75, "p75"))
#: The gauge flow an exceedance is counted against.
THRESHOLD = "gauge_q2"
#: Column suffix of each variant in the issued files.
VARIANTS: dict[str, str] = {"raw": "", "corrected": "_c"}
KEYS = ["source", "station_id", "valid_date"]
ALL_MONTHS = "all"

METHOD = {
    "pairing": "issued rows joined to the gauge's value on the same source, station and valid date, from the "
               "status snapshots (the newest snapshot wins when several carry the same day); observed and "
               "forecast flows must be positive",
    "median_abs_log10_error": "median of |log10(forecast mean) - log10(observed)|; factor is 10 to that power",
    "crps_log10": "mean CRPS in log10 flow, approximated as 2/3 times the sum of the pinball losses of the 0.25, "
                  "0.5 and 0.75 quantiles (quantiles sorted first)",
    "hit_rate": "hits / (hits + misses), an event being a flow above the gauge's 2-year flow (gauge_q2)",
    "false_alarm_ratio": "false alarms / (hits + false alarms), same event",
    "month": "month of the valid date, plus 'all' for every month together",
}


# ── the scores ──────────────────────────────────────────────────────────────


def crps_from_quantiles(quantiles: Any, levels: Iterable[float], observed: Any) -> np.ndarray:
    """CRPS approximated from a few quantiles: ``2 / K * sum_k QS_tau_k(q_k, y)``.

    ``QS_tau(q, y) = (1{y < q} - tau) * (q - y)`` is the pinball (quantile) loss, and the CRPS is twice its
    integral over ``tau`` in (0, 1) (Gneiting and Raftery 2007, eq. 22; Laio and Tamea 2007). With the K levels
    evenly spread, the mean over them is a rectangle-rule estimate of that integral, so the result is coarse with
    three quantiles and it ignores the tails beyond them. ``quantiles`` is (n, K), ``observed`` is (n,); rows are
    sorted first so crossed quantiles cannot make the score negative. Units are those of the inputs.
    """
    q = np.sort(np.asarray(quantiles, dtype=float).reshape(len(np.atleast_1d(observed)), -1), axis=1)
    tau = np.asarray(list(levels), dtype=float)
    y = np.asarray(observed, dtype=float).reshape(-1, 1)
    if q.shape[1] != tau.size:
        raise ValueError(f"{q.shape[1]} quantile columns for {tau.size} levels")
    loss = ((y < q).astype(float) - tau) * (q - y)
    return 2.0 * loss.mean(axis=1)


def observations(status: pd.DataFrame) -> pd.DataFrame:
    """One observed value per source, station and day from status snapshots.

    ``status`` has the status columns (``source``, ``station_id``, ``value_date``, ``value``) and optionally
    ``snapshot`` (the file's date); a day seen in several snapshots keeps the newest one's value.
    """
    if status is None or status.empty:
        return pd.DataFrame(columns=[*KEYS, "observed"])
    df = status.copy()
    if "snapshot" not in df:
        df["snapshot"] = ""
    df["valid_date"] = df["value_date"].astype(str).str[:10]
    df["station_id"] = df["station_id"].astype(str)
    df["observed"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("snapshot", kind="stable").drop_duplicates(KEYS, keep="last")
    return df[[*KEYS, "observed"]].reset_index(drop=True)


def pair(issued: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """Issued forecast rows joined to the observed flow on source, station and valid date.

    Keeps rows whose observed flow is positive; the forecast columns are left as issued (a variant is dropped per
    score where its own value is missing or not positive). Adds ``observed`` and ``month`` (``YYYY-MM`` of the
    valid date).
    """
    if issued is None or issued.empty or obs is None or obs.empty:
        return pd.DataFrame(columns=[*KEYS, "model", "lead_day", "observed", "month"])
    df = issued.copy()
    df["station_id"] = df["station_id"].astype(str)
    df["valid_date"] = df["valid_date"].astype(str).str[:10]
    out = df.merge(obs, on=KEYS, how="inner")
    out = out[out["observed"] > 0].copy()
    out["lead_day"] = pd.to_numeric(out["lead_day"], errors="coerce").astype("Int64")
    out = out[out["lead_day"].notna()]
    out["month"] = out["valid_date"].str[:7]
    return out.reset_index(drop=True)


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _score_group(g: pd.DataFrame, suffix: str, min_pairs: int) -> dict[str, Any] | None:
    point = pd.to_numeric(g.get(POINT + suffix), errors="coerce") if (POINT + suffix) in g else None
    if point is None:
        return None
    ok = point.notna() & (point > 0)
    g, point = g[ok], point[ok]
    n = len(g)
    if n == 0:
        return None
    obs = g["observed"].to_numpy(float)
    err = np.log10(point.to_numpy(float)) - np.log10(obs)
    med_abs = float(np.median(np.abs(err)))
    row: dict[str, Any] = {
        "n_pairs": n, "n_gauges": int(g[["source", "station_id"]].drop_duplicates().shape[0]),
        "first_valid": str(g["valid_date"].min()), "last_valid": str(g["valid_date"].max()),
        "enough_pairs": bool(n >= min_pairs),
        "median_abs_log10_error": round(med_abs, 4), "factor": round(10 ** med_abs, 3),
        "median_log10_error": round(float(np.median(err)), 4),
    }
    cols = [c + suffix for _, c in QUANTILES]
    if all(c in g for c in cols):
        qv = g[cols].apply(pd.to_numeric, errors="coerce")
        qok = (qv.notna() & (qv > 0)).all(axis=1).to_numpy()
        if qok.any():
            crps = crps_from_quantiles(np.log10(qv.to_numpy(float)[qok]), [lv for lv, _ in QUANTILES],
                                       np.log10(obs[qok]))
            row["crps_log10"], row["n_crps"] = round(float(crps.mean()), 4), int(qok.sum())
    row.setdefault("crps_log10", None)
    row.setdefault("n_crps", 0)
    thr = pd.to_numeric(g[THRESHOLD], errors="coerce").to_numpy(float) if THRESHOLD in g else np.full(n, np.nan)
    tok = np.isfinite(thr) & (thr > 0)
    o_ev, f_ev = obs[tok] > thr[tok], point.to_numpy(float)[tok] > thr[tok]
    hits, misses = int((o_ev & f_ev).sum()), int((o_ev & ~f_ev).sum())
    false_alarms, quiet = int((~o_ev & f_ev).sum()), int((~o_ev & ~f_ev).sum())
    row.update({"n_threshold": int(tok.sum()), "hits": hits, "misses": misses, "false_alarms": false_alarms,
                "correct_negatives": quiet, "hit_rate": _ratio(hits, hits + misses),
                "false_alarm_ratio": _ratio(false_alarms, hits + false_alarms)})
    return row


def score(pairs: pd.DataFrame, *, min_pairs: int = MIN_PAIRS) -> pd.DataFrame:
    """The scores per model, variant (``raw`` / ``corrected``), lead day and month, one row each.

    ``month`` is the valid date's ``YYYY-MM``, plus :data:`ALL_MONTHS` pooling every month. A variant with no
    usable pair in a group (GloFAS corrected, while it is not issued) has no row.
    """
    rows: list[dict[str, Any]] = []
    if pairs is None or pairs.empty:
        return pd.DataFrame(rows)
    pooled = pairs.assign(month=ALL_MONTHS)
    for frame in (pairs, pooled):
        for (model, lead, month), g in frame.groupby(["model", "lead_day", "month"], sort=True):
            for variant, suffix in VARIANTS.items():
                got = _score_group(g, suffix, min_pairs)
                if got:
                    rows.append({"model": model, "variant": variant, "lead_day": int(lead), "month": month, **got})
    out = pd.DataFrame(rows)
    return out.sort_values(["model", "variant", "month", "lead_day"], kind="stable").reset_index(drop=True)


def _plain(v: Any) -> Any:
    """A JSON-ready scalar: numpy numbers to Python, NaN to None."""
    if isinstance(v, np.generic):
        v = v.item()
    return None if isinstance(v, float) and np.isnan(v) else v


def summary(table: pd.DataFrame, *, min_pairs: int = MIN_PAIRS, inputs: dict[str, Any] | None = None,
            headline_lead: int = 1) -> dict[str, Any]:
    """A small JSON-ready digest: pair counts per model and variant, the headline lead, and error by lead."""
    doc: dict[str, Any] = {
        "about": "Issued AquaScope forecasts scored against the gauge (#785). Raw and corrected-to-gauge values "
                 "are scored separately; the gauges behind each model and lead differ.",
        "made": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "min_pairs": min_pairs,
        "method": METHOD, "inputs": inputs or {}, "file": f"{FOLDER}/{TABLE}",
        "pairs": {}, "headline": [], "by_lead": {},
    }
    if table is None or table.empty:
        return doc
    pooled = table[table["month"] == ALL_MONTHS]
    for model in sorted(pooled["model"].unique()):
        doc["pairs"][model] = {}
        doc["by_lead"][model] = {}
        for variant in VARIANTS:
            t = pooled[(pooled["model"] == model) & (pooled["variant"] == variant)]
            doc["pairs"][model][variant] = int(t["n_pairs"].sum())
            if t.empty:
                continue
            doc["by_lead"][model][variant] = {str(int(r.lead_day)): r.median_abs_log10_error
                                              for r in t.itertuples()}
            head = t[t["lead_day"] == headline_lead]
            if not head.empty:
                r = head.iloc[0].to_dict()
                doc["headline"].append({k: _plain(v) for k, v in r.items()})
    return doc


# ── reading the Archive ─────────────────────────────────────────────────────


def _date_of(name: str) -> date | None:
    try:
        return date.fromisoformat(Path(name).stem[:10])
    except ValueError:
        return None


def _window(dates: list[date], *, since: date | None, until: date | None, days: int | None) -> list[date]:
    keep = sorted(d for d in dates if (since is None or d >= since) and (until is None or d <= until))
    return keep[-days:] if days else keep


def _read_parquet_bytes(data: bytes | None) -> pd.DataFrame | None:
    return pd.read_parquet(io.BytesIO(data)) if data else None


def read_archive(archive: str | Path | None = None, *, repo_id: str = DEFAULT_REPO, since: date | None = None,
                 until: date | None = None, days: int | None = None, workers: int = 8
                 ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """The issued rows and status snapshots to score, from a local copy of the Archive or the Hub.

    ``archive`` is a folder holding ``forecasts/issued/`` and ``forecasts/status/`` (None reads the published
    dataset). Issues are chosen by date (``since``, ``until``, then the newest ``days`` of them); snapshots are
    read from the first chosen issue to the newest one available, since a forecast is verified after it is issued.
    """
    if archive is not None:
        root = Path(archive) / "forecasts"
        issued_paths = {d: p for p in (root / "issued").glob("*.parquet") if (d := _date_of(p.name))}
        status_paths = {d: p for p in (root / "status").glob("*.parquet") if (d := _date_of(p.name))}
        chosen = _window(list(issued_paths), since=since, until=until, days=days)
        snaps = sorted(d for d in status_paths if chosen and d >= chosen[0])
        issued = [pd.read_parquet(issued_paths[d]) for d in chosen]
        status = [pd.read_parquet(status_paths[d]).assign(snapshot=d.isoformat()) for d in snaps]
    else:
        from aquascope.archive.forecasts import _get, published_url, read_published_json

        manifest = read_published_json("forecasts/manifest.json", repo_id)
        files = {d: i["file"] for i in manifest.get("issues") or [] if i.get("file") and (d := _date_of(i["date"]))}
        chosen = _window(list(files), since=since, until=until, days=days)
        snaps: list[date] = []
        if chosen:
            newest = _date_of(str((manifest.get("status") or {}).get("date") or "")) or chosen[-1]
            snaps = [chosen[0] + timedelta(days=k) for k in range((max(newest, chosen[-1]) - chosen[0]).days + 1)]

        def fetch(path: str) -> pd.DataFrame | None:
            return _read_parquet_bytes(_get(published_url(path, repo_id)))

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            issued = list(pool.map(fetch, [files[d] for d in chosen]))
            got = list(pool.map(fetch, [f"forecasts/status/{d.isoformat()}.parquet" for d in snaps]))
        status = [s.assign(snapshot=d.isoformat()) for d, s in zip(snaps, got) if s is not None]
        snaps = [d for d, s in zip(snaps, got) if s is not None]
        issued = [x for x in issued if x is not None]
    issued_df = pd.concat(issued, ignore_index=True) if issued else pd.DataFrame()
    status_df = pd.concat(status, ignore_index=True) if status else pd.DataFrame()
    meta = {"source": str(archive) if archive is not None else f"hf://datasets/{repo_id}",
            "issues": {"n": len(chosen), "first": chosen[0].isoformat() if chosen else None,
                       "last": chosen[-1].isoformat() if chosen else None, "rows": int(len(issued_df))},
            "snapshots": {"n": len(snaps), "first": snaps[0].isoformat() if snaps else None,
                          "last": snaps[-1].isoformat() if snaps else None}}
    return issued_df, status_df, meta


def run(out: str | Path, *, archive: str | Path | None = None, repo_id: str = DEFAULT_REPO,
        since: date | None = None, until: date | None = None, days: int | None = None,
        min_pairs: int = MIN_PAIRS) -> dict[str, Any]:
    """Read, pair, score and write ``<out>/verify/forecast_skill.parquet`` and ``summary.json``; returns the
    summary."""
    issued, status, meta = read_archive(archive, repo_id=repo_id, since=since, until=until, days=days)
    pairs = pair(issued, observations(status))
    table = score(pairs, min_pairs=min_pairs)
    folder = Path(out) / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    table.to_parquet(folder / TABLE, index=False, compression="zstd")
    doc = summary(table, min_pairs=min_pairs, inputs={**meta, "pairs": int(len(pairs))})
    (folder / SUMMARY).write_text(json.dumps(doc, indent=1, default=str))
    return doc


def _iso(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.archive.verify", description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", required=True, help="Writes <out>/verify/forecast_skill.parquet and summary.json")
    ap.add_argument("--archive", default=None, help="A local copy of the dataset (default: the Hub)")
    ap.add_argument("--repo", default=os.environ.get("HF_DATASET", DEFAULT_REPO))
    ap.add_argument("--since", default=None, help="First issue date, YYYY-MM-DD")
    ap.add_argument("--until", default=None, help="Last issue date, YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=None, help="Only the newest N issues")
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    doc = run(a.out, archive=a.archive, repo_id=a.repo, since=_iso(a.since), until=_iso(a.until), days=a.days,
              min_pairs=a.min_pairs)
    print(json.dumps({k: doc[k] for k in ("inputs", "pairs", "headline")}, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
