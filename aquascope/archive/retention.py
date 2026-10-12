"""How long the daily jobs keep their dated files in the Archive (#597).

Three daily jobs keep a copy of each day's file by date next to ``latest``: ``flowstatus/<date>.bin.gz`` (about
0.7 MB a day for the world), ``forecasts/warnings/<date>.parquet`` (about 0.8 MB a day) and ``scout/<date>.json``.
Kept forever that is a few hundred MB a year of files nothing reads. The rule, applied by each job's own publish
step to its own folder only:

* the ``latest`` file is always kept (it is not dated);
* every dated file of the last :data:`KEEP_DAYS` days, counted back from the newest dated file (not from the
  clock, so a job that stops for a week does not lose its record);
* beyond that, the first dated file of each calendar month.

Pruned files are deleted in the same Hugging Face commit that uploads the day's files, and dropped from the
manifest's ``history`` in that commit too, so the manifest never lists a file that is gone. Deletions are limited
to names of the form ``<folder>/<YYYY-MM-DD><suffix>`` that the job's folder listing shows exist; nothing outside
the folder, and no ``latest`` or ``manifest`` file, can be deleted.

Not pruned: ``forecasts/issued/`` (the archive of record, read to measure forecast skill) and
``forecasts/status/<date>.parquet`` (about 22 kB a day for the live gauges, under 10 MB a year, and the only
record of what each gauge's flow was against normal on the day, provisional data included).

Every publish step takes ``--dry-run``, which prints what would be kept and pruned and uploads nothing.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Every dated file this many days back from the newest is kept; older ones, one a month.
KEEP_DAYS = 60
#: The folders pruned and the suffix of their dated files.
PRUNED = {"flowstatus": ".bin.gz", "forecasts/warnings": ".parquet", "scout": ".json"}
#: Daily folders deliberately left whole.
KEPT_WHOLE = ("forecasts/issued", "forecasts/status")

_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


def retain(dates: Iterable[str], *, days: int = KEEP_DAYS) -> tuple[list[str], list[str]]:
    """Split ISO dates (``YYYY-MM-DD``) into (kept, pruned), both sorted.

    Kept: every date within ``days`` days of the newest (the newest and the ``days - 1`` before it), and the first
    date of each calendar month. Anything that is not an ISO date is neither kept nor pruned (it is left out).
    """
    if days < 1:
        raise ValueError("days must be at least 1")
    good = sorted({d for d in dates if _DATE.match(str(d)) and _is_date(d)})
    if not good:
        return [], []
    newest = date.fromisoformat(good[-1])
    cutoff = newest - timedelta(days=days - 1)
    first_of_month: dict[str, str] = {}
    for d in good:
        first_of_month.setdefault(d[:7], d)
    keep = [d for d in good if date.fromisoformat(d) >= cutoff or first_of_month[d[:7]] == d]
    kept = set(keep)
    return keep, [d for d in good if d not in kept]


def _is_date(text: str) -> bool:
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def date_of(name: str, suffix: str) -> str | None:
    """The date of a dated file's base name (``2026-10-11.bin.gz`` with suffix ``.bin.gz``), else None."""
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    return stem if _DATE.match(stem) and _is_date(stem) else None


def list_published(folder: str, repo_id: str, token: str | None = None) -> list[str] | None:
    """The base names of the files directly under ``folder`` in the dataset; [] when the folder is not there yet,
    None when it cannot be listed (then nothing is deleted this time)."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:  # pragma: no cover - the archive extra is needed to publish at all
        return None
    try:
        items = HfApi(token=token).list_repo_tree(repo_id, path_in_repo=folder, repo_type="dataset")
        return [str(i.path).rsplit("/", 1)[-1] for i in items if type(i).__name__ == "RepoFile"]
    except EntryNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001 - pruning is housekeeping; it never fails the day's publish
        logger.warning("could not list %s in %s (%s); nothing is pruned this time", folder, repo_id, exc)
        return None


def plan(stage: str | Path, folder: str, suffix: str, *, repo_id: str, token: str | None = None,
         days: int = KEEP_DAYS, listing: list[str] | None = None) -> dict[str, Any]:
    """Apply the rule to a staged copy of ``folder`` (``stage`` is the folder itself) and return the plan.

    The dates are those published, those staged and those in the staged manifest's ``history``. Staged dated
    files that fall to the rule are removed from the stage, the staged ``manifest.json`` (when there is one) keeps
    only the history entries whose date is kept, and ``delete`` lists the published files to delete, as paths in
    the dataset. ``listing`` is the folder's published base names (default: read from the dataset).
    """
    stage = Path(stage)
    if listing is None:
        listing = list_published(folder, repo_id, token)
    remote: dict[str, str] = {}
    for name in listing or []:
        d = date_of(name, suffix)
        if d:
            remote[d] = name
    local: dict[str, Path] = {}
    for p in stage.iterdir():
        d = date_of(p.name, suffix) if p.is_file() else None
        if d:
            local[d] = p
    manifest = stage / "manifest.json"
    doc = json.loads(manifest.read_text()) if manifest.exists() else None
    history = (doc or {}).get("history") or []
    in_history = {str(h.get("issue_date")) for h in history if isinstance(h, dict)}
    keep, prune = retain(set(remote) | set(local) | in_history, days=days)
    gone = set(prune)
    for d in gone & set(local):
        local[d].unlink()
    if doc is not None and history:
        doc["history"] = [h for h in history if not (isinstance(h, dict) and str(h.get("issue_date")) in gone)]
        doc["retention"] = describe(days)
        manifest.write_text(json.dumps(doc, indent=1))
    delete = [f"{folder}/{remote[d]}" for d in sorted(gone & set(remote))]
    for path in delete:   # the scope check: only this folder's dated files, never latest or a manifest
        name = path[len(folder) + 1:]
        if not path.startswith(folder + "/") or "/" in name or date_of(name, suffix) is None:
            raise AssertionError(f"refusing to delete {path}: not a dated file of {folder}/")
    return {"folder": folder, "days": days, "listed": listing is not None, "keep": keep, "prune": prune,
            "delete": delete, "history_dropped": sorted(gone & in_history)}


def describe(days: int = KEEP_DAYS) -> str:
    """The rule in one sentence, for a manifest."""
    return (f"The latest file is always kept; dated files are kept for the last {days} days (counted back from the "
            "newest) and, before that, the first of each calendar month. Older dated files are deleted.")


def dry_run(src: str | Path, folder: str, suffix: str, *, repo_id: str, token: str | None = None,
            days: int = KEEP_DAYS) -> dict[str, Any]:
    """What a publish of ``src`` (the job's local folder) would keep and prune; nothing is written or uploaded."""
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        shutil.copytree(src, stage)
        res = plan(stage, folder, suffix, repo_id=repo_id, token=token, days=days)
    res["dry_run"] = True
    return res
