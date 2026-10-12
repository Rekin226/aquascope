"""Publish a harvested folder to a Hugging Face dataset repo."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from aquascope.utils.imports import require

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Rekin226/aquascope-gauges"


def _explain(exc: Exception, repo_id: str, token: str) -> Exception:
    """Hugging Face reports a rejected token as ``RepositoryNotFoundError: 401 ... Please use create_repo``,
    which reads as a missing dataset and sends the reader looking in the wrong place. When the repo is in fact
    public and readable without a token, the token is the problem: say so."""
    text = f"{type(exc).__name__}: {exc}"
    if "401" not in text and "403" not in text and "Invalid username or password" not in text:
        return exc
    public = False
    try:
        import urllib.request

        with urllib.request.urlopen(  # noqa: S310 - a fixed https host
            f"https://huggingface.co/api/datasets/{repo_id}", timeout=10
        ) as resp:
            public = resp.status == 200
    except Exception:  # noqa: BLE001 - the probe is a nicety, never the failure
        public = False
    where = "the repository secret HF_TOKEN" if os.environ.get("GITHUB_ACTIONS") else "HF_TOKEN in your environment"
    tail = (
        f"The dataset {repo_id} exists and is readable without a token, so the token is what was refused: "
        f"it has expired, was revoked, or lacks write access. Rotate {where} "
        "(https://huggingface.co/settings/tokens, a token with write access)."
        if public
        else f"Either the token was refused (expired, revoked or read-only: rotate {where}) or {repo_id} does "
        "not exist and the token cannot create it."
    )
    return PermissionError(f"Hugging Face refused the upload to {repo_id}. {tail} Underlying error: {text}")


def _commit_with_deletions(hub, api, folder: Path, repo_id: str, patterns: list[str], delete: list[str],
                           message: str):
    """One commit that uploads the folder's files matching ``patterns`` and deletes ``delete`` (each a file the
    caller has seen listed; a file being uploaded is never deleted)."""
    from huggingface_hub.utils import filter_repo_objects

    names = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    chosen = list(filter_repo_objects(names, allow_patterns=patterns))
    ops = [hub.CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(folder / n)) for n in chosen]
    ops += [hub.CommitOperationDelete(path_in_repo=d) for d in delete if d not in set(chosen)]
    return api.create_commit(repo_id=repo_id, repo_type="dataset", operations=ops, commit_message=message)


def publish_folder(
    folder: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    *,
    token: str | None = None,
    commit_message: str | None = None,
    create: bool = True,
    allow_patterns: list[str] | tuple[str, ...] | None = None,
    delete: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Upload ``folder`` to the ``repo_id`` dataset and return the commit URL.

    Only files matching ``allow_patterns`` are uploaded (by default the Archive's parquet, GeoJSON, JSON,
    csv.gz, FlatGeobuf, PMTiles and README; the feeds add ``*.xml`` and the monthly bulletin its HTML,
    Markdown and PNG).

    The token comes from ``token``, then ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``,
    then the local ``huggingface_hub`` login. ``delete`` names files in the dataset (paths from its root) to delete
    in the same commit; the daily jobs pass the dated files their retention rule prunes
    (:mod:`aquascope.archive.retention`). Nothing is ever bundled in the
    package. Needs the ``archive`` extra.
    """
    hub = require("huggingface_hub", feature="archive publishing", group="archive")
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a directory")
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise PermissionError(
            "No Hugging Face token: publishing needs HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in the environment, "
            "or a huggingface_hub login. In CI it is the repository secret HF_TOKEN."
        )
    api = hub.HfApi(token=token)
    try:
        if create:
            api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
        patterns = (
            list(allow_patterns)
            if allow_patterns
            else ["*.parquet", "*.geojson", "*.json", "*.csv.gz", "*.fgb", "*.pmtiles", "README.md"]
        )
        info = None
        if delete:
            try:
                info = _commit_with_deletions(hub, api, folder, repo_id, patterns, list(delete),
                                              commit_message or "aquascope harvest")
            except Exception as exc:  # noqa: BLE001 - pruning is housekeeping; the day's files still go up
                logger.warning("the commit that also deletes %d file(s) failed (%s: %s); uploading without "
                               "deleting, the next publish prunes them", len(delete), type(exc).__name__, exc)
        if info is None:
            info = api.upload_folder(
                folder_path=str(folder),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message or "aquascope harvest",
                allow_patterns=patterns,
            )
    except Exception as exc:  # noqa: BLE001 - re-raised below, with the cause named
        raise _explain(exc, repo_id, token) from exc
    url = getattr(info, "commit_url", None) or str(info)
    logger.info("Published %s to https://huggingface.co/datasets/%s (%s)", folder, repo_id, url)
    return url
