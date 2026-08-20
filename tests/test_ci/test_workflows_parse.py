"""Every workflow file has to be valid YAML, and say so before it is pushed.

A workflow GitHub cannot parse does not fail loudly: it registers no triggers
at all, so `workflow_dispatch` reports "this workflow has no such trigger" and
the schedule never fires. The only visible sign is a run with no jobs. That is
how the showcase workflow (#233) shipped broken: a git commit message body sat
at column 0 inside a `run: |` block, which ends the block scalar and takes the
rest of the file with it.

These checks are cheap and would have caught it on the branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML arrives with the dev extra (via pre-commit)")

WORKFLOWS = sorted((Path(__file__).parents[2] / ".github" / "workflows").glob("*.yml"))


def test_there_are_workflows_to_check() -> None:
    assert WORKFLOWS, "no workflow files found, so the checks below would pass vacuously"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_workflow_parses(path: Path) -> None:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        pytest.fail(f"{path.name} is not valid YAML, so GitHub would register no triggers for it:\n{exc}")
    assert isinstance(loaded, dict), f"{path.name} should be a mapping"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_workflow_has_triggers_and_jobs(path: Path) -> None:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML reads the bare key `on:` as the boolean True, which is the YAML 1.1
    # rule GitHub does not follow. Accept either.
    triggers = loaded.get("on", loaded.get(True))
    assert triggers, f"{path.name} declares no triggers, so nothing would ever run it"
    assert loaded.get("jobs"), f"{path.name} declares no jobs"
