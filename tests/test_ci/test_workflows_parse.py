"""Every workflow file has to be valid YAML, and say so before it is pushed.

A workflow GitHub cannot parse does not fail loudly: it registers no triggers
at all, so `workflow_dispatch` reports "this workflow has no such trigger" and
the schedule never fires. The only visible sign is a run with no jobs. That is
how the showcase workflow (#233) shipped broken: a git commit message body sat
at column 0 inside a `run: |` block, which ends the block scalar and takes the
rest of the file with it.

These checks are cheap and would have caught it on the branch.

PyYAML alone is not enough, which the second occurrence proved: a paragraph at
column 0 that happens to contain a colon parses as a perfectly valid top-level
key, so the file "parses" while GitHub rejects it. Hence the allow-list below.
"""

from __future__ import annotations

import re
import subprocess
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


# GitHub Actions' complete set of top-level workflow keys. Anything else means a
# block scalar ended early and swallowed part of the file into the document.
ALLOWED_TOP_LEVEL = {"name", "run-name", "on", "permissions", "env", "defaults", "concurrency", "jobs"}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_workflow_has_no_keys_github_would_not_recognise(path: Path) -> None:
    """The failure this catches: text meant for a `run:` script parsed as YAML.

    `gh pr create --body "line one\n\nline two"` written inline in a `run: |`
    block puts "line two" at column 0 once the block indent is stripped. If it
    contains a colon, PyYAML reads it as a new top-level key and reports success;
    GitHub answers 422 "Unexpected value". Naming the allowed keys catches it.
    """
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML reads the bare key `on` as the boolean True (YAML 1.1); GitHub does not.
    keys = {"on" if k is True else k for k in loaded}
    unexpected = sorted(str(k) for k in keys - ALLOWED_TOP_LEVEL)
    assert not unexpected, (
        f"{path.name} has top-level keys GitHub does not define: {unexpected}. "
        "Usually a line at column 0 inside a `run: |` block ended the block early."
    )


# ── the shell inside the YAML ───────────────────────────────────────────────
#
# A `run:` script is shell that nothing checks until a runner executes it. That
# is how a heredoc whose terminator arrived indented (YAML strips the block's
# indent, the terminator sat inside an `else` and kept two spaces) got as far as
# a live job before failing with "here-document delimited by end-of-file".
# `bash -n` parses without running, and finds exactly that.

_EXPRESSION = re.compile(r"\$\{\{[^}]*\}\}")


def _runs(workflow: dict):
    """Every `run:` script in a workflow, with its step name."""
    for job in (workflow.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            script = step.get("run")
            if not script:
                continue
            shell = step.get("shell", "bash")
            if shell not in ("bash", "sh"):
                continue
            yield step.get("name") or "<unnamed step>", script


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_run_script_is_valid_shell(path: Path) -> None:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name, script in _runs(loaded):
        # ${{ ... }} is GitHub's, not the shell's; stand it in with a bare word
        # so what is left is the shell the runner would actually execute.
        checked = _EXPRESSION.sub("EXPR", script)
        result = subprocess.run(["bash", "-n"], input=checked, text=True, capture_output=True)
        assert result.returncode == 0, (
            f"{path.name}, step {name!r}: the shell does not parse.\n{result.stderr.strip()}"
        )
