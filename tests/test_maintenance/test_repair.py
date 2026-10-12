"""Self-healing harvest: evidence, proposal parsing, and the apply-verify-revert loop on a scratch git repo."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from aquascope.maintenance import repair


def test_evidence_for_a_real_collector_without_probes():
    ev = repair.gather_evidence("uk_ea", "HTTP 404 for https://environment.data.gov.uk/x", "HTTP 404: moved",
                                repo_root=Path(__file__).resolve().parents[2], probe=False)
    assert ev.module_path == "aquascope/collectors/uk_ea.py" and "class" in ev.module_source
    assert any(p.endswith("test_uk_ea.py") for p in ev.test_paths) and ev.registry["key"] == "uk_ea"
    prompt = ev.to_prompt()
    assert "Harvest error" in prompt and "aquascope/collectors/uk_ea.py" in prompt and "Diagnosis: HTTP 404" in prompt


def test_urls_are_extracted_and_probe_failures_are_evidence(monkeypatch):
    urls = repair._urls_in('BASE = "https://api.example.org/v1/"\nx = f"https://api.example.org/{id}"\n')
    assert urls == ["https://api.example.org/v1/", "https://api.example.org/"]  # the templated tail is cut off
    p = repair.probe_url("http://127.0.0.1:9/nothing", timeout=1)
    assert p.status is None and p.error


def test_proposal_parsing_is_tolerant():
    good = '{"action": "patch", "confidence": 0.8, "explanation": "renamed field", "diff": "--- a/x\\n+++ b/x\\n"}'
    p = repair._parse_proposal("Here you go:\n" + good)
    assert p.is_patch and p.confidence == 0.8 and p.diff.endswith("\n")
    nofix = repair._parse_proposal('{"action": "no_fix", "explanation": "agency down"}')
    assert not nofix.is_patch and nofix.action == "no_fix"
    garbage = repair._parse_proposal("I cannot help")
    assert garbage.action == "no_fix" and garbage.confidence == 0.0
    weird = repair._parse_proposal('{"action": "PATCH", "confidence": "2", "diff": ""}')
    assert weird.action == "patch" and weird.confidence == 1.0 and not weird.is_patch  # empty diff is not a patch


def test_propose_repair_uses_the_client(monkeypatch):
    ev = repair.Evidence("x", "err", "diag", "aquascope/collectors/x.py", "print(1)")
    seen = {}

    class Client:
        class chat:  # noqa: N801 - mimics the SDK shape
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    seen.update(kw)
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                        content='{"action":"no_fix","confidence":0.9,"explanation":"endpoint down"}'))])

    prop = repair.propose_repair(ev, client=Client(), model="m")
    assert prop.action == "no_fix" and prop.model == "m" and seen["temperature"] == 0
    assert seen["messages"][0]["role"] == "system" and "aquascope/collectors/x.py" in seen["messages"][1]["content"]


def _scratch_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "aquascope" / "collectors").mkdir(parents=True)
    (root / "tests" / "test_collectors").mkdir(parents=True)
    (root / "aquascope" / "__init__.py").write_text("", encoding="utf-8")
    (root / "aquascope" / "collectors" / "__init__.py").write_text("", encoding="utf-8")
    (root / "aquascope" / "collectors" / "demo.py").write_text(textwrap.dedent('''
        BASE = "https://old.example.org/api"


        def endpoint() -> str:
            return BASE + "/stations"
    ''').lstrip("\n"), encoding="utf-8")
    (root / "tests" / "test_collectors" / "test_demo.py").write_text(textwrap.dedent('''
        from aquascope.collectors.demo import endpoint


        def test_endpoint_moved():
            assert endpoint() == "https://new.example.org/api/stations"
    ''').lstrip("\n"), encoding="utf-8")
    (root / "pyproject.toml").write_text("[tool.ruff]\nline-length = 120\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"], cwd=root,
                   check=True)
    return root


GOOD_DIFF = '''--- a/aquascope/collectors/demo.py
+++ b/aquascope/collectors/demo.py
@@ -1,4 +1,4 @@
-BASE = "https://old.example.org/api"
+BASE = "https://new.example.org/api"


 def endpoint() -> str:
'''


def test_apply_and_verify_accepts_a_good_patch_and_reverts_a_bad_one(tmp_path):
    root = _scratch_repo(tmp_path)
    ev = repair.Evidence("demo", "404", "HTTP 404", "aquascope/collectors/demo.py", "",
                         test_paths=["tests/test_collectors/test_demo.py"])
    good = repair.Proposal("patch", "moved endpoint", 0.9, GOOD_DIFF)
    v = repair.apply_and_verify(good, ev, repo_root=root, live_check=False)
    assert v.applied and v.lint_ok and v.tests_ok and v.ok, v.log
    assert 'https://new.example.org/api' in (root / "aquascope" / "collectors" / "demo.py").read_text(encoding="utf-8")

    # reset, then a patch that applies but breaks the test: reverted
    subprocess.run(["git", "checkout", "-q", "--", "."], cwd=root, check=True)
    bad_diff = GOOD_DIFF.replace("https://new.example.org/api", "https://wrong.example.org/api")
    v2 = repair.apply_and_verify(repair.Proposal("patch", "x", 0.9, bad_diff), ev, repo_root=root, live_check=False)
    assert not v2.ok and v2.tests_ok is False and not v2.applied
    reverted = (root / "aquascope" / "collectors" / "demo.py").read_text(encoding="utf-8")
    assert "https://old.example.org/api" in reverted  # reverted

    # a diff outside the allowed paths is refused before touching anything
    outside = GOOD_DIFF.replace("aquascope/collectors/demo.py", "aquascope/cli.py")
    v3 = repair.apply_and_verify(repair.Proposal("patch", "x", 0.9, outside), ev, repo_root=root, live_check=False)
    assert not v3.applied and "outside the allowed paths" in v3.log[0]
    # a diff that does not apply
    v4 = repair.apply_and_verify(repair.Proposal("patch", "x", 0.9, GOOD_DIFF.replace("old.example", "nope")), ev,
                                 repo_root=root, live_check=False)
    assert not v4.applied and any("git apply --check" in line for line in v4.log)


def test_repair_source_outcomes(tmp_path, monkeypatch):
    root = _scratch_repo(tmp_path)
    ev = repair.Evidence("demo", "404", "HTTP 404", "aquascope/collectors/demo.py", "",
                         test_paths=["tests/test_collectors/test_demo.py"])
    monkeypatch.setattr(repair, "gather_evidence", lambda *a, **k: ev)
    monkeypatch.setattr(repair, "propose_repair",
                        lambda e, **k: repair.Proposal("patch", "moved", 0.9, GOOD_DIFF, model="m"))
    res = repair.repair_source("demo", "404", "HTTP 404", repo_root=root, live_check=False)
    assert res.outcome == "patched" and res.verification.ok and res.to_dict()["outcome"] == "patched"
    subprocess.run(["git", "checkout", "-q", "--", "."], cwd=root, check=True)
    monkeypatch.setattr(repair, "propose_repair", lambda e, **k: repair.Proposal("patch", "moved", 0.2, GOOD_DIFF))
    assert repair.repair_source("demo", "404", "HTTP 404", repo_root=root, live_check=False).outcome == "rejected"
    monkeypatch.setattr(repair, "propose_repair", lambda e, **k: repair.Proposal("no_fix", "agency down", 0.9))
    assert repair.repair_source("demo", "404", "HTTP 404", repo_root=root).outcome == "no_fix"

    def boom(e, **k):
        raise RuntimeError("401 bad key")

    monkeypatch.setattr(repair, "propose_repair", boom)
    r = repair.repair_source("demo", "404", "HTTP 404", repo_root=root)
    assert r.outcome == "error" and "401" in r.note


@pytest.mark.parametrize("diag,expected", [
    ("HTTP 404: the endpoint moved", True), ("HTTP 429: throttled", False), ("TLS failure: chain", False),
    ("Timeout / connection error", False), ("HTTP 5xx from the agency", True),
    ("The response is not the format the collector expects", True), ("Unclassified. The full error", True),
])
def test_repairable_diagnoses(diag, expected):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "harvest_repair", Path(__file__).resolve().parents[2] / ".github" / "scripts" / "harvest_repair.py")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github" / "scripts"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._repairable(diag) is expected
