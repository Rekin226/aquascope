"""Self-healing harvest: when a collector breaks, gather evidence, ask a model for a patch, verify it, hand it over.

The weekly harvest already reports (``.github/scripts/harvest_issues.py``:
one ``collector-health`` issue per failing source, a deterministic
diagnosis). This module is the repair half, kept deliberately narrow:

1. :func:`gather_evidence`: the collector's source, its tests, its registry
   entry, the error and diagnosis, the recent commits touching the file, and
   live probes of the URLs the collector talks to (status, content type, the
   first bytes), so the model reasons over facts, not memory.
2. :func:`propose_repair`: one OpenAI-compatible call (any provider the
   analyst supports) that must answer either ``no_fix`` (endpoint down,
   licence, needs a key: nothing to patch) or a unified diff limited to the
   collector file and its tests, with an explanation and a confidence.
3. :func:`apply_and_verify`: ``git apply --check``, apply, ``ruff`` on the
   touched files, the collector's own tests, and a live smoke call
   (``stations()`` or a small ``collect``); anything red reverts the files.
4. :func:`repair_source`: runs the three and returns a :class:`RepairResult`;
   the workflow script turns a verified patch into a pull request (never a
   merge) and a rejected one into a comment on the health issue.

Everything is reversible and reviewable: the model never pushes to main,
the maintainer sees the evidence, the diff and the verification log.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_SOURCE_CHARS = 60_000
MAX_PROBES = 6
PROBE_BYTES = 3_000

SYSTEM_PROMPT = """You repair one data collector in the open-source Python package aquascope.
You get: the collector's source file, its tests, its registry entry, the error the weekly harvest hit,
a deterministic diagnosis, the last commits touching the file, and live probes of the URLs the collector uses.
Answer with ONE JSON object and nothing else:
{"action": "patch" | "no_fix", "confidence": 0.0-1.0, "explanation": "...", "diff": "<unified diff or empty>"}
Rules:
- "no_fix" when the agency is simply down, throttling, needs a key the workflow lacks, or the probes show the
  same payload the parser already handles. Say why in the explanation.
- "patch" only for a code cause you can see in the evidence: a moved endpoint (probes show the new one), a
  renamed field, a changed response format, a wrong parameter name.
- The diff must be a valid unified diff (paths like a/aquascope/collectors/x.py, b/aquascope/collectors/x.py)
  touching only the collector file and, if fixtures must change, its test file. Minimal, no reformatting,
  no new dependencies, keep the public API and the return models. Update or add a small test when practical.
- Never invent URLs or fields: use only what the probes or the source show.
"""


@dataclass
class Probe:
    url: str
    status: int | None
    content_type: str = ""
    snippet: str = ""
    error: str = ""


@dataclass
class Evidence:
    source: str
    error: str
    diagnosis: str
    module_path: str
    module_source: str
    test_paths: list[str] = field(default_factory=list)
    test_source: str = ""
    registry: dict[str, Any] = field(default_factory=dict)
    git_log: str = ""
    probes: list[Probe] = field(default_factory=list)

    def to_prompt(self) -> str:
        parts = [
            f"# Source: {self.source}",
            f"Registry: {json.dumps(self.registry, ensure_ascii=False)}",
            f"Harvest error:\n{self.error[:2000]}",
            f"Diagnosis: {self.diagnosis}",
            f"Recent commits touching {self.module_path}:\n{self.git_log or '(none)'}",
            "Live probes:",
        ]
        for p in self.probes:
            parts.append(f"- {p.url}\n  status={p.status} content_type={p.content_type} error={p.error}\n"
                         f"  first bytes: {p.snippet[:600]!r}")
        parts.append(f"\n# {self.module_path}\n```python\n{self.module_source[:MAX_SOURCE_CHARS]}\n```")
        if self.test_source:
            parts.append(f"\n# {', '.join(self.test_paths)}\n```python\n{self.test_source[:20_000]}\n```")
        return "\n".join(parts)


@dataclass
class Proposal:
    action: str  # patch | no_fix
    explanation: str = ""
    confidence: float = 0.0
    diff: str = ""
    raw: str = ""
    model: str = ""

    @property
    def is_patch(self) -> bool:
        return self.action == "patch" and bool(self.diff.strip())


@dataclass
class Verification:
    applied: bool = False
    files: list[str] = field(default_factory=list)
    lint_ok: bool | None = None
    tests_ok: bool | None = None
    live_ok: bool | None = None
    log: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.applied and self.lint_ok is not False and self.tests_ok is not False and self.live_ok is not False


@dataclass
class RepairResult:
    source: str
    evidence: Evidence
    proposal: Proposal | None
    verification: Verification | None
    outcome: str  # patched | rejected | no_fix | error
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"]["module_source"] = f"<{len(self.evidence.module_source)} chars>"
        d["evidence"]["test_source"] = f"<{len(self.evidence.test_source)} chars>"
        return d


# ── 1. evidence ─────────────────────────────────────────────────────────────


def _module_path_for(source: str, repo_root: Path) -> Path:
    from aquascope.registry import build_collector

    mod = type(build_collector(source)).__module__  # e.g. aquascope.collectors.usgs (constructors do no I/O)
    return repo_root / Path(*mod.split(".")).with_suffix(".py")


def _urls_in(text: str) -> list[str]:
    urls = re.findall(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+", text)
    seen: list[str] = []
    for u in urls:
        u = u.rstrip("\"',)")
        if "{" in u or "%s" in u or u in seen:
            continue
        seen.append(u)
    return seen


def probe_url(url: str, timeout: float = 20.0) -> Probe:
    """HEAD then GET the first bytes of a URL: status, content type, a snippet. Never raises."""
    import httpx

    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": "aquascope-repair/1"}) as c:
            with c.stream("GET", url) as r:
                ct = r.headers.get("content-type", "")
                chunk = b""
                for part in r.iter_bytes():
                    chunk += part
                    if len(chunk) >= PROBE_BYTES:
                        break
                return Probe(url=url, status=r.status_code, content_type=ct,
                             snippet=chunk[:PROBE_BYTES].decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001 - a probe failing is itself evidence
        return Probe(url=url, status=None, error=f"{type(exc).__name__}: {exc}"[:200])


def gather_evidence(source: str, error: str, diagnosis: str, *, repo_root: str | Path = ".", probe: bool = True,
                    extra_urls: list[str] | None = None) -> Evidence:
    """Everything the model should see about a failing source (see module docstring)."""
    from aquascope.registry import SOURCES

    root = Path(repo_root)
    module_path = _module_path_for(source, root)
    module_source = module_path.read_text(encoding="utf-8") if module_path.exists() else ""
    stem = module_path.stem
    tests = (sorted(p for p in (root / "tests" / "test_collectors").glob(f"test_{stem}*.py"))
             if module_path.exists() else [])
    test_source = "\n\n".join(p.read_text(encoding="utf-8") for p in tests[:2])
    meta = SOURCES[source]
    registry = {"key": source, "label": meta.label, "agency": meta.agency, "country": meta.country,
                "homepage": meta.homepage, "license": meta.license, "variables": list(meta.variables)}
    try:
        git_log = subprocess.run(["git", "log", "-5", "--format=%h %ad %s", "--date=short", "--", str(module_path)],
                                 cwd=root, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        git_log = ""
    probes: list[Probe] = []
    if probe:
        urls = list(extra_urls or []) + _urls_in(module_source)
        for u in urls[:MAX_PROBES]:
            probes.append(probe_url(u))
    return Evidence(source=source, error=error, diagnosis=diagnosis, module_path=module_path.relative_to(root).as_posix(),
                    module_source=module_source, test_paths=[p.relative_to(root).as_posix() for p in tests],
                    test_source=test_source, registry=registry, git_log=git_log, probes=probes)


# ── 2. proposal ─────────────────────────────────────────────────────────────


def _parse_proposal(text: str) -> Proposal:
    raw = text or ""
    body = raw.strip()
    m = re.search(r"\{.*\}", body, re.S)
    data: dict[str, Any] = {}
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            # tolerate a diff with raw newlines inside the JSON string
            try:
                fixed = re.sub(r'("diff"\s*:\s*")(.*)("\s*}\s*$)',
                               lambda mm: mm.group(1) + mm.group(2).replace("\n", "\\n") + mm.group(3),
                               m.group(0), flags=re.S)
                data = json.loads(fixed)
            except Exception:  # noqa: BLE001
                data = {}
    action = str(data.get("action") or "no_fix").lower()
    diff = str(data.get("diff") or "")
    if diff and not diff.endswith("\n"):
        diff += "\n"
    try:
        conf = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return Proposal(action=action if action in ("patch", "no_fix") else "no_fix",
                    explanation=str(data.get("explanation") or ""), confidence=max(0.0, min(1.0, conf)), diff=diff,
                    raw=raw)


def propose_repair(evidence: Evidence, *, client: Any = None, model: str | None = None, provider: str | None = None,
                   api_key: str | None = None, base_url: str | None = None) -> Proposal:
    """One model call over the evidence; returns a parsed :class:`Proposal` (``no_fix`` on any parse trouble)."""
    from aquascope.ai_engine.analyst import resolve_llm
    from aquascope.ai_engine.llm_transport import make_client

    cfg = {"model": model or "unknown"}
    if client is None:
        cfg = resolve_llm(provider, model, api_key, base_url)
        client = make_client(cfg["api_key"], cfg["base_url"])
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": evidence.to_prompt()}],
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    prop = _parse_proposal(text)
    prop.model = str(cfg["model"])
    return prop


# ── 3. verification ─────────────────────────────────────────────────────────


def _diff_files(diff: str) -> list[str]:
    files = []
    for m in re.finditer(r"^\+\+\+ b/(.+)$", diff, re.M):
        files.append(m.group(1).strip())
    return files


def _run(cmd: list[str], cwd: Path, timeout: int, log: list[str]) -> bool:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.append(f"$ {' '.join(cmd)}\n  timed out after {timeout}s")
        return False
    tail = (r.stdout + r.stderr).strip().splitlines()[-25:]
    log.append(f"$ {' '.join(cmd)}\n  exit {r.returncode}\n  " + "\n  ".join(tail))
    return r.returncode == 0


ALLOWED_PREFIXES = ("aquascope/collectors/", "tests/test_collectors/")


def apply_and_verify(proposal: Proposal, evidence: Evidence, *, repo_root: str | Path = ".", run_tests: bool = True,
                     live_check: bool = True, allowed_prefixes: tuple[str, ...] = ALLOWED_PREFIXES,
                     test_timeout: int = 900) -> Verification:
    """Apply the diff in the working tree, lint, run the collector's tests and a live smoke; revert on any failure."""
    root = Path(repo_root)
    v = Verification()
    t0 = time.perf_counter()
    if not proposal.is_patch:
        v.log.append("no patch to apply")
        return v
    files = _diff_files(proposal.diff)
    bad = [f for f in files if not f.startswith(allowed_prefixes)]
    if not files or bad:
        v.log.append(f"diff touches files outside the allowed paths: {bad or '(none listed)'}")
        return v
    v.files = files
    patch_path = root / ".repair.patch"
    patch_path.write_text(proposal.diff, encoding="utf-8")
    try:
        if not _run(["git", "apply", "--check", str(patch_path)], root, 60, v.log):
            return v
        if not _run(["git", "apply", str(patch_path)], root, 60, v.log):
            return v
        v.applied = True
        v.lint_ok = _run(["ruff", "check", *files], root, 120, v.log)
        if v.lint_ok and run_tests:
            tests = [f for f in files if f.startswith("tests/")] or evidence.test_paths
            if tests:
                v.tests_ok = _run(["python", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests], root,
                                  test_timeout, v.log)
            else:
                v.log.append("no collector tests found; skipping pytest")
        if v.lint_ok and v.tests_ok is not False and live_check:
            v.live_ok = _live_smoke(evidence.source, root, v.log)
        if not v.ok:
            _run(["git", "apply", "-R", str(patch_path)], root, 60, v.log)  # revert, new files included
            v.applied = False
    finally:
        patch_path.unlink(missing_ok=True)
        v.seconds = round(time.perf_counter() - t0, 1)
    return v


def _live_smoke(source: str, root: Path, log: list[str]) -> bool:
    """Import the patched collector in a fresh interpreter and ask it for a few stations."""
    code = (
        "import sys, json\n"
        "from aquascope.registry import build_collector\n"
        f"c = build_collector({source!r})\n"
        "try:\n"
        "    out = c.stations(max_items=5) if c.supports_stations() else c.collect(max_items=5)\n"
        "except TypeError:\n"
        "    out = c.stations() if c.supports_stations() else c.collect()\n"
        "print(json.dumps({'n': len(out)}))\n"
        "sys.exit(0 if len(out) > 0 else 3)\n"
    )
    return _run(["python", "-c", code], root, 300, log)


# ── 4. the loop ─────────────────────────────────────────────────────────────


def repair_source(source: str, error: str, diagnosis: str, *, repo_root: str | Path = ".", client: Any = None,
                  model: str | None = None, provider: str | None = None, probe: bool = True, run_tests: bool = True,
                  live_check: bool = True, min_confidence: float = 0.5) -> RepairResult:
    """Evidence -> proposal -> verification for one failing source. Never touches git branches or remotes."""
    ev = gather_evidence(source, error, diagnosis, repo_root=repo_root, probe=probe)
    try:
        prop = propose_repair(ev, client=client, model=model, provider=provider)
    except Exception as exc:  # noqa: BLE001 - report, do not raise out of a maintenance loop
        return RepairResult(source, ev, None, None, "error", f"model call failed: {type(exc).__name__}: {exc}")
    if not prop.is_patch:
        return RepairResult(source, ev, prop, None, "no_fix", prop.explanation)
    if prop.confidence < min_confidence:
        return RepairResult(source, ev, prop, None, "rejected", f"confidence {prop.confidence:.2f} < {min_confidence}")
    ver = apply_and_verify(prop, ev, repo_root=repo_root, run_tests=run_tests, live_check=live_check)
    if ver.ok:
        note = ""
    elif ver.files:
        note = "patch applied but verification failed"
    else:
        note = "patch could not be applied"
    return RepairResult(source, ev, prop, ver, "patched" if ver.ok else "rejected", note)


__all__ = ["Evidence", "Probe", "Proposal", "RepairResult", "Verification", "apply_and_verify", "gather_evidence",
           "probe_url", "propose_repair", "repair_source"]
