"""Worked examples of the Analyst, recorded once so anyone can read them without a key.

The Explorer's Ask panel opens on a credentials form. Most visitors will never
fill it in, so most visitors never see what the Analyst does. That is a poor
trade: the interesting part (which tools ran, what came back, how the answer is
assembled from tool results with its Data and Methods sections) does not need a
model at request time. It needs a model *once*.

So CI runs these questions with the maintainer's key and publishes the traces.
The page replays them: the reader sees the question, every tool call, the answer
and the checks, and can press "run the tools again" to re-run the deterministic
half in their own browser, live, with no key at all. The prose is the only part
that is a recording, and the page says so.

    python -m aquascope.showcase --out explorer/showcase
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aquascope import __version__

logger = logging.getLogger(__name__)

__all__ = ["QUESTIONS", "ShowcaseEntry", "already_recorded", "build", "diagnose", "load_recorded", "write"]

#: The questions worth showing. Each is answerable from the archive, exercises a
#: different part of the tool surface, and is short enough to read.
QUESTIONS: list[dict[str, str]] = [
    {
        "id": "kingston-flood",
        "question": "What is the 100-year flood of the Thames at Kingston, and how sure can we be?",
        "shows": "Finding a gauge, fitting a flood frequency curve, and quoting the confidence interval.",
    },
    {
        "id": "seine-loire-lowflow",
        "question": "Compare the low flows (Q95) of the Seine at Paris and the Loire at Blois.",
        "shows": "Two records in one answer, with the flow-duration curve behind each.",
    },
    {
        "id": "potomac-trend",
        "question": "Is the Potomac at Little Falls getting drier? Use the annual-mean trend.",
        "shows": "A Mann-Kendall trend, and what a p-value does and does not license you to say.",
    },
    {
        "id": "taipei-london-climate",
        "question": "How wet is Taipei compared with London, and what is the aridity class of each?",
        "shows": "Two points with no gauge: ERA5 rainfall, FAO-56 ET0 and the aridity index.",
    },
    {
        "id": "cambridge-groundwater",
        "question": "Which UK boreholes near Cambridge have the longest groundwater records?",
        "shows": "Searching the catalog by place and variable rather than by name.",
    },
    {
        "id": "ungauged-regime",
        "question": "There is no gauge on my river in central Portugal (40.2 N, 8.0 W). What flow should I expect?",
        "shows": "Prediction in ungauged basins: similar catchments, transferred signatures, "
                 "and the skill of that transfer.",
    },
    {
        "id": "catchment-description",
        "question": "Describe the catchment upstream of 47.0 N, 68.6 W: area, climate, land cover and people.",
        "shows": "BasinATLAS attributes for anywhere on land.",
    },
    {
        "id": "flood-vs-neighbours",
        "question": "How does the 100-year flood at USGS-01013500 compare with its most similar gauged basins?",
        "shows": "Chaining donor search into an analysis of each donor.",
    },
]


@dataclass
class ShowcaseEntry:
    id: str
    question: str
    shows: str
    answer: str = ""
    markdown: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    data_used: list[dict[str, Any]] = field(default_factory=list)
    methods: list[dict[str, str]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    study: str = ""
    steps: int = 0
    model: str = ""
    provider: str = ""
    recorded: str = ""
    aquascope_version: str = __version__
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def already_recorded(out_dir: str | Path, *, fresh_for_days: float) -> set[str]:
    """Ids with a recording newer than ``fresh_for_days``, which need not be redone.

    Eight questions cost about a free tier's whole daily token budget, so a run
    that fails halfway must be able to top up rather than start again. Anything
    already recorded and still fresh is left alone.
    """
    out = Path(out_dir)
    if not out.is_dir() or fresh_for_days <= 0:
        return set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=fresh_for_days)
    fresh: set[str] = set()
    for path in out.glob("*.json"):
        if path.name == "index.json":
            continue
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            when = datetime.fromisoformat(entry["recorded"])
        except (OSError, ValueError, KeyError):
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when > cutoff and entry.get("answer"):
            fresh.add(str(entry.get("id") or path.stem))
    return fresh


def load_recorded(out_dir: str | Path) -> list[ShowcaseEntry]:
    """The recordings already on disk, so a top-up run can republish them all."""
    out = Path(out_dir)
    entries: list[ShowcaseEntry] = []
    if not out.is_dir():
        return entries
    known = {f.name for f in fields(ShowcaseEntry)}
    for path in sorted(out.glob("*.json")):
        if path.name == "index.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries.append(ShowcaseEntry(**{k: v for k, v in data.items() if k in known}))
    return entries


def build(
    questions: list[dict[str, str]] | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    max_steps: int = 8,
    on_event: Any = None,
    client: Any = None,
    pause: float = 0.0,
) -> list[ShowcaseEntry]:
    """Run each question once and record the trace.

    ``client`` is an OpenAI-compatible client for tests and for callers that
    already have one; otherwise the usual environment configuration applies.

    ``pause`` waits between questions. A free tier is limited per minute as well
    as per day, and one tool-calling question spends most of a minute's tokens,
    so eight in a row hit the wall on the second. The transport retries a 429,
    but arriving slower is cheaper than retrying.
    """
    import time

    from aquascope.ai_engine.analyst import ask

    say = on_event or (lambda m: logger.info("%s", m))
    out: list[ShowcaseEntry] = []
    for index, spec in enumerate(questions or QUESTIONS):
        if index and pause:
            say(f"waiting {pause:.0f}s so the rate-limit window refills")
            time.sleep(pause)
        entry = ShowcaseEntry(id=spec["id"], question=spec["question"], shows=spec.get("shows", ""))
        say(f"asking: {spec['id']}")
        try:
            kwargs = {"max_steps": max_steps, "on_event": lambda m: say(f"  {m}")}
            if client is not None:
                kwargs["client"] = client
                kwargs["model"] = model or "scripted"
            else:
                kwargs["provider"] = provider
                kwargs["model"] = model
            result = ask(spec["question"], **kwargs)
            entry.answer = result.answer
            entry.markdown = result.to_markdown()
            entry.tool_calls = [
                {"name": c.name, "arguments": c.arguments, "ok": c.ok, "summary": c.summary}
                for c in result.tool_calls
            ]
            entry.data_used = result.data_used
            entry.methods = result.methods
            # checks and study arrive with #234; tolerate an older analyst.
            entry.checks = getattr(result, "checks", [])
            entry.study = getattr(result, "study", "")
            entry.steps = result.steps
            entry.model = result.model
            entry.provider = result.provider
        except Exception as exc:  # noqa: BLE001 - one bad question must not lose the rest
            entry.error = f"{type(exc).__name__}: {exc}"
            say(f"  failed: {entry.error}")
        entry.recorded = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out.append(entry)
    return out


def write(entries: list[ShowcaseEntry], out_dir: str | Path) -> dict[str, str]:
    """Write one JSON per entry plus an index the page reads."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    index = []
    for entry in entries:
        if entry.error and not entry.answer:
            logger.warning("skipping %s: %s", entry.id, entry.error)
            continue
        # A model will happily answer over a failed tool call. That is what
        # `tools_were_used` catches, and an answer with nothing behind it is the
        # last thing to publish as a worked example.
        if any(c.get("name") == "tools_were_used" and not c.get("passed") for c in entry.checks):
            logger.warning("skipping %s: no tool call succeeded, so there is nothing to show", entry.id)
            continue
        path = out / f"{entry.id}.json"
        path.write_text(json.dumps(entry.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        written[entry.id] = str(path)
        index.append({
            "id": entry.id, "question": entry.question, "shows": entry.shows,
            "tools": [c["name"] for c in entry.tool_calls], "steps": entry.steps,
            "model": entry.model, "provider": entry.provider, "recorded": entry.recorded,
            "checks_passed": sum(1 for c in entry.checks if c.get("passed")),
            "checks_total": len(entry.checks),
        })
    (out / "index.json").write_text(
        json.dumps({
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "aquascope_version": __version__,
            "note": "Recorded answers: the prose was written once by the model named in each entry. "
                    "The tools can be re-run live in the page, with no key.",
            "examples": index,
        }, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    written["index"] = str(out / "index.json")
    return written


def diagnose(entries: list[ShowcaseEntry]) -> str:
    """Say what went wrong when nothing recorded, in terms of what to do about it.

    "0/8 recorded" plus eight identical stack traces is a poor error. The common
    causes are specific and each has a specific fix, so name them.
    """
    errors = [e.error or "" for e in entries if e.error]
    if not errors:
        return ""
    joined = " ".join(errors).lower()
    # 401 and 403 are different problems with different fixes, and saying the
    # wrong one sends you looking in the wrong place: 401 is "this key is not a
    # key", 403 is "this key is real but may not call the model".
    if "401" in joined or "invalid api key" in joined or "rejected" in joined:
        return (
            "Every question failed with 401: the provider rejected the key itself. It is not a "
            "permissions problem, the value is wrong. The usual cause is a truncated or "
            "whitespace-padded paste; a Groq key starts with 'gsk_' and is about fifty characters "
            "longer than that. Pipe it in rather than typing it at a prompt:\n"
            "  printf '%s' \"$(pbpaste)\" | gh secret set GROQ_API_KEY --repo <owner>/<repo>\n"
            "Then check the key is still listed at https://console.groq.com/keys"
        )
    if "403" in joined or "sufficient permissions" in joined:
        return (
            "Every question failed with 403: the key is real but is not allowed to call the "
            "model. A Hugging Face token needs the 'Make calls to Inference Providers' "
            "permission, which a write token scoped to repositories does not have. A Groq key "
            "needs nothing else, so setting GROQ_API_KEY is the shorter path: "
            "https://console.groq.com/keys"
        )
    if "429" in joined or "rate" in joined and "limit" in joined:
        return (
            "Every question hit a rate limit. Free tiers are per minute as well as per day, so "
            "re-run with --only to record a few at a time."
        )
    if "no api key" in joined or "not configured" in joined:
        return (
            "No key reached the recorder. Set one of GROQ_API_KEY, OPENAI_API_KEY or HF_TOKEN "
            "(with Inference Providers access) in the environment."
        )
    return f"Nothing recorded. The first failure was: {errors[0]}"


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - a maintenance command
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", default="explorer/showcase")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--only", default=None, help="Comma-separated ids to rebuild")
    ap.add_argument("--pause", type=float, default=25.0,
                    help="Seconds between questions, so a per-minute rate limit can refill")
    ap.add_argument("--refresh-after", type=float, default=30.0, metavar="DAYS",
                    help="Re-record an example only when its recording is older than this "
                         "(0 records every question every time). A full run costs about a free "
                         "tier's daily token budget, so a failed half is topped up, not redone.")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    questions = QUESTIONS
    kept: list[ShowcaseEntry] = []
    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        questions = [q for q in QUESTIONS if q["id"] in wanted]
        kept = [e for e in load_recorded(args.out) if e.id not in wanted]
    else:
        fresh = already_recorded(args.out, fresh_for_days=args.refresh_after)
        if fresh:
            print(f"already recorded and still fresh, skipping: {', '.join(sorted(fresh))}", flush=True)
            kept = [e for e in load_recorded(args.out) if e.id in fresh]
            questions = [q for q in QUESTIONS if q["id"] not in fresh]
    if not questions:
        print(f"nothing to record: all {len(kept)} examples are current", flush=True)
        return
    entries = build(questions, provider=args.provider, model=args.model, max_steps=args.max_steps,
                    pause=args.pause, on_event=lambda m: print(m, flush=True))
    # Republish what was kept alongside what was just recorded, or the index
    # would shrink to this run's handful.
    paths = write(kept + entries, args.out)
    ok = sum(1 for e in entries if not e.error)
    print(f"recorded {ok}/{len(entries)} this run, {len(paths) - 1} published in total", flush=True)
    if ok == 0 and not kept:
        print(diagnose(entries), flush=True)
        raise SystemExit(1)
    if ok == 0:
        print(diagnose(entries), flush=True)
        print("kept the existing recordings; run again when the budget refills", flush=True)
    print("\n".join(f"  {k}: {v}" for k, v in paths.items()))


if __name__ == "__main__":  # pragma: no cover
    main()
