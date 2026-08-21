"""Recording the Analyst's worked examples, so a reader with no key can see what it does.

The recording path is tested with a scripted client, so it needs no key and no
network: what matters here is that a trace is captured completely enough to
replay (question, tool calls with their arguments, answer, checks, study) and
that a failure in one example does not lose the others.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from aquascope import showcase

ANALYSIS = {
    "source": "uk_ea", "station_id": "abc", "agency": "EA", "license": "OGL-UK-3.0",
    "attribution": "Environment Agency", "unit": "m3/s", "start": "1986-08-17", "end": "2026-08-15",
    "years": 40.0, "n": 14555, "name": "Kingston",
    "ffa": {"n_years": 39, "return_periods": [100], "fits": {"lp3": {"q": [500.0], "ci": [[440.0, 570.0]]}}},
    "methods": [{"name": "Log-Pearson III", "text": "t", "citation": "England 2018"}],
}


class ScriptedClient:
    """An OpenAI-compatible client that plays a fixed script (see test_analyst.FakeChat)."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            msg = SimpleNamespace(content=turn, tool_calls=None)
        else:
            calls = [
                SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
                for i, (name, args) in enumerate(turn)
            ]
            msg = SimpleNamespace(content="", tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _record_one(question="What is the 100-year flood at Kingston?"):
    client = ScriptedClient([
        [("flood_frequency", {"source": "uk_ea", "station_id": "abc"})],
        "The 100-year flood at Kingston is about 500 m3/s (90 % interval 440 to 570).",
    ])
    with patch("aquascope.mcp_server.flood_frequency", return_value=ANALYSIS):
        return showcase.build(
            [{"id": "kingston", "question": question, "shows": "a flood curve"}], client=client,
        )


def test_a_recorded_example_carries_everything_needed_to_replay() -> None:
    entries = _record_one()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.error is None
    assert entry.question.startswith("What is the 100-year flood")
    assert [c["name"] for c in entry.tool_calls] == ["flood_frequency"]
    assert entry.tool_calls[0]["arguments"] == {"source": "uk_ea", "station_id": "abc"}
    assert "500" in entry.answer
    assert "## Methods and citations" in entry.markdown
    assert entry.checks, "the checks travel with the recording"
    assert entry.study, "so does the study, so a reader can re-run it themselves"
    assert entry.model == "scripted"


def test_write_produces_one_file_per_example_and_an_index(tmp_path) -> None:
    entries = _record_one()
    paths = showcase.write(entries, tmp_path / "showcase")
    assert set(paths) == {"kingston", "index"}
    entry = json.loads((tmp_path / "showcase" / "kingston.json").read_text(encoding="utf-8"))
    assert entry["tool_calls"][0]["name"] == "flood_frequency"

    index = json.loads((tmp_path / "showcase" / "index.json").read_text(encoding="utf-8"))
    assert index["examples"][0]["id"] == "kingston"
    assert index["examples"][0]["tools"] == ["flood_frequency"]
    assert index["examples"][0]["checks_total"] >= 1
    assert "recorded" in index["note"].lower(), "the index says plainly that the prose is a recording"


def test_a_question_that_fails_does_not_lose_the_others() -> None:
    class HalfBrokenClient(ScriptedClient):
        """Answers the first question, then refuses (a key running out mid-run)."""

        def _create(self, **kwargs):
            if not self.turns:
                raise RuntimeError("no key configured")
            return super()._create(**kwargs)

    client = HalfBrokenClient([
        [("flood_frequency", {"source": "uk_ea", "station_id": "abc"})],
        "Fine.",
    ])
    with patch("aquascope.mcp_server.flood_frequency", return_value=ANALYSIS):
        entries = showcase.build([
            {"id": "ok", "question": "A fine question", "shows": ""},
            {"id": "bad", "question": "Another question", "shows": ""},
        ], client=client)
    assert entries[0].error is None, entries[0].error
    assert entries[1].error is not None and "no key configured" in entries[1].error


def test_a_failed_example_is_not_published(tmp_path) -> None:
    failed = showcase.ShowcaseEntry(id="bad", question="q", shows="", error="RuntimeError: nope")
    paths = showcase.write([failed], tmp_path / "showcase")
    assert "bad" not in paths
    index = json.loads((tmp_path / "showcase" / "index.json").read_text(encoding="utf-8"))
    assert index["examples"] == []


def test_an_authentication_failure_says_what_to_do_about_it() -> None:
    """The first live run recorded 0/8 behind eight identical 403s, which said nothing useful."""
    entries = [showcase.ShowcaseEntry(
        id=str(i), question="q", shows="",
        error="LLMHTTPError: HTTP 403 from https://router.huggingface.co/v1/chat/completions: "
              '{"error":"This authentication method does not have sufficient permissions to '
              'call Inference Providers on behalf of user Rekin226"}',
    ) for i in range(3)]
    said = showcase.diagnose(entries)
    assert "Inference Providers" in said
    assert "GROQ_API_KEY" in said, "name the free-tier way out, not just the diagnosis"


def test_a_rejected_key_is_not_reported_as_a_permission_problem() -> None:
    """The live run hit this: Groq answered 401, and the message talked about HF permissions."""
    entries = [showcase.ShowcaseEntry(
        id="a", question="q", shows="",
        error='LLMHTTPError: HTTP 401 from https://api.groq.com/openai/v1/chat/completions: '
              'the API key was rejected. {"error":{"message":"Invalid API Key"}}',
    )]
    said = showcase.diagnose(entries)
    assert "401" in said and "rejected the key" in said
    assert "gsk_" in said, "say what a good key looks like, since a truncated paste is the usual cause"
    assert "Inference Providers" not in said, "that is the 403 story, and it sends you the wrong way"


def test_a_rate_limit_is_not_reported_as_a_permission_problem() -> None:
    entries = [showcase.ShowcaseEntry(id="a", question="q", shows="", error="LLMHTTPError: HTTP 429 too many requests")]
    said = showcase.diagnose(entries)
    assert "rate limit" in said.lower() and "--only" in said


def test_nothing_is_diagnosed_when_nothing_failed() -> None:
    assert showcase.diagnose([showcase.ShowcaseEntry(id="a", question="q", shows="", answer="fine")]) == ""


def test_the_questions_are_distinct_and_described() -> None:
    ids = [q["id"] for q in showcase.QUESTIONS]
    assert len(ids) == len(set(ids))
    for q in showcase.QUESTIONS:
        assert len(q["question"]) > 30, f"{q['id']} should be a real question"
        assert q["shows"], f"{q['id']} should say what it demonstrates"


# ── topping up rather than starting again (#233) ────────────────────────────
#
# Eight questions cost roughly a free tier's entire daily token budget, so a run
# that dies halfway has to be resumable. Before this, every run re-recorded all
# eight and overwrote the index, so a half-successful run could never be filled in.

def _write_entry(dirpath, entry_id, *, days_old=0.0, answer="an answer"):
    from datetime import datetime, timedelta, timezone
    dirpath.mkdir(parents=True, exist_ok=True)
    when = datetime.now(timezone.utc) - timedelta(days=days_old)
    entry = showcase.ShowcaseEntry(
        id=entry_id, question="q?", shows="", answer=answer,
        recorded=when.isoformat(timespec="seconds"),
    )
    (dirpath / f"{entry_id}.json").write_text(json.dumps(entry.to_dict()), encoding="utf-8")
    return entry


def test_a_fresh_recording_is_not_redone(tmp_path) -> None:
    _write_entry(tmp_path / "showcase", "kingston-flood", days_old=3)
    assert showcase.already_recorded(tmp_path / "showcase", fresh_for_days=30) == {"kingston-flood"}


def test_a_stale_recording_is_redone(tmp_path) -> None:
    _write_entry(tmp_path / "showcase", "kingston-flood", days_old=45)
    assert showcase.already_recorded(tmp_path / "showcase", fresh_for_days=30) == set()


def test_an_entry_with_no_answer_does_not_count_as_recorded(tmp_path) -> None:
    _write_entry(tmp_path / "showcase", "kingston-flood", answer="")
    assert showcase.already_recorded(tmp_path / "showcase", fresh_for_days=30) == set()


def test_zero_days_means_record_everything_again(tmp_path) -> None:
    _write_entry(tmp_path / "showcase", "kingston-flood")
    assert showcase.already_recorded(tmp_path / "showcase", fresh_for_days=0) == set()


def test_a_missing_directory_is_not_an_error(tmp_path) -> None:
    assert showcase.already_recorded(tmp_path / "nothing-here", fresh_for_days=30) == set()


def test_corrupt_json_is_skipped_rather_than_fatal(tmp_path) -> None:
    d = tmp_path / "showcase"
    _write_entry(d, "good")
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    assert showcase.already_recorded(d, fresh_for_days=30) == {"good"}


def test_the_recordings_on_disk_come_back_as_entries(tmp_path) -> None:
    d = tmp_path / "showcase"
    _write_entry(d, "kingston-flood")
    _write_entry(d, "potomac-trend")
    got = showcase.load_recorded(d)
    assert sorted(e.id for e in got) == ["kingston-flood", "potomac-trend"]
    assert all(e.answer for e in got)


def test_republishing_keeps_the_old_and_the_new(tmp_path) -> None:
    """The index must not shrink to whatever this run happened to manage."""
    d = tmp_path / "showcase"
    _write_entry(d, "kingston-flood")
    kept = showcase.load_recorded(d)
    fresh = [showcase.ShowcaseEntry(id="potomac-trend", question="q?", shows="", answer="new")]
    showcase.write(kept + fresh, d)
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    assert sorted(e["id"] for e in index["examples"]) == ["kingston-flood", "potomac-trend"]
