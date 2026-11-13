"""The publish gate: an answer about another river is not published (#731).

``write`` used to skip only a recording whose tool calls all failed. The
Kingston recording answered a question about the Thames with the Hogsmill and
went live, so a wrong river now keeps the example off the panel until a
recording gets it right.
"""

from __future__ import annotations

from pathlib import Path

from aquascope import showcase


def _entry(*, river_passed: bool) -> showcase.ShowcaseEntry:
    return showcase.ShowcaseEntry(
        id="kingston-flood",
        question="What is the 100-year flood of the Thames at Kingston?",
        shows="Finding a gauge and fitting a flood frequency curve.",
        answer="The 100-year flood is 13.31 m3/s.",
        checks=[
            {"name": "tools_were_used", "passed": True, "detail": ""},
            {"name": "river_matches_the_question", "passed": river_passed,
             "detail": "" if river_passed else "The answer quotes the Hogsmill, not the Thames."},
        ],
        steps=3,
        model="m",
        provider="p",
        recorded="2026-09-21T10:48:51+00:00",
    )


def test_a_recording_that_quotes_another_river_is_not_written(tmp_path: Path) -> None:
    written = showcase.write([_entry(river_passed=False)], tmp_path)
    assert "kingston-flood" not in written
    assert not (tmp_path / "kingston-flood.json").exists()
    index = (tmp_path / "index.json").read_text(encoding="utf-8")
    assert "kingston-flood" not in index


def test_a_recording_on_the_named_river_is_written(tmp_path: Path) -> None:
    written = showcase.write([_entry(river_passed=True)], tmp_path)
    assert "kingston-flood" in written
    assert (tmp_path / "kingston-flood.json").exists()
