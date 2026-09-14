"""Guard against count drift between the code and every place a count is
stated in prose (see issue #117).

Three canonical values, all read from the code, never from the docs:

* the data-source registry (``aquascope.registry.SOURCES``) backs every
  source/collector count, and the table in ``docs/data_sources.md`` has to
  list exactly those ids, one row each;
* ``cli.py`` backs the CLI command count;
* ``KC_TABLE`` and ``SignatureReport`` back the crop and signature counts.

The mirrors drift apart on their own: in September 2026 the registry held 33
sources while the README said 31, docs/index.md said 29, docs/features.md said
20 and the CITATION.cff abstract said 20. Every mirror is checked here so the
next gap fails CI instead of shipping.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from aquascope.agri.crop_water import KC_TABLE
from aquascope.hydrology.signatures import SignatureReport
from aquascope.maintenance.docs_counts import (
    CLI_PATTERNS,
    SOURCE_COUNT_PATTERNS,
    TABLE_ROW,
    _cli_command_count,
)
from aquascope.registry import SOURCES

ROOT = Path(__file__).resolve().parents[1]

CROP_COUNT_PATTERNS = {
    "README.md": [r"crop water requirements for (\d+) crops"],
    "docs/index.md": [r"crop water requirements for (\d+) crops"],
    "docs/features.md": [r"\*\*Crop water requirements\*\* — (\d+) crops"],
}

# The test total is a "N+" floor, not an exact count, so it cannot be derived
# from a collection that is already running. What CI can check is that every
# copy of it states the same floor. Re-derive the floor from a full run
# (`python -m pytest -q | tail -1`) and round down, as the release pass does.
TEST_FLOOR_PATTERNS = {
    "README.md": [
        r"tests-(\d+)%2B%20passing",
        r"CAMELS benchmark with ([\d,]+)\+ tests",
        r"\*\*([\d,]+)\+ tests\*\* — covering",
    ],
    "docs/index.md": [
        r"tests-(\d+)%2B%20passing",
        r"CAMELS benchmark with ([\d,]+)\+ tests",
        r"\*\*([\d,]+)\+ tests\*\* across",
    ],
    "docs/features.md": [r"\*\*([\d,]+)\+ tests\*\* with"],
}

SIGNATURE_COUNT_PATTERNS = {
    "README.md": [r"(\d+) hydrological signatures"],
    "docs/index.md": [r"(\d+) hydrological signatures"],
    "docs/features.md": [r"\*\*(\d+) hydrological signatures\*\*"],
    "docs/i18n/README.fr.md": [r"(\d+) signatures couvrant"],
}


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _check(
    patterns: dict[str, list[str]],
    expected: int,
    label: str,
    fix_command: str | None = None,
) -> None:
    for relative, file_patterns in patterns.items():
        text = _read(relative)
        for pattern in file_patterns:
            match = re.search(pattern, text)
            assert match is not None, (
                f"{relative} no longer contains the phrase for {pattern!r}"
            )

            message = (
                f"{relative} says {match.group(1)} {label} for {pattern!r}, "
                f"but the code has {expected}"
            )
            if fix_command:
                message += f". Run `{fix_command}` to repair count drift."

            assert int(match.group(1)) == expected, message


def test_source_table_lists_every_registered_source():
    """The table is the docs' view of the registry, so it has to be one row per id."""
    listed = set(TABLE_ROW.findall(_read("docs/data_sources.md")))
    registered = set(SOURCES)
    assert listed == registered, (
        f"docs/data_sources.md is missing rows for {sorted(registered - listed)} "
        f"and lists unknown ids {sorted(listed - registered)}"
    )


def test_source_counts_match_the_registry():
    expected = len(TABLE_ROW.findall(_read("docs/data_sources.md")))
    _check(
        SOURCE_COUNT_PATTERNS,
        expected,
        "sources",
        "python -m aquascope.maintenance.docs_counts --fix",
    )


def test_crop_counts_match_the_kc_table():
    _check(CROP_COUNT_PATTERNS, len(KC_TABLE), "crops")


def test_signature_counts_match_the_report():
    _check(
        SIGNATURE_COUNT_PATTERNS,
        len(dataclasses.fields(SignatureReport)),
        "signatures",
    )


def test_the_test_floor_is_stated_the_same_everywhere():
    """The badge and the prose copies drifted apart before (1,000+ against 820+)."""
    stated = {}
    for relative, patterns in TEST_FLOOR_PATTERNS.items():
        text = _read(relative)
        for pattern in patterns:
            match = re.search(pattern, text)
            assert match is not None, (
                f"{relative} no longer contains the phrase for {pattern!r}"
            )
            stated[f"{relative}: {pattern}"] = int(
                match.group(1).replace(",", "")
            )

    assert len(set(stated.values())) == 1, (
        f"the stated test floor disagrees across files: {stated}"
    )


def test_cli_counts_match_the_parser():
    expected = _cli_command_count()
    for relative, pattern in CLI_PATTERNS.items():
        match = re.search(pattern, _read(relative))
        assert match is not None, f"{relative} CLI count sentence missing"
        assert int(match.group(1)) == expected, (
            f"{relative} says {match.group(1)} CLI commands but cli.py defines {expected}. "
            "Run `python -m aquascope.maintenance.docs_counts --fix` "
            "to repair count drift."
        )
