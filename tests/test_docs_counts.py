"""Guard against collector-count drift between the docs table and every
place the count is stated in prose (see issue #117).

The canonical count is the number of rows in the data-sources table in
docs/data_sources.md; every hand-written mention must match it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLI_PATTERN = r"AquaScope ships a (\d+)-command CLI"

README_PATTERNS = [
    r"unifies \*\*(\d+) global water-data sources\*\*",
    r"\| (\d+) unified data collectors \|",
    r"any of the (\d+) sources",
    r"(\d+) data collectors spanning four regions",
    r"All (\d+) sources",
]


def _table_row_count() -> int:
    text = (ROOT / "docs" / "data_sources.md").read_text(encoding="utf-8")
    return len(re.findall(r"^\| \[", text, flags=re.MULTILINE))


def _cli_command_count() -> int:
    text = (ROOT / "aquascope" / "cli.py").read_text(encoding="utf-8")
    return len(re.findall(r"\.add_parser\(", text))


def test_docs_intro_count_matches_table():
    text = (ROOT / "docs" / "data_sources.md").read_text(encoding="utf-8")
    match = re.search(r"\*\*(\d+) collectors\*\*", text)
    assert match is not None, "count sentence missing from docs/data_sources.md intro"
    assert int(match.group(1)) == _table_row_count()


def test_readme_counts_match_table():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = _table_row_count()
    for pattern in README_PATTERNS:
        match = re.search(pattern, text)
        assert match is not None, f"README no longer contains the phrase for {pattern!r}"
        assert int(match.group(1)) == expected, (
            f"README says {match.group(1)} for {pattern!r} but the docs/data_sources.md table has {expected} rows"
        )


def test_readme_cli_count_matches_cli():
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    match = re.search(CLI_PATTERN, text)
    assert match is not None, "README CLI count sentence missing"

    expected = _cli_command_count()

    assert int(match.group(1)) == expected, (
        f"README says {match.group(1)} CLI commands but "
        f"cli.py defines {expected}"
    )
