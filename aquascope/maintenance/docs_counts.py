"""Keep prose counts in AquaScope's documentation in sync with canonical sources."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TABLE_ROW = re.compile(
    r"^\| \[[^\]]+\]\([^)]*\) \| `([a-z0-9_]+)` \|",
    re.MULTILINE,
)

CLI_PATTERNS = {
    "README.md": r"AquaScope ships a (\d+)-command CLI",
    "docs/features.md": r"\*\*(\d+) CLI commands\*\*",
    "docs/i18n/README.fr.md": r"CLI de (\d+) commandes",
}

SOURCE_COUNT_PATTERNS = {
    "README.md": [
        r"unifies \*\*(\d+) global water-data sources\*\*",
        r"\| (\d+) unified data collectors \|",
        r"any of the (\d+) sources",
        r"(\d+) data collectors spanning five regions",
        r"All (\d+) sources",
        r"multipage workspace with (\d+) live sources",
    ],
    "docs/index.md": [
        r"unifies \*\*(\d+) global water-data sources\*\*",
        r"\| (\d+) unified data collectors\s*\|",
    ],
    "docs/features.md": [r"## Data Collection \((\d+) sources\)"],
    "docs/data_sources.md": [r"\*\*(\d+) collectors\*\*"],
    "docs/i18n/README.fr.md": [
        r"espace de travail multipage avec (\d+) sources"
    ],
    "CITATION.cff": [
        r"interface to (\d+) global water data sources"
    ],
}


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _table_row_count() -> int:
    return len(TABLE_ROW.findall(_read("docs/data_sources.md")))


def _cli_command_count() -> int:
    text = _read("aquascope/cli.py")
    return len(re.findall(r"(?<![A-Za-z_])sub\.add_parser\(", text))


def canonical_counts() -> dict[str, int]:
    """Return the canonical source and top-level CLI command counts."""
    return {
        "sources": _table_row_count(),
        "cli_commands": _cli_command_count(),
    }


def _replace_one(
    text: str,
    pattern: str,
    expected: int,
    *,
    relative: str,
    label: str,
) -> tuple[str, str | None]:
    matches = list(re.finditer(pattern, text))

    if len(matches) != 1:
        raise ValueError(
            f"{relative} must contain exactly one phrase for "
            f"{pattern!r}; found {len(matches)}"
        )

    match = matches[0]
    current = int(match.group(1))

    if current == expected:
        return text, None

    start, end = match.span(1)
    updated = text[:start] + str(expected) + text[end:]

    return updated, f"{relative}: {label} {current} -> {expected}"


def apply(write: bool = False) -> list[str]:
    """Report or repair documentation count drift."""
    counts = canonical_counts()
    changes: list[str] = []

    source_files = set(SOURCE_COUNT_PATTERNS)
    cli_files = set(CLI_PATTERNS)

    for relative in sorted(source_files | cli_files):
        text = _read(relative)
        updated = text

        for pattern in SOURCE_COUNT_PATTERNS.get(relative, []):
            updated, change = _replace_one(
                updated,
                pattern,
                counts["sources"],
                relative=relative,
                label="sources",
            )
            if change:
                changes.append(change)

        cli_pattern = CLI_PATTERNS.get(relative)
        if cli_pattern is not None:
            updated, change = _replace_one(
                updated,
                cli_pattern,
                counts["cli_commands"],
                relative=relative,
                label="CLI commands",
            )
            if change:
                changes.append(change)

        if write and updated != text:
            (ROOT / relative).write_text(updated, encoding="utf-8")

    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="rewrite stale documentation counts in place",
    )
    args = parser.parse_args(argv)

    changes = apply(write=args.fix)

    if not changes:
        print("Documentation counts are already in sync.")
        return 0

    prefix = "Updated" if args.fix else "Would update"

    for change in changes:
        print(f"{prefix}: {change}")

    if not args.fix:
        print(
            "Run `python -m aquascope.maintenance.docs_counts --fix` "
            "to apply these changes."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
