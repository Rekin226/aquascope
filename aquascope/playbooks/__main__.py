"""``python -m aquascope.playbooks``: write ``explorer/playbooks.json`` from the YAML files."""

from __future__ import annotations

import sys
from pathlib import Path

from aquascope.playbooks import as_json


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "explorer" / "playbooks.json"
    out.write_text(as_json(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
