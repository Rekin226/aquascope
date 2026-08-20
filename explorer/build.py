"""Assemble the static Explorer site: explorer/ files + the aquascope wheel + wheels.json.

Usage::

    python explorer/build.py --out dist-explorer [--wheel dist/aquascope-X.Y.Z-py3-none-any.whl] [--build <token>]

Builds the wheel with ``python -m build`` when ``--wheel`` is not given,
replaces every ``__BUILD__`` placeholder with the build token (git short sha
by default) so browsers never serve a stale app.js after a deploy, and writes
``wheels.json`` for the worker. Pure standard library; runs in CI and locally.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "explorer"
# Every text asset of the app: the page, the ES modules in explorer/src/, and
# the recorded Analyst traces in explorer/showcase/ that the page replays (#233).
TEXT_GLOBS = ("*.html", "*.js", "*.css", "*.json", "src/*.js", "showcase/*.json")
# Assets copied byte for byte: the social preview card (og.png, drawn by
# make_og_image.py) is referenced by the page's og:image.
BINARY_GLOBS = ("*.png", "*.ico")
SKIP = {"build.py", "make_og_image.py"}


def _matching(patterns: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        for path in sorted(SRC.glob(pattern)):
            rel = path.relative_to(SRC)
            if rel.name in SKIP or rel in out:
                continue
            out.append(rel)
    return out


def binary_files() -> list[Path]:
    """Assets copied byte for byte, relative to ``explorer/``."""
    return _matching(BINARY_GLOBS)


def text_files() -> list[Path]:
    """Explorer sources to copy, relative to ``explorer/`` and in a stable order."""
    return _matching(TEXT_GLOBS)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "dev"


def build_wheel() -> Path:
    dist = ROOT / "dist"
    for old in glob.glob(str(dist / "aquascope-*.whl")):
        os.remove(old)
    subprocess.check_call([sys.executable, "-m", "build", "--wheel", "-o", str(dist)], cwd=ROOT)
    wheels = sorted(dist.glob("aquascope-*.whl"))
    if not wheels:
        raise SystemExit("wheel build produced nothing")
    return wheels[-1]


def assemble(out: Path, wheel: Path, build: str, space_readme: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for rel in text_files():
        text = (SRC / rel).read_text(encoding="utf-8").replace("__BUILD__", build)
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    for rel in binary_files():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / rel, dest)
    shutil.copy2(wheel, out / wheel.name)
    (out / "wheels.json").write_text(json.dumps({"wheel": wheel.name, "build": build}), encoding="utf-8")
    if space_readme:
        shutil.copy2(SRC / "SPACE_README.md", out / "README.md")
    plugin_src = ROOT / "integrations" / "geolibre"
    if plugin_src.exists():  # the GeoLibre plugin is served next to the Explorer
        plugin_out = out / "geolibre-plugin"
        plugin_out.mkdir(exist_ok=True)
        for name in ("plugin.json", "index.js", "style.css", "README.md"):
            shutil.copy2(plugin_src / name, plugin_out / name)
    (out / ".nojekyll").write_text("", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", default="dist-explorer")
    ap.add_argument("--wheel", default=None, help="Existing wheel to ship (default: build one)")
    ap.add_argument("--build", default=None, help="Cache-busting token (default: git short sha)")
    ap.add_argument("--no-space-readme", action="store_true", help="Skip the Hugging Face Space README")
    args = ap.parse_args()
    wheel = Path(args.wheel) if args.wheel else build_wheel()
    build = args.build or _git_sha()
    assemble(Path(args.out), wheel, build, space_readme=not args.no_space_readme)
    print(f"explorer assembled in {args.out} (wheel {wheel.name}, build {build})")


if __name__ == "__main__":
    main()
