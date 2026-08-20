"""The Explorer is a static app with no bundler, so these checks stand in for a build.

They catch the failures that only show up in a browser otherwise: an element id
that a module reaches for but the page does not define, an import that points at
a file that was moved, a `__BUILD__` token that the deploy step would not replace
(the browser would then fetch a literal "?v=__BUILD__" forever), and the two
formatting helpers that carry the fixes from #231.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPLORER = ROOT / "explorer"
INDEX = EXPLORER / "index.html"
MODULES = sorted((EXPLORER / "src").glob("*.js")) + [EXPLORER / "app.js"]

# Created at runtime by the module that owns them, so they are not in index.html.
DYNAMIC_IDS = {"ask-this-station", "cite-copy"}


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def _ids_in_html() -> set[str]:
    return set(re.findall(r'id="([^"]+)"', _html()))


def test_every_element_id_used_by_javascript_exists_in_the_page() -> None:
    ids = _ids_in_html()
    missing: dict[str, list[str]] = {}
    for path in MODULES:
        text = path.read_text(encoding="utf-8")
        used = set(re.findall(r'\$\("([^"]+)"\)', text)) | set(re.findall(r'getElementById\("([^"]+)"\)', text))
        gone = sorted(u for u in used - ids if u not in DYNAMIC_IDS)
        if gone:
            missing[path.name] = gone
    assert not missing, f"element ids used in JS but absent from index.html: {missing}"


def test_module_imports_resolve() -> None:
    bad: list[str] = []
    for path in MODULES:
        for spec in re.findall(r'from "([^"]+)"', path.read_text(encoding="utf-8")):
            if spec.startswith("http"):
                continue
            target = (path.parent / spec.split("?")[0]).resolve()
            if not target.exists():
                bad.append(f"{path.name} -> {spec}")
    assert not bad, f"unresolved imports: {bad}"


def test_index_html_references_every_module_entry_point() -> None:
    html = _html()
    assert 'src="./app.js?v=__BUILD__"' in html
    assert 'type="module"' in html
    assert 'href="./style.css?v=__BUILD__"' in html


def test_tabs_and_panes_line_up() -> None:
    """Every tab controls a pane that exists, and every pane is labelled by its tab."""
    html = _html()
    tabs = re.findall(r'role="tab"[^>]*aria-controls="([^"]+)"[^>]*id="([^"]+)"', html)
    assert tabs, "no tabs found in index.html"
    ids = _ids_in_html()
    for pane, tab_id in tabs:
        assert pane in ids, f"tab {tab_id} controls a missing pane {pane}"
        pane_html = re.search(rf'id="{re.escape(pane)}"[^>]*>', html)
        assert pane_html, pane
        assert f'aria-labelledby="{tab_id}"' in pane_html.group(0), f"pane {pane} is not labelled by {tab_id}"


def _build_module():
    sys.path.insert(0, str(EXPLORER))
    import build  # noqa: PLC0415  (the script is a CLI, not a package)

    return build


def test_build_copies_the_modules_and_nothing_ships_with_a_build_token(tmp_path: Path) -> None:
    build = _build_module()
    files = build.text_files()
    assert Path("src/core.js") in files, "the ES modules must ship with the site"
    assert Path("index.html") in files

    # Anything carrying the token must be in the copied set, or it would deploy
    # with a literal "?v=__BUILD__" and never pick up a new version.
    for path in EXPLORER.rglob("*"):
        if path.is_file() and path.suffix in {".js", ".css", ".html"}:
            if "__BUILD__" in path.read_text(encoding="utf-8"):
                assert path.relative_to(EXPLORER) in files, f"{path.name} uses __BUILD__ but build.py skips it"

    wheel = tmp_path / "aquascope-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"not a real wheel, assemble() only copies it")
    out = tmp_path / "site"
    build.assemble(out, wheel, "abc1234", space_readme=False)

    shipped = [p for p in out.rglob("*") if p.is_file() and p.suffix in {".js", ".css", ".html"}]
    assert shipped, "assemble() produced no web assets"
    for path in shipped:
        assert "__BUILD__" not in path.read_text(encoding="utf-8"), f"{path.name} shipped with an unreplaced token"
    assert (out / "src" / "core.js").exists(), "src/ must survive the copy with its directory"
    assert json.loads((out / "wheels.json").read_text(encoding="utf-8"))["build"] == "abc1234"


pytestmark_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


@pytestmark_node
def test_fmt_honours_its_digits_argument() -> None:
    """fmt(x, 0) used to print three decimals below 10, so areas read "303.412 km²"."""
    core = EXPLORER / "src" / "core.js"
    script = f"""
    const m = await import({json.dumps(core.as_uri())});
    console.log(JSON.stringify([
      m.fmt(303.4123, 0), m.fmt(1.509, 2), m.fmt(0.41432), m.fmt(14603),
      m.fmt(3.4217), m.fmt(null), m.fmt(2320.4, 0),
    ]));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    got = json.loads(out.stdout)
    assert got[0] == "303"          # digits honoured below 1000
    assert got[1] == "1.51"         # and below 10
    assert got[2] == "0.414"        # no digits: precision follows the magnitude
    assert got[3] == "14,603"
    assert got[4] == "3.422"
    assert got[5] == "—"
    assert got[6] == "2,320"


@pytestmark_node
def test_article_agrees_with_the_word() -> None:
    """The trend sentence said "a increasing trend"."""
    core = EXPLORER / "src" / "core.js"
    script = f"""
    const m = await import({json.dumps(core.as_uri())});
    console.log(JSON.stringify([m.article("increasing"), m.article("decreasing"), m.article("upward")]));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == ["an", "a", "an"]


@pytestmark_node
def test_fold_text_ignores_accents_and_case() -> None:
    core = EXPLORER / "src" / "core.js"
    script = f"""
    const m = await import({json.dumps(core.as_uri())});
    console.log(JSON.stringify([m.foldText("Le Rhône à Anthon"), m.foldText("MÜNCHEN")]));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == ["le rhone a anthon", "munchen"]


# ── the layer catalogue (#232) ──────────────────────────────────────────────
#
# Every layer on the map has to be free, keyless and correctly credited. These
# checks are the standing guard on that: a layer added later with no licence,
# with a key in its URL, or from a host whose terms forbid this use, fails here.

LAYERS = EXPLORER / "src" / "layers.js"

# Checked on 2026-08-20: keyless, CORS-enabled, and usable from a static page.
ALLOWED_TILE_HOSTS = {
    "tiles.openfreemap.org",          # vector basemaps, no key, no limits stated
    "tiles.maps.eox.at",              # Sentinel-2 cloudless and terrain, CC BY / CC BY-NC-SA
    "gibs.earthdata.nasa.gov",        # NASA GIBS imagery and environmental rasters
    "basemap.nationalmap.gov",        # USGS imagery, public domain, US only
    "elevation-tiles-prod.s3.amazonaws.com",  # AWS Terrain Tiles (Mapzen/Tilezen)
    "wmts.terrascope.be",             # ESA WorldCover, CC BY 4.0
}

# Hosts that must never appear: their terms do not allow this use, or they need
# a billing-enabled key.
FORBIDDEN_HOSTS = (
    "mt.google.com", "khms", "maps.googleapis.com",
    "server.arcgisonline.com", "ibasemaps-api.arcgis.com",
)


def _layers_json() -> dict:
    script = f"""
    const m = await import({json.dumps(LAYERS.as_uri())});
    console.log(JSON.stringify({{
      basemaps: m.BASEMAPS, overlays: m.OVERLAYS, dem: m.TERRAIN_DEM,
      defaultDate: m.defaultDate(new Date("2026-08-20T00:00:00Z")),
      monthly: m.layerDate(m.overlayById("storage"), "2026-08-13"),
      daily: m.layerDate(m.overlayById("precip"), "2026-08-13"),
      precipUrl: m.tileUrls(m.overlayById("precip"), "2026-08-13")[0],
      credits: m.creditLines("satellite", ["precip"], {{terrain: true}}),
      years: m.recordYears({{period_start: "2000-01-01", period_end: "2020-01-01"}}, new Date("2026-01-01T00:00:00Z")),
      openEnded: m.recordYears({{period_start: "2016-01-01"}}, new Date("2026-01-01T00:00:00Z")),
      color: m.breakColor(m.RECORD_BREAKS, 30),
      noColor: m.breakColor(m.RECORD_BREAKS, null),
    }}));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytestmark_node
def test_every_layer_is_credited_and_licensed() -> None:
    data = _layers_json()
    for layer in [*data["basemaps"], *data["overlays"], data["dem"]]:
        assert layer.get("attribution"), f"{layer.get('id')} has no attribution"
        assert layer.get("licence"), f"{layer.get('id')} has no licence"


@pytestmark_node
def test_layer_ids_are_unique() -> None:
    data = _layers_json()
    for key in ("basemaps", "overlays"):
        ids = [layer["id"] for layer in data[key]]
        assert len(ids) == len(set(ids)), f"duplicate ids in {key}: {ids}"
    assert sum(1 for b in data["basemaps"] if b.get("default")) == 1, "exactly one basemap is the default"


@pytestmark_node
def test_tile_hosts_are_keyless_and_allowed() -> None:
    data = _layers_json()
    urls = []
    for layer in [*data["basemaps"], *data["overlays"], data["dem"]]:
        urls.extend(layer.get("tiles") or [])
        if layer.get("url"):
            urls.append(layer["url"])
        if layer.get("legend"):
            urls.append(layer["legend"])
    assert urls
    for url in urls:
        host = re.sub(r"^https?://([^/?]+).*$", r"\1", url)
        assert host in ALLOWED_TILE_HOSTS, f"{host} is not in the checked, keyless allow-list"
        assert not re.search(r"[?&](api_?key|token|access_token)=", url, re.I), f"{url} carries a key"


def test_no_forbidden_tile_host_anywhere_in_the_explorer() -> None:
    """Google tiles and Esri's legacy imagery are out; keep them out."""
    for path in [*MODULES, LAYERS, INDEX]:
        text = path.read_text(encoding="utf-8")
        for host in FORBIDDEN_HOSTS:
            assert host not in text, f"{path.name} references {host}"


@pytestmark_node
def test_time_layers_carry_a_date_and_snap_correctly() -> None:
    data = _layers_json()
    for layer in [*data["basemaps"], *data["overlays"]]:
        for tile in layer.get("tiles") or []:
            same = ("{date}" in tile) == bool(layer.get("time"))
            assert same, f"{layer['id']}: date placeholder and time flag disagree"
    assert data["defaultDate"] == "2026-08-13", "the default date is a week back, for data latency"
    assert data["daily"] == "2026-08-13"
    assert data["monthly"] == "2026-08-01", "monthly products snap to the first of the month"
    assert "{date}" not in data["precipUrl"] and "2026-08-13" in data["precipUrl"]


@pytestmark_node
def test_credits_cover_everything_on_screen() -> None:
    data = _layers_json()
    labels = [c["label"] for c in data["credits"]]
    assert labels == ["Satellite (2016)", "Elevation", "Precipitation rate"]
    assert all(c["attribution"] and c["licence"] for c in data["credits"])


@pytestmark_node
def test_record_length_comes_from_the_catalog_period() -> None:
    data = _layers_json()
    assert abs(data["years"] - 20) < 0.1
    assert abs(data["openEnded"] - 10) < 0.1, "an open-ended record runs to today"
    assert data["color"] != data["noColor"], "a station with no period is not coloured like one with 30 years"


# ── the provider registry ───────────────────────────────────────────────────


def test_explorer_provider_json_matches_the_python_registry() -> None:
    """explorer/providers.json is generated; regenerate it when the registry changes."""
    from aquascope.ai_engine.providers import as_json  # noqa: PLC0415

    shipped = (EXPLORER / "providers.json").read_text(encoding="utf-8")
    assert shipped == as_json(), (
        "explorer/providers.json is stale: run `python -m aquascope.ai_engine.providers`"
    )


def test_the_page_has_no_second_provider_list() -> None:
    """The model ids live in one place; the page keeps only a tiny offline fallback."""
    ask = (EXPLORER / "src" / "ask.js").read_text(encoding="utf-8")
    assert "providers.json" in ask, "the page must read the generated registry"
    # The fallback is deliberately one provider plus `custom`; more than that is drift.
    fallback = ask.split("let ASK_PROVIDERS = {", 1)[1].split("};", 1)[0]
    assert fallback.count("base_url:") <= 2, "the offline fallback grew into a second registry"
