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


def test_the_social_card_exists_is_declared_and_ships(tmp_path: Path) -> None:
    """A link to the Explorer previewed as a blank rectangle: og:image was never set (#231)."""
    card = EXPLORER / "og.png"
    assert card.exists(), "explorer/og.png is missing; regenerate it with make_og_image.py"
    assert card.stat().st_size > 10_000, "the card looks empty"
    assert card.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "og:image has to be a real PNG (SVG is not accepted)"

    page = (EXPLORER / "index.html").read_text(encoding="utf-8")
    for tag in ('property="og:image"', 'name="twitter:image"', 'property="og:image:alt"',
                'property="og:url"'):
        assert tag in page, f"index.html is missing {tag}"
    assert 'content="https://' in page.split('property="og:image"')[1][:80], (
        "og:image must be an absolute URL: scrapers do not resolve relative ones"
    )

    build = _build_module()
    assert Path("og.png") in build.binary_files(), "the card has to be copied into the built site"

    wheel = tmp_path / "aquascope-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"not a real wheel")
    out = tmp_path / "site"
    build.assemble(out, wheel, "abc1234", space_readme=False)
    assert (out / "og.png").read_bytes() == card.read_bytes(), "the card must survive the copy unchanged"


def test_the_recorded_showcase_traces_are_part_of_the_build(tmp_path: Path, monkeypatch) -> None:
    """The traces the page replays live in explorer/showcase/, so they have to ship.

    Without the glob they simply would not be copied, and the panel would be
    empty in production while being full locally.
    """
    build = _build_module()
    src = tmp_path / "explorer"
    (src / "showcase").mkdir(parents=True)
    (src / "index.html").write_text("<!-- page -->", encoding="utf-8")
    (src / "showcase" / "index.json").write_text('{"examples": []}', encoding="utf-8")
    (src / "showcase" / "kingston.json").write_text('{"id": "kingston"}', encoding="utf-8")
    monkeypatch.setattr(build, "SRC", src)

    files = build.text_files()
    assert Path("showcase/index.json") in files
    assert Path("showcase/kingston.json") in files


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
    out = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    got = json.loads(out.stdout)
    assert got[0] == "303"          # digits honoured below 1000
    assert got[1] == "1.51"         # and below 10
    assert got[2] == "0.414"        # no digits: precision follows the magnitude
    assert got[3] == "14,603"
    assert got[4] == "3.422"
    assert got[5] == "—"
    assert got[6] == "2,320"


@pytestmark_node
def test_fmt_p_formats_small_p_values_and_handles_nones() -> None:
    """A test with p = 0.000192 must format as '< 0.001' rather than 'p = 0'."""
    core = EXPLORER / "src" / "core.js"
    script = f"""
    const m = await import({json.dumps(core.as_uri())});
    console.log(JSON.stringify([
      m.fmtP(0.000192), m.fmtP(0.0009), m.fmtP(0.001), m.fmtP(0.042),
      m.fmtP(1.0), m.fmtP(null), m.fmtP(undefined), m.fmtP(Number.NaN),
    ]));
    """
    out = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    got = json.loads(out.stdout)
    assert got[0] == "< 0.001"
    assert got[1] == "< 0.001"
    assert got[2] == "0.001"
    assert got[3] == "0.042"
    assert got[4] == "1.000"
    assert got[5] == "—"
    assert got[6] == "—"
    assert got[7] == "—"


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


# ── the tools the page offers to an in-browser agent (#236) ─────────────────


def test_webmcp_registration_is_feature_detected() -> None:
    """A browser without navigator.modelContext must be unaffected."""
    src = (EXPLORER / "src" / "webmcp.js").read_text(encoding="utf-8")
    assert "navigator.modelContext" in src
    assert "webmcpAvailable()" in src
    assert "if (!webmcpAvailable()) return false" in src


def test_webmcp_tools_are_described_and_schema_d() -> None:
    src = (EXPLORER / "src" / "webmcp.js").read_text(encoding="utf-8")
    for name in ("aquascope_find_stations", "aquascope_analyze_station", "aquascope_anywhere",
                 "aquascope_describe_catchment", "aquascope_show_on_map"):
        assert name in src, f"{name} should be offered to an agent"
    assert src.count("inputSchema") >= 5, "every tool needs a schema an agent can fill in"


def test_place_search_uses_a_geocoder_whose_terms_allow_autocomplete() -> None:
    """Photon allows autocomplete; OSM's Nominatim forbids it, so it must not be called."""
    src = (EXPLORER / "src" / "search.js").read_text(encoding="utf-8")
    assert "photon.komoot.io" in src
    called = re.findall(r'https?://[^\s"\')]+', src)
    assert not [u for u in called if "nominatim" in u.lower()], "Nominatim forbids autocomplete"


# ── focus and announcements (#231 follow-up) ────────────────────────────────
#
# Opening the drawer, the modal or the mobile rail used to leave focus where it
# was and ignore Escape, and closing one dropped focus to the top of the
# document. These exercise src/a11y.js against a hand-built DOM: node has no
# document, and the logic worth testing is the filtering and the Tab cycle.

A11Y = EXPLORER / "src" / "a11y.js"

FAKE_DOM = """
const focused = [];
const el = (name, opts = {}) => ({
  name, hidden: opts.hidden || false, disabled: opts.disabled || false,
  offsetParent: opts.detached ? null : {},
  closest: (sel) => (opts.inHidden && sel === "[hidden]" ? {} : null),
  focus() { focused.push(this.name); globalThis.document.activeElement = this; },
});
const opener = el("opener");
const first = el("first"), middle = el("middle"), last = el("last");
const skipped = el("skipped", { hidden: true });
const container = { children: [first, middle, skipped, last],
  querySelector: () => null,
  querySelectorAll: () => [first, middle, skipped, last],
  contains: (x) => [first, middle, skipped, last].includes(x) };
const handlers = [];
globalThis.document = {
  activeElement: opener,
  addEventListener: (type, fn) => handlers.push(fn),
  removeEventListener: (type, fn) => { const i = handlers.indexOf(fn); if (i >= 0) handlers.splice(i, 1); },
  contains: () => true,
  getElementById: () => null,
  createElement: () => ({ setAttribute(k, v) { this[k] = v; }, className: "", textContent: "" }),
  body: { appendChild: (x) => { globalThis.__region = x; } },
};
const outside = el("outside");
const key = (k, shift = false, target = undefined) => {
  let prevented = false;
  const ev = { key: k, shiftKey: shift, target: target === undefined ? globalThis.document.activeElement : target,
               preventDefault: () => { prevented = true; } };
  for (const h of [...handlers]) h(ev);
  return prevented;
};
"""


@pytestmark_node
def test_a_hidden_control_is_not_a_focus_target() -> None:
    script = f"""
    {FAKE_DOM}
    const m = await import({json.dumps(A11Y.as_uri())});
    const names = m.focusableWithin(container).map((e) => e.name);
    console.log(JSON.stringify(names));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == ["first", "middle", "last"]


@pytestmark_node
def test_opening_moves_focus_in_and_closing_puts_it_back() -> None:
    script = f"""
    {FAKE_DOM}
    const m = await import({json.dumps(A11Y.as_uri())});
    const release = m.captureFocus(container, {{ restoreTo: opener }});
    release();
    console.log(JSON.stringify(focused));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == ["first", "opener"], "focus goes in, then back to what opened it"


@pytestmark_node
def test_escape_closes_and_a_trap_keeps_tab_inside() -> None:
    script = f"""
    {FAKE_DOM}
    const m = await import({json.dumps(A11Y.as_uri())});
    let escapes = 0;
    m.captureFocus(container, {{ trap: true, onEscape: () => {{ escapes += 1; }} }});
    const atLast = (globalThis.document.activeElement = last, key("Tab"));     // wraps to first
    const atFirst = (globalThis.document.activeElement = first, key("Tab", true));  // wraps to last
    const middleTab = (globalThis.document.activeElement = middle, key("Tab"));     // ordinary, untouched
    key("Escape");
    console.log(JSON.stringify({{ focused, escapes, atLast, atFirst, middleTab }}));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    got = json.loads(out.stdout)
    assert got["escapes"] == 1, "Escape has to close the surface"
    assert got["atLast"] and got["atFirst"], "Tab off either end is intercepted"
    assert not got["middleTab"], "Tab in the middle is left alone"
    assert got["focused"] == ["first", "first", "last"], "wraps to the far end each way"


@pytestmark_node
def test_without_a_trap_tab_leaves_the_surface() -> None:
    """The Ask drawer sits beside the page on a wide screen; tabbing out to the map is correct."""
    script = f"""
    {FAKE_DOM}
    const m = await import({json.dumps(A11Y.as_uri())});
    m.captureFocus(container, {{ onEscape: () => {{}} }});
    globalThis.document.activeElement = last;
    console.log(JSON.stringify(key("Tab")));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) is False


@pytestmark_node
def test_announcements_go_to_one_polite_live_region() -> None:
    script = f"""
    {FAKE_DOM}
    const m = await import({json.dumps(A11Y.as_uri())});
    m.announce("GR4J failed, retry");
    const r = globalThis.__region;
    console.log(JSON.stringify({{ role: r.role, live: r["aria-live"], text: r.textContent, cls: r.className }}));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    got = json.loads(out.stdout)
    assert got["role"] == "status" and got["live"] == "polite"
    assert got["text"] == "GR4J failed, retry"
    assert got["cls"] == "visually-hidden", "announced, not shown"


def test_the_visually_hidden_class_the_live_region_uses_exists() -> None:
    css = (EXPLORER / "style.css").read_text(encoding="utf-8")
    assert ".visually-hidden" in css, "the live region would otherwise be visible on the page"


@pytestmark_node
def test_an_untrapped_surface_leaves_escape_alone_outside_itself() -> None:
    """The Ask drawer must not eat the search box's Escape, or the area tool's."""
    script = f"""
    {FAKE_DOM}
    const m = await import({json.dumps(A11Y.as_uri())});
    let escapes = 0;
    m.captureFocus(container, {{ onEscape: () => {{ escapes += 1; }} }});
    const fromOutside = key("Escape", false, outside);
    const inside = key("Escape", false, first);
    console.log(JSON.stringify({{ escapes, fromOutside, inside }}));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    got = json.loads(out.stdout)
    assert got["escapes"] == 1, "Escape inside closes it; Escape elsewhere is somebody else's"
    assert not got["fromOutside"] and got["inside"]


@pytestmark_node
def test_a_trapped_surface_owns_escape_wherever_it_came_from() -> None:
    script = f"""
    {FAKE_DOM}
    const m = await import({json.dumps(A11Y.as_uri())});
    let escapes = 0;
    m.captureFocus(container, {{ trap: true, onEscape: () => {{ escapes += 1; }} }});
    key("Escape", false, outside);
    console.log(JSON.stringify(escapes));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == 1


def test_the_ask_button_is_wired_before_anything_is_awaited() -> None:
    """Clicking Ask right after load used to do nothing at all.

    `initAsk()` began with `await loadProviders()`, a fetch of providers.json,
    and only attached the button's click listener afterwards. Between load and
    that response the app's headline control was inert and silent about it. The
    listener has to be attached before the first await; this is the invariant,
    stated where it will fail if someone moves it back.
    """
    src = (EXPLORER / "src" / "ask.js").read_text(encoding="utf-8")
    body = src[src.index("export async function initAsk()"):]
    # The statement, not the comment that explains it: match on the indentation.
    wired = body.index('\n  $("btn-ask").addEventListener')
    awaited = body.index("\n  await loadProviders();")
    assert wired < awaited, "initAsk() must wire the Ask button before it awaits the provider list"


def test_running_before_the_provider_list_arrives_says_so() -> None:
    """The Run button is live from the start too, so it has to survive an empty picker."""
    src = (EXPLORER / "src" / "ask.js").read_text(encoding="utf-8")
    run = src[src.index("async function runAsk()"):]
    guard = run[:run.index("const key =")]
    assert "ASK_PROVIDERS[provider]" in guard and "askStatus(" in guard, (
        "runAsk() must check the chosen provider exists and report it, not throw on undefined"
    )


def test_the_explorer_asks_for_the_full_record_by_default() -> None:
    """#270: the page passed a 40-year window while the note said 'full period requested'."""
    config = (EXPLORER / "config.js").read_text(encoding="utf-8")
    assert re.search(r"\byears:\s*null\b", config), "CONFIG.years is a cap; null asks for the full record"
    worker = (EXPLORER / "worker.js").read_text(encoding="utf-8")
    assert "|| 40" not in worker, "the worker must not fall back to a hard-coded window"
    assert "period_start" in worker, "the catalog's first date travels with the request"
    panel = (EXPLORER / "src" / "panel-station.js").read_text(encoding="utf-8")
    assert "period_start: r.period_start" in panel


# ── Solve (the plan-first Analyst in the page) ──────────────────────────────


def test_the_solve_drawer_is_wired_end_to_end() -> None:
    """One drawer, two modes; the worker face of team.solve / run_reviewed; the button wired before any await."""
    html = _html()
    for needed in ('id="btn-solve"', 'id="solve-pane"', 'id="ask-pane"', 'name="drawer-mode"', 'id="stage-result"'):
        assert needed in html, needed
    app = (EXPLORER / "app.js").read_text(encoding="utf-8")
    assert "initSolve" in app and "url.solve" in app
    solve = (EXPLORER / "src" / "solve.js").read_text(encoding="utf-8")
    body = solve[solve.index("export async function initSolve()"):]
    assert body.index('$("btn-solve").addEventListener') < body.index("await loadPlaybooks()")
    assert "playbooks.json" in solve, "the chips come from the generated list, not a copy in the page"
    worker = (EXPLORER / "worker.js").read_text(encoding="utf-8")
    for needed in ("solve_plan", "solve_run", "solve_progress", "run_reviewed", "execute=False",
                   "describe_catchment_from_row"):
        assert needed in worker, needed
    client = (EXPLORER / "src" / "worker-client.js").read_text(encoding="utf-8")
    assert "solve_progress" in client and "onSolveProgress" in client
    url = (EXPLORER / "src" / "url.js").read_text(encoding="utf-8")
    assert 'q.set("solve"' in url and 'q.has("solve")' in url


def test_the_solve_surface_writes_with_plain_hyphens() -> None:
    """House style: no em or en dashes in what the Solve surface says."""
    for path in (EXPLORER / "src" / "solve.js", EXPLORER / "playbooks.json"):
        text = path.read_text(encoding="utf-8")
        assert "—" not in text and "–" not in text, path.name
