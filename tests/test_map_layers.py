"""World river status on the map (#544): the months, the legend and the three faces.

One engine, three faces: aquascope.map_layers.river_status_month, the MCP tool of
the same name, `aquascope layers status`, and the Explorer's status-core.js, which
decodes the same colours (checked here). No network: the bucket listing is a fixture.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aquascope import cli
from aquascope import map_layers as ml

ROOT = Path(__file__).resolve().parents[1]
STATUS_JS = ROOT / "explorer" / "src" / "status-core.js"

LISTING = """<?xml version="1.0" encoding="UTF-8"?><ListBucketResult><Name>geoglows-v2</Name>
<Contents><Key>hydrosos/cogs/1990-01.tif</Key></Contents><Contents><Key>hydrosos/cogs/1990-02.tif</Key></Contents>
<Contents><Key>hydrosos/cogs/1990-04.tif</Key></Contents><Contents><Key>hydrosos/cogs/thumb.png</Key></Contents>
<IsTruncated>false</IsTruncated></ListBucketResult>"""


@pytest.fixture
def listed(monkeypatch):
    """The bucket as three months with a gap, without the network."""
    monkeypatch.setattr(ml, "status_months", lambda refresh=False: ["1990-01", "1990-02", "1990-04"])


# ── the months ──────────────────────────────────────────────────────────────


def test_the_listing_gives_months_and_the_next_page():
    months, token = ml.parse_listing(LISTING)
    assert months == ["1990-01", "1990-02", "1990-04"]
    assert token is None
    assert ml.parse_listing("<NextContinuationToken>a/b=</NextContinuationToken>")[1] == "a/b="
    assert ml.missing_months(months) == ["1990-03"]
    assert ml.missing_months([]) == []


def test_status_months_follows_the_pages(monkeypatch):
    pages = {None: LISTING.replace("<IsTruncated>false", "<NextContinuationToken>t1</NextContinuationToken><x"),
             "t1": "<Key>hydrosos/cogs/1990-05.tif</Key>"}
    seen = []

    class FakeClient:
        def __init__(self, **_):
            pass

        def get_text(self, url, params=None, use_cache=True):
            seen.append(params.get("continuation-token"))
            return pages[params.get("continuation-token")]

        def close(self):
            pass

    import aquascope.utils.http_client as hc

    monkeypatch.setattr(hc, "CachedHTTPClient", FakeClient)
    monkeypatch.setitem(ml._listed, "months", None)
    assert ml.status_months(refresh=True) == ["1990-01", "1990-02", "1990-04", "1990-05"]
    assert seen == [None, "t1"]


def test_parse_month_reads_months_days_and_rejects_the_rest():
    assert ml.parse_month("2026-09") == "2026-09"
    assert ml.parse_month("2026-09-15") == "2026-09"
    assert ml.parse_month(None) is None
    for bad in ("2026-13", "Sept", "2026/09"):
        with pytest.raises(ValueError):
            ml.parse_month(bad)


# ── one month ───────────────────────────────────────────────────────────────


def test_the_newest_month_by_default(listed):
    out = ml.river_status_month()
    assert out["available"] and out["month"] == "1990-04"
    assert out["url"] == "https://geoglows-v2.s3.us-west-2.amazonaws.com/hydrosos/cogs/1990-04.tif"
    assert out["valid_range"] == {"first": "1990-01", "latest": "1990-04", "months": 3}
    assert out["missing"] == ["1990-03"] and out["listed"] is True
    assert out["licence"] == "CC BY 4.0" and "GEOGLOWS" in out["attribution"]
    assert [c["id"] for c in out["legend"]] == ["much_below", "below", "normal", "above", "much_above"]


def test_a_gap_or_a_month_outside_the_range_says_so(listed):
    gap = ml.river_status_month("1990-03")
    assert gap["available"] is False and "gap" in gap["error"] and "url" not in gap
    late = ml.river_status_month("2030-01-05")
    assert late["available"] is False and "1990-01 to 1990-04" in late["error"]
    assert ml.river_status_month("nope")["error"].startswith("not a month")


def test_offline_uses_the_recorded_range():
    out = ml.river_status_month("2000-06", live=False)
    assert out["available"] and out["listed"] is False and out["checked"] == ml.CHECKED
    assert out["valid_range"]["first"] == "1990-01" and out["valid_range"]["latest"] == ml.CHECKED_LATEST
    assert ml.river_status_month("2026-03", live=False)["available"] is False


def test_a_failed_listing_falls_back_and_says_why(monkeypatch):
    def boom(refresh=False):
        raise RuntimeError("offline")

    monkeypatch.setattr(ml, "status_months", boom)
    out = ml.river_status_month("1995-01")
    assert out["available"] and "offline" in out["live_error"] and out["listed"] is False


def test_the_legend_is_the_hydrosos_classes_and_their_file_colours():
    from aquascope.nownext import STATUS_CLASSES as NOW

    assert [c["id"] for c in ml.STATUS_CLASSES] == [c["id"] for c in NOW]
    for c in ml.STATUS_CLASSES:
        assert c["hex"] == "#{:02x}{:02x}{:02x}".format(*c["rgb"])
    assert len({c["rgb"][0] for c in ml.STATUS_CLASSES}) == 5, "the red band alone tells the classes apart"


# ── the faces ───────────────────────────────────────────────────────────────


def test_the_mcp_server_offers_the_tool(listed):
    pytest.importorskip("mcp")
    from aquascope import mcp_server as m

    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert "river_status_month" in names
    assert m.river_status_month("1990-02")["url"].endswith("/1990-02.tif")


def test_cli_prints_the_month_and_legend(listed, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aquascope", "layers", "status"])
    cli.main()
    out = capsys.readouterr().out
    assert "1990-04.tif" in out and "no map for 1990-03" in out and "below the 10th percentile" in out
    monkeypatch.setattr(sys, "argv", ["aquascope", "layers", "status", "1990-02", "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["month"] == "1990-02"


def test_cli_exits_non_zero_without_a_map(listed, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aquascope", "layers", "status", "1990-03"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1 and "gap" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_explorer_decodes_the_same_colours_and_lists_the_same_months():
    script = f"""
    const m = await import({json.dumps(STATUS_JS.as_uri())});
    console.log(JSON.stringify({{ rgb: m.FILE_RGB, first: m.STATUS_FIRST, bucket: m.STATUS_BUCKET,
      prefix: m.STATUS_PREFIX, url: m.statusUrl("1990-01"), months: m.parseListing({json.dumps(LISTING)}).months }}));
    """
    out = json.loads(subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True,
                                    check=True).stdout)
    assert out["rgb"] == [c["rgb"] for c in ml.STATUS_CLASSES]
    assert out["first"] == ml.STATUS_FIRST and out["bucket"] == ml.GEOGLOWS_BUCKET and out["prefix"] == ml.STATUS_PREFIX
    assert out["url"] == ml.status_url("1990-01")
    assert out["months"] == ml.parse_listing(LISTING)[0]


# ── the one-line summary (#543 design pass) ─────────────────────────────────

np = pytest.importorskip("numpy")


def _tiff(planes, tile=8):
    """A tiny GeoTIFF laid out as GEOGLOWS writes theirs: little-endian, tiled, deflated, one plane per band."""
    import struct
    import zlib

    height, width = planes[0].shape
    across, down = -(-width // tile), -(-height // tile)
    blobs = []
    for plane in planes:
        padded = np.zeros((down * tile, across * tile), dtype=np.uint8)
        padded[:height, :width] = plane
        for r in range(down):
            for c in range(across):
                blobs.append(zlib.compress(padded[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile].tobytes()))
    n = len(blobs)
    entries = [(256, 3, 1, width), (257, 3, 1, height), (258, 3, 3, None), (259, 3, 1, 8), (262, 3, 1, 2),
               (277, 3, 1, 3), (284, 3, 1, 2), (317, 3, 1, 1), (322, 3, 1, tile), (323, 3, 1, tile),
               (324, 4, n, None), (325, 4, n, None)]
    ifd = 8
    extra = ifd + 2 + 12 * len(entries) + 4
    bits_at, offs_at, counts_at = extra, extra + 6, extra + 6 + 4 * n
    data_at = counts_at + 4 * n
    offsets, pos = [], data_at
    for b in blobs:
        offsets.append(pos)
        pos += len(b)
    out = bytearray(b"II*\x00" + struct.pack("<I", ifd) + struct.pack("<H", len(entries)))
    for tag, typ, count, value in entries:
        if tag == 258:
            value = bits_at
        elif tag == 324:
            value = offs_at
        elif tag == 325:
            value = counts_at
        out += struct.pack("<HHII", tag, typ, count, value) if typ == 4 or count > 1 else \
            struct.pack("<HHIHH", tag, typ, count, value, 0)
    out += struct.pack("<I", 0)
    out += struct.pack("<3H", 8, 8, 8)
    out += struct.pack(f"<{n}I", *offsets) + struct.pack(f"<{n}I", *[len(b) for b in blobs])
    for b in blobs:
        out += b
    return bytes(out)


def test_the_file_reader_takes_the_red_plane_of_a_tiled_planar_tiff():
    red = np.arange(10 * 13, dtype=np.uint8).reshape(10, 13)
    data = _tiff([red, red // 2, red // 3], tile=8)
    assert (ml._tiff_red(data) == red).all()
    with pytest.raises(ValueError):
        ml._tiff_red(b"MM\x00*" + data[4:])


def _world(deg=0.5):
    """Classes on a grid: normal everywhere, the Amazon much below, the Sahel below, southern Africa much
    above, Mexico's box mostly unmapped (sea)."""
    h, w = int(180 / deg), int(360 / deg)
    classes = np.full((h, w), 3, dtype=np.uint8)

    def box(west, south, east, north):
        rows = slice(int((90 - north) / deg), int((90 - south) / deg))
        return rows, slice(int((west + 180) / deg), int((east + 180) / deg))

    classes[box(-80, -15, -44, 5)] = 1
    classes[box(-17, 11, 38, 18)] = 2
    classes[box(10, -35, 41, -13)] = 5
    classes[box(-118, 7, -77, 32)] = 0
    classes[box(-100, 20, -95, 25)] = 1
    return classes


def test_region_shares_weigh_the_mapped_area_and_say_how_much_is_mapped():
    shares = {r["name"]: r for r in ml.region_shares(_world())}
    assert [r["name"] for r in ml.STATUS_REGIONS] == list(shares)
    assert shares["the Amazon"]["below"] == 1.0 and shares["the Amazon"]["above"] == 0.0
    assert shares["the Sahel"]["below"] == 1.0
    assert shares["southern Africa"]["above"] == 1.0
    assert shares["Europe"] == {"name": "Europe", "below": 0.0, "above": 0.0, "cover": 1.0}
    mexico = shares["Mexico and Central America"]
    assert mexico["cover"] < ml.MIN_COVER and mexico["below"] == 1.0   # mostly sea: left out of the headline


def test_the_headline_names_the_mostly_low_and_the_mostly_high():
    regions = ml.region_shares(_world())
    assert ml.status_headline("2026-09", regions) == (
        "River status, September 2026: much of the Amazon and the Sahel below normal, much of southern Africa above")
    wet = [dict(r, below=r["above"], above=r["below"]) for r in regions]
    assert ml.status_headline("1998-01", wet) == (
        "River status, January 1998: much of the Amazon and the Sahel above normal, much of southern Africa below")
    calm = [dict(r, below=0.3, above=0.2) for r in regions]
    assert ml.status_headline("2001-07", calm) == (
        "River status, July 2001: no large region mostly above or below normal")
    # a region at exactly half below and half above is neither
    tie = [dict(r, below=0.5, above=0.5, cover=1.0) for r in regions]
    assert "no large region" in ml.status_headline("2001-07", tie)


def test_the_summary_reads_the_month_and_answers_with_regions_and_a_headline(listed, monkeypatch):
    red = np.zeros((360, 720), dtype=np.uint8)
    for k, c in enumerate(ml.STATUS_CLASSES, start=1):
        red[_world() == k] = c["rgb"][0]
    data = _tiff([red, red, red], tile=256)

    from aquascope.utils import http_client

    monkeypatch.setattr(http_client.CachedHTTPClient, "get_bytes", lambda self, url, **kw: data)
    out = ml.river_status_summary("1990-02")
    assert out["month"] == "1990-02" and out["headline"].startswith("River status, February 1990: much of the Amazon")
    assert {r["name"] for r in out["regions"]} == {r["name"] for r in ml.STATUS_REGIONS}
    assert "not basins" in out["regions_note"]
    assert ml.river_status_summary("1990-03")["available"] is False      # a gap: nothing to read

    def broken(self, url, **kw):
        raise RuntimeError("offline")

    monkeypatch.setattr(http_client.CachedHTTPClient, "get_bytes", broken)
    assert "could not read" in ml.river_status_summary("1990-02")["error"]


def test_the_summary_faces_mcp_and_cli(listed, monkeypatch, capsys):
    canned = {"available": True, "month": "1990-02", "url": ml.status_url("1990-02"), "legend": ml.legend(),
              "headline": "River status, February 1990: much of the Amazon below normal",
              "regions": [{"name": "the Amazon", "below": 0.8, "above": 0.0, "cover": 0.9},
                          {"name": "the Middle East", "below": 0.0, "above": 0.0, "cover": 0.05}],
              "regions_note": "Rough named boxes, not basins."}
    monkeypatch.setattr(ml, "river_status_summary", lambda month=None, live=True: canned)
    monkeypatch.setattr(sys, "argv", ["aquascope", "layers", "status", "1990-02", "--summary"])
    cli.main()
    out = capsys.readouterr().out
    assert "much of the Amazon below normal." in out and "80% below" in out and "Middle East" not in out
    pytest.importorskip("mcp")
    from aquascope import mcp_server as m

    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert "river_status_summary" in names
    assert m.river_status_summary("1990-02")["headline"] == canned["headline"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_explorer_makes_the_same_shares_and_the_same_line(tmp_path):
    red = np.zeros((360, 720), dtype=np.uint8)
    world = _world()
    for k, c in enumerate(ml.STATUS_CLASSES, start=1):
        red[world == k] = c["rgb"][0]
    raw = tmp_path / "red.bin"
    raw.write_bytes(red.tobytes())
    script = f"""
    const m = await import({json.dumps(STATUS_JS.as_uri())});
    const fs = await import("node:fs");
    const red = new Uint8Array(fs.readFileSync({json.dumps(str(raw))}));
    const regions = m.regionShares(red, 720, 360);
    const calm = regions.map((r) => ({{ ...r, below: 0.3, above: 0.2 }}));
    console.log(JSON.stringify({{ list: m.STATUS_REGIONS, share: m.HEADLINE_SHARE, cover: m.MIN_COVER, regions,
      line: m.statusHeadline("2026-09", regions), calm: m.statusHeadline("2001-07", calm) }}));
    """
    out = json.loads(subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True,
                                    check=True).stdout)
    assert out["list"] == ml.STATUS_REGIONS
    assert out["share"] == ml.HEADLINE_SHARE and out["cover"] == ml.MIN_COVER
    py = ml.region_shares(world)
    for a, b in zip(out["regions"], py, strict=True):
        assert a["name"] == b["name"]
        for k in ("below", "above", "cover"):
            assert abs(a[k] - b[k]) <= 0.001, (a, b)
    assert out["line"] == ml.status_headline("2026-09", py)
    assert out["calm"] == ml.status_headline("2001-07", [dict(r, below=0.3, above=0.2) for r in py])
