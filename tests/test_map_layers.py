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
