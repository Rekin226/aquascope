"""Time on the map (#522): the dated map layers, their valid dates, and the frames of a time-lapse.

The Explorer's date control drives a handful of NASA GIBS layers. Each one has a
first day, sometimes a last day, and gaps where a satellite was down. This module
is the package-side record of that, so the CLI, the MCP server and the Explorer
agree on what a date can show:

* :func:`dated_layers` lists the layers, their cadence and their valid range.
  With ``live=True`` it reads the exact intervals (gaps included) from the GIBS
  WMTS capabilities document.
* :func:`layer_frames` turns a layer, a range and a step (day, week, month) into
  the dates and tile URLs of a time-lapse, dropping the dates the layer cannot
  show and capping the count.

The ranges below were read from
``https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/1.0.0/WMTSCapabilities.xml``
on 2026-10-08. ``explorer/src/layers.js`` carries the same ``since`` and
``until`` values for the browser, and a test keeps the two in step.

Pure standard library: this runs in the Explorer's Pyodide worker as it is.
"""

from __future__ import annotations

import bisect
import calendar
import re
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

GIBS = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"
CAPABILITIES_URL = f"{GIBS}/1.0.0/WMTSCapabilities.xml"
CHECKED = "2026-10-08"

STEPS = ("day", "week", "month")
MAX_FRAMES = 60

_NASA_LICENCE = "Open (NASA), acknowledgement requested"

# The GRACE images GIBS holds, as the <Value> intervals of its WMTS capabilities
# (start/end/period) read on 2026-10-08. The months are irregular: some images
# start mid-month (4 February 2004, 8 February 2011, ...), some months have two
# and some none, so "the first of the month" asks GIBS for a day it has no image
# for. The capabilities also list one interval with its end before its start
# (2020-01-20/2020-01-10), left out here, and one running to 2022-12 although no
# tile after 2022-07 exists, which ``until`` clips. explorer/src/layers.js carries
# the same list.
GRACE_PERIODS: tuple[str, ...] = (
    "2002-04-04/2002-04-04/P28D", "2002-05-02/2002-05-02/P17D", "2002-08-01/2003-04-01/P1M",
    "2003-05-01/2003-05-01/P21D", "2003-07-01/2003-12-01/P1M", "2004-01-01/2004-01-01/P13D",
    "2004-02-04/2004-02-04/P26D", "2004-03-01/2010-11-01/P1M", "2010-12-01/2010-12-01/P27D",
    "2011-02-08/2011-02-08/P21D", "2011-03-01/2011-04-01/P1M", "2011-05-01/2011-05-01/P31D",
    "2011-07-05/2011-07-05/P27D", "2011-08-01/2011-09-01/P1M", "2011-10-01/2011-10-01/P16D",
    "2011-10-17/2011-10-17/P31D", "2011-12-17/2011-12-17/P15D", "2012-01-01/2012-02-01/P1M",
    "2012-03-01/2012-03-01/P19D", "2012-03-20/2012-03-20/P31D", "2012-06-01/2012-08-01/P1M",
    "2012-09-01/2012-09-01/P24D", "2012-11-06/2012-11-06/P25D", "2012-12-01/2013-01-01/P1M",
    "2013-02-01/2013-02-01/P26D", "2013-04-11/2013-04-11/P20D", "2013-05-01/2013-06-01/P1M",
    "2013-07-01/2013-07-01/P31D", "2013-10-01/2013-12-01/P1M", "2014-01-01/2014-01-01/P17D",
    "2014-03-01/2014-05-01/P1M", "2014-06-01/2014-06-01/P24D", "2014-08-01/2014-10-01/P1M",
    "2014-11-01/2014-11-01/P32D", "2015-01-13/2015-01-13/P19D", "2015-02-01/2015-03-01/P1M",
    "2015-04-01/2015-04-01/P11D", "2015-04-12/2015-04-12/P30D", "2015-06-29/2015-06-29/P33D",
    "2015-08-01/2015-08-01/P1M", "2015-09-01/2015-09-01/P27D", "2015-12-12/2015-12-12/P23D",
    "2016-01-04/2016-01-04/P25D", "2016-01-29/2016-01-29/P32D", "2016-03-01/2016-03-01/P31D",
    "2016-05-08/2016-05-08/P24D", "2016-06-01/2016-06-01/P1M", "2016-07-01/2016-07-01/P29D",
    "2016-08-08/2016-08-08/P27D", "2016-11-14/2016-12-11/P27D", "2017-01-07/2017-01-07/P28D",
    "2017-03-17/2017-03-17/P24D", "2017-04-10/2017-04-10/P22D", "2017-05-02/2017-05-02/P21D",
    "2017-05-23/2017-05-23/P38D", "2018-06-01/2018-06-01/P1M", "2018-07-01/2018-07-01/P18D",
    "2018-10-22/2018-10-22/P10D", "2018-11-01/2018-12-01/P1M", "2019-01-01/2019-01-01/P25D",
    "2019-01-26/2019-01-26/P34D", "2019-03-01/2022-12-01/P1M", "2020-02-01/2022-07-01/P1M",
)

# id matches explorer/src/layers.js (a basemap for "daily", overlays for the rest).
DATED_LAYERS: list[dict[str, Any]] = [
    {
        "id": "daily", "kind": "basemap", "label": "Satellite (today)",
        "gibs_layer": "VIIRS_SNPP_CorrectedReflectance_TrueColor", "matrix": "GoogleMapsCompatible_Level9",
        "format": "jpg", "cadence": "day", "since": "2015-11-24", "until": None,
        "attribution": "Imagery from NASA Worldview / GIBS (ESDIS)",
    },
    {
        "id": "precip", "kind": "overlay", "label": "Precipitation rate",
        "gibs_layer": "IMERG_Precipitation_Rate", "matrix": "GoogleMapsCompatible_Level6",
        "format": "png", "cadence": "day", "since": "2000-06-01", "until": None,
        "attribution": "GPM IMERG precipitation rate, NASA GIBS (ESDIS)",
    },
    {
        "id": "soil", "kind": "overlay", "label": "Root-zone soil moisture",
        "gibs_layer": "SMAP_L4_Analyzed_Root_Zone_Soil_Moisture", "matrix": "GoogleMapsCompatible_Level6",
        "format": "png", "cadence": "day", "since": "2015-03-31", "until": None,
        "attribution": "SMAP L4 analysed root-zone soil moisture, NASA GIBS (ESDIS)",
    },
    {
        "id": "snow", "kind": "overlay", "label": "Snow cover",
        "gibs_layer": "MODIS_Terra_NDSI_Snow_Cover", "matrix": "GoogleMapsCompatible_Level8",
        "format": "png", "cadence": "day", "since": "2000-02-24", "until": None,
        "attribution": "MODIS/Terra NDSI snow cover, NASA GIBS (ESDIS)",
    },
    {
        "id": "lst", "kind": "overlay", "label": "Land surface temperature",
        "gibs_layer": "MODIS_Terra_Land_Surface_Temp_Day", "matrix": "GoogleMapsCompatible_Level7",
        "format": "png", "cadence": "day", "since": "2000-02-24", "until": None,
        "attribution": "MODIS/Terra daytime land surface temperature, NASA GIBS (ESDIS)",
    },
    {
        "id": "storage", "kind": "overlay", "label": "Water storage anomaly",
        "gibs_layer": "GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI",
        "matrix": "GoogleMapsCompatible_Level6", "format": "png", "cadence": "month",
        "since": "2002-04-04", "until": "2022-07-01", "periods": GRACE_PERIODS,
        "attribution": "GRACE/GRACE-FO Tellus mascon liquid water equivalent thickness, NASA GIBS (ESDIS)",
        "note": ("Monthly, with gaps (the GRACE to GRACE-FO gap runs from 2017 into 2018) and some "
                 "images starting mid-month. GIBS has no month after 2022-07."),
    },
]


def _layer(layer_id: str) -> dict[str, Any] | None:
    return next((dict(lay) for lay in DATED_LAYERS if lay["id"] == layer_id), None)


def parse_date(value: str | date) -> date:
    """A plain YYYY-MM-DD (or the date part of a timestamp) as a :class:`date`."""
    if isinstance(value, date):
        return value
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", str(value).strip())
    if not m:
        raise ValueError(f"not a YYYY-MM-DD date: {value!r}")
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def step_date(value: str | date, step: str = "day", n: int = 1) -> str:
    """Move a date by ``n`` steps. A month step keeps the day of the month where it can
    (31 January plus one month is the last day of February, not 3 March)."""
    if step not in STEPS:
        raise ValueError(f"step must be one of {', '.join(STEPS)}")
    d = parse_date(value)
    if step == "day":
        return (d + timedelta(days=n)).isoformat()
    if step == "week":
        return (d + timedelta(days=7 * n)).isoformat()
    months = d.year * 12 + (d.month - 1) + n
    year, month = divmod(months, 12)
    month += 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day).isoformat()


def frame_dates(start: str | date, end: str | date, step: str = "day", max_frames: int = MAX_FRAMES) -> dict[str, Any]:
    """The dates from ``start`` to ``end`` (both included) at ``step``, at most ``max_frames`` of them."""
    a, b = parse_date(start), parse_date(end)
    if b < a:
        a, b = b, a
    cap = max(1, int(max_frames))
    dates: list[str] = []
    cur = a.isoformat()
    i = 0
    while parse_date(cur) <= b:
        if len(dates) >= cap:
            return {"dates": dates, "truncated": True}
        dates.append(cur)
        i += 1
        cur = step_date(a, step, i)   # from the start each time, so month steps do not drift
    return {"dates": dates, "truncated": False}


@lru_cache(maxsize=8)
def _images(periods: tuple[str, ...], until: str | None) -> tuple[tuple[date, ...], tuple[date, ...]]:
    """The image starts a ``periods`` list stands for, and the day after each one stops covering."""
    spans: dict[date, date] = {}
    last = parse_date(until) if until else None
    for value in periods:
        a, b, period = value.split("/")
        start, end = parse_date(a), parse_date(b)
        m = re.match(r"^P(\d+)([DM])$", period)
        if not m or end < start:
            continue
        n = int(m.group(1))
        i = 0
        while True:
            if m.group(2) == "D":
                s0 = start + timedelta(days=n * i)
                s1 = s0 + timedelta(days=n)
            else:
                s0 = parse_date(step_date(start, "month", n * i))
                s1 = parse_date(step_date(s0, "month", n))
            if s0 > end or (last and s0 > last):
                break
            spans[s0] = max(spans.get(s0, s1), s1)
            i += 1
    starts = tuple(sorted(spans))
    return starts, tuple(spans[s0] for s0 in starts)


def image_for(layer: dict[str, Any], value: str | date) -> date | None:
    """The start of the image a layer with irregular ``periods`` shows for a date: the latest
    image covering it, else the month's latest image starting before it, else the month's first
    image, else None (a gap). The same rule as ``imageFor`` in explorer/src/layers.js."""
    d = parse_date(value)
    starts, ends = _images(tuple(layer.get("periods") or ()), layer.get("until"))
    i = bisect.bisect_right(starts, d)
    for j in range(i - 1, max(-1, i - 4), -1):   # images overlap by a few days at most
        if starts[j] <= d < ends[j]:
            return starts[j]
    month = [s0 for s0 in starts[max(0, i - 3):i + 3] if (s0.year, s0.month) == (d.year, d.month)]
    before = [s0 for s0 in month if s0 <= d]
    return before[-1] if before else (month[0] if month else None)


def layer_date(layer: dict[str, Any], value: str | date) -> str:
    """The date a layer's tile URL carries. A monthly product with known ``periods`` asks for
    the start of the image that covers the day (see :func:`image_for`); otherwise for the first
    of the month (or the layer's first day, when that is later). The same rule as ``layerDate``
    in explorer/src/layers.js. GIBS answers a date inside an image's period with that image."""
    d = parse_date(value)
    if layer.get("cadence") != "month":
        return d.isoformat()
    if layer.get("periods"):
        img = image_for(layer, d)
        if img is not None:
            return img.isoformat()
    first = d.replace(day=1)
    since = parse_date(layer["since"]) if layer.get("since") else first
    return (since if first < since <= d else first).isoformat()


def tile_template(layer: dict[str, Any], value: str | date) -> str:
    """The XYZ tile URL of a layer on a date, with ``{z}/{y}/{x}`` left for the map client."""
    return (f"{GIBS}/{layer['gibs_layer']}/default/{layer_date(layer, value)}/"
            f"{layer['matrix']}/{{z}}/{{y}}/{{x}}.{layer['format']}")


def last_day(layer: dict[str, Any], today: date | None = None) -> date:
    """The last day a layer can show: ``until`` (to the end of that month for a monthly
    product, whose last image covers the whole month), or today for an open range."""
    if not layer.get("until"):
        return today or date.today()
    until = parse_date(layer["until"])
    if layer.get("cadence") == "month":
        return until.replace(day=calendar.monthrange(until.year, until.month)[1])
    return until


def in_range(layer: dict[str, Any], value: str | date, today: date | None = None) -> bool:
    """Whether a layer can show a date: inside its first and last day and, for a layer with
    known ``periods``, not in one of its gaps (other layers' gaps are known only with live=True)."""
    d = parse_date(value)
    if not parse_date(layer["since"]) <= d <= last_day(layer, today):
        return False
    return not layer.get("periods") or image_for(layer, d) is not None


# ── the live intervals ─────────────────────────────────────────────────────


def parse_capabilities(xml: str, identifiers: list[str]) -> dict[str, dict[str, Any]]:
    """The Time dimension of each named layer in a GIBS WMTS capabilities document.

    Returns ``{identifier: {"default": "YYYY-MM-DD", "intervals": [[start, end, period], ...]}}``.
    A plain scan rather than an XML parser: the document is about 6 MB and only a
    few tags per layer matter.
    """
    out: dict[str, dict[str, Any]] = {}
    for ident in identifiers:
        i = xml.find(f"<ows:Identifier>{ident}</ows:Identifier>")
        if i < 0:
            continue
        j = xml.find("<Dimension>", i)
        k = xml.find("</Dimension>", j)
        nxt = xml.find("</Layer>", i)
        if j < 0 or k < 0 or (0 <= nxt < j):
            continue
        block = xml[j:k]
        default = re.search(r"<Default>([^<]+)</Default>", block)
        intervals = [v.split("/") for v in re.findall(r"<Value>([^<]+)</Value>", block)]
        out[ident] = {
            "default": default.group(1)[:10] if default else None,
            "intervals": [iv for iv in intervals if len(iv) == 3],
        }
    return out


def _covered_until(start: str, end: str, period: str) -> date:
    """The first day after a capabilities interval ``start/end/period`` stops covering."""
    last = parse_date(end) if parse_date(end) >= parse_date(start) else parse_date(start)
    m = re.match(r"^P(\d+)([DM])$", period)
    if not m:
        return last + timedelta(days=1)
    n = int(m.group(1))
    if m.group(2) == "D":
        return last + timedelta(days=n)
    return parse_date(step_date(last, "month", n))


def interval_gaps(intervals: list[list[str]]) -> list[list[str]]:
    """The stretches no interval covers, as ``[[first_missing_day, next_covered_day], ...]``."""
    gaps: list[list[str]] = []
    spans = sorted(((parse_date(a), _covered_until(a, b, p)) for a, b, p in intervals), key=lambda t: t[0])
    covered: date | None = None
    for start, until in spans:
        if covered is not None and start > covered:
            gaps.append([covered.isoformat(), start.isoformat()])
        covered = until if covered is None else max(covered, until)
    return gaps


def _live_intervals() -> dict[str, dict[str, Any]]:
    from aquascope.utils.http_client import CachedHTTPClient

    client = CachedHTTPClient(timeout=60.0, retries=2, cache_ttl_seconds=12 * 3600)
    try:
        xml = client.get_text(CAPABILITIES_URL)
    finally:
        client.close()
    return parse_capabilities(xml, [lay["gibs_layer"] for lay in DATED_LAYERS])


# ── the faces ──────────────────────────────────────────────────────────────


def dated_layers(live: bool = False) -> dict[str, Any]:
    """The map layers that change with the date, their cadence and the dates they cover.

    ``live=True`` reads the current intervals (with gaps) and the latest day from
    the GIBS WMTS capabilities (about 6 MB, cached for 12 hours); without it the
    ranges are the ones recorded on ``checked``.
    """
    layers = []
    for lay in DATED_LAYERS:
        row = {k: (list(v) if isinstance(v, tuple) else v) for k, v in lay.items() if k not in ("matrix", "format")}
        row["licence"] = _NASA_LICENCE
        row["tile_template"] = (f"{GIBS}/{lay['gibs_layer']}/default/{{date}}/{lay['matrix']}/"
                                f"{{z}}/{{y}}/{{x}}.{lay['format']}")
        layers.append(row)
    out: dict[str, Any] = {
        "layers": layers, "steps": list(STEPS), "max_frames": MAX_FRAMES,
        "checked": CHECKED, "source": CAPABILITIES_URL,
        "note": "Dates are UTC days. A date outside a layer's range, or in one of its gaps, draws nothing.",
    }
    if live:
        try:
            found = _live_intervals()
        except Exception as exc:  # noqa: BLE001 - the ranges above still stand
            out["live_error"] = f"{type(exc).__name__}: {exc}"
            return out
        for row in layers:
            info = found.get(row["gibs_layer"])
            if not info:
                continue
            row["intervals"] = info["intervals"]
            row["gaps"] = interval_gaps(info["intervals"])
            row["latest"] = info["default"]
            if info["intervals"]:
                row["since"] = info["intervals"][0][0][:10]
        out["live"] = True
    return out


def layer_frames(
    layer: str,
    start: str,
    end: str,
    step: str = "day",
    max_frames: int = MAX_FRAMES,
) -> dict[str, Any]:
    """The frames of a time-lapse of one dated map layer: each date from ``start`` to ``end``
    at ``step`` (day, week or month) with its tile URL, skipping dates the layer cannot show
    (outside its range, or in a known gap) and keeping at most ``max_frames`` (capped at 60).
    This is what the Explorer plays and turns into a GIF; any XYZ map client can draw the
    tile templates.
    """
    spec = _layer(layer)
    if spec is None:
        return {"error": f"unknown layer {layer!r}; one of {', '.join(lay['id'] for lay in DATED_LAYERS)}"}
    if step not in STEPS:
        return {"error": f"step must be one of {', '.join(STEPS)}"}
    try:
        a, b = parse_date(start), parse_date(end)
    except ValueError as exc:
        return {"error": str(exc)}
    try:
        cap = max(1, min(int(max_frames), MAX_FRAMES))
    except (TypeError, ValueError):
        return {"error": "max_frames must be a whole number"}
    today = date.today()
    lo = parse_date(spec["since"])
    hi = last_day(spec, today)
    if b < a:
        a, b = b, a
    first, last = max(a, lo), min(b, hi)
    out: dict[str, Any] = {
        "layer": spec["id"], "label": spec["label"], "step": step,
        "requested": {"start": a.isoformat(), "end": b.isoformat()},
        "valid": {"since": spec["since"], "until": spec.get("until")},
        "attribution": spec["attribution"], "licence": _NASA_LICENCE,
    }
    if first > last:
        out.update(frames=[], truncated=False,
                   note=f"{spec['label']} has no data between {a.isoformat()} and {b.isoformat()}.")
        return out
    # Step from the requested start so a range clipped by the layer keeps the reader's rhythm,
    # but begin at the first step inside the layer's range (a request from 1900 does not walk
    # a century of empty days first).
    total = _steps_before(a, step, b + timedelta(days=1))
    i = _steps_before(a, step, first)
    frames = []
    seen: set[str] = set()
    usable = 0
    while True:
        d = parse_date(step_date(a, step, i))
        if d > last:
            break
        i += 1
        if spec.get("periods") and image_for(spec, d) is None:
            continue      # a gap inside the range: no image to show
        usable += 1
        ld = layer_date(spec, d)
        if spec["cadence"] == "month" and ld in seen:
            continue      # one frame per image of a monthly product
        seen.add(ld)
        frames.append({"date": d.isoformat(), "layer_date": ld, "tiles": tile_template(spec, d)})
    out["truncated"] = len(frames) > cap
    out["frames"] = frames[:cap]
    out["skipped"] = total - usable
    return out


def _steps_before(start: date, step: str, target: date) -> int:
    """How many dates of the series start, start + 1 step, ... fall before ``target``."""
    if target <= start:
        return 0
    if step in ("day", "week"):
        n = 1 if step == "day" else 7
        return -(-(target - start).days // n)
    k = max(0, (target.year - start.year) * 12 + target.month - start.month - 1)
    while parse_date(step_date(start, "month", k)) < target:
        k += 1
    return k
