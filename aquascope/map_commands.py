"""Talk to the map (#561): the map's action schema and a keyless phrase grammar.

The Explorer's AI acts on the globe rather than writing a report: it moves the
view, sets the map date, switches layers, lights up a river, draws an area and
drops pins with notes. Every one of those is an *action*, a small dict with a
``type``, and this module is where actions are defined, checked and read out of
plain English, so the browser, the MCP server and the CLI agree on them:

* :data:`ACTION_TYPES`, :data:`LAYERS`, :data:`BASEMAPS` and :data:`REGIONS`
  are the vocabulary; :func:`action_schema` is the JSON Schema a model's reply
  is held to.
* :func:`validate_action` and :func:`validate_actions` check an action before it
  runs (the page never applies one that did not pass), whoever wrote it: the
  grammar, the on-device model, the reader's own key, or an agent over WebMCP.
* :func:`parse_command` is the keyless path: a phrase grammar for the common
  requests ("show the Rhine", "September 2023", "play the last 5 years", "turn on
  floods past", "trace the Nile to the sea", "where are rivers much above
  normal"). It needs no model and no network.
* :func:`resolve_actions` turns place names into boxes and points with the same
  gazetteer the Explorer's search uses (Photon by komoot, OpenStreetMap data).
* :func:`model_prompt`, :func:`parse_model_reply` and :func:`model_command` are
  the model path: the prompt (with worked examples taken from the grammar
  itself), reading a reply into checked actions, and one call with the reader's
  own key through the provider registry.

Pure standard library apart from the gazetteer and model calls (``urllib``), so
it imports on a bare install and runs in the Explorer's light Pyodide worker.
"""

from __future__ import annotations

import calendar
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import Any

from aquascope.map_layers import STATUS_CLASSES

__all__ = [
    "ACTION_TYPES", "BASEMAPS", "DIRECTIONS", "EXAMPLES", "LAYERS", "PHOTON_URL", "PLACE_CREDIT", "REGIONS",
    "action_schema", "describe_action", "model_command", "model_prompt", "parse_command", "parse_model_reply",
    "resolve_actions", "resolve_place", "validate_action", "validate_actions",
]

ACTION_TYPES = (
    "fly_to", "set_time", "set_layer", "focus_status", "set_basemap", "highlight_river", "draw_area", "add_pin",
)
DIRECTIONS = ("upstream", "downstream", "both")
STEPS = ("day", "week", "month")
STATUS_IDS = tuple(c["id"] for c in STATUS_CLASSES)
EARLIEST = date(1940, 1, 1)   # GEOGLOWS v2's simulated record starts in 1940; nothing on the map is older

#: The map's layers by the id the Explorer uses (explorer/src/map-actions.js maps each to its control; a
#: test keeps the two lists equal). ``aliases`` are the words the grammar accepts for each.
LAYERS: dict[str, dict[str, Any]] = {
    "status": {"label": "World river status", "aliases": (
        "world river status", "river status", "status map", "status layer", "status", "hydrosos",
        "rivers against normal", "river anomalies")},
    "floods_past": {"label": "Floods past", "aliases": (
        "floods past", "past floods", "flood events", "flood news", "floods in the news", "radar floods",
        "flood history", "historical floods")},
    "floods_ahead": {"label": "Floods ahead", "aliases": (
        "floods ahead", "flood forecast", "flood forecasts", "forecast floods", "flood warnings", "warnings",
        "future floods", "upcoming floods")},
    "rivers": {"label": "Rivers", "aliases": ("rivers", "river network", "the river network", "streams")},
    "flow": {"label": "Flow direction", "aliases": (
        "flow direction", "flow animation", "animated flow", "flow arrows", "flow")},
    "basins": {"label": "Catchments", "aliases": (
        "catchments", "catchment boundaries", "basins", "basin boundaries", "watersheds", "basinatlas")},
    "precip": {"label": "Precipitation rate", "aliases": ("rain", "rainfall", "precipitation", "imerg")},
    "soil": {"label": "Root-zone soil moisture", "aliases": ("soil moisture", "soil", "smap")},
    "snow": {"label": "Snow cover", "aliases": ("snow cover", "snow")},
    "lst": {"label": "Land surface temperature", "aliases": (
        "land surface temperature", "land temperature", "temperature", "lst")},
    "storage": {"label": "Water storage anomaly", "aliases": (
        "water storage anomaly", "water storage", "groundwater storage", "grace", "terrestrial water storage")},
    "surface-water": {"label": "Surface water since 1984", "aliases": (
        "surface water", "water occurrence", "global surface water")},
    "landcover": {"label": "Land cover (2021)", "aliases": ("land cover", "landcover", "worldcover")},
    "hillshade": {"label": "Hillshade", "aliases": ("hillshade", "shaded relief", "relief")},
    "terrain": {"label": "3D terrain", "aliases": ("3d terrain", "terrain in 3d", "3d")},
    "heat": {"label": "Gauge density heat map", "aliases": ("heat map", "heatmap", "density heat map", "density")},
    "globe": {"label": "Globe", "aliases": ("globe", "globe view", "the globe")},
}

#: The basemaps by the id in explorer/src/layers.js, with the words that pick each.
BASEMAPS: dict[str, dict[str, Any]] = {
    "light": {"label": "Light", "aliases": ("light", "light map", "plain", "default")},
    "dark": {"label": "Dark", "aliases": ("dark", "dark map", "night", "dark mode")},
    "streets": {"label": "Streets", "aliases": ("streets", "street map", "roads", "street")},
    "satellite": {"label": "Satellite (2016)", "aliases": ("satellite 2016",)},
    "satellite-recent": {"label": "Satellite (2025)", "aliases": (
        "satellite", "satellite imagery", "imagery", "aerial", "satellite 2025", "recent satellite")},
    "terrain": {"label": "Terrain", "aliases": ("terrain", "terrain map", "topographic", "topo")},
    "daily": {"label": "Satellite (today)", "aliases": ("todays satellite", "satellite today", "daily satellite")},
    "usgs": {"label": "US imagery", "aliases": ("usgs", "us imagery", "usgs imagery")},
}

#: View frames for regions a gazetteer does not hold as one place, as [west, south, east, north] in degrees.
#: They frame the camera and nothing else: rough boxes drawn by hand, not borders, and no data is read from them.
REGIONS: dict[str, dict[str, Any]] = {
    "world": {"label": "the world", "bbox": None, "aliases": (
        "the world", "the whole world", "whole world", "world", "the globe view", "everywhere", "earth",
        "the earth", "home")},
    "africa": {"label": "Africa", "bbox": [-18.0, -35.0, 52.0, 37.5], "aliases": ("africa",)},
    "europe": {"label": "Europe", "bbox": [-11.0, 35.0, 40.0, 71.0], "aliases": ("europe",)},
    "asia": {"label": "Asia", "bbox": [26.0, -11.0, 150.0, 60.0], "aliases": ("asia",)},
    "south_asia": {"label": "South Asia", "bbox": [60.5, 5.5, 97.5, 37.0], "aliases": (
        "south asia", "the indian subcontinent", "indian subcontinent")},
    "southeast_asia": {"label": "Southeast Asia", "bbox": [92.0, -11.0, 141.0, 28.5], "aliases": (
        "southeast asia", "south east asia", "south-east asia")},
    "east_asia": {"label": "East Asia", "bbox": [73.0, 18.0, 146.0, 54.0], "aliases": ("east asia",)},
    "central_asia": {"label": "Central Asia", "bbox": [46.5, 35.0, 87.5, 55.5], "aliases": ("central asia",)},
    "middle_east": {"label": "the Middle East", "bbox": [25.0, 12.0, 63.5, 42.0], "aliases": (
        "the middle east", "middle east", "the near east")},
    "north_america": {"label": "North America", "bbox": [-168.0, 7.0, -52.0, 72.0], "aliases": ("north america",)},
    "central_america": {"label": "Central America", "bbox": [-92.5, 7.0, -77.0, 18.5], "aliases": (
        "central america",)},
    "south_america": {"label": "South America", "bbox": [-82.0, -56.0, -34.0, 13.0], "aliases": ("south america",)},
    "oceania": {"label": "Oceania", "bbox": [110.0, -48.0, 180.0, 0.0], "aliases": ("oceania", "australasia")},
    "west_africa": {"label": "West Africa", "bbox": [-18.0, 4.0, 16.0, 25.0], "aliases": ("west africa",)},
    "east_africa": {"label": "East Africa", "bbox": [28.5, -12.0, 52.0, 18.0], "aliases": (
        "east africa", "the horn of africa", "horn of africa")},
    "southern_africa": {"label": "southern Africa", "bbox": [11.0, -35.0, 41.0, -8.0], "aliases": (
        "southern africa",)},
    "the_arctic": {"label": "the Arctic", "bbox": [-180.0, 60.0, 180.0, 85.0], "aliases": ("the arctic", "arctic")},
}

PHOTON_URL = "https://photon.komoot.io/api/"
PLACE_CREDIT = "Places: Photon by komoot, data © OpenStreetMap contributors (ODbL)"

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
MONTHS["sept"] = 9
_MONTH_RE = r"(?P<mon>" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b\.?"
_NUM = r"-?\d{1,3}(?:\.\d+)?"
_UNITS = {"day": "day", "days": "day", "week": "week", "weeks": "week", "month": "month", "months": "month",
          "year": "year", "years": "year", "decade": "decade", "decades": "decade"}
_WORD_NUMBERS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30}

# Words that may be left over once a clause has been read, without the clause being misunderstood.
_FILLER = frozenset("""
a an the me us map on in at of for to is are was were be please show showing see view where which what how
now rivers river layer layers can you could would i want like let lets just also with set go around over
time date then and it its this that there here up some all only flowing running water look looking
""".split())


# ── small helpers ───────────────────────────────────────────────────────────


def _fold(text: str) -> str:
    """Lower case, accents off, curly quotes straight, single spaces."""
    s = unicodedata.normalize("NFD", str(text or ""))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def _iso(d: date) -> str:
    return d.isoformat()


def _today(today: str | date | None) -> date:
    if isinstance(today, date):
        return today
    if today:
        return date.fromisoformat(str(today)[:10])
    return date.today()


def _mid_month(y: int, m: int, today: date) -> date:
    """The 15th of a month (the day the Explorer opens a month on); in the month under way, never after today.
    A month still to come keeps its 15th, which the checks then refuse."""
    d = date(y, m, 15)
    return min(d, today) if (y, m) == (today.year, today.month) else d


def _month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def _add_months(d: date, n: int) -> date:
    total = d.year * 12 + d.month - 1 + n
    y, m0 = divmod(total, 12)
    return date(y, m0 + 1, min(d.day, calendar.monthrange(y, m0 + 1)[1]))


def _number(word: str) -> int | None:
    if word.isdigit():
        return int(word)
    return _WORD_NUMBERS.get(word)


def _alias_index(table: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """(alias, id) pairs, longest alias first, so "soil moisture" wins over "soil"."""
    pairs = [(a, key) for key, spec in table.items() for a in spec["aliases"]]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


_LAYER_ALIASES = _alias_index(LAYERS)
_BASEMAP_ALIASES = _alias_index(BASEMAPS)
_REGION_ALIASES = _alias_index(REGIONS)


def _layer_id(words: str) -> str | None:
    w = re.sub(r"^(?:the|a)\s+", "", words.strip())
    w = re.sub(r"\s+(?:layer|layers|overlay|map)$", "", w)
    return next((key for alias, key in _LAYER_ALIASES if w == alias), None)


def _layer_ids(words: str) -> list[str] | None:
    """Several layers ("rain and snow", "floods past, floods ahead"), or None if any part is not a layer."""
    parts = [p for p in re.split(r"\s*(?:,|\band\b|&)\s*", words) if p.strip()]
    ids = [_layer_id(p) for p in parts]
    return ids if ids and all(ids) else None  # type: ignore[return-value]


def _region(words: str) -> str | None:
    w = words.strip()
    return next((key for alias, key in _REGION_ALIASES if w == alias), None)


def _place_text(raw: str) -> str:
    """A place name as said, tidied: no leading article, no trailing punctuation, title case if all lower."""
    p = re.sub(r"^(?:the)\s+", "", raw.strip(" .,!?;:'\""))
    p = re.sub(r"(?:\s+(?:and|then|also|please|on the map))+$", "", p, flags=re.I)
    p = re.sub(r"\s+river$", "", p, flags=re.I)
    return p.strip(" ,")


# ── the action vocabulary, the schema and the checks ────────────────────────


def action_schema() -> dict[str, Any]:
    """The JSON Schema a model's reply is held to: ``{"actions": [ {type, ...}, ... ]}``.

    One flat object per action (every argument optional, ``type`` required), which small on-device models
    follow far better than a ``oneOf``; :func:`validate_action` then checks the arguments each type needs.
    """
    num = {"type": "number"}
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": list(ACTION_TYPES)},
                        "place": {"type": "string", "description": "A place, river, country or region by name"},
                        "bbox": {"type": "array", "items": num, "minItems": 4, "maxItems": 4,
                                 "description": "[west, south, east, north] in degrees"},
                        "center": {"type": "array", "items": num, "minItems": 2, "maxItems": 2,
                                   "description": "[lat, lon]"},
                        "zoom": num,
                        "zoom_by": num,
                        "date": {"type": "string", "description": "YYYY-MM-DD, not after today"},
                        "step": {"type": "string", "enum": list(STEPS)},
                        "range": {"type": "object", "properties": {"from": {"type": "string"},
                                                                   "to": {"type": "string"}}},
                        "playing": {"type": "boolean"},
                        "layer": {"type": "string", "enum": list(LAYERS)},
                        "on": {"type": "boolean"},
                        "classes": {"type": "array", "items": {"type": "string", "enum": list(STATUS_IDS)}},
                        "basemap": {"type": "string", "enum": list(BASEMAPS)},
                        "direction": {"type": "string", "enum": list(DIRECTIONS)},
                        "lat": num,
                        "lon": num,
                        "label": {"type": "string"},
                        "title": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["type"],
                },
            },
        },
        "required": ["actions"],
    }


def _is_iso(value: Any) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def _text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    s = re.sub(r"\s+", " ", str(value)).strip()
    return s[:limit] if s else None


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_finite(v) for v in value):
        return None
    w, s, e, n = (float(v) for v in value)
    if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s < n <= 90) or w == e:
        return None
    return [round(w, 5), round(s, 5), round(e, 5), round(n, 5)]


def _latlon(lat: Any, lon: Any) -> tuple[float, float] | None:
    if not (_finite(lat) and _finite(lon)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return round(float(lat), 5), round(float(lon), 5)


def validate_action(action: Any, today: str | date | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """``(clean, None)`` for an action the map can run, ``(None, why)`` for one it cannot.

    The clean copy keeps only the arguments its type uses, rounded and trimmed. Dates after ``today`` are
    refused: the map date is a day that has happened (the forecast has its own layer).
    """
    if not isinstance(action, dict):
        return None, "an action must be an object with a type"
    kind = action.get("type")
    if kind not in ACTION_TYPES:
        return None, f"unknown action type {kind!r}"
    t = _today(today)
    out: dict[str, Any] = {"type": kind}
    if action.get("optional") is True:
        out["optional"] = True
    if action.get("river") is True and kind in ("fly_to", "draw_area", "add_pin"):
        out["river"] = True   # a lookup preference: the place is a river
    place = _text(action.get("place"), 80)

    if kind == "fly_to":
        bbox = _bbox(action.get("bbox")) if action.get("bbox") is not None else None
        center = action.get("center")
        zoom = float(action["zoom"]) if _finite(action.get("zoom")) else None
        zoom_by = float(action["zoom_by"]) if _finite(action.get("zoom_by")) else None
        point = _latlon(*center) if isinstance(center, (list, tuple)) and len(center) == 2 else None
        if action.get("region") in REGIONS:
            out["region"] = action["region"]
        if bbox:
            out["bbox"] = bbox
        elif point:
            out["center"] = list(point)
        elif place:
            out["place"] = place
        elif zoom_by is not None and -8 <= zoom_by <= 8 and zoom_by != 0:
            out["zoom_by"] = round(zoom_by, 2)
        elif out.get("region") == "world":
            pass
        else:
            return None, "fly_to needs a place, a bbox [west, south, east, north], a center [lat, lon] or zoom_by"
        if zoom is not None and 0 <= zoom <= 18 and "zoom_by" not in out:
            out["zoom"] = round(zoom, 2)
        if action.get("label"):
            out["label"] = _text(action["label"], 80)
        return out, None

    if kind == "set_time":
        if action.get("date") is not None:
            if not _is_iso(action["date"]):
                return None, "set_time's date must be YYYY-MM-DD"
            d = date.fromisoformat(action["date"])
            if d > t:
                return None, f"{action['date']} is after today; the map date is a day that has happened"
            if d < EARLIEST:
                return None, f"{action['date']} is before anything the map holds ({_iso(EARLIEST)})"
            out["date"] = action["date"]
        if action.get("step") is not None:
            if action["step"] not in STEPS:
                return None, "set_time's step must be day, week or month"
            out["step"] = action["step"]
        rng = action.get("range")
        if rng is not None:
            if not isinstance(rng, dict) or not _is_iso(rng.get("from")) or not _is_iso(rng.get("to")):
                return None, "set_time's range must be {from: YYYY-MM-DD, to: YYYY-MM-DD}"
            a, b = sorted([rng["from"], rng["to"]])
            if date.fromisoformat(a) > t:
                return None, "set_time's range starts after today"
            out["range"] = {"from": max(a, _iso(EARLIEST)), "to": min(b, _iso(t))}
        if action.get("playing") is not None:
            if not isinstance(action["playing"], bool):
                return None, "set_time's playing must be true or false"
            out["playing"] = action["playing"]
        if len(out) == 1 or (len(out) == 2 and "optional" in out):
            return None, "set_time needs a date, a step, a range or playing"
        return out, None

    if kind == "set_layer":
        layer = action.get("layer")
        if layer not in LAYERS:
            return None, f"unknown layer {layer!r}; the layers are {', '.join(LAYERS)}"
        if not isinstance(action.get("on"), bool):
            return None, "set_layer's on must be true or false"
        out.update(layer=layer, on=action["on"])
        return out, None

    if kind == "focus_status":
        classes = action.get("classes") or []
        if not isinstance(classes, (list, tuple)) or any(c not in STATUS_IDS for c in classes):
            return None, f"focus_status's classes must be among {', '.join(STATUS_IDS)}"
        out["classes"] = [c for c in STATUS_IDS if c in classes]
        return out, None

    if kind == "set_basemap":
        if action.get("basemap") not in BASEMAPS:
            return None, f"unknown basemap {action.get('basemap')!r}; the basemaps are {', '.join(BASEMAPS)}"
        out["basemap"] = action["basemap"]
        return out, None

    if kind == "highlight_river":
        direction = action.get("direction") or "both"
        if _text(action.get("label"), 80):
            out["label"] = _text(action.get("label"), 80)
        if direction not in DIRECTIONS:
            return None, "highlight_river's direction must be upstream, downstream or both"
        point = _latlon(action.get("lat"), action.get("lon"))
        if point:
            out["lat"], out["lon"] = point
        elif place:
            out["place"] = place
        else:
            return None, "highlight_river needs a place or a lat and lon"
        out["direction"] = direction
        return out, None

    if kind == "draw_area":
        bbox = _bbox(action.get("bbox")) if action.get("bbox") is not None else None
        if bbox:
            out["bbox"] = bbox
        elif place:
            out["place"] = place
        else:
            return None, "draw_area needs a bbox [west, south, east, north] or a place"
        label = _text(action.get("label"), 80)
        if label:
            out["label"] = label
        return out, None

    # add_pin
    at_center = action.get("at") == "center"
    point = _latlon(action.get("lat"), action.get("lon"))
    if point:
        out["lat"], out["lon"] = point
    elif at_center:
        out["at"] = "center"
    elif place:
        out["place"] = place
    else:
        return None, "add_pin needs a lat and lon, a place, or at: center"
    title = _text(action.get("title"), 80)
    if not title:
        return None, "add_pin needs a title"
    out["title"] = title
    text = _text(action.get("text"), 600)
    if text:
        out["text"] = text
    facts = action.get("facts")
    if facts is not None:
        if not isinstance(facts, list) or len(facts) > 8:
            return None, "add_pin's facts must be a list of at most 8 {label, value, unit}"
        clean = []
        for f in facts:
            if not isinstance(f, dict) or not _text(f.get("label"), 60) or f.get("value") is None:
                return None, "each fact needs a label and a value"
            value = f["value"] if _finite(f["value"]) else _text(f["value"], 60)
            item = {"label": _text(f["label"], 60), "value": value}
            if _text(f.get("unit"), 20):
                item["unit"] = _text(f["unit"], 20)
            clean.append(item)
        out["facts"] = clean
    source = _text(action.get("source"), 200)
    if source:
        out["source"] = source
    return out, None


def validate_actions(actions: Any, today: str | date | None = None) -> dict[str, Any]:
    """Check a list of actions: ``{"actions": [the ones that passed], "errors": [why the others did not]}``."""
    if isinstance(actions, dict):
        actions = actions.get("actions", [actions] if "type" in actions else [])
    if not isinstance(actions, list):
        return {"actions": [], "errors": ["expected a list of actions"]}
    good, errors = [], []
    for i, a in enumerate(actions[:8]):
        clean, why = validate_action(a, today)
        if clean:
            good.append(clean)
        else:
            errors.append(f"action {i + 1}: {why}")
    if len(actions) > 8:
        errors.append(f"only the first 8 of {len(actions)} actions are run")
    return {"actions": good, "errors": errors}


def _month_words(d: str) -> str:
    y, m, _ = d.split("-")
    return f"{calendar.month_name[int(m)]} {y}"


def describe_action(a: dict[str, Any]) -> str:
    """The action in a few words, for the action log ("Fly to the Rhine", "Map date: September 2023")."""
    kind = a.get("type")
    if kind == "fly_to":
        if a.get("label"):
            return f"Fly to {a['label']}"
        if a.get("region") == "world":
            return "Show the whole world"
        if a.get("region"):
            return f"Fly to {REGIONS[a['region']]['label']}"
        if a.get("place"):
            return f"Fly to {a['place']}"
        if a.get("zoom_by"):
            return "Zoom in" if a["zoom_by"] > 0 else "Zoom out"
        if a.get("center"):
            return f"Fly to {a['center'][0]:.2f}, {a['center'][1]:.2f}"
        return "Move the view"
    if kind == "set_time":
        rng, d, step = a.get("range"), a.get("date"), a.get("step")
        by = f", by {step}" if step else ""
        if a.get("playing") and rng:
            return f"Play {_month_words(rng['from'])} to {_month_words(rng['to'])}{by}"
        if a.get("playing"):
            return "Play the time bar"
        if a.get("playing") is False and not d:
            return "Stop playing"
        when = (_month_words(d) if step == "month" else d) if d else None
        if rng:
            return f"{_month_words(rng['from'])} to {_month_words(rng['to'])}" + (f", showing {when}" if when else "")
        if when:
            return f"Map date: {when}"
        return f"Step by {step}" if step else "Set the map date"
    if kind == "set_layer":
        return f"{'Turn on' if a['on'] else 'Turn off'} {LAYERS[a['layer']]['label']}"
    if kind == "focus_status":
        labels = [c["label"] for c in STATUS_CLASSES if c["id"] in a.get("classes", [])]
        return f"River status: only {' or '.join(labels)}" if labels else "River status: every class"
    if kind == "set_basemap":
        return f"Basemap: {BASEMAPS[a['basemap']]['label']}"
    if kind == "highlight_river":
        where = a.get("label") or a.get("place") or f"{a.get('lat'):.2f}, {a.get('lon'):.2f}"
        return {"downstream": f"Trace {where} to the sea", "upstream": f"Light what drains to {where}",
                "both": f"Light up {where}"}[a.get("direction", "both")]
    if kind == "draw_area":
        return f"Outline {a.get('label') or a.get('place') or 'an area'}"
    if kind == "add_pin":
        return f"Pin: {a.get('title')}"
    return str(kind)


# ── the keyless grammar ──────────────────────────────────────────────────────

class _Clause:
    """One clause being read: the folded text, the actions found, and what is left over."""

    def __init__(self, text: str, raw: str, today: date):
        self.text = text
        self.raw = raw
        self.today = today
        self.actions: list[dict[str, Any]] = []
        self.rules: list[str] = []

    def take(self, pattern: str, flags: int = 0) -> re.Match[str] | None:
        m = re.search(pattern, self.text, flags)
        if m:
            self.text = (self.text[: m.start()] + " " + self.text[m.end():]).strip()
            self.text = re.sub(r"\s+", " ", self.text)
        return m

    def add(self, rule: str, *actions: dict[str, Any]) -> None:
        self.rules.append(rule)
        self.actions.extend(actions)

    def leftover(self) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9']+", self.text) if w not in _FILLER]


def _date_phrase(c: _Clause) -> None:
    """Dates, months, years and spans, anywhere in the clause."""
    t = c.today
    play = bool(re.search(r"\b(?:play|animate|replay|run through|loop)\b", c.text))

    # the last N years / months / weeks / days (a span: the range the time bar plays)
    # ("last month" alone is the calendar month, below; "the last month" and "the past month" are spans)
    m = c.take(r"\b(?:(?:over|for|during|in)\s+)?(?:(?:the\s+)?(?:last|past|previous)\s+(?P<n>\d{1,3}|"
               + "|".join(_WORD_NUMBERS) + r")\s*|the\s+(?:last|past|previous)\s+|(?:past|previous)\s+)"
               r"(?P<unit>days?|weeks?|months?|years?|decades?)\b")
    if m:
        n = _number(m.group("n") or "1") or 1
        unit = _UNITS[m.group("unit")]
        if unit == "day":
            start, step = t - timedelta(days=max(1, n) - 1), "day"
        elif unit == "week":
            start, step = t - timedelta(weeks=n), "day" if n <= 4 else "week"
        elif unit == "month":
            start, step = _add_months(t, -n), "week" if n <= 3 else "month"
        else:
            years = n * (10 if unit == "decade" else 1)
            start, step = _add_months(t, -12 * years), "month"
        start = max(start, EARLIEST)
        if step == "month":
            start = start.replace(day=15) if start.day <= 15 else _add_months(start.replace(day=15), 1)
        rng = {"from": _iso(start), "to": _iso(t)}
        c.add("span", {"type": "set_time", "date": rng["from"] if play else rng["to"], "step": step,
                       "range": rng, **({"playing": True} if play else {})})
        return

    # from X to Y / between X and Y (years, or month and year)
    span = (r"(?:from|between)\s+(?:" + _MONTH_RE.replace("mon", "m1") + r"\s+)?(?P<y1>\d{4})"
            r"\s+(?:to|and|until|-)\s+(?:" + _MONTH_RE.replace("mon", "m2") + r"\s+)?(?P<y2>\d{4})\b")
    m = c.take(span)
    if m:
        y1, y2 = int(m.group("y1")), int(m.group("y2"))
        m1 = MONTHS[m.group("m1")] if m.group("m1") else 1
        m2 = MONTHS[m.group("m2")] if m.group("m2") else 12
        a, b = sorted([date(y1, m1, 1), _month_end(y2, m2)])
        b = min(b, t)
        if a <= t:
            rng = {"from": _iso(max(a, EARLIEST)), "to": _iso(b)}
            c.add("span", {"type": "set_time", "date": rng["from"] if play else rng["to"], "step": "month",
                           "range": rng, **({"playing": True} if play else {})})
            return

    # an ISO day or month
    m = c.take(r"\b(?P<y>\d{4})-(?P<m>\d{2})(?:-(?P<d>\d{2}))?\b")
    if m:
        y, mo = int(m.group("y")), int(m.group("m"))
        if 1 <= mo <= 12:
            if m.group("d"):
                c.add("date", {"type": "set_time", "date": f"{y:04d}-{mo:02d}-{m.group('d')}", "step": "day"})
            else:
                c.add("month", {"type": "set_time", "date": _iso(_mid_month(y, mo, t)), "step": "month"})
            return

    # 14 september 2023 / september 14, 2023
    m = c.take(r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH_RE + r"\s*,?\s*(?P<y>\d{4})\b") or \
        c.take(r"\b" + _MONTH_RE + r"\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(?P<y>\d{4})\b")
    if m:
        y, mo, d = int(m.group("y")), MONTHS[m.group("mon")], int(m.group("d"))
        if 1 <= d <= calendar.monthrange(y, mo)[1]:
            c.add("date", {"type": "set_time", "date": f"{y:04d}-{mo:02d}-{d:02d}", "step": "day"})
            return

    # september 2023
    m = c.take(r"\b(?:in\s+)?" + _MONTH_RE + r"\s*,?\s*(?:of\s+)?(?P<y>\d{4})\b")
    if m:
        c.add("month", {"type": "set_time", "date": _iso(_mid_month(int(m.group("y")), MONTHS[m.group("mon")], t)),
                        "step": "month"})
        return

    # last july / this march / in july (the most recent one that has begun)
    m = c.take(r"\b(?:(?P<which>last|this|in|during)\s+)" + _MONTH_RE + r"(?!\s*\d)")
    if m:
        mo = MONTHS[m.group("mon")]
        y = t.year if mo < t.month or (mo == t.month and m.group("which") != "last") else t.year - 1
        c.add("month", {"type": "set_time", "date": _iso(_mid_month(y, mo, t)), "step": "month"})
        return

    # relative days and months
    prev = _add_months(t.replace(day=15), -1)
    rel = {
        "today": (t, "day"), "right now": (t, "day"), "yesterday": (t - timedelta(days=1), "day"),
        "last week": (t - timedelta(days=7), "day"), "this month": (_mid_month(t.year, t.month, t), "month"),
        "last month": (prev, "month"),
    }
    m = c.take(r"\b(?:(?:for|as of|on)\s+)?(?P<w>right now|today|yesterday|last week|this month|last month)\b")
    if m:
        day, step = rel[m.group("w")]
        c.add("relative", {"type": "set_time", "date": _iso(day), "step": step})
        return

    # last year / a year (a whole calendar year: the range, ending in December)
    m = c.take(r"\b(?:(?:in|during|for|of)\s+)?(?P<w>last year|this year)\b") or \
        c.take(r"\b(?:in|during|for|of|year)\s+(?P<y>(?:19[4-9]|20\d)\d)\b") or c.take(r"\b(?P<y>(?:19[4-9]|20\d)\d)\b")
    if m:
        y = int(m.group("y")) if m.groupdict().get("y") else (t.year - 1 if m.group("w") == "last year" else t.year)
        if date(y, 1, 1) <= t:
            rng = {"from": f"{y:04d}-01-15", "to": _iso(min(_month_end(y, 12), t))}
            c.add("year", {"type": "set_time", "date": rng["from"] if play else _iso(_mid_month(y, 12, t)),
                           "step": "month", "range": rng, **({"playing": True} if play else {})})


def _play_phrase(c: _Clause) -> None:
    if c.take(r"\b(?:stop|pause|halt)(?:\s+(?:playing|the animation|the time bar|it|play))?\b"):
        c.add("stop", {"type": "set_time", "playing": False})
        return
    if c.take(r"\b(?:play|animate|replay|run through|loop)(?:\s+(?:it|through|the time bar|the months|them))?\b"):
        for a in c.actions:
            if a["type"] == "set_time" and "range" in a:
                a["playing"] = True
                a["date"] = a["range"]["from"]
                c.rules.append("play")
                return
        c.add("play", {"type": "set_time", "playing": True})


_STATUS_WORDS = [
    (r"much above normal|far above normal|very high|extremely high|unusually high|notably high", ["much_above"]),
    (r"much below normal|far below normal|very low|extremely low|unusually low|notably low|very dry", ["much_below"]),
    (r"above normal|above average|higher than normal|high|wet|swollen|in flood", ["above", "much_above"]),
    (r"below normal|below average|lower than normal|low|dry|in drought|drought", ["below", "much_below"]),
    (r"(?:about |near |at )?normal|average", ["normal"]),
]


def _status_phrase(c: _Clause) -> None:
    if c.take(r"\b(?:show\s+)?(?:all|every)\s+(?:river\s+)?status\s+(?:classes|colours|colors)\b|"
              r"\b(?:clear|reset|remove)\s+(?:the\s+)?(?:status\s+)?(?:focus|filter)\b"):
        c.add("status-all", {"type": "focus_status", "classes": []})
        return
    for words, classes in _STATUS_WORDS:
        pattern = (r"\b(?:where\s+(?:are|is)\s+)?(?:the\s+)?(?:rivers?|flows?|streams?|basins?)\s+"
                   r"(?:are\s+|is\s+|running\s+|flowing\s+)?(?:" + words + r")\b")
        if c.take(pattern):
            c.add("status", {"type": "set_layer", "layer": "status", "on": True},
                  {"type": "focus_status", "classes": classes})
            return
        if c.take(r"\b(?:where\s+(?:are|is)\s+)?(?:" + words + r")\s+(?:rivers?|flows?|streams?|basins?)\b") or \
                c.take(r"\bwhere\s+(?:is it|are things|is the water)\s+(?:" + words + r")\b"):
            c.add("status", {"type": "set_layer", "layer": "status", "on": True},
                  {"type": "focus_status", "classes": classes})
            return


def _layer_phrase(c: _Clause) -> None:
    patterns = [
        r"^(?:please\s+)?(?:turn|switch|toggle|put)\s+(?P<on>on|off)\s+(?:the\s+)?(?P<l>.+)$",
        r"^(?:please\s+)?(?:turn|switch|put)\s+(?:the\s+)?(?P<l>.+?)\s+(?P<on>on|off)$",
        r"^(?:please\s+)?(?P<verb>hide|remove|clear|drop|disable)\s+(?:the\s+)?(?P<l>.+)$",
        r"^(?:please\s+)?(?P<verb>show|add|enable|display|overlay|with)\s+(?:me\s+)?(?:the\s+)?(?P<l>.+)$",
        r"^(?P<l>.+?)\s+(?P<on>on|off)$",
    ]
    for p in patterns:
        m = re.search(p, c.text)
        if not m:
            continue
        ids = _layer_ids(m.group("l"))
        if not ids:
            continue
        on = m.group("on") == "on" if m.groupdict().get("on") else m.group("verb") not in (
            "hide", "remove", "clear", "drop", "disable")
        c.take(re.escape(m.group(0)))
        if ids == ["globe"]:
            c.add("globe", {"type": "set_layer", "layer": "globe", "on": on})
        else:
            c.add("layer", *({"type": "set_layer", "layer": i, "on": on} for i in ids))
        return
    if c.take(r"\b(?:a\s+)?flat(?:\s+map|\s+view|ten(?:\s+the\s+(?:globe|map))?)\b"):
        c.add("globe", {"type": "set_layer", "layer": "globe", "on": False})


def _basemap_phrase(c: _Clause) -> None:
    for alias, key in _BASEMAP_ALIASES:
        if re.search(r"(?:^|\b)(?:switch to|change to|use|show|go to)?\s*(?:the\s+|a\s+)?" + re.escape(alias)
                     + r"\s+(?:basemap|base map|background|map|view|mode|imagery)$", c.text) or \
                re.search(r"^(?:switch|change)\s+(?:the\s+)?(?:basemap|base map|background|map)\s+to\s+(?:the\s+)?"
                          + re.escape(alias) + r"$", c.text) or \
                re.search(r"^(?:switch to|use)\s+(?:the\s+)?" + re.escape(alias) + r"$", c.text):
            c.text = ""
            c.add("basemap", {"type": "set_basemap", "basemap": key})
            return


def _zoom_phrase(c: _Clause) -> None:
    m = c.take(r"^(?:zoom|move|go)\s+(?P<dir>in|out|closer|further out|back out)"
               r"(?:\s+(?:a\s+)?(?P<more>lot|bit|little))?$")
    if m:
        sign = 1 if m.group("dir") in ("in", "closer") else -1
        size = {"lot": 3.5, "bit": 1, "little": 1}.get(m.group("more") or "", 2)
        c.add("zoom", {"type": "fly_to", "zoom_by": sign * size})


def _river_phrase(c: _Clause) -> None:
    patterns = [
        (r"^(?:show(?: me)?\s+)?(?:the\s+)?(?:way|path|route)\s+(?:of|from)\s+(?:the\s+)?(?P<p>.+?)"
         r"\s+to\s+the\s+(?:sea|ocean)$", "downstream"),
        (r"^(?:trace|follow|show(?: me)?)\s+(?:the\s+)?(?P<p>.+?)(?:\s+river)?\s+(?:down\s+)?(?:all the way\s+)?"
         r"(?:to|into|out to)\s+the\s+(?:sea|ocean|coast|mouth|outlet|delta)$", "downstream"),
        (r"^(?:show(?: me)?\s+|light up\s+|highlight\s+)?(?:what|everything that|all that)\s+(?:drains|flows)\s+"
         r"(?:in)?to\s+(?:the\s+)?(?P<p>.+)$", "upstream"),
        (r"^(?:show(?: me)?\s+|light up\s+|highlight\s+)?(?:the\s+)?"
         r"(?:catchment|basin|watershed|everything upstream|upstream)\s+(?:of|from)\s+(?:the\s+)?(?P<p>.+)$",
         "upstream"),
        (r"^(?:show(?: me)?\s+|light up\s+|highlight\s+)?(?:the\s+)?(?:everything\s+)?downstream\s+(?:of|from)\s+"
         r"(?:the\s+)?(?P<p>.+)$", "downstream"),
        (r"^(?:trace|follow|light up|highlight)\s+(?:the\s+)?(?P<p>.+?)(?:\s+river)?$", "both"),
    ]
    for p, direction in patterns:
        m = re.search(p, c.text)
        if m:
            place = _place_text(_raw_span(c, m.group("p")))
            if not place:
                continue
            c.text = ""
            c.add("river", {"type": "fly_to", "place": place, "river": True},
                  {"type": "highlight_river", "place": place, "direction": direction})
            return


def _raw_span(c: _Clause, folded: str) -> str:
    """The words as the reader typed them (case kept), when the folded clause came from them unchanged."""
    m = re.search(re.escape(folded), _fold(c.raw))
    if m and len(_fold(c.raw)) == len(c.raw):
        return c.raw[m.start(): m.end()]
    return folded.title() if folded.islower() else folded


def _area_phrase(c: _Clause) -> None:
    m = c.take(r"^(?:draw|outline|mark|box)\s+(?:a\s+|an\s+)?(?:box|area|rectangle|region)?\s*(?:from\s+)?"
               r"(?P<a>" + _NUM + r")\s*,\s*(?P<b>" + _NUM + r")\s+(?:to|and)\s+"
               r"(?P<c>" + _NUM + r")\s*,\s*(?P<d>" + _NUM + r")$")
    if m:
        lat1, lon1, lat2, lon2 = (float(m.group(k)) for k in "abcd")
        bbox = [min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)]
        c.add("area", {"type": "draw_area", "bbox": bbox})
        return
    m = re.search(r"^(?:draw|outline|mark|box)\s+(?:a\s+|an\s+|the\s+)?"
                  r"(?:box|area|rectangle|outline|region|boundary)?\s*(?:of|around|over|for)?\s*(?:the\s+)?(?P<p>.+)$",
                  c.text)
    if m and re.match(r"^(?:draw|outline|box)", c.text):
        raw = m.group("p")
        region = _region(raw)
        c.text = ""
        if region and REGIONS[region]["bbox"]:
            c.add("area", {"type": "draw_area", "bbox": REGIONS[region]["bbox"], "label": REGIONS[region]["label"]})
        else:
            place = _place_text(_raw_span(c, raw))
            c.add("area", {"type": "draw_area", "place": place, "label": place})


def _pin_phrase(c: _Clause) -> None:
    # On the raw text, so the note keeps its capitals.
    raw = c.raw.strip()
    m = re.match(r"^(?:please\s+)?(?:drop|put|add|place|leave|make)\s+(?:a\s+)?(?:pin|note|marker)"
                 r"(?:\s+(?:at|on)\s+(?P<lat>" + _NUM + r")\s*,\s*(?P<lon>" + _NUM + r"))?(?:\s+here)?"
                 r"(?:\s*(?:saying|that says|reading|with|:|-)\s*(?P<text>.+))?$", raw, re.I)
    if not m:
        return
    note = (m.group("text") or "").strip(" \"'")
    head, sep, rest = note.partition(": ")
    if sep and 0 < len(head) <= 40 and rest.strip():     # "Dhaka: check the gauges" is a title and a note
        title, text = head.strip(), rest.strip()
    else:
        title, text = (note[:60] or "Note"), (note if len(note) > 60 else "")
    action: dict[str, Any] = {"type": "add_pin", "title": title}
    if text:
        action["text"] = text
    if m.group("lat") is not None:
        action.update(lat=float(m.group("lat")), lon=float(m.group("lon")))
    else:
        action["at"] = "center"
    c.text = ""
    c.add("pin", action)


_NOT_IN_NAMES = frozenset("""a an my your our some something anything everything nothing way how why what which clever
please it things stuff help more weather forecast data""".split())


def _plausible_place(place: str) -> bool:
    """A name a gazetteer might hold: a few words, none of them the kind a sentence is made of."""
    words = re.findall(r"[a-z0-9']+", _fold(place))
    return 0 < len(words) <= 5 and not any(w in _NOT_IN_NAMES for w in words) and place.lower() not in _FILLER


def _place_phrase(c: _Clause) -> None:
    """Go to a place: a region, coordinates or a name for the gazetteer. Last, on what the others left."""
    m = c.take(r"^(?:go|fly|take me|move|jump|pan|zoom|head|center|centre)(?:\s+(?:in|over|back))?\s+(?:to|on|into)\s+"
               r"(?P<a>" + _NUM + r")\s*,\s*(?P<b>" + _NUM + r")$") \
        or c.take(r"^(?P<a>" + _NUM + r")\s*,\s*(?P<b>" + _NUM + r")$")
    if m:
        c.add("coordinates", {"type": "fly_to", "center": [float(m.group("a")), float(m.group("b"))], "zoom": 8})
        return
    patterns = [
        r"^(?:go|fly|take me|move|jump|pan|zoom|head|center|centre)(?:\s+(?:in|over|back|out))?"
        r"\s+(?:to|on|into|over)\s+(?P<p>.+)$",
        r"^(?:show|find|locate|where is|where's|focus on|look at|see|open)\s+(?:me\s+)?"
        r"(?:the\s+map\s+(?:of|for)\s+)?(?P<p>.+)$",
        r"^(?:in|over|across|around|for|near)\s+(?P<p>.+)$",
        r"\b(?:in|over|across|around|for)\s+(?P<p>(?!\d)[a-z][a-z' .-]+)$",
    ]
    for i, p in enumerate(patterns):
        m = re.search(p, c.text)
        if not m:
            continue
        # "show in Europe" (what is left of "show high rivers in Europe"): the place, not "in Europe".
        words = re.sub(r"^(?:in|over|across|around|near|for|of)\s+(?:the\s+)?", "", m.group("p").strip())
        region = _region(words)
        if region:
            c.take(re.escape(m.group(0)))
            spec = REGIONS[region]
            c.add("region", {"type": "fly_to", "region": region, **({"bbox": spec["bbox"]} if spec["bbox"] else {})})
            return
        # The bare "in X" at the end only names a place when the clause said something else too.
        if i >= 2 and not c.actions:
            continue
        if _layer_ids(words) or not re.search(r"[a-z]", words):
            continue
        place = _place_text(_raw_span(c, words))
        if not _plausible_place(place):
            continue
        c.take(re.escape(m.group(0)))
        river = bool(re.match(r"^(?:show|find|locate|focus on|look at|see)\s+(?:me\s+)?the\s+", m.group(0)))
        c.add("place", {"type": "fly_to", "place": place, **({"river": True} if river else {})},
              *([{"type": "highlight_river", "place": place, "direction": "both", "optional": True}] if river else []))
        return


_ORDER = {k: i for i, k in enumerate(("set_basemap", "set_layer", "focus_status", "set_time", "fly_to", "draw_area",
                                      "highlight_river", "add_pin"))}


def _parse_clause(raw: str, today: date) -> _Clause:
    c = _Clause(_fold(raw).strip(" .!?"), raw.strip(" .!?"), today)
    c.text = re.sub(r"^(?:please|hey|ok|okay|now|can you|could you|would you|i want to|i'd like to|let me|let's)\s+",
                    "", c.text)
    c.text = re.sub(r"\s+(?:please|for me|on the map)$", "", c.text)
    for step in (_pin_phrase, _zoom_phrase, _basemap_phrase, _area_phrase, _river_phrase):
        step(c)
        if c.actions:
            return c
    _date_phrase(c)
    _play_phrase(c)
    _status_phrase(c)
    _layer_phrase(c)
    _place_phrase(c)
    return c


def _merge_time(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One set_time per command: "September 2023" then "play" become one action."""
    out, time_action = [], None
    for a in actions:
        if a["type"] == "set_time":
            if time_action is None:
                time_action = dict(a)
                out.append(time_action)
            else:
                time_action.update(a)
        else:
            out.append(a)
    return out


def parse_command(text: str, today: str | date | None = None) -> dict[str, Any]:
    """Read one plain-English request into checked map actions, with no model and no network.

    Returns ``{"text", "matched", "actions", "said", "rules", "errors", "control"}``: ``matched`` is True only
    when every word of the request was understood (so "show me the Rhine and something clever" is left to a
    model rather than half done); ``said`` is one line per action for the log; ``control`` is "undo" or
    "undo_all" when the request was about the log itself. Place names stay names (``place``) until
    :func:`resolve_actions` looks them up.
    """
    t = _today(today)
    folded = _fold(text).strip(" .!?")
    out: dict[str, Any] = {"text": str(text or "").strip(), "matched": False, "actions": [], "said": [],
                           "rules": [], "errors": [], "control": None}
    if not folded:
        return out
    if re.fullmatch(r"(?:please\s+)?(?:undo|undo (?:that|it|the last(?: one| action)?)|go back|take that back|revert)",
                    folded):
        return {**out, "matched": True, "control": "undo", "rules": ["undo"]}
    if re.fullmatch(r"(?:please\s+)?(?:undo (?:all|everything|it all)|clear (?:the map|everything|all)|start over"
                    r"|reset(?: the map)?)", folded):
        return {**out, "matched": True, "control": "undo_all", "rules": ["undo_all"]}

    # The whole request first, then clause by clause: "Bosnia and Herzegovina" is one place.
    clauses = [_parse_clause(text, t)]
    if clauses[0].leftover() or not clauses[0].actions:
        splitter = r"\s*(?:;|,\s*(?:and\s+)?(?:then\s+)?|\band then\b|\bthen\b|\band\b)\s*"
        parts = [p for p in re.split(splitter, str(text), flags=re.I) if p.strip()]
        if len(parts) > 1:
            clauses = [_parse_clause(p, t) for p in parts]
    leftover = [w for c in clauses for w in c.leftover()]
    actions = [a for c in clauses for a in c.actions]
    if not actions or leftover:
        out["rules"] = [r for c in clauses for r in c.rules]
        out["unknown"] = leftover
        return out
    actions = sorted(_merge_time(actions), key=lambda a: _ORDER[a["type"]])
    checked = validate_actions(actions, t)
    out.update(matched=not checked["errors"] and bool(checked["actions"]), actions=checked["actions"],
               errors=checked["errors"], rules=[r for c in clauses for r in c.rules])
    out["said"] = [describe_action(a) for a in out["actions"]]
    return out


# ── places: the gazetteer the Explorer's search uses ────────────────────────


def _photon(query: str, limit: int = 5, timeout: float = 10.0) -> list[dict[str, Any]]:
    from aquascope import __version__

    url = f"{PHOTON_URL}?{urllib.parse.urlencode({'q': query, 'limit': limit, 'lang': 'en'})}"
    req = urllib.request.Request(url, headers={"User-Agent": f"aquascope/{__version__} (map commands)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - a fixed https host
        data = json.loads(resp.read().decode("utf-8"))
    return list(data.get("features") or [])


_ZOOM_BY_KIND = {"country": 5.0, "state": 6.5, "region": 6.5, "county": 8.0, "city": 10.0, "town": 11.0,
                 "village": 12.0, "district": 11.0, "locality": 12.0}


_NAME_NOISE = frozenset("the river rio lake mount mountains of de la le du des".split())


def _names_match(asked: str, found: Any) -> bool:
    """Whether a gazetteer hit is the place asked for, not a fuzzy neighbour ("something clever" is not
    "Something Fishy"): every word asked is in the name, the two are nearly the same, or the ask is an
    abbreviation like "USA"."""
    if not found:
        return False
    a = [w for w in re.findall(r"[a-z0-9]+", _fold(asked)) if w not in _NAME_NOISE]
    f = [w for w in re.findall(r"[a-z0-9]+", _fold(str(found))) if w not in _NAME_NOISE]
    # Every word asked, in a name not much longer ("in Europe" is not "The Leuven Institute for Ireland In Europe").
    if a and all(w in f for w in a) and len(f) <= len(a) + 2:
        return True
    import difflib

    # A spelling slip ("Bangaldesh"), not a longer word that starts the same ("weather" is not "Weatherby").
    x, y = _fold(asked), _fold(str(found))
    return abs(len(x) - len(y)) <= 1 and difflib.SequenceMatcher(None, x, y).ratio() >= 0.8


#: Short names the gazetteer does not know by themselves ("USA" finds Usa in Japan).
_PLACE_ALIASES = {
    "usa": "United States", "us": "United States", "u.s.": "United States", "u.s.a.": "United States",
    "united states of america": "United States", "america": "United States", "uk": "United Kingdom",
    "britain": "United Kingdom", "great britain": "United Kingdom", "uae": "United Arab Emirates",
    "drc": "Democratic Republic of the Congo", "dr congo": "Democratic Republic of the Congo",
}


def _frame(bbox: list[float] | None, lat: float | None, lon: float | None) -> list[float] | None:
    """A gazetteer extent made fit for the camera, or None when it cannot frame the place.

    Countries' extents carry their far islands and territories (the Netherlands' runs to the Caribbean,
    Norway's to Bouvet Island), so where the place's own point sits near one edge of an axis, that axis is
    cut back to the point's near side, mirrored. An extent wider than 100 degrees (one that crosses the
    antimeridian, or the United States with its Pacific islands) cannot be framed at all.
    """
    if not bbox or lat is None or lon is None:
        return bbox
    w, s, e, n = bbox
    if e - w > 100:
        return None
    rx, ry = (lon - w) / (e - w), (lat - s) / (n - s)
    if rx > 0.8:
        w = lon - (e - lon) * 1.2
    elif rx < 0.2:
        e = lon + (lon - w) * 1.2
    if ry > 0.8:
        s = lat - (n - lat) * 1.2
    elif ry < 0.2:
        n = lat + (lat - s) * 1.2
    return _bbox([w, s, e, n])


def resolve_place(name: str, want: str = "any", fetch: Any = None) -> dict[str, Any]:
    """Look a place up in the gazetteer: ``{name, detail, lat, lon, bbox, kind, is_river, credit}``.

    ``want="river"`` prefers a waterway among the first few hits ("the Nile" is a river before it is a
    county in Washington). ``bbox`` is [west, south, east, north] when the gazetteer gives an extent.
    ``fetch(query) -> features`` replaces the HTTP call in tests. Regions in :data:`REGIONS` are answered
    without a lookup.
    """
    region = _region(_fold(name))
    if region:
        spec = REGIONS[region]
        return {"name": spec["label"], "detail": "region", "region": region, "bbox": spec["bbox"], "lat": None,
                "lon": None, "kind": "region", "is_river": False, "credit": None}
    name = _PLACE_ALIASES.get(_fold(name), name)
    try:
        feats = (fetch or _photon)(name)
    except Exception as exc:  # noqa: BLE001 - the gazetteer being down is said, not raised
        return {"error": f"the place search is unavailable just now ({type(exc).__name__})", "name": name}
    if not feats:
        return {"error": f"no place called {name!r} was found", "name": name}

    def is_river(f: dict[str, Any]) -> bool:
        p = f.get("properties") or {}
        return p.get("osm_key") == "waterway" and p.get("osm_value") in ("river", "canal", "stream")

    feats = [f for f in feats if _names_match(name, (f.get("properties") or {}).get("name"))]
    if not feats:
        return {"error": f"no place called {name!r} was found", "name": name}
    pick = next((f for f in feats if is_river(f)), None) if want == "river" else None
    pick = pick or feats[0]
    p = pick.get("properties") or {}
    lon, lat = (pick.get("geometry") or {}).get("coordinates", [None, None])[:2]
    ext = p.get("extent")
    bbox = None
    if isinstance(ext, list) and len(ext) == 4:
        bbox = _bbox([ext[0], ext[3], ext[2], ext[1]])   # Photon: [minLon, maxLat, maxLon, minLat]
    kind = p.get("type") or p.get("osm_value") or "place"
    lat = round(float(lat), 5) if _finite(lat) else None
    lon = round(float(lon), 5) if _finite(lon) else None
    stub = False
    if not is_river(pick):   # a river's extent is the river: it is never cut back
        bbox = _frame(bbox, lat, lon)
    elif bbox and max(bbox[2] - bbox[0], bbox[3] - bbox[1]) < 0.5:
        # Only one stretch of the river (the Amazon's is a few km at its mouth): frame its point, not the stub.
        bbox, stub = None, True
    zoom = 8.0 if stub else 4.0 if kind == "country" and not bbox else _ZOOM_BY_KIND.get(str(kind), 9.0)
    return {
        "name": p.get("name") or name,
        "detail": ", ".join(x for x in (p.get("state"), p.get("country")) if x and x != p.get("name")),
        "lat": lat, "lon": lon,
        "bbox": bbox, "kind": kind, "is_river": is_river(pick),
        "zoom": zoom,
        "credit": PLACE_CREDIT,
    }


def resolve_actions(actions: list[dict[str, Any]], fetch: Any = None) -> dict[str, Any]:
    """Turn the place names in checked actions into boxes and points, ready for the map.

    Returns ``{"actions", "notes", "credit"}``. A place that cannot be found drops its actions with a note;
    an ``optional`` river highlight is dropped quietly when the place turns out not to be a river.
    """
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    out, notes, used = [], [], False
    for a in actions:
        if not a.get("place"):
            out.append({k: v for k, v in a.items() if k != "river"})
            continue
        want = "river" if a.get("river") or a["type"] == "highlight_river" else "any"
        # "the Nile at Khartoum": the river, found where the second place is.
        river_at = re.match(r"^(?P<river>.+?)\s+(?:at|near|by|in)\s+(?P<where>.+)$", a["place"]) \
            if want == "river" else None
        name = river_at.group("where") if river_at else a["place"]
        key = (name.lower(), "any" if river_at else want)
        if key not in cache:
            cache[key] = resolve_place(name, want=key[1], fetch=fetch)
        hit = cache[key]
        if river_at and not hit.get("error"):
            hit = {**hit, "name": f"{river_at.group('river')} at {hit['name']}", "is_river": True,
                   "bbox": None, "zoom": 8.0}
        if hit.get("error"):
            if not a.get("optional"):
                notes.append(hit["error"])
            continue
        used = used or bool(hit.get("credit"))
        b = {k: v for k, v in a.items() if k not in ("place", "river", "optional")}
        label = hit["name"] + (f", {hit['detail']}" if hit.get("detail") and not hit.get("region") else "")
        if a["type"] == "fly_to":
            if hit.get("bbox"):
                b["bbox"] = hit["bbox"]
            elif hit.get("lat") is not None:
                b["center"], b["zoom"] = [hit["lat"], hit["lon"]], hit.get("zoom", 9.0)
            elif hit.get("region") == "world":
                b["region"] = "world"
            else:
                notes.append(f"{a['place']} has no position in the gazetteer")
                continue
            b["label"] = hit["name"]
        elif a["type"] == "highlight_river":
            if a.get("optional") and not hit.get("is_river"):
                continue
            if hit.get("lat") is None:
                notes.append(f"{a['place']} has no position to start the river from")
                continue
            b.update(lat=hit["lat"], lon=hit["lon"], label=hit["name"])
        elif a["type"] == "draw_area":
            if not hit.get("bbox"):
                notes.append(f"the gazetteer gives no outline for {a['place']}, only a point")
                continue
            b["bbox"] = hit["bbox"]
            b.setdefault("label", hit["name"])
        elif a["type"] == "add_pin":
            if hit.get("lat") is None:
                notes.append(f"{a['place']} has no position for a pin")
                continue
            b.update(lat=hit["lat"], lon=hit["lon"])
        b["where"] = label
        out.append(b)
    return {"actions": out, "notes": notes, "credit": PLACE_CREDIT if used else None,
            "said": [describe_action(a) for a in out]}


# ── the model path ───────────────────────────────────────────────────────────

#: Requests the grammar reads, shown to a model as worked examples (their actions are computed, never typed).
EXAMPLES = (
    "show the Rhine",
    "where are rivers much above normal in South Asia last July",
    "play the last 5 years",
    "turn on floods past and turn off floods ahead",
    "trace the Nile to the sea",
)


def model_prompt(context: str = "", today: str | date | None = None) -> dict[str, Any]:
    """The system prompt and the reply schema for a model that turns a request into map actions.

    The worked examples are :data:`EXAMPLES` run through :func:`parse_command`, so the prompt can never
    disagree with the grammar.
    """
    t = _today(today)
    layers = "; ".join(f"{k} ({v['label']})" for k, v in LAYERS.items())
    basemaps = "; ".join(f"{k} ({v['label']})" for k, v in BASEMAPS.items())
    classes = ", ".join(STATUS_IDS)
    shots = []
    for ex in EXAMPLES:
        acts = [{k: v for k, v in a.items() if k not in ("river", "optional")}
                for a in parse_command(ex, t)["actions"]]
        shots.append(f"Request: {ex}\nReply: {json.dumps({'actions': acts})}")
    lines = [
        "You control a world map of rivers. Turn the reader's request into map actions.",
        'Reply with ONE JSON object and nothing else: {"actions": [ ... ]}, '
        "at most 8 actions, in the order to run them.",
        f"Today is {_iso(t)}. Dates are YYYY-MM-DD and never after today. "
        'A month is shown on its 15th with step "month".',
        "",
        "The actions:",
        '- fly_to: move the view. Give "place" (a place, river, country or region by name), or "bbox" '
        '[west, south, east, north], or "center" [lat, lon] with "zoom" (0 to 18), or "zoom_by" (+2 zooms in, -2 out).',
        '- set_time: the map date. "date", "step" (day, week, month), "range" {"from", "to"}, '
        '"playing" (true plays the range, false stops).',
        f'- set_layer: switch a layer. "layer" and "on" (true or false). Layers: {layers}.',
        f'- focus_status: show only some classes of the world river status ("classes" among {classes}; '
        "[] shows all). Turn the status layer on too.",
        f'- set_basemap: "basemap" among {basemaps}.',
        '- highlight_river: light a river on the map. "place" (or "lat" and "lon") and "direction": '
        "upstream (what drains to it), downstream (its way to the sea) or both.",
        '- draw_area: outline an area. "bbox" or "place", and a short "label".',
        '- add_pin: a pin with a note. "lat" and "lon" (or "place"), a short "title" and "text".',
        "",
        "Never invent numbers, data or places that were not asked for. "
        "Use place names as the reader wrote them; the map looks them up.",
        'If the request is not about the map, reply {"actions": []}.',
        "",
        *shots,
    ]
    system = "\n".join(lines)
    if context:
        system += f"\n\nWhat the reader is looking at: {context}"
    return {"system": system, "schema": action_schema()}


def parse_model_reply(reply: Any, today: str | date | None = None) -> dict[str, Any]:
    """A model's reply (text or parsed JSON) as checked actions: ``{"actions", "errors", "said"}``."""
    data = reply
    if isinstance(reply, str):
        raw = reply.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        start = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0], default=-1)
        if start < 0:
            return {"actions": [], "errors": ["the model did not reply with JSON"], "said": []}
        try:
            data, _ = json.JSONDecoder().raw_decode(raw[start:])
        except ValueError:
            return {"actions": [], "errors": ["the model's JSON could not be read"], "said": []}
    checked = validate_actions(data, today)
    checked["said"] = [describe_action(a) for a in checked["actions"]]
    return checked


def model_command(text: str, *, provider: str | None = None, model: str | None = None, api_key: str | None = None,
                  base_url: str | None = None, context: str = "", today: str | date | None = None,
                  client: Any = None) -> dict[str, Any]:
    """Ask the reader's own model (the provider registry, their key) for the actions, and check them.

    Returns :func:`parse_model_reply`'s dict plus ``model`` (``"<model> via <provider>"``). Nothing here falls
    back to a key of ours: with no key and no client it raises, and the caller says so.
    """
    from aquascope.ai_engine.analyst import resolve_llm
    from aquascope.ai_engine.llm_transport import make_client

    if client is None:
        cfg = resolve_llm(provider, model, api_key, base_url)
        client = make_client(cfg["api_key"], cfg["base_url"], provider=cfg["provider"])
        model, provider = cfg["model"], cfg["provider"]
    prompt = model_prompt(context, today)
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=700,
        messages=[{"role": "system", "content": prompt["system"]}, {"role": "user", "content": str(text)}],
    )
    content = resp.choices[0].message.content or ""
    out = parse_model_reply(content, today)
    out["model"] = f"{model} via {provider}"
    return out
