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
    "FLOOD_DEFAULT_RP", "FLOOD_FLOWS", "ROUTE_DEFAULT_RP", "SKETCH_LIMITS", "ROUTE_LIMITS", "ROUTE_SHAPES",
    "WELL_LIMITS",
    "STORM_DEFAULTS", "STORM_LIMITS", "RESERVOIR_DEFAULTS", "RESERVOIR_LIMITS",
    "action_schema", "describe_action", "model_command", "model_prompt", "parse_command", "parse_model_reply",
    "reply_schema", "resolve_actions", "resolve_place", "validate_action", "validate_actions",
]

ACTION_TYPES = (
    "fly_to", "set_time", "set_layer", "focus_status", "set_basemap", "highlight_river", "draw_area", "add_pin",
    "route",
    "paint",
    "flood_reach",
    "draw_levee", "draw_breach",
    "replay",
    "drop_well",
    "change_landuse",
    "draw_storm",
    "draw_reservoir",
)
#: Route it (#656): the shapes of the water entering, and the bounds a route action is held to.
ROUTE_SHAPES = ("triangle", "block")
ROUTE_LIMITS = {"peak_cms": (0.0, 1_000_000.0), "return_period": (2.0, 500.0), "duration_h": (0.0, 720.0)}
#: The flood routed when the words give no size: the 10-year flow of the reach (floodsim.design_flow).
ROUTE_DEFAULT_RP = 10.0
#: Flood it (#658): the flows a flood_reach action may name besides a return period or a number: GEOGLOWS's
#: forecast peak, its 15-day forecast as a hydrograph, or Route it's wave; and the return period when none is said.
FLOOD_FLOWS = ("forecast_peak", "forecast", "route")
FLOOD_DEFAULT_RP = 100.0
FLOOD_RETURN_PERIODS = (2.0, 500.0)
#: Levees and breaches on Flood it's grid (#663): a line of at most this many points, a crest or an invert in m
#: above mean sea level, a freeboard above the highest flood along the line in m.
SKETCH_LIMITS = {"points": 200, "level_m": (-100.0, 9000.0), "freeboard_m": (0.0, 20.0)}
SKETCHES = ("draw_levee", "draw_breach")
#: Drop a well (#660): the bounds a drop_well action is held to (aquascope.sketch.drawdown.LIMITS; a test keeps
#: them equal, and this module stays importable without scipy, in the light worker).
WELL_LIMITS = {"rate_m3_day": (0.1, 1_000_000.0), "days": (0.001, 36_500.0),
               "transmissivity_m2_day": (0.01, 1_000_000.0), "storativity": (1e-7, 0.5)}
WELL_DEFAULT_RATE = 1000.0
#: Change land use (#661): what a drawn area can become (aquascope.sketch.landuse.USES), and the bounds.
LANDUSE_LIMITS = {"rain_mm": (1.0, 2000.0), "impervious_pct": (0.0, 100.0), "points": 200}
#: Words for each use, for the keyless rules ("build houses here", "pave this area", "plant woods").
LANDUSE_WORDS: dict[str, str] = {
    "residential_1_8": r"town ?houses|terraced houses|terraces|dense housing|flats|apartments",
    "residential_1_4": r"houses|homes|housing|residential|a (?:housing )?estate|a suburb|suburbs|"
                       r"built (?:over|up|on)|urbani[sz]ed|developed|development",
    "residential_1_2": r"large gardens|half[- ]acre lots",
    "residential_1_acre": r"acre lots|large lots",
    "commercial": r"shops|offices|commercial|business(?: park)?|retail|a (?:shopping )?mall|shopping cent(?:re|er)",
    "industrial": r"factor(?:y|ies)|industrial(?: estate| park| zone)?|warehouses?|an industrial estate",
    "paved": r"a car ?park|parking(?: lots?)?|paved|tarmac|asphalt|concrete|roofs",
    "newly_graded": r"a construction site|construction|bare earth|graded|bulldozed|cleared",
    "open_space_good": r"a park|parks?|lawns?|open space|playing fields|a golf course",
    "row_crops": r"crops|cropland|farmland|arable(?: land)?|fields|maize|corn",
    "pasture_good": r"pasture|grassland|grass|grazing",
    "meadow": r"(?:a )?meadow|hay",
    "woods_good": r"woods?|woodland|(?:a )?forest|trees",
}
#: Draw a storm (#670): a storm's rain in mm, its hours and its runoff coefficient, the points its outline may
#: have, and what is used when the words do not say (aquascope.routing.route_storm's own defaults).
STORM_LIMITS = {"rain_mm": (0.0, 2000.0), "duration_h": (0.0, 720.0), "runoff_coefficient": (0.0, 1.0),
                "points": 200}
STORM_DEFAULTS = {"duration_h": 6.0, "runoff_coefficient": 0.3}
#: Draw a reservoir (#662): the dam's levels (m above the lowest ground in the lake) and its spillway's width, and
#: what is used when the words do not say (aquascope.sketch.reservoir.DEFAULTS; the flood: the 100-year flow).
RESERVOIR_LIMITS = {"crest_m": (0.0, 300.0), "top_m": (0.0, 320.0), "width_m": (0.0, 2000.0), "points": 200}
RESERVOIR_DEFAULTS = {"crest_m": 10.0, "rise_m": 5.0, "width_m": 30.0, "return_period": 100.0, "duration_h": 24.0}
#: Replay an event (#659): which of Floods past's floods near the place a replay action may pick (by rank).
REPLAY_PICKS = (1, 5)
DIRECTIONS = ("upstream", "downstream", "both")
STEPS = ("day", "week", "month")
STATUS_IDS = tuple(c["id"] for c in STATUS_CLASSES)
EARLIEST = date(1940, 1, 1)   # GEOGLOWS v2's simulated record starts in 1940; nothing on the map is older

#: Paint the globe (#657): the questions a ``paint`` action may ask, by the id :data:`aquascope.paint.QUESTIONS`
#: uses (a test keeps the two equal; this module stays free of pandas for the light worker), with the words the
#: grammar reads for each.
PAINT_QUESTIONS: dict[str, dict[str, Any]] = {
    "flood_trend": {"label": "the trend in annual floods", "aliases": (
        "flood trends", "flood trend", "trend in annual floods", "trends in annual floods", "trend in floods",
        "trends in floods", "annual flood trends", "annual flood trend", "trend in annual maxima",
        "annual maximum trend", "trend in peak flows", "peak flow trends", "flood trend map")},
    "low_flow_trend": {"label": "the trend in low flows", "aliases": (
        "low flow trends", "low flow trend", "low-flow trends", "low-flow trend", "trend in low flows",
        "trends in low flows", "trend in low flow", "drought trend", "drought trends", "low flows")},
    "baseflow_index": {"label": "the baseflow index", "aliases": (
        "baseflow index", "base flow index", "the bfi", "bfi", "baseflow")},
}
PAINT_MIN_YEARS = (5, 100)

#: The map's layers by the id the Explorer uses (explorer/src/map-actions.js maps each to its control; a
#: test keeps the two lists equal). ``aliases`` are the words the grammar accepts for each.
LAYERS: dict[str, dict[str, Any]] = {
    "status": {"label": "World river status", "aliases": (
        "world river status", "river status", "status map", "status layer", "status", "hydrosos",
        "rivers against normal", "river anomalies")},
    "floods_past": {"label": "Floods past", "aliases": (
        "floods past", "past floods", "flood events", "flood news", "floods in the news", "radar floods",
        "flood history", "historical floods")},
    "groundwater": {"label": "Groundwater", "aliases": (
        "groundwater", "groundwater level", "groundwater levels", "groundwater surface", "water table",
        "head surface", "heads", "well levels", "wells", "modelmuse", "modflow", "modflow heads")},
    "water_balance": {"label": "Water balance", "aliases": (
        "water balance", "water budget", "basin water balance", "subbasin water balance", "subbasins",
        "evaporation", "evapotranspiration", "runoff", "closure error", "hec-hms", "qswat", "swat")},
    "floods_ahead": {"label": "Floods ahead", "aliases": (
        "floods ahead", "flood forecast", "flood forecasts", "forecast floods", "flood warnings", "warnings",
        "future floods", "upcoming floods")},
    "flood_depth": {"label": "Flood depth where floods are forecast", "aliases": (
        "flood depth", "flood depths", "flood depth map", "depth map", "water depth", "inundation",
        "inundation map", "flood extent", "ras mapper")},
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


#: The controls a model may answer with instead of actions: the review of the model result on the map (#564) and
#: the notes on the view (#562).
MODEL_CONTROLS = ("review", "explain")


def _landuse_uses() -> tuple[str, ...]:
    from aquascope.sketch.landuse import USES

    return USES


def action_schema(places: list[str] | None = None, layers: list[str] | None = None) -> dict[str, Any]:
    """The JSON Schema a model's reply is held to: ``{"actions": [ {type, ...}, ... ]}``.

    One flat object per action (every argument optional, ``type`` required), which small on-device models
    follow far better than a ``oneOf``; :func:`validate_action` then checks the arguments each type needs.
    With ``places`` and ``layers`` (:func:`reply_schema`, for one request), the place a model may name and the
    layers it may switch are enums of what the words name, and the free-text fields (label, title, text) are
    left out: the package writes those.
    """
    num = {"type": "number"}
    schema = {
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
                        "peak_cms": {"type": "number", "description": "route: the peak flow entering, m3/s"},
                        "return_period": {"type": "number",
                                          "description": "route: or the reach's T-year flow, in years"},
                        "duration_h": {"type": "number", "description": "route: hours the water enters for"},
                        "shape": {"type": "string", "enum": list(ROUTE_SHAPES),
                                  "description": "route: triangle (a flood) or block (a release)"},
                        "at": {"type": "string", "enum": ["here", "center"],
                               "description": "route, flood_reach: from the river selected on the map (here)"},
                        "flow_cms": {"type": "number", "description": "flood_reach: a flow at the reach, m3/s"},
                        "flow": {"type": "string", "enum": list(FLOOD_FLOWS),
                                 "description": "flood_reach: the forecast peak, the 15-day forecast, or Route "
                                                "it's wave, instead of a return period"},
                        "line": {"description": "draw_levee, draw_breach: \"drawn\" (the reader draws it on the "
                                                "map), or a list of [lon, lat] points the reader gave",
                                 "anyOf": [{"type": "string", "enum": ["drawn"]},
                                           {"type": "array", "items": {"type": "array", "items": num,
                                                                       "minItems": 2, "maxItems": 2}}]},
                        "from_place": {"type": "string", "description": "draw_levee, draw_breach: the line's start, "
                                                                        "a place by name"},
                        "to_place": {"type": "string", "description": "draw_levee, draw_breach: the line's end"},
                        "crest": {"type": "number", "description": "draw_levee: the crest, m above mean sea level, "
                                                                   "only when the reader gave it"},
                        "freeboard_m": {"type": "number", "description": "draw_levee: or the crest this many m "
                                                                         "above the highest flood along the line"},
                        "invert": {"type": "number", "description": "draw_breach: the gap's floor, m above mean "
                                                                    "sea level (none: the natural ground)"},
                        "rate_m3_day": {"type": "number", "description": "drop_well: the pumping rate, m3/day, "
                                                                          "only when the reader gave it"},
                        "days": {"type": "number", "description": "drop_well: days of pumping"},
                        "transmissivity": {"type": "number", "description": "drop_well: m2/day, only when given"},
                        "storativity": {"type": "number", "description": "drop_well: dimensionless, only when given"},
                        "use": {"type": "string", "enum": list(_landuse_uses()),
                                "description": "change_landuse: what the drawn area becomes (TR-55's covers)"},
                        "soil_group": {"type": "string", "enum": ["A", "B", "C", "D"],
                                       "description": "change_landuse: the hydrologic soil group, only when given"},
                        "impervious_pct": {"type": "number", "description": "change_landuse with use impervious: "
                                                                            "the percent impervious"},
                        "label": {"type": "string"},
                        "title": {"type": "string"},
                        "text": {"type": "string"},
                        "rain_mm": {"type": "number",
                                    "description": "draw_storm: the rain, mm, as the reader wrote it; "
                                                   "change_landuse: the 24-hour design storm, only when given"},
                        "runoff_coefficient": {"type": "number",
                                               "description": "draw_storm: the share of the rain that runs off, "
                                                              "only when the reader gave it"},
                        "area": {"type": "string", "enum": ["drawn"],
                                 "description": "draw_storm, draw_reservoir, change_landuse: the area the "
                                                "reader draws, or the one on the map"},
                        "polygon": {"type": "array", "items": {"type": "array", "items": num, "minItems": 2,
                                                               "maxItems": 2},
                                    "description": "draw_storm, draw_reservoir, change_landuse: a ring of "
                                                   "[lon, lat] the reader gave"},
                        "crest_m": {"type": "number", "description": "draw_reservoir: the spillway's crest, m above "
                                                                     "the lowest ground, only when the reader gave it"},
                        "top_m": {"type": "number", "description": "draw_reservoir: the top of the dam, m above the "
                                                                   "lowest ground, only when the reader gave it"},
                        "width_m": {"type": "number", "description": "draw_reservoir: the spillway's width, m"},
                        "inflow": {"type": "string", "enum": ["wave"],
                                   "description": "draw_reservoir: send the flood wave or storm on the map through"},
                        "month": {"type": "string",
                                  "description": "replay: the month of the flood, YYYY-MM, not after today"},
                        "pick": {"type": "integer", "minimum": REPLAY_PICKS[0], "maximum": REPLAY_PICKS[1],
                                 "description": "replay: which of the floods the code lists there (1: the most "
                                                "reported)"},
                        "question": {"type": "string", "enum": list(PAINT_QUESTIONS),
                                     "description": "paint: the question to answer at every gauge"},
                        "min_years": {"type": "integer", "minimum": PAINT_MIN_YEARS[0],
                                      "maximum": PAINT_MIN_YEARS[1]},
                    },
                    "required": ["type"],
                },
            },
            # A request about the model result on the map, or about the view itself, is a control (#590).
            "control": {"type": "string", "enum": list(MODEL_CONTROLS)},
        },
        "required": ["actions"],
    }
    if places is None and layers is None:
        return schema
    item = schema["properties"]["actions"]["items"]["properties"]
    for free in ("label", "title", "text"):
        item.pop(free)
    if places is not None:
        if places:
            item["place"] = {"type": "string", "enum": list(places), "description": "A place the request names"}
        else:
            item.pop("place")
    if layers is not None:
        if layers:
            item["layer"] = {"type": "string", "enum": list(layers)}
        else:
            item.pop("layer")
            item.pop("on")
        if "status" not in (layers or []):
            item.pop("classes")
            item["type"]["enum"] = [t for t in item["type"]["enum"] if t != "focus_status"]
        if not layers:
            item["type"]["enum"] = [t for t in item["type"]["enum"] if t != "set_layer"]
    return schema


#: Words that ask for a layer without naming it: "running low" asks for the river status, "flood" for a flood
#: layer. A layer the words neither name nor ask for this way is not switched by a model (#590 check3: "are there
#: any numerical instabilities in this run?" turned Floods past and the river status on).
_LAYER_CUES: dict[str, str] = {
    "status": r"normal|above|below|low|lower|lowest|high|higher|highest|dry|drier|dryer|driest|wet|wetter|"
              r"wettest|drought|droughts|swollen|anomal\w*",
    "floods_past": r"flood\w*", "floods_ahead": r"flood\w*|forecasts?", "flood_depth": r"flood\w*|inundat\w*",
    "groundwater": r"aquifers?|water table", "precip": r"rain\w*|storms?", "snow": r"snow\w*",
    "basins": r"catchments?|basins?|watersheds?", "rivers": r"rivers?|streams?",
    "surface-water": r"lakes?|reservoirs?", "lst": r"temperatures?|heat ?waves?",
    "landcover": r"forests?|crops?|cropland|urban", "terrain": r"mountains?",
}


def _layers_named(text: str) -> list[str]:
    """The layers a request names by an alias, or asks for by a word in :data:`_LAYER_CUES`."""
    folded = _fold(text)
    out = []
    for lid, spec in LAYERS.items():
        words = [re.escape(a) for a in spec["aliases"]] + ([_LAYER_CUES[lid]] if lid in _LAYER_CUES else [])
        if re.search(r"(?<![a-z0-9])(?:" + "|".join(words) + r")(?![a-z0-9])", folded):
            out.append(lid)
    return out


#: Words of a request that are never part of a place's name.
_NOT_PLACE = frozenset("""find gauges gauge stations station compared compare comparing right now zoom today month year
week doing any numerical run model results result tell plain words showing me interesting driest wettest highest
lowest turn take fly go show trace light highlight play set switch open move pan head""".split())


def _places_named(text: str) -> list[str]:
    """Runs of the request's own words that could be a place ("Kingston upon Thames", "Danube"), as typed,
    and the regions it names: what a model may give as ``place``."""
    raw = re.sub(r"[?!.;:()\"]", " ", str(text or ""))
    words = re.findall(r"[^\s,]+", raw)
    out: list[str] = []
    # Names first (runs that start and end with a capital); every run only when the request has none.
    for named in (True, False):
        for n in (4, 3, 2, 1):
            for i in range(len(words) - n + 1):
                span = words[i:i + n]
                low = [_fold(w).strip("'") for w in span]
                if low[0] in _FILLER or low[-1] in _FILLER or any(w in _NOT_IN_NAMES or w in _NOT_PLACE for w in low):
                    continue
                if any(re.search(r"\d", w) for w in low) or len(" ".join(span)) < 3:
                    continue
                if named and not (span[0][:1].isupper() and span[-1][:1].isupper()):
                    continue
                if not named and "and" in low:
                    continue   # "Bosnia and Herzegovina" is a name; "mekong delta and turn" is two clauses
                phrase = " ".join(span)
                if phrase not in out:
                    out.append(phrase)
        if out:
            break
    region = _region_in(_fold(text))
    if region and REGIONS[region]["label"] not in out:
        out.append(REGIONS[region]["label"])
    return out[:40]


def reply_schema(text: str) -> dict[str, Any]:
    """:func:`action_schema` for one request: the place a model may name is a run of the request's words, the
    layers it may switch are those the words ask for, and it writes no label, title or text (#590 round 3:
    the model picks, the package writes)."""
    return action_schema(places=_places_named(text), layers=_layers_named(text))


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
    if action.get("river") is True and kind in ("fly_to", "draw_area", "add_pin", "paint"):
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

    if kind == "route":
        return _validate_route(action, out, place)
    if kind == "flood_reach":
        return _validate_flood(action, out, place)
    if kind in SKETCHES:
        return _validate_sketch(action, out)
    if kind == "replay":
        return _validate_replay(action, out, place, t)
    if kind == "drop_well":
        return _validate_well(action, out, place)
    if kind == "change_landuse":
        return _validate_landuse(action, out)
    if kind == "draw_storm":
        return _validate_storm(action, out, place)
    if kind == "draw_reservoir":
        return _validate_reservoir(action, out)
    if kind == "paint":
        if action.get("question") not in PAINT_QUESTIONS:
            return None, f"paint's question must be one of {', '.join(PAINT_QUESTIONS)}"
        out["question"] = action["question"]
        bbox = _bbox(action.get("bbox")) if action.get("bbox") is not None else None
        if bbox:
            out["bbox"] = bbox
        elif place:
            out["place"] = place
        if action.get("region") in REGIONS and REGIONS[action["region"]]["bbox"]:
            out["region"] = action["region"]
            out.setdefault("bbox", REGIONS[action["region"]]["bbox"])
        if action.get("min_years") is not None:
            if not _finite(action["min_years"]) or not PAINT_MIN_YEARS[0] <= float(action["min_years"]) <= \
                    PAINT_MIN_YEARS[1]:
                return None, f"paint's min_years must be {PAINT_MIN_YEARS[0]} to {PAINT_MIN_YEARS[1]}"
            out["min_years"] = int(round(float(action["min_years"])))
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


def _validate_route(action: dict[str, Any], out: dict[str, Any], place: str | None
                    ) -> tuple[dict[str, Any] | None, str | None]:
    """Route it (#656): where the water enters (a point, a place, or ``at: here``, the river selected on the map)
    and what enters (``peak_cms`` or ``return_period``, the 10-year flow when neither, ``duration_h``, ``shape``)."""
    point = _latlon(action.get("lat"), action.get("lon"))
    if point:
        out["lat"], out["lon"] = point
    elif place:
        out["place"] = place
    elif action.get("at") in ("here", "center"):
        out["at"] = "here"
    else:
        return None, "route needs a place, a lat and lon, or at: here (the river selected on the map)"
    for key in ("peak_cms", "return_period", "duration_h"):
        v = action.get(key)
        if v is None:
            continue
        if not _finite(v):
            return None, f"route's {key} must be a number"
        lo, hi = ROUTE_LIMITS[key]
        if not (lo < float(v) <= hi):
            return None, f"route's {key} must be above {lo:g} and at most {hi:g}"
        out[key] = round(float(v), 3)
    if "peak_cms" in out:
        out.pop("return_period", None)
    elif "return_period" not in out:
        out["return_period"] = ROUTE_DEFAULT_RP
    out.setdefault("duration_h", 24.0)
    shape = action.get("shape") or "triangle"
    if shape not in ROUTE_SHAPES:
        return None, f"route's shape must be one of {', '.join(ROUTE_SHAPES)}"
    out["shape"] = shape
    label = _text(action.get("label"), 80)
    if label:
        out["label"] = label
    return out, None


def _validate_flood(action: dict[str, Any], out: dict[str, Any], place: str | None
                    ) -> tuple[dict[str, Any] | None, str | None]:
    """Flood it (#658): where (a point, a place, or ``at: here``, the river selected on the map) and the flow: a
    ``return_period`` (2 to 500 years; 100 when none), a ``flow_cms`` at the reach, or a ``flow`` of
    :data:`FLOOD_FLOWS`. The map computes the flood; the model only chooses these."""
    point = _latlon(action.get("lat"), action.get("lon"))
    if point:
        out["lat"], out["lon"] = point
    elif place:
        out["place"] = place
    elif action.get("at") in ("here", "center"):
        out["at"] = "here"
    else:
        return None, "flood_reach needs a place, a lat and lon, or at: here (the river selected on the map)"
    flow = action.get("flow")
    if flow is not None:
        if flow not in FLOOD_FLOWS:
            return None, f"flood_reach's flow must be one of {', '.join(FLOOD_FLOWS)}"
        out["flow"] = flow
    q = action.get("flow_cms")
    if q is not None and flow is None:
        if not _finite(q) or not 0 < float(q) <= 1_000_000:
            return None, "flood_reach's flow_cms must be above 0 and at most 1,000,000 m3/s"
        out["flow_cms"] = round(float(q), 3)
    rp = action.get("return_period")
    if rp is not None and "flow_cms" not in out and flow in (None, "route"):
        if not _finite(rp) or not FLOOD_RETURN_PERIODS[0] <= float(rp) <= FLOOD_RETURN_PERIODS[1]:
            return None, f"flood_reach's return_period must be {FLOOD_RETURN_PERIODS[0]:g} to " \
                         f"{FLOOD_RETURN_PERIODS[1]:g} years"
        out["return_period"] = float(rp)
    if "flow_cms" not in out and flow is None and "return_period" not in out:
        out["return_period"] = FLOOD_DEFAULT_RP
    label = _text(action.get("label"), 80)
    if label:
        out["label"] = label
    return out, None


def _validate_replay(action: dict[str, Any], out: dict[str, Any], place: str | None, t: date
                     ) -> tuple[dict[str, Any] | None, str | None]:
    """Replay an event (#659): where (a place, a point, or ``at: here``, the river selected on the map) and when (a
    ``month``, YYYY-MM). The code lists Floods past's floods there that month (:func:`aquascope.replay.candidates`)
    and takes the ``pick``-th (the most reported by default); the model never writes the story or a number."""
    point = _latlon(action.get("lat"), action.get("lon"))
    if point:
        out["lat"], out["lon"] = point
    elif place:
        out["place"] = place
    elif action.get("at") in ("here", "center"):
        out["at"] = "here"
    else:
        return None, "replay needs a place, a lat and lon, or at: here (the river selected on the map)"
    month = str(action.get("month") or "")[:7]
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        return None, "replay needs a month, YYYY-MM"
    if not (date(2000, 1, 1) <= date(int(month[:4]), int(month[5:7]), 1) <= t):
        return None, "replay's month must be from January 2000 (where Floods past starts) to this month"
    out["month"] = month
    pick = action.get("pick")
    if pick is not None:
        if not _finite(pick) or not REPLAY_PICKS[0] <= int(pick) <= REPLAY_PICKS[1]:
            return None, f"replay's pick must be {REPLAY_PICKS[0]} to {REPLAY_PICKS[1]}"
        out["pick"] = int(pick)
    label = _text(action.get("label"), 80)
    if label:
        out["label"] = label
    return out, None


def _validate_storm(action: dict[str, Any], out: dict[str, Any], place: str | None
                    ) -> tuple[dict[str, Any] | None, str | None]:
    """Draw a storm (#670): the rain (``rain_mm``, required), its hours (``duration_h``, 6 when none) and the share
    that runs off (``runoff_coefficient``, 0.3 when none: aquascope.routing.route_storm's default), over an area: a
    ``polygon`` of [lon, lat], a ``bbox``, a ``place`` (its outline from the gazetteer) or ``area: "drawn"`` (the
    reader draws it, or the storm area already on the map). The map turns it into runoff and routes it."""
    for key in ("rain_mm", "duration_h", "runoff_coefficient"):
        v = action.get(key)
        if v is None:
            continue
        lo, hi = STORM_LIMITS[key]
        if not _finite(v) or not (lo < float(v) <= hi):
            return None, f"draw_storm's {key} must be a number above {lo:g} and at most {hi:g}"
        out[key] = round(float(v), 3)
    if "rain_mm" not in out:
        return None, "draw_storm needs rain_mm, the rain in mm"
    for key, v in STORM_DEFAULTS.items():
        out.setdefault(key, v)
    poly = action.get("polygon")
    bbox = _bbox(action.get("bbox")) if action.get("bbox") is not None else None
    if isinstance(poly, (list, tuple)):
        if not 3 <= len(poly) <= STORM_LIMITS["points"]:
            return None, f"draw_storm's polygon needs 3 to {STORM_LIMITS['points']} points"
        pts = []
        for p in poly:
            ll = _latlon(p[1], p[0]) if isinstance(p, (list, tuple)) and len(p) == 2 else None
            if not ll:
                return None, "draw_storm's polygon is a list of [lon, lat] points"
            pts.append([ll[1], ll[0]])
        out["polygon"] = pts
    elif bbox:
        w, s_, e, n = bbox
        out["polygon"] = [[w, s_], [e, s_], [e, n], [w, n]]
    elif place:
        out["place"] = place
    elif action.get("area") in (None, "drawn", "here"):
        out["area"] = "drawn"
    else:
        return None, "draw_storm's area must be \"drawn\", a polygon of [lon, lat], a bbox or a place"
    label = _text(action.get("label"), 80)
    if label:
        out["label"] = label
    return out, None


def _validate_reservoir(action: dict[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Draw a reservoir (#662): the lake (``area: "drawn"``, the reader draws it or the one on the map, or a
    ``polygon``), the dam (``crest_m`` and ``top_m`` above the lowest ground, ``width_m`` of spillway; the package's
    defaults when not said) and the flood (``return_period``, ``peak_cms`` with ``duration_h``, or ``inflow:
    "wave"``, the wave on the map; the 100-year flow when none)."""
    for key in ("crest_m", "top_m", "width_m"):
        v = action.get(key)
        if v is None:
            continue
        lo, hi = RESERVOIR_LIMITS[key]
        if not _finite(v) or not (lo < float(v) <= hi):
            return None, f"draw_reservoir's {key} must be a number above {lo:g} and at most {hi:g}"
        out[key] = round(float(v), 2)
    out.setdefault("crest_m", RESERVOIR_DEFAULTS["crest_m"])
    out.setdefault("top_m", round(out["crest_m"] + RESERVOIR_DEFAULTS["rise_m"], 2))
    out.setdefault("width_m", RESERVOIR_DEFAULTS["width_m"])
    if out["top_m"] <= out["crest_m"]:
        return None, "draw_reservoir's top_m must be above its crest_m"
    if action.get("inflow") == "wave":
        out["inflow"] = "wave"
    else:
        for key in ("peak_cms", "return_period", "duration_h"):
            v = action.get(key)
            if v is None:
                continue
            lo, hi = ROUTE_LIMITS[key]
            if not _finite(v) or not (lo < float(v) <= hi):
                return None, f"draw_reservoir's {key} must be above {lo:g} and at most {hi:g}"
            out[key] = round(float(v), 3)
        if "peak_cms" in out:
            out.pop("return_period", None)
        else:
            out.setdefault("return_period", RESERVOIR_DEFAULTS["return_period"])
        out.setdefault("duration_h", RESERVOIR_DEFAULTS["duration_h"])
    poly = action.get("polygon")
    if isinstance(poly, (list, tuple)):
        if not 3 <= len(poly) <= RESERVOIR_LIMITS["points"]:
            return None, f"draw_reservoir's polygon needs 3 to {RESERVOIR_LIMITS['points']} points"
        pts = []
        for p in poly:
            ll = _latlon(p[1], p[0]) if isinstance(p, (list, tuple)) and len(p) == 2 else None
            if not ll:
                return None, "draw_reservoir's polygon is a list of [lon, lat] points"
            pts.append([ll[1], ll[0]])
        out["polygon"] = pts
    else:
        out["area"] = "drawn"
    return out, None


def _validate_sketch(action: dict[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """A levee or a breach on Flood it's grid (#663): the line (``"drawn"``: the reader draws it on the map; a list
    of [lon, lat]; or ``from_place`` and ``to_place``, two places the map looks up), and for a levee its crest (m
    above mean sea level) or a freeboard above the flood along the line (the code computes the crest; 0.5 m when
    neither is given), for a breach an optional invert. The map rasterises the line and runs Flood it again."""
    kind = out["type"]
    line = action.get("line")
    frm, to = _text(action.get("from_place"), 80), _text(action.get("to_place"), 80)
    if isinstance(line, (list, tuple)):
        if not 2 <= len(line) <= SKETCH_LIMITS["points"]:
            return None, f"{kind}'s line needs 2 to {SKETCH_LIMITS['points']} points"
        pts = []
        for p in line:
            ll = _latlon(p[1], p[0]) if isinstance(p, (list, tuple)) and len(p) == 2 else None
            if not ll:
                return None, f"{kind}'s line is a list of [lon, lat] points"
            pts.append([ll[1], ll[0]])
        out["line"] = pts
    elif frm and to:
        out["from_place"], out["to_place"] = frm, to
    elif line in (None, "drawn"):
        out["line"] = "drawn"
    else:
        return None, f"{kind}'s line must be \"drawn\", a list of [lon, lat], or from_place and to_place"
    lo, hi = SKETCH_LIMITS["level_m"]
    for key in (("crest", "freeboard_m") if kind == "draw_levee" else ("invert",)):
        v = action.get(key)
        if v is None:
            continue
        rng = SKETCH_LIMITS["freeboard_m"] if key == "freeboard_m" else (lo, hi)
        if not _finite(v) or not rng[0] <= float(v) <= rng[1]:
            return None, f"{kind}'s {key} must be a number from {rng[0]:g} to {rng[1]:g} m"
        out[key] = round(float(v), 2)
    if kind == "draw_levee" and "crest" in out:
        out.pop("freeboard_m", None)
    if kind == "draw_levee" and "crest" not in out and "freeboard_m" not in out:
        out["freeboard_m"] = 0.5
    label = _text(action.get("label"), 80)
    if label:
        out["label"] = label
    return out, None


def _validate_well(action: dict[str, Any], out: dict[str, Any], place: str | None
                   ) -> tuple[dict[str, Any] | None, str | None]:
    """Drop a well (#660): where (a point, a place, or ``at: center``, the middle of the map) and, when the reader
    gave them, the rate (m3/day), the days of pumping and the aquifer (transmissivity, storativity). The well joins
    the wells already on the map; the map computes the cone (aquascope.sketch.drawdown)."""
    point = _latlon(action.get("lat"), action.get("lon"))
    if point:
        out["lat"], out["lon"] = point
    elif place:
        out["place"] = place
    elif action.get("at") in (None, "here", "center"):
        out["at"] = "center"
    else:
        return None, "drop_well needs a place, a lat and lon, or at: center"
    for key, limit in (("rate_m3_day", "rate_m3_day"), ("days", "days"),
                       ("transmissivity", "transmissivity_m2_day"), ("storativity", "storativity")):
        v = action.get(key)
        if v is None:
            continue
        lo, hi = WELL_LIMITS[limit]
        if not _finite(v) or not lo <= float(v) <= hi:
            return None, f"drop_well's {key} must be a number from {lo:g} to {hi:g}"
        out[key] = round(float(v), 9 if key == "storativity" else 3)
    out.setdefault("rate_m3_day", WELL_DEFAULT_RATE)
    label = _text(action.get("label"), 80)
    if label:
        out["label"] = label
    return out, None


def _validate_landuse(action: dict[str, Any], out: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Change land use (#661): the area (``area: "drawn"``: the reader draws it on the map, or the one on the map;
    or ``polygon``, a ring of [lon, lat]), what
    it becomes (``use``), and when the reader gave them the design storm (``rain_mm``), the soil group and, for use
    ``impervious``, the percent impervious. The map computes the change (aquascope.sketch.landuse)."""
    uses = _landuse_uses()
    use = action.get("use")
    if use not in uses:
        return None, f"change_landuse's use must be one of {', '.join(uses)}"
    out["use"] = use
    poly = action.get("polygon")
    if isinstance(poly, (list, tuple)):
        if not 3 <= len(poly) <= LANDUSE_LIMITS["points"]:
            return None, f"change_landuse's polygon needs 3 to {LANDUSE_LIMITS['points']} points"
        pts = []
        for p in poly:
            ll = _latlon(p[1], p[0]) if isinstance(p, (list, tuple)) and len(p) == 2 else None
            if not ll:
                return None, "change_landuse's polygon is a list of [lon, lat] points"
            pts.append([ll[1], ll[0]])
        out["polygon"] = pts
    elif poly is None and action.get("area") in (None, "drawn"):
        out["area"] = "drawn"
    else:
        return None, "change_landuse needs area: drawn, or a polygon of [lon, lat]"
    for key in ("rain_mm", "impervious_pct"):
        v = action.get(key)
        if v is None:
            continue
        lo, hi = LANDUSE_LIMITS[key]
        if not _finite(v) or not lo <= float(v) <= hi:
            return None, f"change_landuse's {key} must be a number from {lo:g} to {hi:g}"
        out[key] = round(float(v), 2)
    if use == "impervious" and "impervious_pct" not in out:
        return None, "change_landuse's use impervious needs impervious_pct"
    if action.get("soil_group") is not None:
        g = str(action["soil_group"]).upper()
        if g not in ("A", "B", "C", "D"):
            return None, "change_landuse's soil_group is A, B, C or D"
        out["soil_group"] = g
    label = _text(action.get("label"), 80)
    if label:
        out["label"] = label
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
    if kind == "route":
        where = a.get("label") or a.get("place") or ("the selected river" if a.get("at") == "here" else
                                                     f"{a.get('lat'):.2f}, {a.get('lon'):.2f}")
        hours = float(a.get("duration_h") or 24.0)
        span = f"{hours / 24:g} days" if hours >= 48 and hours % 24 == 0 else f"{hours:g} h"
        if a.get("peak_cms") is not None:
            what = f"{float(a['peak_cms']):,.0f} m³/s" + (" released" if a.get("shape") == "block" else "")
        else:
            what = f"the {float(a.get('return_period') or ROUTE_DEFAULT_RP):g}-year flood"
        return f"Route {what} over {span} from {where}"
    if kind == "flood_reach":
        where = a.get("label") or a.get("place") or ("the selected river" if a.get("at") == "here" else
                                                     f"{a.get('lat'):.2f}, {a.get('lon'):.2f}")
        if a.get("flow_cms") is not None:
            what = f"{float(a['flow_cms']):,.0f} m³/s"
        elif a.get("flow") == "forecast_peak":
            what = "the forecast peak"
        elif a.get("flow") == "forecast":
            what = "the 15-day forecast"
        elif a.get("flow") == "route":
            what = f"Route it's {float(a.get('return_period') or FLOOD_DEFAULT_RP):g}-year wave"
        else:
            what = f"the {float(a.get('return_period') or FLOOD_DEFAULT_RP):g}-year flow"
        return f"Flood {where} at {what}"
    if kind in SKETCHES:
        what = "a levee" if kind == "draw_levee" else "a breach"
        if a.get("from_place"):
            where = f"from {a['from_place']} to {a['to_place']}"
        elif isinstance(a.get("line"), list):
            where = f"along {len(a['line'])} points"
        else:
            where = "on the map"
        if kind == "draw_levee":
            top = (f", crest {float(a['crest']):g} m" if a.get("crest") is not None
                   else f", {float(a.get('freeboard_m') or 0.5):g} m above the flood")
        else:
            top = f", down to {float(a['invert']):g} m" if a.get("invert") is not None else ""
        return f"Draw {what} {where}{top}, and flood again"
    if kind == "change_landuse":
        from aquascope.sketch.landuse import TR55

        what = (TR55[a["use"]]["label"].split(" (")[0].lower() if a["use"] in TR55
                else f"{float(a.get('impervious_pct') or 0):g}% impervious")
        where = "the area drawn" if a.get("area") == "drawn" else f"an area of {len(a.get('polygon') or [])} points"
        rain = f", {float(a['rain_mm']):g} mm of rain" if a.get("rain_mm") is not None else ""
        soil = f", soil group {a['soil_group']}" if a.get("soil_group") else ""
        return f"Change {where} to {what}{rain}{soil}"
    if kind == "drop_well":
        where = a.get("label") or a.get("place") or ("the middle of the map" if a.get("at") == "center" else
                                                     f"{a.get('lat'):.3f}, {a.get('lon'):.3f}")
        q = float(a.get("rate_m3_day") or WELL_DEFAULT_RATE)
        days = f" for {float(a['days']):g} days" if a.get("days") is not None else ""
        return f"Drop a well at {where} pumping {q:,.0f} m³/day ({q / 86.4:.3g} L/s){days}"
    if kind == "draw_storm":
        if a.get("place") or a.get("label"):
            where = a.get("label") or a.get("place")
        elif a.get("polygon"):
            where = f"an area of {len(a['polygon'])} points"
        else:
            where = "the area drawn on the map"
        c = float(a.get("runoff_coefficient") or STORM_DEFAULTS["runoff_coefficient"])
        return (f"Rain {float(a['rain_mm']):g} mm in {float(a.get('duration_h') or STORM_DEFAULTS['duration_h']):g} h "
                f"over {where}, {c:g} of it running off, and route it")
    if kind == "draw_reservoir":
        where = f"an area of {len(a['polygon'])} points" if a.get("polygon") else "the lake drawn on the map"
        if a.get("inflow") == "wave":
            what = "the wave on the map"
        elif a.get("peak_cms") is not None:
            what = f"{float(a['peak_cms']):,.0f} m³/s"
        else:
            what = f"the {float(a.get('return_period') or RESERVOIR_DEFAULTS['return_period']):g}-year flood"
        return (f"A reservoir on {where} (crest {float(a['crest_m']):g} m, top {float(a['top_m']):g} m, spillway "
                f"{float(a['width_m']):g} m wide), and route {what} through it")
    if kind == "replay":
        where = a.get("label") or a.get("place") or ("the selected river" if a.get("at") == "here" else
                                                     f"{a.get('lat'):.2f}, {a.get('lon'):.2f}")
        pick = f" (flood {a['pick']} there)" if a.get("pick", 1) != 1 else ""
        return f"Replay the flood of {_month_words(a['month'] + '-01')} at {where}{pick}"
    if kind == "paint":
        where = a.get("label") or a.get("place") or ("the box" if a.get("bbox") else "this view")
        return f"Paint {where}: {PAINT_QUESTIONS[a['question']]['label']}"
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


#: "which rivers in Spain are running low this month?" read the place as "Spain are running low" (#590 check3):
#: the place between the rivers and their state is moved after it, so the status and the place are read apart.
_STATUS_STATE = (r"(?:much |far |very |extremely |unusually )?(?:above normal|below normal|above average|"
                 r"below average|higher than normal|lower than normal|normal|average|high|low|dry|wet|swollen|"
                 r"in flood|in drought)")


def _status_phrase(c: _Clause) -> None:
    m = re.search(r"\b(?P<noun>rivers?|flows?|streams?|basins?)\s+(?P<where>(?:in|across|over|around)\s+"
                  r"(?!the map\b)[a-z][a-z' .-]*?)\s+(?:(?:are|is)\s+)?(?:(?:running|flowing|doing)\s+)?"
                  r"(?P<state>" + _STATUS_STATE + r")\b", c.text)
    if m:
        c.text = (c.text[: m.start()] + f"{m.group('noun')} {m.group('state')} {m.group('where')}"
                  + c.text[m.end():]).strip()
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


_FLOW_UNIT = r"(?:m3/s|m³/s|m3 s-1|m3s-1|cumecs|cms|cubic met(?:re|er)s? (?:a|per) second)"


def _route_phrase(c: _Clause) -> None:
    """Route it (#656): "route a flood of 500 m3/s from Basel", "release 800 m3/s for 2 days from here",
    "route the 100-year flood down the Rhine from Basel", "send a flood down the river from here"."""
    m = re.search(
        r"^(?P<verb>route|send|release|run|let go)\s+(?:a\s+|an\s+|the\s+)?"
        r"(?:(?P<rp>\d{1,4})[- ]?(?:year|yr)s?\s+)?(?P<what>flood|wave|flood wave|release|pulse|water)?\s*"
        r"(?:of\s+)?(?:(?P<q>\d[\d,]*(?:\.\d+)?)\s*" + _FLOW_UNIT + r"\s*)?"
        r"(?:(?:flood|wave|release|pulse|water)\s+)?"
        r"(?:(?:for|over)\s+(?P<d>\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|ten)\s*"
        r"(?P<du>hours?|hrs?|h|days?|d)\s+)?"
        r"(?P<tail>.*)$", c.text)
    if not m or not (m.group("what") or m.group("q") or m.group("verb") == "release"):
        return
    tail = re.sub(r"\s*(?:all the way\s+)?(?:down\s+)?to\s+the\s+(?:sea|ocean)$", "", m.group("tail").strip())
    t = re.fullmatch(r"(?:down\s+(?:the\s+)?(?P<river>.+?))?\s*(?:(?:from|at|starting at)\s+(?P<p>.+))?", tail)
    if t is None:
        return
    if m.group("verb") in ("run", "let go") and not m.group("what"):
        return
    action: dict[str, Any] = {"type": "route"}
    if m.group("q"):
        action["peak_cms"] = float(m.group("q").replace(",", ""))
    elif m.group("rp"):
        action["return_period"] = float(m.group("rp"))
    if m.group("d"):
        n = m.group("d")
        num = float(n) if re.match(r"\d", n) else float(_WORD_NUMBERS.get(n, 1))
        action["duration_h"] = num * (24.0 if m.group("du").startswith("d") else 1.0)
    if m.group("verb") == "release" or m.group("what") in ("release", "pulse"):
        action["shape"] = "block"
    where = (t.group("p") or "").strip()
    river = (t.group("river") or "").strip()
    if not where or where in ("here", "this river", "this reach", "the selected river", "this point"):
        if river and river not in ("river", "this river"):
            action["place"] = _place_text(_raw_span(c, river))
        else:
            action["at"] = "here"
    else:
        name = _place_text(_raw_span(c, where))
        if river and river not in ("river", "this river"):
            name = f"{_place_text(_raw_span(c, river))} at {name}"
        action["place"] = name
    c.text = ""
    c.add("route", action)


def _flood_phrase(c: _Clause) -> None:
    """Flood it (#658): "flood Upton upon Severn at the 100-year flow", "flood the Severn at Tewkesbury with 500
    m3/s", "flood here", "flood this river with the forecast peak", "flood it"."""
    m = re.search(r"^(?:flood(?:\s+it)?|run\s+flood\s+it|run\s+a\s+screening\s+flood)(?:\s+(?:on|at|of|for))?\s*"
                  r"(?P<tail>.*)$", c.text)
    layer_words = r"^(?:depths?|warnings?|ahead|past|risk|layers?|maps?|extents?|zones?|plains?|s\b)"
    if not m or re.match(layer_words, m.group("tail")):
        return
    tail = m.group("tail").strip()
    action: dict[str, Any] = {"type": "flood_reach"}
    flow = re.search(r"\s*(?:at|with|for|to)\s+(?:the\s+|a\s+)?(?:(?P<rp>\d{1,3})[- ]?(?:year|yr)s?\s*(?:flow|flood)?"
                     r"|(?P<q>\d[\d,]*(?:\.\d+)?)\s*" + _FLOW_UNIT +
                     r"|(?P<peak>forecast(?:ed)?\s+peak|peak\s+of\s+the\s+forecast)"
                     r"|(?P<fc>(?:15[- ]day\s+)?forecast(?:\s+hydrograph)?)"
                     r"|(?P<route>route it'?s?\s+wave|routed\s+wave))\s*$", tail)
    if flow:
        if flow.group("rp"):
            action["return_period"] = float(flow.group("rp"))
        elif flow.group("q"):
            action["flow_cms"] = float(flow.group("q").replace(",", ""))
        elif flow.group("peak"):
            action["flow"] = "forecast_peak"
        elif flow.group("fc"):
            action["flow"] = "forecast"
        elif flow.group("route"):
            action["flow"] = "route"
        tail = tail[:flow.start()].strip()
    if action.get("flow") is None and action.get("flow_cms") is None:
        action.setdefault("return_period", FLOOD_DEFAULT_RP)
    if not tail or tail in ("here", "this river", "this reach", "the selected river", "the river", "it"):
        action["at"] = "here"
    else:
        action["place"] = _place_text(_raw_span(c, re.sub(r"^the\s+river\s+", "", tail)))
    c.text = ""
    c.add("flood_reach", action)


def _replay_phrase(c: _Clause) -> None:
    """Replay an event (#659): "replay the Severn flood of November 2024", "replay the flood of September 2024 at
    Asheville", "replay the November 2024 flood here", "replay the floods of Nov 2024 on the Severn"."""
    m = re.search(r"^(?:replay|re-play|tell|show)\s+(?:me\s+)?(?:the\s+|a\s+)?(?P<pre>.*?)\s*floods?\b\s*(?P<tail>.*)$",
                  c.text)
    if not m or not re.match(r"^(?:replay|re-play)", c.text):
        return
    words = f"{m.group('pre')} {m.group('tail')}".strip()
    when = re.search(r"(?:\b(?:of|in|from)\s+)?" + _MONTH_RE + r"\s+(?P<y>(?:19|20)\d{2})\b", words)
    if not when:
        when = re.search(r"(?:\b(?:of|in|from)\s+)?(?P<y>(?:19|20)\d{2})-(?P<m>0[1-9]|1[0-2])\b", words)
        if not when:
            return
        month = f"{when.group('y')}-{when.group('m')}"
    else:
        month = f"{when.group('y')}-{MONTHS[when.group('mon')]:02d}"
    rest = (words[: when.start()] + " " + words[when.end():]).strip()
    rest = re.sub(r"^(?:on|at|in|of|along|near|for)\s+", "", re.sub(r"\s+", " ", rest)).strip()
    rest = re.sub(r"\s+(?:on|at|in|of|along|near)$", "", rest).strip()
    rest = re.sub(r"^(?:the\s+)?(?:river\s+)?", "", rest) if rest.startswith(("the river ", "river ")) else rest
    action: dict[str, Any] = {"type": "replay", "month": month}
    if not rest or rest in ("here", "this river", "this reach", "the selected river", "the river", "it", "this"):
        action["at"] = "here"
    else:
        action["place"] = _place_text(_raw_span(c, re.sub(r"^the\s+", "", rest)))
    c.text = ""
    c.add("replay", action)


_HOURS = (r"(?P<d>\d+(?:\.\d+)?|an?|one|two|three|four|five|six|eight|ten|twelve|twenty)\s*"
          r"(?P<du>hours?|hrs?|h|days?|d)\b")


def _storm_phrase(c: _Clause) -> None:
    """Draw a storm (#670): "50 mm of rain over this area in 6 hours", "draw a storm of 80 mm over 12 h",
    "rain 100 mm on Wales in 2 days with a runoff coefficient of 0.5", "a 40 mm storm here". The area is the one
    the reader draws (or the one on the map) unless a place is named."""
    if not re.search(r"\d\s*(?:mm|millimet(?:re|er)s?)\b", c.text) or not re.search(r"\b(?:rain\w*|storms?|downpour)\b",
                                                                                     c.text):
        return
    m = re.search(r"^(?:(?:what if|imagine|suppose)\s+)?(?:(?:draw|make|drop|put|add|let|have|run|simulate)\s+)?"
                  r"(?:it\s+)?(?:rain\s+)?(?:a\s+|an\s+)?"
                  r"(?:(?:storm|downpour|rainstorm)\s+(?:of\s+)?)?(?P<mm>\d+(?:\.\d+)?)\s*(?:mm|millimet(?:re|er)s?)\b"
                  r"\s*(?:of\s+)?(?:rain(?:fall)?|storm|downpour|rainstorm)?\s*(?:storm\s+)?(?:falling|fell|falls?)?\s*"
                  r"(?P<tail>.*)$", c.text)
    if not m:
        return
    tail = m.group("tail").strip()
    action: dict[str, Any] = {"type": "draw_storm", "rain_mm": float(m.group("mm"))}
    coef = re.search(r",?\s*(?:with\s+|and\s+)?(?:a\s+)?runoff\s+coefficient\s+(?:of\s+)?"
                     r"(?P<c>0?\.\d+|1(?:\.0+)?|0)\b", tail)
    if coef:
        action["runoff_coefficient"] = float(coef.group("c"))
        tail = (tail[:coef.start()] + " " + tail[coef.end():]).strip()
    dur = re.search(r"\b(?:in|over|for|during|within|lasting)\s+" + _HOURS, tail)
    if dur:
        n = dur.group("d")
        num = float(n) if re.match(r"\d", n) else float(_WORD_NUMBERS.get(n, 1))
        action["duration_h"] = num * (24.0 if dur.group("du").startswith("d") else 1.0)
        tail = (tail[:dur.start()] + " " + tail[dur.end():]).strip()
    tail = re.sub(r"\s+", " ", tail).strip(" ,")
    tail = re.sub(r"^(?:over|on|across|onto|in|above|upon)\s+", "", tail)
    if not tail or tail in ("here", "this area", "the area", "that area", "this polygon", "the drawn area",
                            "the area drawn", "this", "it", "there", "the area on the map", "this catchment"):
        action["area"] = "drawn"
    else:
        action["place"] = _place_text(_raw_span(c, re.sub(r"^the\s+", "", tail)))
    c.text = ""
    c.add("draw_storm", action)


def _reservoir_phrase(c: _Clause) -> None:
    """Draw a reservoir (#662): "draw a reservoir here", "draw a reservoir with a 40 m spillway at 12 m", "put a dam
    here and route the 100-year flood through it", "draw a reservoir and send the wave through it". The lake is the
    one the reader draws (or the one on the map)."""
    m = re.search(r"^(?:draw|build|add|put|make|place|create)\s+(?:a\s+|an\s+|the\s+)?(?:reservoir|dam|lake)\b"
                  r"(?P<tail>.*)$", c.text)
    if not m:
        return
    tail = " " + m.group("tail").strip() + " "
    action: dict[str, Any] = {"type": "draw_reservoir"}
    num = r"(?P<v>\d+(?:\.\d+)?)\s*(?:m|metres?|meters?)\b"
    for key, pat in (("width_m", r"(?:with\s+)?(?:a\s+)?" + num + r"\s*(?:wide\s+)?spillway"),
                     ("width_m", r"spillway\s+(?:of\s+)?" + num + r"\s*wide"),
                     ("top_m", num + r"\s*(?:high|tall)"),
                     ("top_m", r"(?:dam\s+)?top\s+(?:at\s+|of\s+)?" + num),
                     ("crest_m", r"(?:spillway\s+)?crest\s+(?:at\s+|of\s+)?" + num),
                     ("crest_m", r"\bat\s+" + num)):
        if key in action:
            continue
        hit = re.search(pat, tail)
        if hit:
            action[key] = float(hit.group("v"))
            tail = tail[:hit.start()] + " " + tail[hit.end():]
    flood = re.search(r"(?:and\s+)?(?:route|send|pass|run)\s+(?:the\s+|a\s+)?(?:(?P<rp>\d{1,4})[- ]?(?:year|yr)s?\s+)?"
                      r"(?:(?P<q>\d[\d,]*(?:\.\d+)?)\s*" + _FLOW_UNIT + r"\s*)?(?P<what>flood|wave|storm|flow)?\s*"
                      r"(?:wave\s+)?(?:on\s+the\s+map\s+)?through(?:\s+it)?", tail)
    if flood:
        if flood.group("q"):
            action["peak_cms"] = float(flood.group("q").replace(",", ""))
        elif flood.group("rp"):
            action["return_period"] = float(flood.group("rp"))
        elif flood.group("what") in ("wave", "storm"):
            action["inflow"] = "wave"
        tail = tail[:flood.start()] + " " + tail[flood.end():]
    rest = re.sub(r"\s+", " ", tail).strip(" ,")
    rest = re.sub(r"^(?:and\s+)?", "", rest)
    if rest and rest not in ("here", "on this river", "on the river", "across the river", "across this river",
                             "on this reach", "there", "and", "with", "on it"):
        return
    action["area"] = "drawn"
    c.text = ""
    c.add("draw_reservoir", action)


def _sketch_phrase(c: _Clause) -> None:
    """Levees and breaches (#663): "draw a levee", "draw a levee from Upton upon Severn to Hanley Castle at 14 m",
    "build a flood wall 1 m above the flood", "draw a breach", "breach the levee". The line is drawn on the map
    unless two places are named; the crest is the reader's number or a freeboard above the flood."""
    m = re.search(r"^(?:draw|build|add|put|make|raise)\s+(?:a|an|the)?\s*(?P<what>levee|dyke|dike|flood\s*wall|"
                  r"embankment|bund|breach|gap)\b(?P<tail>.*)$", c.text)
    if not m:
        m = re.search(r"^(?P<what>breach)\s+(?:the\s+|a\s+)?(?:levee|dyke|dike|flood\s*wall|embankment|bund)"
                      r"(?P<tail>.*)$", c.text)
    if not m:
        return
    kind = "draw_breach" if m.group("what") in ("breach", "gap") else "draw_levee"
    tail = m.group("tail").strip()
    action: dict[str, Any] = {"type": kind}
    num = r"(?P<v>-?\d+(?:\.\d+)?)\s*(?:m|metres?|meters?)\b"
    fb = re.search(num + r"\s+above\s+the\s+(?:flood|water)", tail)
    lvl = re.search(r"(?:at|with\s+a\s+crest\s+(?:of|at)|crest(?:\s+(?:of|at))?|to|down\s+to|"
                    r"invert(?:\s+(?:of|at))?)\s+" + num, tail) if not fb else None
    if fb:
        if kind == "draw_levee":
            action["freeboard_m"] = float(fb.group("v"))
        tail = (tail[:fb.start()] + tail[fb.end():]).strip()
    elif lvl:
        action["crest" if kind == "draw_levee" else "invert"] = float(lvl.group("v"))
        tail = (tail[:lvl.start()] + tail[lvl.end():]).strip()
    ends = re.search(r"\bfrom\s+(?P<a>.+?)\s+to\s+(?P<b>.+?)\s*,?$", tail)
    if ends:
        action["from_place"] = _place_text(_raw_span(c, ends.group("a")))
        action["to_place"] = _place_text(_raw_span(c, ends.group("b")))
    else:
        action["line"] = "drawn"
    if kind == "draw_levee" and "crest" not in action:
        action.setdefault("freeboard_m", 0.5)
    c.text = ""
    c.add(kind, action)


_WELL_RATE = (r"(?P<q>\d[\d,]*(?:\.\d+)?)\s*(?P<u>m3/day|m3/d|m³/day|m³/d|m3 a day|m3 per day|"
              r"cubic met(?:re|er)s? (?:a|per) day|l/s|lps|litres? (?:a|per) second|liters? (?:a|per) second)")
_WELL_SPAN = (r"for\s+(?P<n>\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
              r"(?P<per>days?|weeks?|months?|years?)")


def _well_phrase(c: _Clause) -> None:
    """Drop a well (#660): "drop a well here", "drop a well at Lyon pumping 2000 m3/day", "drill a borehole near
    Lille pumping 20 l/s for 90 days", "pump 1500 m3/day here for a year". The rate and the days only when said."""
    m = re.search(r"^(?:drop|put|add|place|drill|sink|dig)\s+(?:a\s+|an\s+|another\s+|one\s+)?(?:pumping\s+)?"
                  r"(?:well|borehole|bore ?hole)\b(?P<tail>.*)$", c.text)
    if not m:
        m = re.search(r"^pump\s+(?P<tail>(?:at\s+)?" + _WELL_RATE + r".*)$", c.text)
    if not m:
        return
    tail = m.group("tail").strip()
    action: dict[str, Any] = {"type": "drop_well"}
    q = re.search(r"(?:(?:pumping|that pumps|at|of|with)\s+)?" + _WELL_RATE, tail)
    if q:
        v = float(q.group("q").replace(",", ""))
        action["rate_m3_day"] = round(v * 86.4, 3) if q.group("u") in ("l/s", "lps") or "second" in q.group("u") \
            else v
        tail = (tail[:q.start()] + " " + tail[q.end():]).strip()
    d = re.search(_WELL_SPAN, tail)
    if d:
        n = d.group("n")
        num = float(n) if re.match(r"\d", n) else float(_WORD_NUMBERS.get(n, 1))
        per = {"d": 1.0, "w": 7.0, "m": 30.0, "y": 365.0}[d.group("per")[0]]
        action["days"] = round(num * per, 3)
        tail = (tail[:d.start()] + " " + tail[d.end():]).strip()
    tail = re.sub(r"\s+", " ", tail).strip(" ,")
    where = re.fullmatch(r"(?:(?:at|in|near|on|by)\s+)?(?P<p>.+)", tail) if tail else None
    p = where.group("p").strip() if where else ""
    if not p or p in ("here", "this point", "the centre", "the center", "the middle", "the map", "this spot"):
        action["at"] = "center"
    else:
        action["place"] = _place_text(_raw_span(c, p))
    c.text = ""
    c.add("drop_well", action)


_LU_RAIN = r"(?:with|for|under|in)\s+(?:a\s+)?(?P<rain>\d+(?:\.\d+)?)\s*mm(?:\s+(?:of\s+)?(?:rain|storm|rainfall))?"
_LU_AREA = r"(?:this|the|that)\s+(?:area|field|plot|site|land|polygon|place)|here|it"


def _landuse_use(words: str) -> str | None:
    for key, pat in LANDUSE_WORDS.items():
        if re.fullmatch(r"(?:" + pat + r")", words.strip()):
            return key
    return None


def _landuse_phrase(c: _Clause) -> None:
    """Change land use (#661): "change this area to houses", "turn this field into a car park", "what if this area is
    built over", "build houses here", "pave this area", "plant woods here", each with an optional "with 80 mm of
    rain". The area is drawn on the map."""
    text = c.text
    rain = re.search(_LU_RAIN, text)
    if rain:
        text = (text[:rain.start()] + text[rain.end():]).strip()
    text = re.sub(r"\s+", " ", text).strip(" ,")
    use = None
    for pat in (r"^(?:change|turn|convert|make)\s+(?:" + _LU_AREA + r")\s+(?:to|into)\s+(?P<w>.+)$",
                r"^(?:change\s+)?(?:the\s+)?land\s*use\s+(?:here\s+)?(?:to|into)\s+(?P<w>.+)$",
                r"^what if\s+(?:" + _LU_AREA + r")\s+(?:is|was|were|became|becomes|gets|got)\s+(?P<w>.+)$",
                r"^(?:build|put|plant|develop)\s+(?P<w>.+?)\s+(?:here|on\s+(?:" + _LU_AREA + r")|in\s+(?:" + _LU_AREA
                + r"))$",
                r"^(?P<w>pave|build on|build over|develop|urbani[sz]e)\s+(?:" + _LU_AREA + r")$"):
        m = re.search(pat, text)
        if not m:
            continue
        w = re.sub(r"^(?:a|an|the|some)\s+", "", m.group("w").strip())
        w = {"pave": "paved", "build on": "built on", "build over": "built over", "develop": "developed",
             "urbanise": "urbanised", "urbanize": "urbanized"}.get(w, w)
        use = _landuse_use(w) or _landuse_use(m.group("w").strip())
        if use:
            break
    if not use:
        return
    action: dict[str, Any] = {"type": "change_landuse", "area": "drawn", "use": use}
    if rain:
        action["rain_mm"] = float(rain.group("rain"))
    c.text = ""
    c.add("change_landuse", action)


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


_PAINT_ALIASES = sorted(((alias, key) for key, v in PAINT_QUESTIONS.items() for alias in v["aliases"]),
                        key=lambda x: -len(x[0]))
_PAINT_Q = "(?P<q>" + "|".join(re.escape(a) for a, _ in _PAINT_ALIASES) + ")"
_WHERE = r"(?:\s+(?:in|across|over|for|around)\s+(?:the\s+)?(?P<where>[a-z][a-z' .-]*))?"


def _paint_where(c: _Clause, where: str | None, question: str) -> None:
    """The paint action, and a flight to the region or place it names (none: this view)."""
    paint: dict[str, Any] = {"type": "paint", "question": question}
    if not where:
        c.add("paint", paint)
        return
    region = _region(where)
    if region and REGIONS[region]["bbox"]:
        spec = REGIONS[region]
        c.add("paint", {"type": "fly_to", "region": region, "bbox": spec["bbox"]},
              {**paint, "region": region, "bbox": spec["bbox"], "label": spec["label"]})
        return
    place = _place_text(_raw_span(c, where))
    c.add("paint", {"type": "fly_to", "place": place}, {**paint, "place": place, "label": place})


def _paint_phrase(c: _Clause) -> None:
    """Paint the globe (#657): "paint the flood trend", "map low-flow trends in France", "where are floods getting
    bigger in Europe?", "are droughts getting worse in Poland?". The question is one of :data:`PAINT_QUESTIONS`."""
    m = re.search(r"^(?:paint|colou?r|map|show)\s+(?:(?:this|the|my)\s+(?:view|map|screen)\s+)?(?:with\s+|by\s+)?"
                  r"(?:the\s+)?" + _PAINT_Q + r"(?:\s+map)?(?:\s+(?:here|in view|on the map))?" + _WHERE + r"$", c.text)
    if not m:
        m = re.search(r"^(?:paint|colou?r)\s+(?:this|the)\s+(?:view|map|screen)\s+(?:with|by)\s+(?:the\s+)?"
                      + _PAINT_Q + r"$", c.text)
    if m:
        question = dict(_PAINT_ALIASES)[m.group("q")]
        c.text = ""
        _paint_where(c, m.groupdict().get("where"), question)
        return
    m = re.search(r"^(?:where\s+)?(?:are|is)\s+(?:the\s+)?(?P<thing>floods?|flood peaks|peak flows|annual floods|"
                  r"low flows|low-flows|droughts?|dry spells)\s+(?:getting|becoming|growing)\s+"
                  r"(?:bigger|larger|worse|smaller|more severe|less severe|lower|higher|stronger|weaker)"
                  + _WHERE + r"$",
                  c.text) \
        or re.search(r"^(?:where\s+)?(?:are|is)\s+(?:the\s+)?(?P<thing>floods?|flood peaks|peak flows|annual floods|"
                     r"low flows|low-flows|droughts?)\s+(?:increasing|decreasing|rising|falling|changing|growing|"
                     r"shrinking)" + _WHERE + r"$", c.text)
    if m:
        thing = m.group("thing")
        question = "low_flow_trend" if re.match(r"low|drought|dry", thing) else "flood_trend"
        c.text = ""
        _paint_where(c, m.group("where"), question)


_NOT_IN_NAMES = frozenset("""a an my your our some something anything everything nothing way how why what which clever
please it things stuff help more weather forecast data are is was were running flowing doing""".split())


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
                                      "highlight_river", "route", "flood_reach", "draw_levee", "draw_breach",
                                      "drop_well", "change_landuse", "replay",
                                      "draw_storm", "draw_reservoir",
                                      "add_pin", "paint"))}


def _parse_clause(raw: str, today: date) -> _Clause:
    c = _Clause(_fold(raw).strip(" .!?"), raw.strip(" .!?"), today)
    c.text = re.sub(r"^(?:please|hey|ok|okay|now|can you|could you|would you|i want to|i'd like to|let me|let's)\s+",
                    "", c.text)
    c.text = re.sub(r"\s+(?:please|for me|on the map)$", "", c.text)
    for step in (_pin_phrase, _paint_phrase, _zoom_phrase, _basemap_phrase, _replay_phrase, _sketch_phrase,
                 _well_phrase, _landuse_phrase, _storm_phrase, _reservoir_phrase,
                 _area_phrase, _flood_phrase, _route_phrase, _river_phrase):
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


# "What am I seeing?": explain the view (#562, aquascope.map_explain), a control like undo rather than an action.
_EXPLAIN = re.compile(
    r"(?:please\s+|can you\s+)?(?:"
    r"what am i (?:seeing|looking at)(?: here| now| on (?:the|this) (?:map|screen|view))?"
    r"|what(?: is|s) (?:this|that|on (?:the )?(?:screen|map)|going on(?: here)?|happening(?: here)?)"
    r"|what does (?:this|the)(?: (?:view|map|screen))? show(?: me)?"
    r"|what do (?:the|these) colou?rs mean(?: here)?"
    r"|what do i see(?: here)?"
    r"|explain(?: (?:this|the|my|what is on the|whats on the))?(?: (?:view|map|screen))?(?: to me)?(?: please)?"
    r"|explain what i(?: a)?m (?:seeing|looking at)"
    r"|(?:describe|summari[sz]e|walk me through)(?: (?:this|the|my))? (?:view|map|screen)(?: for me)?"
    r"|tell me about (?:this|the) (?:view|map|screen)"
    r"|tell me what i(?: a)?m? (?:see|seeing|looking at)"
    # "tell me in plain words what this map is showing me" went to the device model, which flew to the selected
    # gauge (#590 recheck): asking what the map shows, in any of these words, is the notes on the view.
    r"|(?:tell me|explain|say)(?: in (?:plain|simple) (?:words|english|terms|language))?"
    r" what (?:this|the|my) (?:map|view|screen) (?:is )?(?:show(?:s|ing)?|telling me|saying)(?: me)?"
    r"(?: in (?:plain|simple) (?:words|english|terms|language))?"
    r"|what(?: is|s) (?:this|the) (?:map|view|screen) (?:showing|telling me|saying)(?: me)?"
    r")")


# "Review my model": the checks on the model result on the map (#564, aquascope.mesh_review), a control too.
#: "Is anything wrong with my flood model?" went to a model on the device, which turned Floods past on (#590): the
#: ways people ask whether a model result is right are read here, so they reach the review.
_KINDS = r"(?:flood|hec ?-?ras|hecras|modflow|swmm|groundwater|hydraulic|2d|1d|river|catchment)"
_MODEL_KIND = r"(?:" + _KINDS + r" )?"
_MODEL_NOUN = _MODEL_KIND + r"(?:model(?: (?:results?|run|output))?|run|results?|mesh|output|simulation)"
#: "Are the depths in my flood model realistic?" and "does this simulation look plausible to you?" did not reach the
#: review on the device (#590 recheck), and the first flew to the Nile: these words are read here too.
_VERDICT = (r"(?:look )?(?:right|ok|okay|sensible|plausible|correct|reasonable|realistic|believable|accurate|sane)"
            r"(?: to you)?")
_QUANTITY = (r"(?:water )?(?:depths?|velocit(?:y|ies)|speeds?|heads?|(?:water )?levels?|flows?|discharges?|"
             r"values|numbers|results?|outputs?)")
#: What goes wrong in a model run, as a reader names it: the review's checks look for these.
_TROUBLE = (r"(?:(?:numerical|model) )?(?:instabilit(?:y|ies)|oscillations?|wiggles?|spikes?|"
            r"(?:wet[- ]dry )?flicker(?:ing)?|convergence (?:problems?|issues?|errors?)|(?:mass|volume) (?:balance )?"
            r"errors?|negative depths?|unstable cells?)")
_REVIEW = re.compile(
    r"(?:please\s+|can you\s+|could you\s+)?(?:"
    r"(?:review|check|audit|validate|sanity check|inspect)(?: (?:my|the|this|these|over))?(?: " + _KINDS
    + r")?(?: model)?(?: (?:results?|run|mesh|output|simulation))?(?: for (?:me|errors|problems|mistakes))?(?: please)?"
    r"|(?:does|do|is|are) (?:my|the|this|these) " + _MODEL_NOUN + r" " + _VERDICT
    + r"|(?:does|do|is|are) (?:my|the|this|these) " + _QUANTITY
    + r"(?: (?:in|of|from|on) (?:my|the|this|these) " + _MODEL_NOUN + r")? " + _VERDICT
    + r"|(?:is|are) (?:there )?(?:anything|something|any (?:problems?|issues?|errors?|mistakes?)) (?:wrong )?"
    r"(?:with|in) (?:my|the|this|these) " + _MODEL_NOUN
    + r"|what(?: is|s) wrong with (?:my|the|this) " + _MODEL_NOUN
    + r"|any (?:problems?|issues?|errors?|mistakes?) (?:with|in) (?:my|the|this|these) " + _MODEL_NOUN
    + r"|is (?:my|the|this) " + _MODEL_NOUN + r" (?:wrong|broken)"
    # "are there any numerical instabilities in this run?" turned two layers on unasked (#590 check3).
    + r"|(?:are|is) there (?:any )?(?:signs? of )?" + _TROUBLE + r"(?: (?:in|with|on) (?:my|the|this|these) "
    + _MODEL_NOUN + r")?"
    + r"|any " + _TROUBLE + r"(?: (?:in|with|on) (?:my|the|this|these) " + _MODEL_NOUN + r")?"
    + r"|(?:does|do) (?:my|the|this|these) " + _MODEL_NOUN + r" (?:have|show) (?:any )?" + _TROUBLE
    + r"|(?:check|look|test) (?:my|the|this|these) " + _MODEL_NOUN + r" for " + _TROUBLE
    + r"|(?:check|look) for " + _TROUBLE + r"(?: (?:in|with|on) (?:my|the|this|these) " + _MODEL_NOUN + r")?"
    + r"|(?:is|are) (?:my|the|this|these) " + _MODEL_NOUN + r" (?:numerically )?(?:stable|unstable|converged)"
    r")")


def parse_command(text: str, today: str | date | None = None) -> dict[str, Any]:
    """Read one plain-English request into checked map actions, with no model and no network.

    Returns ``{"text", "matched", "actions", "said", "rules", "errors", "control"}``: ``matched`` is True only
    when every word of the request was understood (so "show me the Rhine and something clever" is left to a
    model rather than half done); ``said`` is one line per action for the log; ``control`` is "undo" or
    "undo_all" when the request was about the log itself, "explain" for "what am I seeing?" (#562) and
    "review" for "review my model" (#564). Place
    names stay names (``place``) until :func:`resolve_actions` looks them up.
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
    if _EXPLAIN.fullmatch(folded.replace("'", "")):
        return {**out, "matched": True, "control": "explain", "rules": ["explain"]}
    if _REVIEW.fullmatch(folded.replace("'", "")) and re.search(
            r"model|result|run|mesh|output|review|simulation|depth|velocit|head|instabilit|oscillat|converge|"
            r"flicker|mass|volume", folded):
        return {**out, "matched": True, "control": "review", "rules": ["review"]}

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
        if a.get("type") in SKETCHES and a.get("from_place"):
            ends = []
            for name in (a["from_place"], a["to_place"]):
                key = (name.lower(), "any")
                if key not in cache:
                    cache[key] = resolve_place(name, fetch=fetch)
                ends.append(cache[key])
            bad = [e.get("error") or f"{n} has no position" for e, n in zip(ends, (a["from_place"], a["to_place"]))
                   if e.get("error") or e.get("lat") is None]
            if bad:
                notes.extend(bad)
                continue
            used = used or any(bool(e.get("credit")) for e in ends)
            b = {k: v for k, v in a.items() if k not in ("from_place", "to_place")}
            b["line"] = [[ends[0]["lon"], ends[0]["lat"]], [ends[1]["lon"], ends[1]["lat"]]]
            b["where"] = f"{ends[0]['name']} to {ends[1]['name']}"
            out.append(b)
            continue
        if not a.get("place"):
            out.append({k: v for k, v in a.items() if k != "river"})
            continue
        want = "river" if a.get("river") or a["type"] in ("highlight_river", "route", "flood_reach", "replay") \
            else "any"
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
        elif a["type"] == "drop_well":
            if hit.get("lat") is None:
                notes.append(f"{a['place']} has no position for a well")
                continue
            b.update(lat=hit["lat"], lon=hit["lon"], label=hit["name"])
        elif a["type"] in ("route", "flood_reach", "replay"):
            if hit.get("lat") is None:
                notes.append(f"{a['place']} has no position to start the flood from")
                continue
            b.update(lat=hit["lat"], lon=hit["lon"], label=hit["name"])
        elif a["type"] == "draw_storm":
            if not hit.get("bbox"):
                notes.append(f"the gazetteer gives no outline for {a['place']}, only a point: draw the storm on "
                             "the map")
                continue
            w, s_, e, n = hit["bbox"]
            b["polygon"] = [[w, s_], [e, s_], [e, n], [w, n]]
            b.setdefault("label", hit["name"])
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
        elif a["type"] == "paint":
            if hit.get("bbox"):
                b["bbox"] = hit["bbox"]
            elif hit.get("lat") is not None:
                # A point (a town): the gauges within a degree of it.
                b["bbox"] = [round(hit["lon"] - 1.0, 4), round(hit["lat"] - 1.0, 4), round(hit["lon"] + 1.0, 4),
                             round(hit["lat"] + 1.0, 4)]
            else:
                notes.append(f"{a['place']} has no position to paint around")
                continue
            b.setdefault("label", hit["name"])
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
    "where are floods getting bigger in Europe",
)


def _example_shots(t: date) -> list[tuple[str, list[dict[str, Any]]]]:
    """:data:`EXAMPLES` and their computed actions, each named place swapped for a stand-in.

    "Are the depths in my flood model realistic?" flew to the Nile on the device (#590 recheck): the model took
    the place from the example "trace the Nile to the sea". The examples now say <river>, <place> and <region>
    where they named places, in the request and the reply alike, so they teach the shape and give no place.
    """
    out = []
    for ex in EXAMPLES:
        request = ex
        acts = []
        raw = [a for a in parse_command(ex, t)["actions"]]
        rivers = {a.get("place") for a in raw if a["type"] == "highlight_river"}
        for a in raw:
            b = {k: v for k, v in a.items() if k not in ("river", "optional")}
            if b.get("region") in REGIONS:
                label = REGIONS[b["region"]]["label"]
                request = re.sub(re.escape(label), "<region>", request, flags=re.I)
                b = {**{k: v for k, v in b.items() if k not in ("region", "bbox", "label")}, "place": "<region>"}
            elif b.get("place"):
                stand_in = "<river>" if b["place"] in rivers else "<place>"
                request = re.sub(r"\b" + re.escape(b["place"]) + r"\b", stand_in, request, flags=re.I)
                b["place"] = stand_in
            acts.append(b)
        out.append((request, acts))
    return out


def model_prompt(context: str = "", today: str | date | None = None) -> dict[str, Any]:
    """The system prompt and the reply schema for a model that turns a request into map actions.

    The worked examples are :data:`EXAMPLES` run through :func:`parse_command`, so the prompt can never
    disagree with the grammar.
    """
    t = _today(today)
    layers = "; ".join(f"{k} ({v['label']})" for k, v in LAYERS.items())
    basemaps = "; ".join(f"{k} ({v['label']})" for k, v in BASEMAPS.items())
    classes = ", ".join(STATUS_IDS)
    shots = [f"Request: {ex}\nReply: {json.dumps({'actions': acts})}" for ex, acts in _example_shots(t)]
    paint_questions = ", ".join(f"{k} ({v['label']})" for k, v in PAINT_QUESTIONS.items())
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
        '- paint: answer one question at every gauge in a region and paint the map with it. "question" among '
        + paint_questions + '; "place" for the region, or none to paint the view. Fly there first. '
        "The numbers are computed by the map, never by you.",
        '- flood_reach: flood a river\'s valley from a reach (Flood it, a screening flood the map computes). "place" '
        '(or "lat" and "lon", or "at": "here" for the river selected on the map) and the flow: "return_period" '
        '(2 to 500 years; 100 when the request gives none), or "flow_cms" (m3/s, only when the reader wrote it), '
        'or "flow": forecast_peak, forecast (the 15-day forecast) or route (Route it\'s wave).',
        '- draw_levee: a levee on the flooded valley, then flood again (Flood it must be on the map). "line": '
        '"drawn" (the reader draws it), a list of [lon, lat] the reader gave, or "from_place" and "to_place". '
        '"crest" in m above mean sea level only when the reader gave a level; otherwise "freeboard_m", the height '
        'above the highest flood along the line (0.5 when none is said). The map computes the levee and the flood.',
        '- draw_breach: a gap in a levee, then flood again. "line" as for draw_levee; "invert" only when the reader '
        'gave the gap\'s floor level.',
        '- draw_storm: rain on an area, turned into runoff and routed down every river below it (Route it). '
        '"rain_mm" and "duration_h" (6 when none is said) as the reader wrote them; "runoff_coefficient" (the share '
        'of the rain that runs off; 0.3 when none is said) only when the reader gave it; the area: "area": "drawn" '
        '(the reader draws it, or the storm area already on the map; "this area", "here"), or "place", or "bbox". '
        'The map computes the runoff and the wave.',
        '- draw_reservoir: a reservoir on a river, a flood routed through it and on down the river (level-pool '
        'routing the map computes). "area": "drawn" (the reader draws the lake, or the one on the map) or "polygon"; '
        '"crest_m" (the spillway crest) and "top_m" (the top of the dam), m above the lowest ground, and "width_m" '
        '(the spillway) only when the reader gave them; the flood: "return_period" (100 when none is said), '
        '"peak_cms" (only when the reader wrote it), or "inflow": "wave" (the flood wave or storm on the map).',
        '- replay: replay a past flood as a story on the map (the rain, the gauge against the model, what the '
        'forecast said, the wave downstream, a screening flood). "place" (a river or a town; or "lat" and "lon", or '
        '"at": "here") and "month" (YYYY-MM). The map lists the floods it has there that month and takes the most '
        'reported; "pick" (2 to 5) only when the reader asked for another. The map writes the story and every number.',
        '- drop_well: drop a pumping well and draw its drawdown cone (Theis). "place" (or "lat" and "lon", or "at": '
        '"center" for the middle of the map); "rate_m3_day" only when the reader gave a rate (L/s times 86.4); "days", '
        '"transmissivity" (m2/day) and "storativity" only when given. The map computes the cone and every number.',
        '- change_landuse: a drawn area becomes a new land use; the map shows the runoff and peak flow before and '
        'after (SCS curve number, TR-55). "area": "drawn" (the reader draws it), or "polygon" when the reader '
        'gave one; "use": one of the listed uses ("impervious" with "impervious_pct"); "rain_mm" and "soil_group" '
        'only when the reader gave them.',
        "",
        "Never invent numbers, data or places that were not asked for. "
        "Use place names as the reader wrote them; the map looks them up. Give a bbox or a center only when "
        "the reader wrote its numbers. Name only places the request names: when it names none, move nothing.",
        "In the examples, <river>, <place> and <region> stand for the names a request gives; they are not "
        "places to go to.",
        'If the reader asks whether the model results on the map are right ("is anything wrong with my flood '
        'model?", "check my model"), reply {"actions": [], "control": "review"}. If they ask what the view '
        'shows ("what am I seeing?"), reply {"actions": [], "control": "explain"}.',
        'If the request is not about the map, reply {"actions": []}.',
        "",
        *shots,
    ]
    system = "\n".join(lines)
    if context:
        system += f"\n\nWhat the reader is looking at: {context}"
    return {"system": system, "schema": action_schema()}


_COORD_KEYS = ("bbox", "center", "lat", "lon")


def _region_in(folded: str) -> str | None:
    """The region a request names anywhere in it ("where is it driest in Africa" names Africa), longest first."""
    for alias, key in _REGION_ALIASES:
        if key == "world" and not re.search(r"world|globe", alias):
            continue   # "home", "earth" and "everywhere" are words, not a request for the world view
        if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", folded):
            return key
    return None


def _coords(action: dict[str, Any]) -> list[float]:
    vals: list[Any] = []
    for key in ("bbox", "center"):
        if isinstance(action.get(key), (list, tuple)):
            vals.extend(action[key])
    vals.extend(action[k] for k in ("lat", "lon") if action.get(k) is not None)
    return [float(v) for v in vals if _finite(v)]


def ground_places(actions: Any, text: str) -> tuple[Any, list[str]]:
    """Hold a model's boxes and points to the request's words (#590).

    "Where is it driest in Africa right now? zoom there" came back from the on-device model as a fly_to with a
    bbox of its own, [15, 20, 30, 40], and the map flew there as if it were the answer. A bbox, center or
    lat/lon is kept only when the words hold its numbers; otherwise, in this order, the action keeps the place
    the model also named (the gazetteer looks it up), takes the extent of a region the words name, or takes its
    label as a place when the words say it. Failing all three it is refused, with what is needed. A named place
    the words do not name (the prompt's example river, a selected gauge from the context) is refused too.
    Returns the actions and the lines to show for what was replaced or refused.
    """
    if isinstance(actions, dict):
        actions = actions.get("actions", [actions] if "type" in actions else [])
    if not isinstance(actions, list):
        return actions, []
    folded = _fold(text)
    said = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", folded)]
    region = _region_in(folded)
    out: list[Any] = []
    notes: list[str] = []
    for i, a in enumerate(actions[:8]):
        if not isinstance(a, dict):
            out.append(a)
            continue
        vals = _coords(a)
        if not vals or all(any(abs(abs(v) - n) <= 0.05 for n in said) for v in vals):
            out.append(a)
            continue
        bare = {k: v for k, v in a.items() if k not in _COORD_KEYS}
        kind = a.get("type")
        label = _text(a.get("label"), 80)
        what = "box" if a.get("bbox") is not None else "point"
        if _text(a.get("place"), 80):
            out.append(bare)
        elif region and kind in ("fly_to", "draw_area", "paint"):
            spec = REGIONS[region]
            out.append({**bare, "region": region, **({"bbox": spec["bbox"]} if spec["bbox"] else {}),
                        "label": spec["label"]})
            notes.append(f"action {i + 1}: the model's own {what}, replaced by the extent of {spec['label']}")
        elif label and len(label) >= 3 and _fold(label) in folded:
            # The label becomes the place for the gazetteer; without it, so a failed lookup still reads as one.
            out.append({**{k: v for k, v in bare.items() if k != "label"}, "place": label})
        else:
            notes.append(f"action {i + 1}: a {what} the model made up (the words name no place it could come "
                         "from); name a place or a region")
    # A place the words do not name: "are the depths in my flood model realistic?" flew to and lit up the Nile,
    # the example in the prompt (#590 recheck). Refused, as a made-up box is.
    kept: list[Any] = []
    for i, a in enumerate(out):
        place = _text(a.get("place"), 80) if isinstance(a, dict) else None
        if place and not _place_in_words(place, folded):
            if a.get("type") == "paint":
                # The question still stands: the view is painted, not a place the words never named (#657).
                kept.append({k: v for k, v in a.items() if k not in ("place", "label", "river")})
                notes.append(f"action {i + 1}: {place}, a place the words do not name; this view is painted")
                continue
            notes.append(f"action {i + 1}: {place}, a place the words do not name; name the place to go to")
            continue
        kept.append(a)
    return kept + actions[8:], notes


def _actions_list(actions: Any) -> list[Any] | None:
    if isinstance(actions, dict):
        actions = actions.get("actions", [actions] if "type" in actions else [])
    return actions if isinstance(actions, list) else None


def _no_free_text(actions: Any) -> Any:
    """A model's own words (a pin's title and note, its facts and source) never reach the map; a label stays
    until :func:`ground_places` has read it as a place, and :func:`_package_labels` decides on it after."""
    items = _actions_list(actions)
    if items is None:
        return actions
    return [{k: v for k, v in a.items() if k not in ("title", "text", "facts", "source")} if isinstance(a, dict)
            else a for a in items]


def _ground_layers(actions: Any, text: str) -> tuple[Any, list[str]]:
    """Drop a layer switch (and a status focus) the words do not ask for (#590 check3: "are there any numerical
    instabilities in this run?" turned Floods past and the river status on)."""
    items = _actions_list(actions)
    if items is None:
        return actions, []
    asked = set(_layers_named(text))
    kept, notes = [], []
    for i, a in enumerate(items):
        if isinstance(a, dict) and a.get("type") in ("set_layer", "focus_status"):
            layer = a.get("layer") if a.get("type") == "set_layer" else "status"
            if layer in LAYERS and layer not in asked:
                notes.append(f"action {i + 1}: {LAYERS[layer]['label']}, a layer the words do not ask for")
                continue
        kept.append(a)
    return kept, notes


def _package_labels(actions: Any, text: str) -> Any:
    """Titles and labels written here: a pin is titled with its place (or its point), an outline labelled with
    its region or place, and a model's label is kept only when it is the request's own words."""
    items = _actions_list(actions)
    if items is None:
        return actions
    folded = _fold(text)
    out = []
    for a in items:
        if not isinstance(a, dict):
            out.append(a)
            continue
        b = dict(a)
        label = _text(b.get("label"), 80)
        if label and not (b.get("region") in REGIONS and label == REGIONS[b["region"]]["label"]) \
                and _fold(label) not in folded:
            b.pop("label")
        if b.get("type") == "add_pin":
            where = _text(b.get("place"), 80) or (REGIONS[b["region"]]["label"] if b.get("region") in REGIONS
                                                  else None)
            if not where and _finite(b.get("lat")) and _finite(b.get("lon")):
                where = f"{float(b['lat']):g}, {float(b['lon']):g}"
            b["title"] = where or "Pin"
        out.append(b)
    return out


#: Words in a place's name that do not say which place it is.
_PLACE_GENERIC = frozenset({"the", "and", "river", "rivers", "sea", "lake", "city", "basin", "region", "county",
                            "state", "province", "district", "upon", "near", "of", "de", "la", "del", "der", "el"})


def _place_in_words(place: str, folded_words: str) -> bool:
    """Do the request's words (folded) name this place? "Rhine River" for "show the Rhine" does; "Nile" for
    "are the depths in my flood model realistic?" does not."""
    fp = _fold(place).strip()
    if not fp:
        return False
    if re.search(r"(?<![a-z0-9])" + re.escape(fp) + r"(?![a-z0-9])", folded_words):
        return True
    words = [w for w in re.findall(r"[a-z0-9]+", fp) if len(w) >= 3 and w not in _PLACE_GENERIC]
    return any(re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", folded_words) for w in words)


def parse_model_reply(reply: Any, today: str | date | None = None, text: str | None = None) -> dict[str, Any]:
    """A model's reply (text or parsed JSON) as checked actions: ``{"actions", "errors", "said", "control"}``.

    ``control`` is "review" or "explain" when the model answered with one of :data:`MODEL_CONTROLS` (the
    request was about the model result on the map, or the view), and the actions are then left out. With
    ``text`` (the request), a box or point the words do not hold goes through :func:`ground_places`, a layer
    the words do not ask for is not switched, and the model's own titles, notes and labels are dropped: a pin
    or an outline is titled with the place the words name (#590 round 3, the model picks, the package writes).
    """
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
    control = data.get("control") if isinstance(data, dict) else None
    if control in MODEL_CONTROLS:
        return {"actions": [], "errors": [], "said": [], "control": control}
    notes: list[str] = []
    if text is not None:
        data = _no_free_text(data)
        data, notes = ground_places(data, text)
        data, layer_notes = _ground_layers(data, text)
        data = _package_labels(data, text)
        notes += layer_notes
    checked = validate_actions(data, today)
    checked["errors"] = notes + checked["errors"]
    checked["said"] = [describe_action(a) for a in checked["actions"]]
    checked["control"] = None
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
    out = parse_model_reply(content, today, text=str(text))
    out["model"] = f"{model} via {provider}"
    return out
