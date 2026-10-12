"""Watch a river: what changed at the gauges, reaches and areas you follow since you last looked (#521).

One engine for the Explorer's "Since you were here" panel, the MCP tool and ``aquascope watch``:

* :func:`watch_digest` takes the watched items (gauges, river reaches, drawn areas) and when each was last seen,
  and says per item what changed: new observations and the latest value, today's status class against the one
  seen before (:func:`aquascope.nownext.flow_status`), a forecast that passes a threshold or a return period in
  the next 15 days (the GEOGLOWS forecast corrected to the gauge where it can be, :mod:`aquascope.nownext`), and
  flood events in the news nearby that started since (:func:`aquascope.context.events.flood_history`). Each item
  gets one line, and the digest one summary. Each item also returns ``seen``, the state to keep for next time.
* Thresholds are optional, per gauge: a value (``300``, in the record's unit) or a return period (``"10y"``, the
  10-year flow from the gauge's own annual maxima). Without one, forecasts are checked against the 2-year flow.
* :func:`atom_feed` writes an Atom 1.0 feed (RFC 4287); the daily feed job (:mod:`aquascope.archive.feeds`) uses
  it for every gauge with a live record.

Everything returns plain JSON. Forecasts are model output and say so. Nothing needs a key or an account: the
Explorer keeps the watch list in the browser.
"""

from __future__ import annotations

import logging
import math
import re
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_RETURN_PERIOD",
    "area_id",
    "atom_feed",
    "digest_summary",
    "load_issued",
    "load_snapshot",
    "parse_item",
    "parse_threshold",
    "threshold_flow",
    "watch_digest",
]

IS_EMSCRIPTEN = sys.platform == "emscripten"
#: A forecast is checked against this return period's flow when the item has no threshold of its own.
DEFAULT_RETURN_PERIOD = 2.0
#: With no last visit, the digest looks back this many days.
DEFAULT_SINCE_DAYS = 7
#: Flood events in the news count as nearby within this distance of a gauge or reach.
FLOOD_RADIUS_KM = 25.0
#: Forecast days looked at.
HORIZON_DAYS = 15
#: Status classes above and below normal (an area counts its gauges in the first).
HIGH_CLASSES = ("above", "much_above")

FLOOD_CREDIT = "Flood events in the news: Google Groundsource (CC BY 4.0)"
EXPLORER_URL = "https://rekin226-aquascope-explorer.static.hf.space/"


# ── small helpers ────────────────────────────────────────────────────────────


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _as_date(x: Any) -> date | None:
    if x is None or x == "":
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    try:
        return date.fromisoformat(str(x).strip()[:10])
    except ValueError as exc:
        raise ValueError(f"not a date: {x!r} (use YYYY-MM-DD)") from exc


def _num(x: Any, digits: int = 4) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def _fmt(x: float | None, unit: str | None = None) -> str:
    from aquascope.nownext import _fmt_q

    text = _fmt_q(x)
    return f"{text} {_unit(unit)}".strip() if unit else text


def _unit(unit: str | None) -> str:
    return {"m3/s": "m³/s", "m3 s-1": "m³/s", "ft3/s": "ft³/s"}.get(str(unit or ""), str(unit or ""))


def _day(d: date, today: date) -> str:
    from aquascope.nownext import _day_month

    return _day_month(d, year=d.year != today.year)


def _plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


# ── items and thresholds ─────────────────────────────────────────────────────

_RP = re.compile(r"^\s*(?:t|q)?\s*(\d+(?:\.\d+)?)\s*(?:-?\s*(?:y|yr|yrs|year|years)(?:\s*flow)?)\s*$", re.I)
_RP_PREFIX = re.compile(r"^\s*(?:t|q)\s*(\d+(?:\.\d+)?)\s*$", re.I)


def parse_threshold(spec: Any) -> dict[str, Any] | None:
    """A threshold from ``300`` (a value in the record's unit), ``"10y"``/``"10-year"``/``"T10"`` (a return
    period in years), or a dict with ``value`` or ``return_period``. None, ``""`` and ``"none"`` mean no threshold.
    """
    if spec is None or spec == "" or (isinstance(spec, str) and spec.strip().lower() in ("none", "off")):
        return None
    if isinstance(spec, dict):
        if spec.get("return_period") not in (None, ""):
            return _rp(spec["return_period"])
        if spec.get("value") not in (None, ""):
            return _value(spec["value"])
        return None
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return _value(spec)
    text = str(spec)
    m = _RP.match(text) or _RP_PREFIX.match(text)
    if m:
        return _rp(m.group(1))
    try:
        return _value(float(text.replace(",", "")))
    except ValueError as exc:
        raise ValueError(f"not a threshold: {spec!r} (give a value like 300, or a return period like 10y)") from exc


def _value(x: Any) -> dict[str, Any]:
    v = _num(x, 6)
    if v is None:
        raise ValueError(f"not a threshold value: {x!r}")
    return {"value": v}


def _rp(x: Any) -> dict[str, Any]:
    t = _num(x, 3)
    if t is None or not 1.01 <= t <= 1000:
        raise ValueError(f"a return period is between 1.01 and 1,000 years, not {x!r}")
    return {"return_period": t}


def _bbox(x: Any) -> list[float]:
    vals = [float(v) for v in (x.split(",") if isinstance(x, str) else x)]
    if len(vals) != 4:
        raise ValueError("an area is west,south,east,north")
    w, s, e, n = vals
    if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s < n <= 90):
        raise ValueError(f"not a box: {x!r} (west,south,east,north in degrees)")
    return [round(w, 4), round(s, 4), round(e, 4), round(n, 4)]


def _coord(v: float) -> str:
    text = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def area_id(bbox: list[float]) -> str:
    """An area's id, ``area:west,south,east,north`` at 4 decimals with trailing zeros dropped (the Explorer
    writes the same)."""
    return "area:" + ",".join(_coord(v) for v in bbox)


def parse_item(spec: Any) -> dict[str, Any]:
    """One watched item, normalised.

    A string is a gauge ``"source/station_id"``, a reach ``"river:<id>"`` (or ``"reach:<id>"``), or an area
    ``"area:west,south,east,north"``. A dict carries ``kind`` (``gauge``, ``reach`` or ``area``) with ``source``
    and ``station_id``, ``river_id``, or ``bbox``, and may add ``name``, ``lat``, ``lon``, ``river_id`` (a gauge's
    reach), ``threshold`` and, for a gauge, ``series`` (a record already at hand). Returns the dict with a
    canonical ``id``.
    """
    if isinstance(spec, str):
        text = spec.strip()
        low = text.lower()
        if low.startswith(("river:", "reach:")):
            spec = {"kind": "reach", "river_id": text.split(":", 1)[1]}
        elif low.startswith("area:"):
            spec = {"kind": "area", "bbox": text.split(":", 1)[1]}
        elif "/" in text:
            source, sid = text.split("/", 1)
            spec = {"kind": "gauge", "source": source, "station_id": sid}
        else:
            raise ValueError(f"not a watch id: {spec!r} (source/station_id, river:<id> or area:w,s,e,n)")
    if not isinstance(spec, dict):
        raise ValueError(f"not a watch item: {spec!r}")
    item = dict(spec)
    kind = str(item.get("kind") or "").lower()
    if not kind:
        if item.get("bbox") is not None:
            kind = "area"
        elif item.get("source") or "/" in str(item.get("id") or ""):
            kind = "gauge"
        elif item.get("river_id") is not None or str(item.get("id") or "").startswith(("river:", "reach:")):
            kind = "reach"
        elif str(item.get("id") or "").startswith("area:"):
            kind = "area"
    if kind == "gauge":
        if not item.get("source") and "/" in str(item.get("id") or ""):
            item["source"], item["station_id"] = str(item["id"]).split("/", 1)
        if not item.get("source") or not item.get("station_id"):
            raise ValueError("a watched gauge needs its source and station_id")
        item["source"], item["station_id"] = str(item["source"]), str(item["station_id"])
        item["id"] = f"{item['source']}/{item['station_id']}"
    elif kind == "reach":
        rid = item.get("river_id")
        if rid in (None, "") and ":" in str(item.get("id") or ""):
            rid = str(item["id"]).split(":", 1)[1]
        try:
            item["river_id"] = int(str(rid).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"not a river reach id: {rid!r}") from exc
        item["id"] = f"river:{item['river_id']}"
    elif kind == "area":
        box = item.get("bbox")
        if box is None and str(item.get("id") or "").startswith("area:"):
            box = str(item["id"]).split(":", 1)[1]
        item["bbox"] = _bbox(box)
        item["id"] = area_id(item["bbox"])
    else:
        raise ValueError(f"a watched item is a gauge, a reach or an area, not {kind or spec!r}")
    item["kind"] = kind
    item["threshold"] = parse_threshold(item.get("threshold"))
    for k in ("lat", "lon"):
        if item.get(k) is not None:
            item[k] = _num(item[k], 6)
    if kind == "gauge" and item.get("river_id") not in (None, ""):
        try:
            item["river_id"] = int(item["river_id"])
        except (TypeError, ValueError):
            item["river_id"] = None
    return item


def threshold_flow(threshold: dict[str, Any] | None, *, record: Any = None, table: dict[str, Any] | None = None,
                   unit: str | None = "m3/s", whose: str = "the gauge's") -> dict[str, Any]:
    """The value a threshold stands for, and how to say it.

    A value is itself. A return period is read from ``table`` (``{"return_periods": [...], "q": [...]}``, a fit
    at hand) when it holds that period, else fitted to ``record``'s annual maxima (Log-Pearson III, as
    :mod:`aquascope.nownext` does). With no threshold the :data:`DEFAULT_RETURN_PERIOD` flow is used.
    Returns ``value``, ``label``, ``return_period`` (or None), ``default`` and ``error`` when it cannot be said.
    """
    spec = threshold or {"return_period": DEFAULT_RETURN_PERIOD}
    default = threshold is None
    if spec.get("value") is not None:
        v = float(spec["value"])
        return {"value": v, "return_period": None, "default": False, "label": f"your threshold ({_fmt(v, unit)})"}
    t = float(spec["return_period"])
    q = None
    for rp, x in zip((table or {}).get("return_periods") or [], (table or {}).get("q") or []):
        if rp is not None and abs(float(rp) - t) < 1e-9 and x is not None:
            q = float(x)
    method = (table or {}).get("method")
    if q is None and record is not None:
        from aquascope.nownext import _thresholds

        fit = _thresholds(record, [t])
        if fit.get("error"):
            return {"value": None, "return_period": t, "default": default, "error": fit["error"],
                    "label": f"the {t:g}-year flow"}
        rps, qs = fit.get("return_periods") or [], fit.get("q") or []
        q = next((float(x) for rp, x in zip(rps, qs) if x is not None and abs(float(rp) - t) < 1e-9), None)
        method = fit.get("method")
    if q is None:
        return {"value": None, "return_period": t, "default": default, "label": f"the {t:g}-year flow",
                "error": f"No {t:g}-year flow is known here."}
    name = f"the {t:g}-year flow" if whose == "the gauge's" else f"{whose} {t:g}-year flow"
    return {"value": q, "return_period": t, "default": default, "method": method, "source": whose,
            "label": f"{name} ({_fmt(q, unit)})"}


# ── reading the Archive's daily files (CLI and MCP; the page reads them with DuckDB) ──


def load_snapshot(repo_id: str | None = None) -> list[dict[str, Any]]:
    """The daily status snapshot (``forecasts/status/latest.parquet``), or [] when it is not published or
    pyarrow is not installed."""
    try:
        from aquascope.archive import forecasts as fa

        return fa.read_published_rows(f"{fa.FOLDER}/status/latest.parquet", repo_id or fa.DEFAULT_REPO)
    except ImportError:
        return []
    except Exception as exc:  # noqa: BLE001 - the digest still answers from the records
        logger.info("status snapshot unreadable: %s", exc)
        return []


def load_issued(repo_id: str | None = None) -> list[dict[str, Any]]:
    """The newest issue of the forecast archive (``forecasts/issued/<date>.parquet``), or []."""
    try:
        from aquascope.archive import forecasts as fa

        repo = repo_id or fa.DEFAULT_REPO
        issues = [i for i in (fa.read_published_json(f"{fa.FOLDER}/manifest.json", repo).get("issues") or [])
                  if i.get("file")]
        if not issues:
            return []
        return fa.read_published_rows(issues[-1]["file"], repo)
    except ImportError:
        return []
    except Exception as exc:  # noqa: BLE001
        logger.info("forecast issue unreadable: %s", exc)
        return []


def _fill_gauges(items: list[dict[str, Any]]) -> None:
    """A gauge given by its id alone (the CLI, MCP) takes its name and position from the catalog, so its
    floods nearby and its live forecast can be checked."""
    want = [i for i in items if i["kind"] == "gauge" and (not i.get("name") or i.get("lat") is None
                                                          or i.get("lon") is None)]
    if not want:
        return
    try:
        from aquascope.archive.catalog import load_stations

        rows = {(str(r.get("source")), str(r.get("station_id"))): r for r in load_stations()}
    except Exception as exc:  # noqa: BLE001 - the digest still answers without names and positions
        logger.info("catalog unreadable: %s", exc)
        return
    for item in want:
        row = rows.get((item["source"], item["station_id"]))
        if not row:
            continue
        item["name"] = item.get("name") or row.get("name")
        if item.get("lat") is None or item.get("lon") is None:
            item["lat"], item["lon"] = _num(row.get("latitude"), 6), _num(row.get("longitude"), 6)


def _positions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Snapshot rows with lat and lon from the catalog, for an area's count."""
    if not rows or all(r.get("lat") is not None for r in rows):
        return rows
    try:
        from aquascope.archive.catalog import load_stations

        pos = {(str(r["source"]), str(r["station_id"])): (r.get("latitude"), r.get("longitude"))
               for r in load_stations()}
    except Exception as exc:  # noqa: BLE001 - an area without counts still has its floods
        logger.info("catalog unreadable: %s", exc)
        return rows
    out = []
    for r in rows:
        la, lo = pos.get((str(r.get("source")), str(r.get("station_id"))), (None, None))
        out.append({**r, "lat": r.get("lat", la), "lon": r.get("lon", lo)})
    return out


# ── the pieces of one item's digest ──────────────────────────────────────────


def _new_floods(lat: float, lon: float, since: date, *, radius_km: float = FLOOD_RADIUS_KM,
                bbox: list[float] | None = None) -> dict[str, Any]:
    from aquascope.context import events

    try:
        if bbox is not None:
            fh = events.flood_history_area(*bbox, limit=200, max_points=0)
        else:
            fh = events.flood_history(lat, lon, radius_km=radius_km, limit=200)
    except Exception as exc:  # noqa: BLE001 - the flood check is one part of the digest
        return {"available": False, "note": f"Flood history could not be read ({exc})."}
    news = (fh or {}).get("news") or {}
    if not news.get("available"):
        return {"available": False, "note": news.get("note") or (fh or {}).get("error") or
                "Flood history is not available here yet."}
    new = [e for e in news.get("recent") or [] if (_safe_date(e.get("start")) or date.min) > since]
    out: dict[str, Any] = {"available": True, "n_new": len(new), "new": new[:10], "latest": news.get("latest"),
                           "radius_km": None if bbox is not None else radius_km}
    return out


def _safe_date(x: Any) -> date | None:
    try:
        return _as_date(x)
    except ValueError:
        return None


def _flood_text(fl: dict[str, Any], today: date, where: str) -> str:
    n = fl.get("n_new") or 0
    if not n:
        return ""
    latest = max((_safe_date(e.get("start")) for e in fl.get("new") or [] if _safe_date(e.get("start"))),
                 default=None)
    tail = f", latest {_day(latest, today)}" if latest else ""
    return f"{_plural(n, 'new flood event')} in the news {where}{tail}."


def _check_forecast(dates: list[Any], values: list[Any], thr: dict[str, Any], today: date,
                    horizon: int = HORIZON_DAYS) -> dict[str, Any]:
    """The peak of the next ``horizon`` days and whether it passes ``thr``."""
    end = today + timedelta(days=int(horizon))
    pairs = []
    for d, v in zip(dates, values):
        dd, vv = _safe_date(d), _num(v)
        if dd is not None and vv is not None and today <= dd < end:
            pairs.append((dd, vv))
    if not pairs:
        return {"peak": None}
    peak_d, peak = max(pairs, key=lambda p: p[1])
    out: dict[str, Any] = {"peak": peak, "peak_date": peak_d.isoformat(), "days": len(pairs)}
    q = thr.get("value")
    if q is not None:
        above = [d for d, v in pairs if v >= q]
        out.update({"exceeds": bool(above), "first_above": above[0].isoformat() if above else None,
                    "days_above": len(above)})
    return out


def _forecast_text(fc: dict[str, Any], today: date, unit: str | None) -> str:
    if not fc or fc.get("peak") is None:
        return ""
    thr = fc.get("threshold") or {}
    head = f"The forecast (modelled{', ' + fc['scale'] if fc.get('scale') else ''}) peaks at " \
           f"{_fmt(fc['peak'], unit)} on {_day(_as_date(fc['peak_date']), today)}"
    if thr.get("value") is None:
        return head + "."
    if fc.get("exceeds"):
        return f"{head}, above {thr['label']}."
    return f"{head}, under {thr['label']}."


def _status_text(status: dict[str, Any], today: date) -> str:
    if not status or not status.get("class"):
        return ""
    now_label = status["label"]
    prev = status.get("previous")
    word = "modelled flow" if status.get("modelled") else ""
    when = _safe_date(status.get("date"))
    stale = when is not None and (today - when).days > 3
    if stale:
        return f"The latest status, on {_day(when, today)}, was {now_label}."
    subject = f"Now {word + ' ' if word else ''}{now_label}"
    if prev and prev.get("class") and prev["class"] != status["class"]:
        pd_ = _safe_date(prev.get("date"))
        on = f" on {_day(pd_, today)}" if pd_ else ""
        return f"{subject} (was {prev['label']}{on})."
    if prev and prev.get("class") == status["class"]:
        return f"Still {word + ' ' if word else ''}{now_label}."
    return f"{subject}."


def _class_label(cls: str | None) -> str | None:
    from aquascope.nownext import STATUS_CLASSES

    return next((c["label"] for c in STATUS_CLASSES if c["id"] == cls), None)


def _issued_for(rows: list[dict[str, Any]], key: tuple[str, str]) -> list[dict[str, Any]]:
    mine = [r for r in rows if (str(r.get("source")), str(r.get("station_id"))) == key
            and r.get("model") == "geoglows"]
    if not mine:
        return []
    last = max(str(r.get("issue_date")) for r in mine)
    return sorted((r for r in mine if str(r.get("issue_date")) == last), key=lambda r: str(r.get("valid_date")))


def _gauge(item: dict[str, Any], since: date, prev: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from aquascope import nownext

    today = ctx["today"]
    source, sid = item["source"], item["station_id"]
    out: dict[str, Any] = {"notes": []}
    series, variable, unit = item.get("series"), item.get("variable"), item.get("unit")
    if series is None and ctx["fetch"]:
        from aquascope.explore import fetch_series

        try:
            got = fetch_series(source, sid, variable=variable)
            series, variable, unit = got.get("series"), got.get("variable") or variable, got.get("unit") or unit
        except Exception as exc:  # noqa: BLE001 - the snapshot may still say something
            out["notes"].append(f"The record could not be read ({exc}).")
    variable = variable or "discharge"
    if not unit:
        from aquascope.archive.observations import ARCHIVE_UNITS

        unit = ARCHIVE_UNITS.get(variable)
    s = nownext._as_daily(series)
    if not s.empty and ctx["refresh"]:
        s, said = nownext.top_up(s, source, sid, variable=variable, today=today)
        if said:
            out["notes"].append(said)
    out.update(variable=variable, unit=unit)
    snap = ctx["snapshot"].get((source, sid))

    # new observations and the latest value
    latest = None
    if not s.empty:
        new = s[s.index.normalize() > pd.Timestamp(since)]
        latest = {"date": s.index.max().date().isoformat(), "value": _num(s.iloc[-1])}
        out["new_days"] = int(len(new))
    elif snap and snap.get("value_date"):
        latest = {"date": str(snap["value_date"])[:10], "value": _num(snap.get("value"))}
        out["new_days"] = None
        out["notes"].append("From the daily status snapshot; the record itself could not be read.")
    else:
        out["new_days"] = None
    out["latest"] = latest

    # today's status, and the one seen before
    status = None
    if snap and snap.get("class") and (not latest or str(snap.get("value_date"))[:10] >= latest["date"]):
        status = {"class": snap["class"], "label": _class_label(snap["class"]),
                  "percentile": _num(snap.get("percentile"), 1), "date": str(snap.get("value_date"))[:10],
                  "from": "the daily status snapshot"}
    elif not s.empty:
        st = nownext.flow_status(s, variable=variable, unit=unit, today=today)
        if st.get("class"):
            status = {"class": st["class"], "label": st["label"], "percentile": st["percentile"], "date": st["date"],
                      "sentence": st.get("sentence")}
        elif st.get("error"):
            out["notes"].append(st["error"])
    if status:
        previous = None
        if prev.get("class"):
            previous = {"class": prev["class"], "label": _class_label(prev["class"]), "date": prev.get("date")}
        elif not s.empty:
            back = s[s.index.normalize() <= pd.Timestamp(since)]
            if not back.empty and (since - back.index.max().date()).days <= 3:
                pst = nownext.flow_status(s, back.index.max().date(), variable=variable, today=today)
                if pst.get("class"):
                    previous = {"class": pst["class"], "label": pst["label"], "date": pst["date"]}
        status["previous"] = previous
        status["changed"] = bool(previous and previous["class"] != status["class"])
    out["status"] = status

    # the threshold, against the latest value and the forecast
    thr_table = None
    rows = _issued_for(ctx["issued"], (source, sid)) if ctx["forecast"] in ("auto", "archive") else []
    if rows:
        thr_table = {"return_periods": [], "q": []}
        from aquascope.archive.forecasts import GAUGE_RETURN_PERIODS

        for t in GAUGE_RETURN_PERIODS:
            if rows[0].get(f"gauge_q{t}") is not None:
                thr_table["return_periods"].append(t)
                thr_table["q"].append(rows[0][f"gauge_q{t}"])
    thr: dict[str, Any] = {}
    if item.get("threshold") or (ctx["forecast"] != "off" and variable == "discharge"):
        thr = threshold_flow(item.get("threshold"), record=None if s.empty else s, table=thr_table, unit=unit)
    out["threshold"] = thr or None
    if item.get("threshold") and thr.get("value") is not None and latest and latest.get("value") is not None:
        out["latest_above"] = latest["value"] >= thr["value"]

    fc = None
    if variable != "discharge":
        if ctx["forecast"] != "off":
            out["notes"].append("The forecast is for flow; this gauge records "
                                f"{variable.replace('_', ' ')}, so no forecast is checked.")
    elif rows:
        corrected = all(r.get("mean_c") is not None for r in rows)
        vals = [r.get("mean_c") if corrected else r.get("mean") for r in rows]
        if not corrected and thr.get("return_period") is not None:
            thr = {"value": None, "label": thr.get("label"), "error": "The archived forecast is not corrected to "
                   "this gauge, so it is not compared with the gauge's flows."}
        fc = {**_check_forecast([r.get("valid_date") for r in rows], vals, thr, today, ctx["horizon"]),
              "from": "the forecast archive", "issue_date": str(rows[0].get("issue_date")),
              "scale": "corrected to the gauge" if corrected else "not corrected to the gauge", "threshold": thr}
    elif ctx["forecast"] in ("auto", "live"):
        lat, lon = item.get("lat"), item.get("lon")
        if (lat is None or lon is None) and item.get("river_id") is None and not IS_EMSCRIPTEN:
            row = nownext._station_row(source, sid)
            if row:
                lat, lon = row.get("latitude"), row.get("longitude")
        if lat is None and item.get("river_id") is None:
            out["notes"].append("The gauge's position is not known, so no forecast is checked.")
        else:
            try:
                live = nownext.forecast(lat, lon, river_id=item.get("river_id"), obs=None if s.empty else s,
                                        glofas=False, days=ctx["horizon"],
                                        match_mean_flow=None if s.empty else float(s.mean()))
                fc = _from_live(live, item.get("threshold"), s, unit, today, ctx["horizon"], gauge=True)
            except Exception as exc:  # noqa: BLE001 - the rest of the digest stands
                out["notes"].append(f"The forecast did not answer ({exc}).")
    out["forecast"] = fc

    lat, lon = item.get("lat"), item.get("lon")
    if ctx["floods"] and lat is not None and lon is not None:
        out["floods"] = _new_floods(lat, lon, since, radius_km=ctx["radius_km"])
    return out


def _from_live(live: dict[str, Any], threshold: dict[str, Any] | None, record: Any, unit: str | None, today: date,
               horizon: int, *, gauge: bool) -> dict[str, Any] | None:
    """The forecast part of a digest from :func:`aquascope.nownext.forecast`'s answer."""
    corrected = ((live.get("correction") or {}).get("forecast") or {}) if gauge else {}
    raw = live.get("geoglows") or {}
    if corrected.get("mean"):
        part, scale = corrected, "corrected to the gauge"
        thr = threshold_flow(threshold, record=record, table=live.get("gauge_thresholds"), unit=unit)
    elif raw.get("mean"):
        part, scale = raw, ("the reach, not corrected to the gauge" if gauge else "")
        if threshold and threshold.get("value") is not None and gauge:
            thr = threshold_flow(threshold, unit=unit)
        else:
            thr = threshold_flow(threshold, table=live.get("thresholds"), unit="m3/s", whose="the reach's simulated")
    else:
        return {"peak": None, "error": (raw.get("error") or live.get("sentence") or "No forecast answered."),
                "from": "GEOGLOWS"}
    res = _check_forecast(part.get("date") or [], part.get("mean") or [], thr, today, horizon)
    return {**res, "from": "GEOGLOWS v2, asked now", "scale": scale, "threshold": thr,
            "initialized": raw.get("initialized"), "river_id": live.get("river_id")}


def _reach(item: dict[str, Any], since: date, prev: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    from aquascope import nownext

    today = ctx["today"]
    out: dict[str, Any] = {"notes": [], "new_days": None, "latest": None, "unit": "m3/s"}
    status = None
    if ctx["forecast"] in ("auto", "live"):
        try:
            live = nownext.forecast(item.get("lat"), item.get("lon"), river_id=item["river_id"], glofas=False,
                                    snap=False, days=ctx["horizon"])
            out["forecast"] = _from_live(live, item.get("threshold"), None, "m3/s", today, ctx["horizon"],
                                         gauge=False)
            st = live.get("status") or {}
            if st.get("class"):
                status = {"class": st["class"], "label": st["label"], "percentile": st.get("percentile"),
                          "date": st.get("date"), "modelled": True}
        except Exception as exc:  # noqa: BLE001
            out["notes"].append(f"GEOGLOWS did not answer ({exc}).")
            out["forecast"] = None
    else:
        out["forecast"] = None
    if status:
        previous = None
        if prev.get("class"):
            previous = {"class": prev["class"], "label": _class_label(prev["class"]), "date": prev.get("date")}
        status["previous"] = previous
        status["changed"] = bool(previous and previous["class"] != status["class"])
    out["status"] = status
    out["threshold"] = (out.get("forecast") or {}).get("threshold")
    if ctx["floods"] and item.get("lat") is not None and item.get("lon") is not None:
        out["floods"] = _new_floods(item["lat"], item["lon"], since, radius_km=ctx["radius_km"])
    return out


def _in_box(lat: Any, lon: Any, bbox: list[float]) -> bool:
    la, lo = _num(lat), _num(lon)
    if la is None or lo is None:
        return False
    w, s, e, n = bbox
    return s <= la <= n and ((w <= lo <= e) if w <= e else (lo >= w or lo <= e))


def _area(item: dict[str, Any], since: date, prev: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    from aquascope.nownext import STATUS_CLASSES

    out: dict[str, Any] = {"notes": [], "new_days": None, "latest": None, "status": None, "forecast": None}
    rows = [r for r in ctx["snapshot_rows"] if _in_box(r.get("lat"), r.get("lon"), item["bbox"])]
    if rows:
        counts = {c["id"]: sum(1 for r in rows if r.get("class") == c["id"]) for c in STATUS_CLASSES}
        high = sum(counts[c] for c in HIGH_CLASSES)
        out["gauges"] = {"n": len(rows), "counts": counts, "high": high, "previous_high": prev.get("high"),
                         "rose": prev.get("high") is not None and high > int(prev["high"])}
    elif ctx["snapshot_rows"]:
        out["notes"].append("No gauge in this area has a live status today.")
    if ctx["floods"]:
        out["floods"] = _new_floods(0.0, 0.0, since, bbox=item["bbox"])
    return out


def _line(kind: str, part: dict[str, Any], since: date, today: date) -> str:
    bits: list[str] = []
    unit = part.get("unit")
    latest = part.get("latest")
    if kind == "gauge":
        n = part.get("new_days")
        if latest and latest.get("date"):
            when = _day(_as_date(latest["date"]), today)
            if n:
                bits.append(f"{_plural(n, 'new day')} of data since {_day(since, today)}, latest "
                            f"{_fmt(latest.get('value'), unit)} on {when}.")
            elif n == 0:
                bits.append(f"No new data since {_day(since, today)}; the latest value is from {when}.")
            else:
                bits.append(f"Latest {_fmt(latest.get('value'), unit)} on {when}.")
        else:
            bits.append("No record could be read.")
        if part.get("latest_above"):
            bits.append(f"The latest value is above {part['threshold']['label']}.")
    if kind == "area":
        g = part.get("gauges")
        if g:
            bits.append(f"{g['high']} of {_plural(g['n'], 'gauge')} with a live record "
                        f"{'is' if g['high'] == 1 else 'are'} above normal today"
                        + (f" (was {g['previous_high']})." if g.get("rose") else "."))
    st = _status_text(part.get("status") or {}, today)
    if st:
        bits.append(st)
    fc = _forecast_text(part.get("forecast") or {}, today, "m3/s" if kind != "gauge" else (unit or "m3/s"))
    failed = ""
    if fc:
        bits.append(fc)
    elif (part.get("forecast") or {}).get("error"):
        failed = "The forecast did not answer this time."
    fl = part.get("floods") or {}
    where = "in this area" if kind == "area" else f"within {fl.get('radius_km') or FLOOD_RADIUS_KM:g} km"
    ft = _flood_text(fl, today, where)
    if ft:
        bits.append(ft)
    if kind != "gauge" and not bits:
        bits.append(f"Nothing new since {_day(since, today)}.")
    if failed:
        bits.append(failed)
    return " ".join(bits)


def _alerts(kind: str, part: dict[str, Any]) -> list[str]:
    out = []
    if part.get("latest_above"):
        out.append("latest value above the threshold")
    fc = part.get("forecast") or {}
    if fc.get("exceeds"):
        out.append(f"forecast above {(fc.get('threshold') or {}).get('label', 'the threshold')}")
    if (part.get("status") or {}).get("changed"):
        out.append(f"now {part['status']['label']}")
    if (part.get("floods") or {}).get("n_new"):
        out.append("new flood event nearby")
    if (part.get("gauges") or {}).get("rose"):
        out.append("more gauges above normal")
    return out


# ── the digest ───────────────────────────────────────────────────────────────


def _seen_for(item_id: str, last_seen: Any, default: date) -> tuple[date, dict[str, Any]]:
    """(the date the item was last seen, the state kept then) from ``last_seen``."""
    if isinstance(last_seen, dict):
        entry = last_seen.get(item_id)
        if isinstance(entry, dict):
            return _as_date(entry.get("date")) or default, entry
        if entry:
            return _as_date(entry) or default, {}
        return default, {}
    return default, {}


def watch_digest(items: list[Any], last_seen: Any = None, *, today: Any = None,
                 snapshot: list[dict[str, Any]] | None = None, issued: list[dict[str, Any]] | None = None,
                 fetch: bool = True, refresh: bool = True, forecast: str = "auto", floods: bool = True,
                 flood_radius_km: float = FLOOD_RADIUS_KM, horizon_days: int = HORIZON_DAYS,
                 archive: bool = True) -> dict[str, Any]:
    """What changed at each watched item since it was last seen.

    ``items``: gauges, reaches and areas as :func:`parse_item` takes them (``"usgs/USGS-01646500"``,
    ``"river:230260670"``, ``"area:-77.5,38.1,-76.8,39.0"`` or dicts, which may carry ``threshold``).
    ``last_seen``: one date for every item (``"2026-10-01"``), or a dict from item id to a date or to the
    ``seen`` state an earlier digest returned (``{"date", "class", "value_date", "high"}``). With nothing, the
    digest looks back :data:`DEFAULT_SINCE_DAYS` days.

    Per gauge: new observations since then (``new_days``) and the ``latest`` value (the Archive's copy first,
    topped up with the agency's newest days when ``refresh``); today's ``status`` class with the ``previous``
    one (kept in ``last_seen``, or placed on the last-seen day from the record) and whether it ``changed``; the
    ``forecast`` peak in the next ``horizon_days`` days against the ``threshold`` (the item's, else the 2-year
    flow), from the forecast archive's newest issue (``issued`` rows, corrected to the gauge) or GEOGLOWS asked
    now; and ``floods``, flood events in the news within ``flood_radius_km`` that started since. A reach gets
    its modelled status and forecast against its simulated return periods; an area its new flood events and how
    many of its gauges in the daily ``snapshot`` are above normal.

    ``forecast`` is ``"auto"`` (the archive where it covers the gauge, else GEOGLOWS now), ``"archive"``,
    ``"live"`` or ``"off"``. ``snapshot`` and ``issued`` are the Archive's daily rows when the caller has them
    (the Explorer reads them with DuckDB); with ``archive`` they are read here when not given (needs pyarrow).

    Returns ``items`` (each with ``line``, ``changed``, ``alerts`` and ``seen``, the state to keep for next
    time), ``n_changed``, ``n_alerts``, ``summary``, ``since``, ``today`` and ``attribution``.
    """
    today_d = _as_date(today) or _today()
    if isinstance(last_seen, (str, date, datetime)):
        default_since = _as_date(last_seen) or today_d
    else:
        default_since = today_d - timedelta(days=DEFAULT_SINCE_DAYS)
    forecast = str(forecast or "auto").lower()
    if forecast not in ("auto", "archive", "live", "off"):
        raise ValueError("forecast is auto, archive, live or off")
    parsed, errors = [], []
    for spec in items or []:
        try:
            parsed.append(parse_item(spec))
        except ValueError as exc:
            errors.append({"item": spec if isinstance(spec, str) else str(spec)[:80], "error": str(exc)})
    kinds = {p["kind"] for p in parsed}
    if archive and not IS_EMSCRIPTEN:
        _fill_gauges(parsed)
    if snapshot is None and archive and not IS_EMSCRIPTEN and kinds & {"gauge", "area"}:
        snapshot = load_snapshot()
    if issued is None and archive and not IS_EMSCRIPTEN and "gauge" in kinds and forecast in ("auto", "archive"):
        issued = load_issued()
    snap_rows = list(snapshot or [])
    if "area" in kinds and snap_rows:
        snap_rows = _positions(snap_rows)
    ctx = {"today": today_d, "fetch": fetch, "refresh": refresh, "forecast": forecast, "floods": floods,
           "radius_km": float(flood_radius_km), "horizon": max(1, min(int(horizon_days), 15)),
           "snapshot": {(str(r.get("source")), str(r.get("station_id"))): r for r in snap_rows},
           "snapshot_rows": snap_rows, "issued": list(issued or [])}

    results = []
    for item in parsed:
        since, prev = _seen_for(item["id"], last_seen, default_since)
        base = {"id": item["id"], "kind": item["kind"], "name": item.get("name") or _default_name(item),
                "since": since.isoformat()}
        try:
            fn = {"gauge": _gauge, "reach": _reach, "area": _area}[item["kind"]]
            part = fn(item, since, prev, ctx)
        except Exception as exc:  # noqa: BLE001 - one item failing must not hide the others
            logger.info("watch digest failed for %s: %s", item["id"], exc)
            results.append({**base, "error": str(exc), "line": f"Could not be checked ({exc}).", "changed": False,
                            "alerts": [], "seen": {**prev, "date": prev.get("date") or since.isoformat()}})
            continue
        alerts = _alerts(item["kind"], part)
        changed = bool(alerts or part.get("new_days"))
        seen = {"date": today_d.isoformat()}
        if (part.get("status") or {}).get("class"):
            seen["class"] = part["status"]["class"]
        if part.get("latest"):
            seen["value_date"] = part["latest"].get("date")
        if part.get("gauges"):
            seen["high"] = part["gauges"]["high"]
        keep = ("status", "forecast", "new_days", "latest")
        res = {**base, **{k: v for k, v in part.items() if v is not None or k in keep},
               "threshold_set": item.get("threshold"), "alerts": alerts, "changed": changed,
               "line": _line(item["kind"], part, since, today_d), "seen": seen}
        results.append(res)
    since_all = min((_as_date(r["since"]) for r in results), default=default_since)
    out = {"today": today_d.isoformat(), "since": since_all.isoformat(), "items": results,
           "n_items": len(results), "n_changed": sum(1 for r in results if r.get("changed")),
           "n_alerts": sum(1 for r in results if r.get("alerts")),
           "attribution": ["Observations: each gauge's agency, through the AquaScope Archive",
                           "Forecasts: GEOGLOWS v2 (GEOGloWS ECMWF Streamflow Service), CC BY 4.0, modelled",
                           FLOOD_CREDIT]}
    notes = sorted({(r.get("floods") or {}).get("note") for r in results} - {None})
    if notes:
        out["notes"] = notes
    if errors:
        out["errors"] = errors
    out["summary"] = digest_summary(results, since_all, today=today_d)
    return out


def _default_name(item: dict[str, Any]) -> str:
    if item["kind"] == "gauge":
        return item["id"]
    if item["kind"] == "reach":
        return f"River reach {item['river_id']}"
    w, s, e, n = item["bbox"]
    return f"Area {s:.2f} to {n:.2f} °N, {w:.2f} to {e:.2f} °E"


def digest_summary(items: list[dict[str, Any]], since: Any = None, *, today: Any = None) -> str:
    """One line for a whole digest: how many watched places changed, and the alerts by name."""
    today_d = _as_date(today) or _today()
    if not items:
        return "Nothing is watched yet."
    since_d = _as_date(since) or min((_as_date(i.get("since")) for i in items if i.get("since")), default=today_d)
    n = len(items)
    changed = [i for i in items if i.get("changed")]
    alerted = [i for i in items if i.get("alerts")]
    when = _day(since_d, today_d)
    if not changed:
        return f"Nothing changed at your {_plural(n, 'watched place')} since {when}."
    head = f"Since {when}: {len(changed)} of {_plural(n, 'watched place')} changed"
    if not alerted:
        return head + "."
    named = "; ".join(f"{i.get('name') or i['id']}: {', '.join(i['alerts'])}" for i in alerted[:3])
    more = f"; and {len(alerted) - 3} more" if len(alerted) > 3 else ""
    return f"{head}, {len(alerted)} with something to look at ({named}{more})."


# ── Atom ─────────────────────────────────────────────────────────────────────

ATOM_NS = "http://www.w3.org/2005/Atom"


def _rfc3339(x: Any) -> str:
    if isinstance(x, datetime):
        dt = x if x.tzinfo else x.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = str(x)
    if len(text) == 10:
        return f"{text}T00:00:00Z"
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return _rfc3339(dt)


def atom_feed(*, feed_id: str, title: str, updated: Any, entries: list[dict[str, Any]],
              self_url: str | None = None, link: str | None = None, subtitle: str | None = None,
              author: str = "AquaScope", rights: str | None = None) -> str:
    """An Atom 1.0 feed (RFC 4287) as UTF-8 text.

    ``entries`` hold ``id``, ``title``, ``updated`` and optionally ``summary`` and ``link``; they are written
    newest first. ``feed_id`` and entry ids should be stable URIs (a ``tag:`` URI is fine).
    """
    import xml.etree.ElementTree as ET

    ET.register_namespace("", ATOM_NS)

    def el(parent: Any, tag: str, text: str | None = None, **attrs: str) -> Any:
        e = ET.SubElement(parent, f"{{{ATOM_NS}}}{tag}", attrs)
        if text is not None:
            e.text = text
        return e

    feed = ET.Element(f"{{{ATOM_NS}}}feed")
    el(feed, "id", feed_id)
    el(feed, "title", title)
    if subtitle:
        el(feed, "subtitle", subtitle)
    el(feed, "updated", _rfc3339(updated))
    who = el(feed, "author")
    el(who, "name", author)
    el(who, "uri", "https://github.com/Rekin226/aquascope")
    if self_url:
        el(feed, "link", rel="self", href=self_url, type="application/atom+xml")
    if link:
        el(feed, "link", rel="alternate", href=link, type="text/html")
    el(feed, "generator", "aquascope", uri="https://github.com/Rekin226/aquascope")
    if rights:
        el(feed, "rights", rights)
    for e in sorted(entries, key=lambda x: _rfc3339(x["updated"]), reverse=True):
        entry = el(feed, "entry")
        el(entry, "id", str(e["id"]))
        el(entry, "title", str(e["title"]))
        el(entry, "updated", _rfc3339(e["updated"]))
        if e.get("link"):
            el(entry, "link", rel="alternate", href=str(e["link"]), type="text/html")
        if e.get("summary"):
            el(entry, "summary", str(e["summary"]), type="text")
    body = ET.tostring(feed, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + body + "\n"
