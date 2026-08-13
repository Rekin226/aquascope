"""Live fetch sweep across all 27 dashboard sources, mirroring the Collect page defaults."""
import json, signal, time, traceback
from datetime import date, timedelta

import aquascope.collectors as c
from aquascope.dashboard.views.collect import _FACTORIES, SOURCES

today = date.today()
d7, d30 = str(today - timedelta(days=7)), str(today - timedelta(days=30))

# (fetch_kwargs, per-source timeout seconds, note)
PARAMS = {
    "usgs":               ({"days": 3, "bbox": "-77.6,38.7,-76.9,39.1", "max_items": 500}, 240, ""),
    "grdc":               ({"source_type": "in_situ"}, 600, "DARUS download"),
    "openmeteo":          ({"latitude": 25.0, "longitude": 121.5, "start_date": d7, "end_date": str(today)}, 180, ""),
    "sdg6":               ({}, 240, "defaults"),
    "gemstat":            ({"country": "Germany", "max_records": 500}, 300, ""),
    "aquastat":           ({"country_code": "EGY", "start_year": 2000, "end_year": 2023}, 240, ""),
    "wapor":              ({"variable": "RET", "start_date": "2026-04-01", "end_date": "2026-05-01",
                            "bbox": (30.5, 29.8, 31.1, 30.2)}, 400, ""),
    "copernicus":         (None, 0, "NEEDS_KEY (Copernicus CDS)"),
    "wqp":                ({"state_code": "US:11"}, 300, "DC = small payload"),
    "hubeau_hydrometrie": ({"max_items": 500}, 240, ""),
    "eu_wfd":             ({"country": "DE", "water_body_type": "river", "year": 2018}, 300, ""),
    "taiwan_moenv":       (None, 0, "NEEDS_KEY (Taiwan MOENV)"),
    "taiwan_wra_level":   ({}, 240, ""),
    "taiwan_wra_reservoir": ({}, 240, ""),
    "taiwan_wra_fhy":     ({}, 240, ""),
    "taiwan_wra_iot":     ({}, 240, ""),
    "taiwan_datagov":     ({"limit": 200}, 240, ""),
    "taiwan_civil_iot":   ({}, 300, ""),
    "japan_mlit":         ({"prefecture": "Tokyo", "parameter": "water_level"}, 300, ""),
    "korea_wamis":        ({"basin": "Han", "parameter": "water_level"}, 300, ""),
    "india_wris":         ({"state_name": "Assam", "district_name": "Kamrup", "agency_name": "CWC",
                            "startdate": d30, "enddate": str(today)}, 300, ""),
    "noaa_nwps":          ({"lid": "ANAW1"}, 240, ""),
    "ireland_opw":        ({"max_stations": 3}, 300, ""),
    "pegelonline":        ({"days": 2}, 300, ""),
    "camels_cl":          ({"station_ids": ["1001001"], "start": "2000-01-01", "end": "2000-12-31"}, 900, "~275MB first download"),
    "camels_br":          ({"station_ids": ["10500000"], "start": "2000-01-01", "end": "2000-03-31"}, 600, "~62MB, may be cached"),
    "uk_ea":              ({"observed_property": "waterFlow", "collection": "15min", "days": 2}, 300, ""),
}

class Timeout(Exception): pass
def _alarm(sig, frame): raise Timeout()
signal.signal(signal.SIGALRM, _alarm)

results = {}
missing = set(SOURCES) - set(PARAMS)
if missing:
    print("PARAMS missing sources:", missing)

for key in SOURCES:
    fetch, tmo, note = PARAMS[key]
    if fetch is None:
        results[key] = {"status": "NEEDS_KEY", "note": note, "secs": 0}
        print(f"{key:22} NEEDS_KEY")
    else:
        t0 = time.time()
        try:
            signal.alarm(tmo)
            col = _FACTORIES[key](None, {}, c)
            recs = col.collect(**fetch)
            signal.alarm(0)
            n = len(recs)
            rtype = type(recs[0]).__name__ if n else "-"
            status = "OK" if n > 0 else "EMPTY"
            results[key] = {"status": status, "records": n, "rtype": rtype,
                            "secs": round(time.time() - t0, 1), "note": note}
            print(f"{key:22} {status:6} {n:>6} {rtype}  {results[key]['secs']}s")
        except Timeout:
            results[key] = {"status": "TIMEOUT", "secs": tmo, "note": note}
            print(f"{key:22} TIMEOUT after {tmo}s")
        except Exception as e:
            signal.alarm(0)
            results[key] = {"status": "FAIL", "error": f"{type(e).__name__}: {str(e)[:220]}",
                            "secs": round(time.time() - t0, 1), "note": note,
                            "trace": traceback.format_exc()[-600:]}
            print(f"{key:22} FAIL   {type(e).__name__}: {str(e)[:140]}")
    with open("/tmp/sweep_results.json", "w") as f:
        json.dump(results, f, indent=1)

ok = sum(1 for r in results.values() if r["status"] == "OK")
print(f"\n=== {ok}/{len(results)} OK ===")
for k, r in results.items():
    if r["status"] not in ("OK", "NEEDS_KEY"):
        print(f"  {r['status']:8} {k}: {r.get('error', r.get('note',''))[:160]}")
