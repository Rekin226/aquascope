# The monthly bulletin: the state of the rivers

Once a month AquaScope writes a short bulletin on how the rivers stood the month
before, in the style of the WMO Hydrological Status and Outlook System (HydroSOS):
every Archive gauge with a mirrored record covering the month, placed against the
same month in its other years, in five classes from much below normal to much above
normal. It is the same in the package, the CLI, the MCP server and the Explorer
(`aquascope.bulletin`, issue #523).

## What it says

For each gauge:

- the month's **mean daily flow**, when the month has at least **25 days** of data;
- its **percentile** against the same calendar month in every other year of the record
  that also has 25 days (mid-rank, as for today against normal);
- the **class**: much below normal (under 10), below (10 to 24), normal (25 to 75),
  above (76 to 90), much above (over 90), the USGS National Water Dashboard and
  HydroSOS thresholds;
- whether it is a **new record** for that month (above or below every other year).

A gauge needs at least **10 other years** of that month. One that has fewer, or a
month with fewer than 25 days, is left out and counted, so the coverage line always
says how many gauges the bulletin rests on.

The bulletin then rolls the gauges up **per country** (the catalogue's country) and
**per river basin** (BasinATLAS river basins, the level-12 sub-basin at the river's
outlet, with at least 3 classed gauges, named after their largest gauge), names the
**notable gauges** (new monthly highs and lows, those of gauges with a usual flow of at
least 1 m³/s named first; the furthest above and below their usual flow for the month,
among gauges classed above or below normal with a usual flow of at least 1 m³/s) and
writes a summary paragraph from those numbers by rules. No language model writes any of it.

It renders through the Studio's document composer, so it reads like a report: the
same print-ready HTML (Print or save as PDF from the browser) and Markdown, with one
map of the gauges coloured by class over Natural Earth country outlines.

## Use it

```bash
aquascope bulletin 2026-09                  # the summary and the country table
aquascope bulletin 2026-09 --out bulletin   # bulletins/2026-09/bulletin.html, .md, .json, map.png, status.parquet
aquascope bulletin 2026-09 --json           # everything but the per-gauge list (add --gauges for it)
```

Without a month it is the latest published bulletin (last month when none is published
yet). The published bulletin is read when there is one;
`--rebuild` (or `--sources`, or `--archive DIR` for a local copy of the dataset) builds
it from the Archive's discharge records instead, which downloads a few hundred MB the
first time. `--top-up N` asks the agencies for up to N gauges' days the weekly mirror
does not have yet.

```python
from aquascope.bulletin import status_bulletin, write_bulletin

b = status_bulletin("2026-09")
print(b["summary"])
write_bulletin(b, "bulletin")
```

The MCP tool is `status_bulletin(month, sources, country)`; it leaves the per-gauge list
out unless `country` asks for one country's gauges.

## In the Explorer

**Bulletin** in the Tools menu opens the latest bulletin in a reader (the document, with
Print or save as PDF and the Markdown beside it). **Last month's status** in the layers
panel's gauge colouring colours the gauges by their class in that bulletin; gauges it
did not class are light grey. Before the first bulletin is published, both say so and
the gauges keep their agency colours.

## The monthly workflow

`.github/workflows/bulletin.yml` runs on the 3rd of every month (and by hand with a
`month`). It downloads the catalogue, the discharge bundles and the basin tables, asks
the agencies for the missing days of up to 2,500 gauges within 90 minutes, and writes
only these paths in the Archive dataset:

| path | what |
| --- | --- |
| `bulletins/<YYYY-MM>/bulletin.html` | the bulletin, print-ready |
| `bulletins/<YYYY-MM>/bulletin.md` | the same in Markdown, with `map.png` beside it |
| `bulletins/<YYYY-MM>/bulletin.json` | every number, the per-gauge list included |
| `bulletins/<YYYY-MM>/status.parquet` | one row per classed gauge: `source`, `station_id`, `month`, `value`, `n_days`, `percentile`, `class`, `n_years`, `median`, `ratio`, `record`, `country`, `basin_id` (the Explorer's colouring) |
| `bulletins/index.json` | every month published, newest first, with its headline |

A `smoke` run takes a few gauges per source and never publishes. Locally:
`python -m aquascope.bulletin build --archive DIR --out DIR [--month YYYY-MM]`.

## Data and licences

Only sources whose terms ask for nothing beyond attribution are used (public domain,
CC BY, the UK Open Government Licence, Licence Ouverte, the IMGW-PIB terms and the
like). A share-alike source (Greece OpenHi.net, CC BY-SA 4.0) is left out and named,
because a status derived from it would carry its terms. Each bulletin lists its
sources with their attribution. Basins are BasinATLAS (HydroATLAS v1.0, CC BY 4.0); the
country outlines are Natural Earth (public domain). The observations are as the agencies
publish them, and the newest are often provisional.

## Limits

- Coverage is what the Archive mirrors: today the United States, England and France
  (and Poland when its archive reaches the month). A month the mirror has not reached
  yet, and the agencies cannot fill, is left out.
- "England" is the catalogue's country GBR: the Environment Agency covers England only.
- A tidal or canal gauge can report a negative monthly mean; it is classed but not named
  among the notable gauges.
- The percentile compares a gauge with itself, not with its neighbours; a regulated
  river is compared with its own regulated past.
