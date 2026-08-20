# MCP server: the world's gauges as tools for Claude, Cursor and friends

`aquascope mcp` serves aquascope over the [Model Context Protocol](https://modelcontextprotocol.io/)
(stdio by default). Any MCP-speaking assistant can then find stations anywhere
on Earth, pull the observed record, and get flood frequency, flow duration and
trend with citations, without writing Python. It is the same code the CLI and
the [Explorer](explorer.md) run: the registry, the [Archive](archive.md) catalog,
and `aquascope.explore`.

## Install and connect

```bash
pip install "aquascope[mcp]"
```

Claude Code:

```bash
claude mcp add aquascope -- aquascope mcp
```

Claude Desktop (`claude_desktop_config.json`), Cursor and other clients take the
same shape:

```json
{
  "mcpServers": {
    "aquascope": { "command": "aquascope", "args": ["mcp"] }
  }
}
```

If `aquascope` is not on the client's PATH, use the interpreter explicitly:
`"command": "/path/to/python", "args": ["-m", "aquascope.cli", "mcp"]`.

## Tools

| tool | what it does | touches an agency? |
| --- | --- | --- |
| `list_sources()` | every source with agency, country, variables, licence, whether it has a station catalog | no |
| `find_stations(query, bbox, near, variable, sources, limit)` | search the published catalog (45k+ stations): name/id substring, `[west, south, east, north]` box, nearest-first from `[lat, lon]`, variable filter; at most 200 results | no (reads the Archive, cached daily) |
| `get_timeseries(source, station_id, years, resample, max_points)` | the observed record through aquascope's collector, resampled (`D`/`W`/`M`/`Y`) and thinned to at most 2,000 points, with stats, unit, licence and attribution | yes |
| `analyze_station(source, station_id, years, bootstrap_ci)` | record summary, annual maxima, GEV (L-moments) and Log-Pearson III return levels with 90 % CI, optional bootstrap GEV band, FDC percentiles, Mann-Kendall trend, method citations (raw arrays omitted) | yes |
| `flood_frequency(source, station_id, years, bootstrap_ci)` | just the return-period table and its methods | yes |
| `describe_methods()` | what each analysis computes and the reference to cite | no |
| `describe_catchment(lat, lon, upstream=True)` | the BasinATLAS (HydroATLAS, CC BY 4.0) catchment of a point: sub-basin, upstream area, elevation, climate, land cover, soils, population, dams; `upstream=False` for the local sub-basin | no (Archive `basins/` files) |
| `similar_basins(lat, lon | source, station_id, k, method, sources)` | the gauged basins whose catchments most resemble a point's or a station's (BasinATLAS attribute space and/or distance): donor selection for ungauged sites | no (Archive `basins/station_catchments.parquet`) |
| `regionalize_signatures(lat, lon, k, method)` | the estimated flow regime of an ungauged point (mean/median/Q95/Q05 flow in mm/d, annual maximum, runoff ratio, baseflow index, FDC slope, flow frequencies, seasonality, flashiness) transferred from the most similar gauged donors, with a band and the leave-one-out skill; `method`: similarity, regression or both | no (Archive `basins/station_signatures.parquet` + `regionalization_skill.json`) |
| `archive_health()` | per-source status of the last catalog harvest | no |
| `list_analyses()` | the sixteen `aquascope.workbench` analyses with their parameters: quality, preprocessing, insights, the WHO drinking-water screen, flow duration, three baseflow separations, recession, GEV flood frequency, flow signatures, return periods, FAO-56 ET0 and irrigation, SGI drought, WTF recharge, Theis drawdown | no |
| `analyse_table(csv, analysis, params)` | run one of those on a table the assistant already has (a user's own export, for instance): the date and value columns are detected, units converted to SI, and the result carries its methods and citations | no |
| `station_view(source, station_id, years)` | the `analyze_station` result plus a self-contained HTML view (inline hydrograph, headline numbers, attribution) under `_meta["mcp/view"]`, for clients that support the MCP Apps extension; clients that do not simply ignore the extra key | yes |

Resources: `aquascope://sources` and `aquascope://methods` (JSON).

In a browser, the same tools are available a second way: where WebMCP
(`navigator.modelContext`) exists, the [Explorer](explorer.md) registers
`find_stations`, `analyze_station`, `anywhere`, `describe_catchment` and
`show_on_map` in the page itself, with nothing installed at all.

Response sizes are bounded on purpose (station caps, thinning, no raw daily
arrays in analyses): an assistant's context is not a data lake. Ask for
`get_timeseries` when you need the numbers.

## Example conversation

> Which gauges measure discharge near Paris? → `find_stations(near=[48.85, 2.35], variable="discharge")`
> → What is the 100-year flood at the Seine at Austerlitz? → `flood_frequency("hubeau_hydrometrie", "F700000103")`
> → the return-period table (GEV L-moments and LP3 with 90 % CI), the record used (2006 to today, daily
> mean discharge from Hub'Eau obs_elab), the licence (Licence Ouverte 2.0) and the citations to put in
> the report.

## Keys and terms

Every tool works keyless. `USGS_API_KEY` in the environment lifts the shared demo-key throttling for USGS;
`HF_TOKEN` is not needed to read the public catalog. Data licences are returned with every result;
sources whose terms do not allow redistribution are still searchable but their observations are only
ever fetched live from the agency, never mirrored.
