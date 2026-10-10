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
| `find_stations(query, bbox, near, variable, sources, limit)` | search the published catalog (45k+ stations): words from the name, id or river (accent-insensitive; "Kingston Thames" finds the Thames at Kingston), `[west, south, east, north]` box, nearest-first from `[lat, lon]`, variable filter; at most 200 results | no (reads the Archive, cached daily) |
| `get_timeseries(source, station_id, years, resample, max_points)` | the observed record through aquascope's collector, resampled (`D`/`W`/`M`/`Y`) and thinned to at most 2,000 points, with stats, unit, licence and attribution | yes |
| `water_quality_samples(source, station_id, years, parameters, use)` | sampled water-quality parameters at a station as tidy rows with per-parameter counts, units and period (USGS daily means for temperature, conductivity, dissolved oxygen and pH; Water Quality Portal discrete samples for a short list per use): a screening over the last five years by default, with licence and attribution; feed the rows to `analyse_table` with `wqi`, `iwqi` or `who_screen` | yes |
| `analyze_station(source, station_id, years, bootstrap_ci)` | record summary, annual maxima, GEV (L-moments) and Log-Pearson III return levels with 90 % CI, optional bootstrap GEV band, FDC percentiles, Mann-Kendall trend, method citations (raw arrays omitted); the full record is requested by default, `years` caps it to the last N, and `fetch_note` says what was asked for and what the agency served | yes |
| `flood_frequency(source, station_id, years, bootstrap_ci)` | just the return-period table and its methods | yes |
| `describe_methods()` | what each analysis computes and the reference to cite | no |
| `assess_site(lat, lon, radius_km, problem, return_period)` | what can be answered at a place before any analysis: the gauges within reach with their true catalog spans, the BasinATLAS catchment, the site context, and a sufficiency table marking every method defensible, marginal or not defensible here with the reason and the station it would use; call it first for a place or a station | no (catalog and Archive `basins/` files) |
| `describe_catchment(lat, lon, upstream=True)` | the BasinATLAS (HydroATLAS, CC BY 4.0) catchment of a point: sub-basin, upstream area, elevation, climate, land cover, soils, population, dams; `upstream=False` for the local sub-basin | no (Archive `basins/` files) |
| `place_context(lat, lon, layers)` | the context of a place, each layer with a one-line summary and its licence: flood events in the news (Groundsource) and Sentinel-1 radar flood months 2014-2024, how often the ground was water since 1984 (JRC Global Surface Water), modelled flood depth at the 10 to 500-year floods (JRC CEMS-GloFAS), dams nearby (Global Dam Watch), soil texture and plant-available water (SoilGrids), actual evapotranspiration (FAO WaPOR v3) and the nearest GHCN-Daily rain gauge with its record | no agency; open data hosts and the Archive `context/` mirror |
| `area_context(west, south, east, north, layers)` | the same layers over a box: flood events and radar months inside it, dams and their storage, rain gauges, and the rasters sampled on a small grid | no agency; as above |
| `similar_basins(lat, lon | source, station_id, k, method, sources)` | the gauged basins whose catchments most resemble a point's or a station's (BasinATLAS attribute space and/or distance): donor selection for ungauged sites | no (Archive `basins/station_catchments.parquet`) |
| `regionalize_signatures(lat, lon, k, method)` | the estimated flow regime of an ungauged point (mean/median/Q95/Q05 flow in mm/d, annual maximum, runoff ratio, baseflow index, FDC slope, flow frequencies, seasonality, flashiness) transferred from the most similar gauged donors, with a band and the leave-one-out skill; `method`: similarity, regression or both | no (Archive `basins/station_signatures.parquet` + `regionalization_skill.json`) |
| `drought_indices(lat, lon, years, timescales, source, station_id, pet)` | drought status at a place: SPI and SPEI at 1, 3 and 12 months (or `timescales`) with the divergence between them, from a rain gauge (`source` + `station_id`, its whole record, ERA5 for the PET) or the ERA5 cell over the last `years`; `pet`: thornthwaite (default), fao56 or none; plus the ERA5 temperature trend and the drought events | Open-Meteo (ERA5); the gauge's agency or the Archive |
| `drought_propagation(source, station_id, lat, lon, years, max_lag)` | the Standardised Groundwater Index at a well (current, worst, events) and the SPI accumulation period and lag in months, on ERA5 precipitation for the cell, whose cross-correlation with it is highest | the well's agency or the Archive; Open-Meteo |
| `low_flow_context(source, station_id, years)` | Q95, Q50, Q10 (and Q05, Q25, Q75, Q90), the baseflow index, 7Q10 when the record has ten years, and where the last 30 and 90 days sit in the record | the gauge's agency or the Archive |
| `supply_reliability(demand_m3s or demand_ml_day, source, station_id or lat, lon, share, reserve, months)` | can a river supply a demand as a run-of-river abstraction: the fraction of days, of years without a shortfall and of the volume met while `reserve` (Q95 by default) stays in the river and at most `share` of the flow is taken, over the year or over `months`; ungauged, the reliability read off donor-transferred Q95, median and Q05 as a band | the gauge's agency or the Archive; Archive basins for the ungauged case |
| `crop_water_demand(lat, lon, crop, area_ha, planting_month, efficiency, years)` | a crop's seasonal irrigation demand: FAO-56 single Kc (Table 12 keys) on ERA5 FAO-56 ET0, effective rainfall subtracted, divided by the efficiency, the season repeated over the years of the window; mm, m3 over the area, mean and peak-month m3/s, the season's months | Open-Meteo (ERA5) |
| `snap_to_river(lat, lon, max_distance_m, prefer, area_km2)` | the GEOGLOWS v2 river reach a point stands for (its `river_id`, stream order and distance): the main channel within the tolerance, naming a smaller stream that was nearer (`prefer="nearest"` takes the nearest line), or with `area_km2` the reach whose upstream area matches a gauge's catchment; "no stream within N m" with the nearest reach named, so a hillside is not taken for a river | GEOGLOWS stream tiles (range reads) |
| `reach_record(river_id or lat, lon, years, return_periods)` | a reach's simulated daily discharge since 1940, analysed like a gauge: return periods (GEV and LP3 with 90 % CI), Q95/Q50/Q10, the monthly regime, the trend; labelled modelled, CC BY 4.0; daily arrays omitted | the GEOGLOWS API |
| `upstream_area(river_id, lat, lon)` | the area draining to a reach (km2) and the reaches upstream, summed from the model's unit catchments; approximate, within about 7 % of four agency-published gauge areas in our checks | GEOGLOWS routing tables |
| `trace_downstream(river_id or lat, lon, gauge_km, dam_km)` | the reach followed to its outlet: reaches, length, the path (TDX-Hydro, CC BY-SA 4.0, for display), the catalog gauges within `gauge_km` of it in downstream order, the Global Dam Watch dams within `dam_km` (capacity, purpose, degree of regulation where GDW gives it, km along the path), the countries it crosses (Natural Earth), the dams upstream and the upstream area | GEOGLOWS tiles and routing tables; the Archive's GDW mirror; Natural Earth via jsDelivr |
| `upstream_ids(river_id, max_n, lat, lon)` | the `river_id`s of every reach that drains to a reach (its own first, breadth-first), how many there are and the area they drain; past `max_n` (200 by default, at most 20,000) the reaches with the largest drainage area are kept and `min_area_km2` says where the cut fell | GEOGLOWS routing tables |
| `downstream_ids(river_id, max_n, lat, lon)` | the `river_id`s from a reach to its outlet, in the order the water goes, and the outlet's id; lighter than `trace_downstream` (no geometry, gauges or dams) | GEOGLOWS routing tables |
| `upstream_dams(river_id or lat, lon, with_flow)` | is the river regulated upstream of a reach: the Global Dam Watch dams that drain to it, their storage, and with `with_flow` the degree of regulation (storage as a % of a year's mean flow, GEOGLOWS, modelled); approximate, each dam matched to its nearest reach | the Archive's GDW mirror; GEOGLOWS tiles, routing tables and API |
| `model_skill(source, station_id or lat, lon, area_km2, models, years)` | how well each global model reproduces a gauge: GEOGLOWS v2 and GloFAS scored live, NWM v3 (US) and Google GRRR from the monthly table; KGE with r, alpha and beta, NSE, percent bias, the error at the 2-, 10- and 100-year flows, a grade A to D and a sentence on where they disagree ([evidence.md](evidence.md)) | the gauge's agency or the Archive; GEOGLOWS, Open-Meteo, the Archive's `skill/` table |
| `model_to_lean_on(lat, lon, radius_km)` | which global model tracked the graded gauges near a site best (median KGE), and the sentence that says so | the Archive's `skill/` table |
| `flow_status(source, station_id, date)` | where a gauge's latest day (topped up with the agency's newest days), or `date`, sits against the same days of the year in its other years: the percentile, the class (much below normal to much above normal, the USGS and WMO HydroSOS classes) and one sentence; needs 10 years in the window | the Archive or the gauge's agency |
| `status_bulletin(month, sources, country)` | the month's state of the rivers (default last month): every Archive gauge's monthly mean against the same month in its other years (25 days, 10 years), the five HydroSOS classes, the roll-up per country and per BasinATLAS river basin, the new monthly records, the gauges furthest from normal, the coverage and a summary written by rules; `country` (ISO3) lists that country's gauges ([bulletin.md](bulletin.md)) | the Archive's published bulletin, else its discharge records |
| `flood_warnings(bbox, min_rp, limit)` | Floods ahead: the river reaches the GEOGLOWS v2 global forecast expects to reach their 2-year flow in the next 15 days (Strahler order 5 and up, and every reach an Archive gauge sits on), optionally inside a `[west, south, east, north]` box: the issue date, counts by return-period class, the reaches (highest class first) with peak flow, peak day and the share of members that agree, and the method. Model output, not an official warning ([archive.md](archive.md#floods-ahead-forecastswarnings)) | the Archive's daily `forecasts/warnings/` issue |
| `flow_forecast(lat, lon, river_id, station, days)` | the next 15 days from GEOGLOWS v2 (ensemble mean, median, 25-75 and min-max bands, high-res run) and GloFAS v4 (daily ensemble statistics), with the reach's 2- to 100-year flows; with `station`, also today's status and the forecast corrected to the gauge; modelled | the GEOGLOWS API and Open-Meteo |
| `watch_digest(items, since, thresholds, forecast)` | what changed at watched gauges (`source/station_id`), river reaches (`river:<id>`) and areas (`area:w,s,e,n`) since a date: new days and the latest value, today's status class against the one then, the 15-day forecast peak against a per-item threshold (a value or a return period such as `"10y"`, else the 2-year flow; modelled) and flood events in the news nearby; one line each and a summary | the Archive, the gauges' agencies, the GEOGLOWS API and the Archive's Groundsource mirror |
| `correct_to_gauge(source, station_id, river_id, days)` | the GEOGLOWS forecast at a gauge's reach corrected to the gauge's record (monthly flow-duration quantile mapping) and its hindcast skill, raw against corrected: KGE with r, alpha, beta, percent bias, hit rate and false alarms above the gauge's 2-year flow | the GEOGLOWS API, the Archive or the gauge's agency |
| `archive_health()` | per-source status of the last catalog harvest | no |
| `dated_layers(live)` | the map layers that change with the date (NASA GIBS VIIRS true colour, IMERG rain, SMAP soil moisture, MODIS snow and land temperature, GRACE water storage): cadence, first and last day, licence, tile template; `live=True` reads the exact intervals and gaps from the GIBS capabilities | no (GIBS, only with `live`) |
| `layer_frames(layer, start, end, step, max_frames)` | the frames of a time-lapse of one dated layer: each date at a day, week or month step with its XYZ tile URL, skipping dates the layer cannot show, at most 60 | no |
| `river_status_month(month)` | the world river status map for a month (default the newest): GEOGLOWS v2's monthly HydroSOS GeoTIFF, every HydroBASINS level-4 basin in one of the five classes from much below to much above normal; its URL, the legend with the file's colours and percentile bounds, the months that exist (1990 on, with gaps), the method and the licence (CC BY 4.0) | no (lists the GEOGLOWS bucket) |
| `river_status_summary(month)` | where the rivers are low or high in a month of that map, in one line: reads the month's GeoTIFF and gives, for a few named regions (rough boxes such as the Amazon, the Sahel, South Asia), the share of the mapped area below and above normal, and the headline the Explorer shows over the globe |
| `list_analyses()` | the eighteen `aquascope.workbench` analyses with their parameters: quality, preprocessing, insights, the WHO drinking-water screen, the water quality index (CCME WQI 1.0 against WHO 2022 drinking-water, FAO 29 irrigation or CCME aquatic-life guidelines, plus the NSF WQI) and the FAO 29 irrigation suitability index, flow duration, three baseflow separations, recession, GEV flood frequency, flow signatures, return periods, FAO-56 ET0 and irrigation, SGI drought, WTF recharge, Theis drawdown | no |
| `analyse_table(csv, analysis, params)` | run one of those on a table the assistant already has (a user's own export, for instance): the date and value columns are detected, units converted to SI, and the result carries its methods and citations | no |
| `list_playbooks()` | the problem playbooks (flood risk, ungauged flow, groundwater decline, drought status, supply reliability, irrigation feasibility, water quality): id, title, branches, intake fields | no |
| `describe_playbook(id)` | one playbook in full: intake, branches with conditions and steps, gates, fallbacks, declines, caveats, citations | no |
| `solve_plan(problem, lat, lon, playbook, intake)` | reconnaissance of the point, the playbook and branch the tree picks, and the study (version 2) it fills, with a gate per step; nothing is executed and no model is called; `declined` carries the playbook's reason when it refuses | no (the catalog) |
| `solve_run(study)` | execute a study from `solve_plan` (edited or not): every gate outcome, the report, the study with its results, which `aquascope run` reproduces | yes (the study's tools) |
| `engineering_export(source, station_id, tool, years, variable, regional_skew, regional_skew_mse, out_dir)` | the record as inputs for an engineering tool (`hec-hms`, `hec-ras`, `hec-ssp`, `dss`, `swmm`, `modflow6`, `fews`, `raven`, or `all`): each file's text, cut at `max_chars`, with the notes to read first; `out_dir` also writes them ([formats](engineering_exports.md)) | yes |
| `station_view(source, station_id, years)` | the `analyze_station` result plus a self-contained HTML view (inline hydrograph, headline numbers, attribution) under `_meta["mcp/view"]`, for clients that support the MCP Apps extension; clients that do not simply ignore the extra key | yes |

Resources: `aquascope://sources` and `aquascope://methods` (JSON).

In a browser, the same tools are available a second way: where WebMCP
(`navigator.modelContext`) exists, the [Explorer](explorer.md) registers
`find_stations`, `analyze_station`, `anywhere`, `describe_catchment`,
`show_on_map` and `set_map_date` in the page itself, with nothing installed at all.

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

Every tool works keyless. `USGS_API_KEY` in the environment raises the keyless USGS rate limit;
`HF_TOKEN` is not needed to read the public catalog. Data licences are returned with every result;
sources whose terms do not allow redistribution are still searchable but their observations are only
ever fetched live from the agency, never mirrored.
