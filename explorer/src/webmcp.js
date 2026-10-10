// Offer the page's tools to an agent running in the browser (WebMCP).
//
// `navigator.modelContext.registerTool` is the browser-native counterpart of an
// MCP server: a page declares what it can do, and an assistant in the same
// browser can call it. It is a W3C Web Machine Learning CG draft, in a Chrome
// origin trial at the time of writing, so this is entirely feature-detected:
// where it does not exist, nothing happens and nothing breaks.
//
// The tools are the same functions the MCP server and the Analyst use, running
// in this page's Pyodide worker, so an agent gets the world's gauges without
// aquascope being installed anywhere.

import { setTime, state } from "./core.js?v=__BUILD__";
import { STEPS, isIsoDate, normaliseRange, todayIso } from "./timeline.js?v=__BUILD__";
import { call, callLight } from "./worker-client.js?v=__BUILD__";

const TOOLS = [
  {
    name: "aquascope_find_stations",
    tool: "find_stations",
    description: "Search the world catalog of public water gauges by name, area or nearest point. "
      + "Returns source, station id, name, coordinates, variables and period of record.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Name or id fragment" },
        near: { type: "array", items: { type: "number" }, minItems: 2, maxItems: 2, description: "[lat, lon]" },
        bbox: { type: "array", items: { type: "number" }, minItems: 4, maxItems: 4 },
        variable: { type: "string", description: "discharge, water_level, precipitation, groundwater_level" },
        limit: { type: "integer" },
      },
    },
  },
  {
    name: "aquascope_analyze_station",
    tool: "analyze_station",
    description: "Fetch one gauge's observed record and compute its summary, annual maxima, flood frequency "
      + "(GEV and Log-Pearson III with confidence limits), flow-duration percentiles and Mann-Kendall trend.",
    inputSchema: {
      type: "object",
      properties: {
        source: { type: "string" }, station_id: { type: "string" },
        years: { type: "integer", description: "Optional cap on the record (the last N years); leave it out for the full record." },
        variable: { type: "string" },
      },
      required: ["source", "station_id"],
    },
  },
  {
    name: "aquascope_anywhere",
    tool: "anywhere",
    description: "Climate and modelled river discharge for any point on Earth with no gauge: ERA5 rainfall and "
      + "temperature, FAO-56 reference ET0, the aridity index and GloFAS discharge.",
    inputSchema: {
      type: "object",
      properties: { lat: { type: "number" }, lon: { type: "number" }, years: { type: "integer" } },
      required: ["lat", "lon"],
    },
  },
  {
    name: "aquascope_describe_catchment",
    tool: "describe_catchment",
    description: "The catchment upstream of a point from BasinATLAS: area, elevation, climate, land cover, "
      + "soils, population and regulation by dams.",
    inputSchema: {
      type: "object",
      properties: { lat: { type: "number" }, lon: { type: "number" }, upstream: { type: "boolean" } },
      required: ["lat", "lon"],
    },
  },
  {
    name: "aquascope_filter_gauges",
    description: "Filter the gauges on the map by their flow signatures: years of daily data, flood trend "
      + "(Mann-Kendall on annual maxima: rising, falling or none) and baseflow index range. Give the fields, "
      + "or a question in plain words such as '50+ years with a rising flood trend'. Returns how many match.",
    inputSchema: {
      type: "object",
      properties: {
        question: { type: "string", description: "The filter in plain words" },
        min_years: { type: "number" },
        flood_trend: { type: "string", enum: ["rising", "falling", "none"] },
        bfi_min: { type: "number" }, bfi_max: { type: "number" },
      },
    },
  },
  {
    name: "aquascope_show_on_map",
    description: "Show a gauge or a point on the map the reader is looking at, and open its analysis panel.",
    inputSchema: {
      type: "object",
      properties: {
        source: { type: "string" }, station_id: { type: "string" },
        lat: { type: "number" }, lon: { type: "number" },
      },
    },
  },
  {
    name: "aquascope_set_map_date",
    description: "Set the date the map's dated layers show (NASA GIBS rain, soil moisture, snow, land temperature, "
      + "satellite imagery, GRACE water storage), and optionally the play step and range. Dates are YYYY-MM-DD.",
    inputSchema: {
      type: "object",
      properties: {
        date: { type: "string", description: "YYYY-MM-DD" },
        step: { type: "string", enum: ["day", "week", "month"] },
        from: { type: "string", description: "Range start, YYYY-MM-DD" },
        to: { type: "string", description: "Range end, YYYY-MM-DD" },
      },
      required: ["date"],
    },
  },
  {
    name: "aquascope_map_actions",
    description: "Act on the map the reader is looking at, with every action listed in the page's action log "
      + "where the reader can undo it. Actions (aquascope.map_commands): fly_to {place | bbox [w,s,e,n] | "
      + "center [lat,lon], zoom | zoom_by}, set_time {date YYYY-MM-DD, step day|week|month, range {from,to}, "
      + "playing}, set_layer {layer, on}, focus_status {classes: much_below, below, normal, above, much_above}, "
      + "set_basemap {basemap}, highlight_river {place | lat, lon, direction upstream|downstream|both}, "
      + "draw_area {bbox | place, label}, add_pin {lat, lon, title, text, facts [{label, value, unit}], source}. "
      + "Each is checked before it runs; place names are looked up in the gazetteer.",
    inputSchema: {
      type: "object",
      properties: { actions: { type: "array", items: { type: "object" }, maxItems: 8 } },
      required: ["actions"],
    },
  },
  {
    name: "aquascope_scout",
    description: "Scout the view the reader is looking at (#563): drop up to ten numbered pins on what stands out "
      + "(the largest areas much above or below normal in the month's river status, the strongest floods "
      + "ahead, floods in the news, gauges at extremes today, gauges no model matches), each with its reason, "
      + "numbers and source, found by fixed rules in aquascope.map_scout. The pins go in the action log, where "
      + "the reader can undo them. Returns the pins.",
    inputSchema: { type: "object", properties: {} },
  },
];

export function webmcpAvailable() {
  return Boolean(navigator.modelContext && typeof navigator.modelContext.registerTool === "function");
}

function textResult(payload) {
  return { content: [{ type: "text", text: JSON.stringify(payload) }] };
}

export function registerWebMcpTools({ actions }) {
  if (!webmcpAvailable()) return false;
  try {
    for (const spec of TOOLS) {
      navigator.modelContext.registerTool({
        name: spec.name,
        description: spec.description,
        inputSchema: spec.inputSchema,
        async execute(args = {}) {
          if (spec.name === "aquascope_filter_gauges") {  // page-side: signature-filter.js
            if (!actions.setSignatureFilter) return textResult({ error: "The signature filter is not loaded." });
            const { question, ...fields } = args;
            return textResult(question
              ? await actions.setSignatureFilterFromQuestion(question, fields)
              : await actions.setSignatureFilter(fields));
          }
          if (spec.name === "aquascope_set_map_date") {  // page-side: the time bar (#522)
            if (!isIsoDate(args.date) || args.date > todayIso()) return textResult({ error: "Give a past date as YYYY-MM-DD." });
            const patch = { date: args.date };
            if (STEPS.includes(args.step)) patch.step = args.step;
            const range = normaliseRange({ from: args.from, to: args.to });
            if (range) patch.range = range;
            setTime(patch, { source: "agent" });
            return textResult({ date: state.date, step: state.timeStep, range: state.timeRange });
          }
          if (spec.name === "aquascope_map_actions") {  // page-side: map-actions.js (#561), checked by the package
            if (!actions.applyMapActions) return textResult({ error: "The map is not ready." });
            const checked = await callLight("map_command", { op: "validate", actions: args.actions || [], today: todayIso() });
            const resolved = await callLight("map_command", { op: "resolve", actions: checked.actions || [] });
            const done = await actions.applyMapActions(resolved.actions, { by: "agent", said: resolved.said || [] });
            return textResult({ applied: done.applied.map((e) => ({ id: e.id, label: e.label })),
              failed: done.failed, rejected: checked.errors || [], notes: resolved.notes || [], credit: resolved.credit });
          }
          if (spec.name === "aquascope_scout") {  // page-side: scout.js (#563)
            if (!actions.runScout) return textResult({ error: "The map is not ready." });
            const done = await actions.runScout();
            if (!done) return textResult({ error: "The scout could not run; the line under the bar says why." });
            return textResult({ mode: done.mode, by: done.by, pins: done.picks.map((f) => ({ rank: f.rank,
              title: f.title, reason: f.reason, lat: f.lat, lon: f.lon, source: f.source })) });
          }
          if (spec.name === "aquascope_show_on_map") {
            if (args.source && args.station_id) {
              actions.selectStation(`${args.source}/${args.station_id}`, { fly: true });
              return textResult({ shown: `${args.source}/${args.station_id}` });
            }
            if (typeof args.lat === "number" && typeof args.lon === "number") {
              actions.selectPoint(args.lat, args.lon, { fly: true });
              return textResult({ shown: [args.lat, args.lon] });
            }
            return textResult({ error: "Give a source and station_id, or a lat and lon." });
          }
          const payload = await call("tool", { name: spec.tool, arguments: args });
          return textResult(payload);
        },
      });
    }
    state.webmcp = TOOLS.length;
    console.info(`WebMCP: registered ${TOOLS.length} aquascope tools for an in-browser agent`);
    return true;
  } catch (err) {
    console.info("WebMCP registration failed:", err && err.message);
    return false;
  }
}
