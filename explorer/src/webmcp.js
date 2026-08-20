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

import { state } from "./core.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";

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
        years: { type: "integer" }, variable: { type: "string" },
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
          if (spec.name === "aquascope_show_on_map") {
            if (args.source && args.station_id) {
              actions.selectStation(`${args.source}/${args.station_id}`, { fly: true });
              return textResult({ shown: `${args.source}/${args.station_id}` });
            }
            if (typeof args.lat === "number" && typeof args.lon === "number") {
              actions.selectPoint(args.lat, args.lon);
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
