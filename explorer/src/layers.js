// The layer catalogue: basemaps, terrain and the environmental overlays.
//
// Everything here is free, keyless and reachable from a static page: each entry
// was checked for terms, an API key and CORS on 2026-08-20 (#232). Two rules
// this file exists to keep honest:
//
//   * every layer carries its attribution and its licence, and the page shows
//     them, because several of these are CC BY or CC BY-NC-SA;
//   * nothing here needs a key, a token or a server of ours.
//
// Deliberately absent: Google Maps and Earth tiles (the Maps Platform terms
// forbid using them "with or near a non-Google map", and the Map Tiles API
// needs a billing-enabled key), and Esri's legacy World Imagery (it answers
// without a token, but Esri's own docs say a token is required).
//
// No maplibregl, no window: this module is data plus pure helpers, so the
// tests can import it under node.

export const OPENFREEMAP = "https://tiles.openfreemap.org/styles";
export const EOX = "https://tiles.maps.eox.at/wmts/1.0.0";
export const GIBS = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best";
export const GIBS_LEGENDS = "https://gibs.earthdata.nasa.gov/legends";

// Where a raster basemap needs labels and the style is a plain image source.
const OSM_ATTR = '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

export const BASEMAPS = [
  {
    id: "light",
    label: "Light",
    kind: "style",
    url: `${OPENFREEMAP}/positron`,
    attribution: `OpenFreeMap · © OpenMapTiles · ${OSM_ATTR}`,
    licence: "Open data (ODbL), free public instance, no key",
    default: true,
  },
  {
    id: "dark",
    label: "Dark",
    kind: "style",
    url: `${OPENFREEMAP}/dark`,
    attribution: `OpenFreeMap · © OpenMapTiles · ${OSM_ATTR}`,
    licence: "Open data (ODbL), free public instance, no key",
  },
  {
    id: "streets",
    label: "Streets",
    kind: "style",
    url: `${OPENFREEMAP}/liberty`,
    attribution: `OpenFreeMap · © OpenMapTiles · ${OSM_ATTR}`,
    licence: "Open data (ODbL), free public instance, no key",
  },
  {
    id: "satellite",
    label: "Satellite (2016)",
    kind: "raster",
    tiles: [`${EOX}/s2cloudless_3857/default/g/{z}/{y}/{x}.jpg`],
    maxzoom: 18,
    attribution: 'Sentinel-2 cloudless 2016 by <a href="https://s2maps.eu">EOX IT Services</a> (Contains modified Copernicus Sentinel data 2016)',
    licence: "CC BY 4.0",
    note: "Cloud-free Sentinel-2 mosaic. The 2016 build is the one under a CC BY licence.",
  },
  {
    id: "satellite-recent",
    label: "Satellite (2025)",
    kind: "raster",
    tiles: [`${EOX}/s2cloudless-2025_3857/default/g/{z}/{y}/{x}.jpg`],
    maxzoom: 18,
    attribution: 'Sentinel-2 cloudless 2025 by <a href="https://s2maps.eu">EOX IT Services</a> (Contains modified Copernicus Sentinel data 2025)',
    licence: "CC BY-NC-SA 4.0 (non-commercial)",
    note: "Newer mosaic, but its licence is non-commercial: fine to look at and cite, not for a commercial product.",
  },
  {
    id: "terrain",
    label: "Terrain",
    kind: "raster",
    tiles: [`${EOX}/terrain-light_3857/default/g/{z}/{y}/{x}.jpg`],
    maxzoom: 16,
    attribution: 'Terrain Light © <a href="https://maps.eox.at">EOX IT Services</a> (Contains modified SRTM and Natural Earth data)',
    licence: "CC BY-SA 4.0",
  },
  {
    id: "daily",
    label: "Satellite (today)",
    kind: "raster",
    tiles: [`${GIBS}/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`],
    maxzoom: 9,
    time: true,
    attribution: "Imagery from NASA Worldview / GIBS (ESDIS)",
    licence: "Open (NASA), acknowledgement requested",
    note: "VIIRS true colour for the chosen date, coarse (about 250 m). Move the date to see a flood or a storm.",
  },
  {
    id: "usgs",
    label: "US imagery",
    kind: "raster",
    tiles: ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"],
    maxzoom: 16,
    attribution: "Map services and data available from U.S. Geological Survey, National Geospatial Program",
    licence: "Public domain",
    note: "United States only.",
  },
];

// The DEM behind 3D terrain, hillshade and contours.
export const TERRAIN_DEM = {
  id: "aws-terrarium",
  tiles: ["https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"],
  encoding: "terrarium",
  tileSize: 256,
  maxzoom: 13,
  attribution: '<a href="https://registry.opendata.aws/terrain-tiles/">AWS Terrain Tiles</a> (Mapzen/Tilezen; SRTM, 3DEP, GMTED, ETOPO1 and national DEMs)',
  licence: "Open data, per-source attribution (tilezen/joerd)",
};

// Raster overlays. `time: true` means the URL carries {date} and the shared
// date control drives it. Legends are the GIBS colour maps, which are SVG.
export const OVERLAYS = [
  {
    id: "precip",
    label: "Precipitation rate",
    group: "Water and climate",
    tiles: [`${GIBS}/IMERG_Precipitation_Rate/default/{date}/GoogleMapsCompatible_Level6/{z}/{y}/{x}.png`],
    maxzoom: 6,
    time: true,
    opacity: 0.75,
    legend: `${GIBS_LEGENDS}/GPM_Precipitation_Rate_H.svg`,
    attribution: "GPM IMERG precipitation rate, NASA GIBS (ESDIS)",
    licence: "Open (NASA), acknowledgement requested",
    note: "Half-hourly IMERG rain rate for the chosen day, mm/h.",
  },
  {
    id: "soil",
    label: "Root-zone soil moisture",
    group: "Water and climate",
    tiles: [`${GIBS}/SMAP_L4_Analyzed_Root_Zone_Soil_Moisture/default/{date}/GoogleMapsCompatible_Level6/{z}/{y}/{x}.png`],
    maxzoom: 6,
    time: true,
    opacity: 0.75,
    legend: `${GIBS_LEGENDS}/SMAP_Analyzed_Soil_Moisture_H.svg`,
    attribution: "SMAP L4 analysed root-zone soil moisture, NASA GIBS (ESDIS)",
    licence: "Open (NASA), acknowledgement requested",
    note: "Model-assimilated SMAP soil moisture in the top metre, m³/m³. A few days behind today.",
  },
  {
    id: "snow",
    label: "Snow cover",
    group: "Water and climate",
    tiles: [`${GIBS}/MODIS_Terra_NDSI_Snow_Cover/default/{date}/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png`],
    maxzoom: 8,
    time: true,
    opacity: 0.8,
    legend: `${GIBS_LEGENDS}/MODIS_NDSI_Snow_Cover_H.svg`,
    attribution: "MODIS/Terra NDSI snow cover, NASA GIBS (ESDIS)",
    licence: "Open (NASA), acknowledgement requested",
    note: "Cloud gaps are normal: this is one day of one satellite.",
  },
  {
    id: "lst",
    label: "Land surface temperature",
    group: "Water and climate",
    tiles: [`${GIBS}/MODIS_Terra_Land_Surface_Temp_Day/default/{date}/GoogleMapsCompatible_Level7/{z}/{y}/{x}.png`],
    maxzoom: 7,
    time: true,
    opacity: 0.7,
    legend: `${GIBS_LEGENDS}/MODIS_Land_Surface_Temp_H.svg`,
    attribution: "MODIS/Terra daytime land surface temperature, NASA GIBS (ESDIS)",
    licence: "Open (NASA), acknowledgement requested",
  },
  {
    id: "storage",
    label: "Water storage anomaly",
    group: "Water and climate",
    tiles: [`${GIBS}/GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI/default/{date}/GoogleMapsCompatible_Level6/{z}/{y}/{x}.png`],
    maxzoom: 6,
    time: true,
    monthly: true,
    opacity: 0.75,
    legend: `${GIBS_LEGENDS}/GRACE_Tellus_Liquid_Water_Equivalent_Thickness_Mascon_CRI_H.svg`,
    attribution: "GRACE/GRACE-FO Tellus mascon liquid water equivalent thickness, NASA GIBS (ESDIS)",
    licence: "Open (NASA), acknowledgement requested",
    note: "Total water storage anomaly in cm of equivalent water, monthly. The signal groundwater depletion shows up in.",
  },
  {
    id: "landcover",
    label: "Land cover (2021)",
    group: "Land",
    tiles: ["https://wmts.terrascope.be/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0" +
      "&LAYER=esa-worldcover-map-10m-2021-v2_map&STYLE=default&TILEMATRIXSET=EPSG:3857" +
      "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/png&TIME=2021-01-01"],
    minzoom: 6,
    maxzoom: 14,
    opacity: 0.8,
    attribution: "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium",
    licence: "CC BY 4.0",
    note: "10 m land cover, from zoom 6 in.",
  },
];

export const OVERLAY_GROUPS = ["Water and climate", "Land"];

export const basemapById = (id) => BASEMAPS.find((b) => b.id === id) || BASEMAPS.find((b) => b.default);
export const overlayById = (id) => OVERLAYS.find((o) => o.id === id);

// GIBS wants a plain YYYY-MM-DD, and monthly products want the first of the
// month. Data lands a few days late, so the default date is a week back.
export function defaultDate(today = new Date()) {
  const d = new Date(today.getTime() - 7 * 86400000);
  return d.toISOString().slice(0, 10);
}

export function layerDate(layer, date) {
  if (!date) return date;
  return layer && layer.monthly ? `${date.slice(0, 7)}-01` : date;
}

export function tileUrls(layer, date) {
  const d = layerDate(layer, date);
  return (layer.tiles || []).map((t) => t.replace("{date}", d || ""));
}

// One line per visible layer for the credits panel.
export function creditLines(basemapId, overlayIds, { terrain = false } = {}) {
  const out = [];
  const base = basemapById(basemapId);
  if (base) out.push({ label: base.label, attribution: base.attribution, licence: base.licence });
  if (terrain) out.push({ label: "Elevation", attribution: TERRAIN_DEM.attribution, licence: TERRAIN_DEM.licence });
  for (const id of overlayIds || []) {
    const o = overlayById(id);
    if (o) out.push({ label: o.label, attribution: o.attribution, licence: o.licence });
  }
  return out;
}

// ── how the gauges themselves are coloured ──────────────────────────────────

export const GAUGE_STYLES = [
  { id: "source", label: "Agency" },
  { id: "record", label: "Record length" },
  { id: "recent", label: "Last observation" },
];

// Years of record from the catalog's own period columns (no extra data needed).
export function recordYears(station, now = new Date()) {
  if (!station || !station.period_start) return null;
  const start = Date.parse(station.period_start);
  const end = station.period_end ? Date.parse(station.period_end) : now.getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
  return (end - start) / (365.2425 * 86400000);
}

export function yearsSinceLast(station, now = new Date()) {
  if (!station || !station.period_end) return null;
  const end = Date.parse(station.period_end);
  if (!Number.isFinite(end)) return null;
  return (now.getTime() - end) / (365.2425 * 86400000);
}

// Sequential ramp for record length, diverging-ish for staleness. Both are
// colour-blind safe (viridis-like and a warm ramp), and both are also given as
// a legend so colour is never the only encoding.
export const RECORD_BREAKS = [
  { max: 5, color: "#fde725", label: "under 5 yr" },
  { max: 10, color: "#7ad151", label: "5 to 10" },
  { max: 20, color: "#22a884", label: "10 to 20" },
  { max: 50, color: "#2a788e", label: "20 to 50" },
  { max: 100, color: "#414487", label: "50 to 100" },
  { max: Infinity, color: "#440154", label: "100 yr and more" },
];

export const RECENT_BREAKS = [
  { max: 0.1, color: "#1a9850", label: "this month" },
  { max: 1, color: "#91cf60", label: "this year" },
  { max: 5, color: "#fee08b", label: "1 to 5 yr ago" },
  { max: 20, color: "#fc8d59", label: "5 to 20 yr ago" },
  { max: Infinity, color: "#d73027", label: "over 20 yr ago" },
];

export function breakColor(breaks, value, fallback = "#9e9e9e") {
  if (value === null || value === undefined || !Number.isFinite(value)) return fallback;
  for (const b of breaks) if (value < b.max) return b.color;
  return breaks[breaks.length - 1].color;
}
