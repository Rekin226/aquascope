// World river status (#544), the pure part: no DOM, no map, so node can test
// it and the decoding worker can import it.
//
// GEOGLOWS v2 publishes one global map of river status a month since January
// 1990, hydrosos/cogs/YYYY-MM.tif: an RGB GeoTIFF, 7200 x 3600 cells of 0.05
// degree, each HydroBASINS level-4 basin painted in one of the five WMO
// HydroSOS classes. aquascope/map_layers.py is the package side (the CLI and
// the MCP tool); this file only reads the months, turns the file's colours
// back into classes and lays them out on a Web Mercator canvas.

import { STATUS_CLASSES } from "./now-core.js?v=__BUILD__";

export const STATUS_BUCKET = "https://geoglows-v2.s3.us-west-2.amazonaws.com";
export const STATUS_PREFIX = "hydrosos/cogs/";
export const STATUS_FIRST = "1990-01";
export const STATUS_LIST_URL = `${STATUS_BUCKET}/?list-type=2&prefix=${encodeURIComponent(STATUS_PREFIX)}`;
export const statusUrl = (month) => `${STATUS_BUCKET}/${STATUS_PREFIX}${month}.tif`;
// geotiff.js (MIT), pinned: it reads the files in the browser.
export const GEOTIFF_MODULE = "https://cdn.jsdelivr.net/npm/geotiff@3.0.5/+esm";

export const STATUS_CREDIT = {
  label: "World river status",
  attribution: 'GEOGLOWS v2 HydroSOS monthly map (<a href="https://geoglows.org">GEOGloWS ECMWF Streamflow Service</a>)',
  licence: "CC BY 4.0",
};

// The colours in the file, class 1 (much below normal) to 5 (much above), as
// GEOGLOWS's monthly_products.py writes them and aquascope/map_layers.py
// records them (a test keeps the two equal). Their red values all differ, so
// one band is enough to tell the classes apart: a third of the decoding.
export const FILE_RGB = [[205, 35, 63], [255, 168, 133], [231, 226, 188], [142, 206, 238], [44, 125, 205]];
const RED_TO_CLASS = new Uint8Array(256);
FILE_RGB.forEach(([r], i) => { RED_TO_CLASS[r] = i + 1; });
export const classOfRed = (r) => RED_TO_CLASS[r] || 0;

// ── which months exist ──────────────────────────────────────────────────────

/** The months in one page of the bucket listing (S3 ListObjectsV2), sorted, and the next page's token. */
export function parseListing(xml) {
  const months = new Set();
  const re = /<Key>hydrosos\/cogs\/(\d{4}-\d{2})\.tif<\/Key>/g;
  let m;
  while ((m = re.exec(String(xml || "")))) months.add(m[1]);
  const token = /<NextContinuationToken>([^<]+)<\/NextContinuationToken>/.exec(String(xml || ""));
  return { months: [...months].sort(), token: token ? token[1] : null };
}

const nextMonth = (ym) => {
  let y = +ym.slice(0, 4), m = +ym.slice(5, 7) + 1;
  if (m > 12) { y += 1; m = 1; }
  return `${y}-${String(m).padStart(2, "0")}`;
};

/** The months between the first and the last that have no map (2026-03 when this was written). */
export function missingMonths(months) {
  if (!months || !months.length) return [];
  const have = new Set(months);
  const out = [];
  for (let ym = months[0]; ym <= months[months.length - 1]; ym = nextMonth(ym)) if (!have.has(ym)) out.push(ym);
  return out;
}

/** The months as runs in the time bar's `periods` form ("1990-01-01/2026-02-01/P1M"), so a gap reads as a gap. */
export function monthsToPeriods(months) {
  const out = [];
  let start = null, prev = null;
  for (const ym of months || []) {
    if (start && nextMonth(prev) === ym) { prev = ym; continue; }
    if (start) out.push(`${start}-01/${prev}-01/P1M`);
    start = prev = ym;
  }
  if (start) out.push(`${start}-01/${prev}-01/P1M`);
  return out;
}

/** The month a map date shows, or null when there is no map for it. */
export function statusMonthFor(date, months) {
  const ym = String(date || "").slice(0, 7);
  if (!/^\d{4}-\d{2}$/.test(ym) || !months || !months.length) return null;
  return months.includes(ym) ? ym : null;
}

/** The day the map moves to when it opens on the layer: the middle of the newest month. */
export const latestDay = (months) => (months && months.length ? `${months[months.length - 1]}-15` : null);

/**
 * The layer as the time bar sees it (layers.js datedLayersOn): its first and
 * last month and its gaps, so the bar's notes say when it runs.
 */
export function statusDatedLayer(months) {
  const list = months && months.length ? months : null;
  return {
    id: "status", label: "World river status", time: true, monthly: true,
    since: `${list ? list[0] : STATUS_FIRST}-01`,
    until: list ? `${list[list.length - 1]}-01` : null,
    periods: list ? monthsToPeriods(list) : null,
    attribution: STATUS_CREDIT.attribution, licence: STATUS_CREDIT.licence, credit: "GEOGLOWS, CC BY 4.0",
  };
}

/** "Sep 2026" */
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const monthLabel = (ym) => (/^\d{4}-\d{2}$/.test(ym || "") ? `${MONTHS[+ym.slice(5, 7) - 1]} ${ym.slice(0, 4)}` : "");

// ── drawing ─────────────────────────────────────────────────────────────────

// The canvas is a Web Mercator square; MapLibre stretches it over these corners
// and onto the globe. Mercator stops short of the poles, and so does the map.
export const MERC_MAX_LAT = 85.0511287798066;
export const STATUS_CORNERS = [[-180, MERC_MAX_LAT], [180, MERC_MAX_LAT], [180, -MERC_MAX_LAT], [-180, -MERC_MAX_LAT]];

/** The latitude at the middle of canvas row y of h (Web Mercator, top row north). */
export function mercatorRowLat(y, h) {
  const n = Math.PI * (1 - (2 * (y + 0.5)) / h);
  return (Math.atan(Math.sinh(n)) * 180) / Math.PI;
}

/**
 * The file's red band (srcW x srcH, 90 N to 90 S, 180 W to 180 E) as one class
 * per cell of a w x h Web Mercator canvas: 0 for no data, 1 to 5 for much below
 * to much above normal. Nearest cell, which keeps the basins' edges clean.
 */
export function classGrid(red, srcW, srcH, w, h) {
  const out = new Uint8Array(w * h);
  const cols = new Int32Array(w);
  for (let x = 0; x < w; x++) cols[x] = Math.min(srcW - 1, Math.floor(((x + 0.5) / w) * srcW));
  for (let y = 0; y < h; y++) {
    const lat = mercatorRowLat(y, h);
    const row = Math.min(srcH - 1, Math.max(0, Math.floor(((90 - lat) / 180) * srcH)));
    const base = row * srcW, o = y * w;
    for (let x = 0; x < w; x++) out[o + x] = RED_TO_CLASS[red[base + cols[x]]];
  }
  return out;
}

/** "#a6611a" to [166, 97, 26] */
export function hexRgb(hex) {
  const s = String(hex).replace("#", "");
  return [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
}

// How strongly each class is painted, out of 255. Normal is most of the world
// most of the time, so it is a light wash and the months that are not normal
// stand out; the layer's own opacity slider scales all of them.
export const CLASS_ALPHA = [0, 235, 205, 70, 205, 235];

/**
 * The colours the canvas uses, index 0 to 5, as [r, g, b, a]: the same five as
 * the gauges' "Today vs normal" (now-core.js), so a dot and the basin around
 * it mean the same thing. GEOGLOWS draws the classes red to blue; the legend
 * says so.
 */
export function statusPalette() {
  return [[0, 0, 0, 0], ...STATUS_CLASSES.map((c, i) => [...hexRgb(c.color), CLASS_ALPHA[i + 1]])];
}

/**
 * The palette with only the `focus` classes painted (ids such as "much_above"), the others left off the
 * map: "where are rivers much above normal" answered by the map itself (#561). Empty or null paints all.
 */
export function focusPalette(focus) {
  const base = statusPalette();
  if (!focus || !focus.length) return base;
  return base.map((c, i) => (i === 0 || focus.includes(STATUS_CLASSES[i - 1].id) ? c : [c[0], c[1], c[2], 0]));
}

/** Fill RGBA pixels (a canvas ImageData's data) from a class grid. */
export function paintGrid(grid, rgba, palette = statusPalette()) {
  const p32 = new Uint32Array(palette.length);
  const probe = new Uint8ClampedArray(4);
  const view = new Uint32Array(probe.buffer);
  palette.forEach((c, i) => { probe.set(c); p32[i] = view[0]; });
  const out = new Uint32Array(rgba.buffer, rgba.byteOffset, grid.length);
  for (let i = 0; i < grid.length; i++) out[i] = p32[grid[i]] || 0;
  return rgba;
}

/**
 * Decode one month's file into a class grid, with geotiff.js handed in (the
 * worker loads it; so does the page when it has no worker). Only the red band.
 * With `withRegions`, also the named regions' shares (regionShares), for the
 * caption over the globe: { grid, regions }.
 */
export async function decodeStatus(geotiff, buffer, w, h, { withRegions = false } = {}) {
  const tiff = await geotiff.fromArrayBuffer(buffer);
  const image = await tiff.getImage();
  const [red] = await image.readRasters({ samples: [0] });
  const grid = classGrid(red, image.getWidth(), image.getHeight(), w, h);
  return withRegions ? { grid, regions: regionShares(red, image.getWidth(), image.getHeight()) } : grid;
}

// ── the one-line summary (#543 design pass) ─────────────────────────────────
//
// aquascope/map_layers.py (region_shares, status_headline) is the package side, which the CLI and the MCP tool
// use; a test keeps the regions, the shares and the sentence the same here, made from the file the worker has
// already decoded. The regions are rough named boxes [west, south, east, north], not basins.

export const STATUS_REGIONS = [
  { name: "the Amazon", bbox: [-80, -15, -44, 5] },
  { name: "the La Plata basin", bbox: [-66, -35, -43, -15] },
  { name: "Mexico and Central America", bbox: [-118, 7, -77, 32] },
  { name: "the western US", bbox: [-125, 31, -102, 49] },
  { name: "the eastern US", bbox: [-102, 25, -67, 49] },
  { name: "Canada", bbox: [-141, 49, -52, 70] },
  { name: "Europe", bbox: [-10, 36, 40, 71] },
  { name: "the Sahel", bbox: [-17, 11, 38, 18] },
  { name: "the Congo basin", bbox: [12, -13, 32, 8] },
  { name: "East Africa", bbox: [29, -12, 52, 11] },
  { name: "southern Africa", bbox: [10, -35, 41, -13] },
  { name: "the Middle East", bbox: [34, 12, 63, 42] },
  { name: "Central Asia", bbox: [50, 36, 90, 55] },
  { name: "Siberia", bbox: [60, 50, 180, 75] },
  { name: "South Asia", bbox: [66, 6, 92, 36] },
  { name: "Southeast Asia", bbox: [92, -10, 141, 22] },
  { name: "China", bbox: [98, 22, 123, 45] },
  { name: "Australia", bbox: [112, -44, 154, -10] },
];
export const HEADLINE_SHARE = 0.5;
export const MIN_COVER = 0.2;
const MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
  "October", "November", "December"];
const r3 = (x) => Math.round(x * 1000) / 1000;

/**
 * For each named region, from the file's red band (srcW x srcH, 90 N to 90 S): the share of its mapped area
 * below normal and above normal, and how much of its box is mapped. Area-weighted by latitude.
 */
export function regionShares(red, srcW, srcH) {
  const deg = 180 / srcH;
  const weight = new Float64Array(srcH);
  for (let r = 0; r < srcH; r++) weight[r] = Math.cos(((90 - (r + 0.5) * deg) * Math.PI) / 180);
  return STATUS_REGIONS.map(({ name, bbox: [w, s, e, n] }) => {
    const r0 = Math.max(0, Math.ceil((90 - n) / deg - 0.5)), r1 = Math.min(srcH, Math.floor((90 - s) / deg - 0.5) + 1);
    const c0 = Math.max(0, Math.ceil((w + 180) / deg - 0.5)), c1 = Math.min(srcW, Math.floor((e + 180) / deg - 0.5) + 1);
    const per = new Float64Array(6);
    let box = 0;
    for (let r = r0; r < r1; r++) {
      const counts = new Float64Array(6);
      const base = r * srcW;
      for (let c = c0; c < c1; c++) counts[RED_TO_CLASS[red[base + c]]] += 1;
      for (let k = 0; k < 6; k++) per[k] += counts[k] * weight[r];
      box += weight[r] * (c1 - c0);
    }
    const mapped = per[1] + per[2] + per[3] + per[4] + per[5];
    return {
      name,
      below: mapped ? r3((per[1] + per[2]) / mapped) : 0,
      above: mapped ? r3((per[4] + per[5]) / mapped) : 0,
      cover: box ? r3(mapped / box) : 0,
    };
  });
}

const names = (list) => (list.length === 1 ? list[0] : `${list.slice(0, -1).join(", ")} and ${list[list.length - 1]}`);

/** One line for a month from regionShares: aquascope.map_layers.status_headline, word for word. */
export function statusHeadline(month, regions) {
  const label = `${MONTH_NAMES[+month.slice(5, 7) - 1]} ${month.slice(0, 4)}`;
  const sides = { below: [], above: [] };
  (regions || []).forEach((r, i) => {
    if (r.cover < MIN_COVER) return;
    for (const [side, other] of [["below", "above"], ["above", "below"]]) {
      if (r[side] >= HEADLINE_SHARE && r[side] > r[other]) sides[side].push([-r[side], i, r.name]);
    }
  });
  const byShare = (a, b) => a[0] - b[0] || a[1] - b[1];
  if (!sides.below.length && !sides.above.length) return `River status, ${label}: no large region mostly above or below normal`;
  const first = sides.below.length >= sides.above.length ? "below" : "above";
  const second = first === "below" ? "above" : "below";
  const parts = [`much of ${names(sides[first].sort(byShare).slice(0, 2).map((x) => x[2]))} ${first} normal`];
  if (sides[second].length) parts.push(`much of ${sides[second].sort(byShare)[0][2]} ${second}`);
  return `River status, ${label}: ${parts.join(", ")}`;
}

/**
 * A class grid as a PNG, painted in the status colours, on whatever canvas the
 * caller can make: an OffscreenCanvas in the worker, a page canvas otherwise.
 * The map shows it as an image source, which draws the same on the globe and
 * flat; a few hundred kB a month, so several months fit in memory at once.
 */
export async function gridToPng(grid, w, h, makeCanvas, palette = statusPalette()) {
  const canvas = makeCanvas(w, h);
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(w, h);
  paintGrid(grid, image.data, palette);
  ctx.putImageData(image, 0, 0);
  if (canvas.convertToBlob) return canvas.convertToBlob({ type: "image/png" });
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("no PNG"))), "image/png"));
}
