// Groundwater on the map (#558), the pure part: no DOM and no map, so node tests it
// (explorer/tests/groundwater-core.test.mjs). The numbers come from Python
// (aquascope.groundwater.surface, in the worker); this file decides how they are
// coloured, painted and said.

export const GW_CREDIT = {
  label: "Groundwater",
  attribution: "Environment Agency groundwater levels (England), mirrored by the AquaScope Archive " +
    "(obs/groundwater_level/uk_ea), interpolated between wells in your browser",
  licence: "Open Government Licence v3.0",
};
export const GW_SHORT_CREDIT = "Environment Agency well levels, OGL v3.0, via the AquaScope Archive. Interpolated (inverse distance), not a model.";
// The months the Archive's wells cover well enough to draw: over 200 wells a year from 1986.
export const GW_SINCE = "1986-01-01";
// The wells' extent (England), with room for the 20 km mask: the view has to touch it for a surface.
export const GW_EXTENT = [-6.2, 49.8, 2.0, 55.9];
export const GW_SOURCE = "uk_ea";

/** Whether a view [w, s, e, n] touches the wells' extent. */
export function viewTouches(view, extent = GW_EXTENT) {
  return Boolean(view) && view[0] < extent[2] && view[2] > extent[0] && view[1] < extent[3] && view[3] > extent[1];
}

/** The months of readings a surface needs: from 26 months before the month (12 for the fallback to a covered
 * month, 12 more for the change, and the nearest-reading margin) to a month after it. */
export function readingsWindow(month) {
  const [y, m] = month.split("-").map(Number);
  const at = (k) => { const t = y * 12 + (m - 1) + k; return `${Math.floor(t / 12)}-${String((t % 12) + 1).padStart(2, "0")}-01`; };
  return { from: at(-26), to: at(2) };
}

/** The map date as a month, YYYY-MM. */
export const monthOf = (date) => String(date || "").slice(0, 7);

/** Reads gw=1 (or 0) from a hash; null when the link does not say. */
export function readGwParam(hash) {
  const q = new URLSearchParams(String(hash || "").replace(/^#/, ""));
  if (!q.has("gw")) return null;
  return q.get("gw") === "1";
}

// Head: a calm sequential ramp, pale where low, deep blue where high (ColorBrewer YlGnBu).
export const HEAD_RAMP = ["#ffffd9", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"];
// Change: diverging, brown where the level fell, blue-green where it rose (ColorBrewer BrBG).
export const CHANGE_RAMP = ["#8c510a", "#d8b365", "#f6e8c3", "#f5f5f5", "#c7eae5", "#5ab4ac", "#01665e"];
export const MASK_GREY = [120, 128, 136];
export const SURFACE_ALPHA = 205;

const hex = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];

/** The colour at t (0 to 1) along a ramp of hex stops, as [r, g, b]. */
export function rampColor(ramp, t) {
  const stops = ramp.map(hex);
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
}

export const rampFor = (mode) => (mode === "change" ? CHANGE_RAMP : HEAD_RAMP);
export const rampCss = (mode) => `linear-gradient(90deg,${rampFor(mode).join(",")})`;

/** Where a value sits on the colour range: 0 to 1. */
export function rangeT(v, range) {
  if (!range || !Number.isFinite(Number(v))) return 0.5;
  const [lo, hi] = range;
  return hi > lo ? (Number(v) - lo) / (hi - lo) : 0.5;
}

export function valueColor(v, range, mode) {
  return rampColor(rampFor(mode), rangeT(v, range));
}

export const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

/**
 * The surface as RGBA pixels, one per grid cell (north row first), for an image on the map that MapLibre
 * smooths: drawn cells in the ramp, everything else transparent.
 */
export function paintSurface(grid, range, mode) {
  const n = grid.nx * grid.ny;
  const px = new Uint8ClampedArray(n * 4);
  const ramp = rampFor(mode);
  for (let k = 0; k < n; k++) {
    const v = grid.values[k];
    if (grid.status[k] !== 2 || v === null || v === undefined) continue;
    const t = rangeT(v, range);
    const c = rampColor(ramp, t);
    px[k * 4] = c[0]; px[k * 4 + 1] = c[1]; px[k * 4 + 2] = c[2]; px[k * 4 + 3] = surfaceAlpha(t, mode);
  }
  return px;
}

/**
 * A cell's opacity. The head is one even wash. The change fades toward "about the same": the diverging ramp's
 * middle is near white, and drawn at full strength it hid the map like fog, so rises and falls stand out and
 * the drawn area still shows faintly where nothing changed.
 */
export const CHANGE_ALPHA_MIN = 70;
export function surfaceAlpha(t, mode) {
  if (mode !== "change") return SURFACE_ALPHA;
  const k = Math.min(1, Math.abs((Number.isFinite(t) ? t : 0.5) - 0.5) * 2);
  return Math.round(CHANGE_ALPHA_MIN + (SURFACE_ALPHA - CHANGE_ALPHA_MIN) * Math.sqrt(k));
}

/**
 * The density mask as RGBA pixels at `scale` pixels per cell: thin grey hatching where only one or two wells
 * are near (status 1), nothing elsewhere. Hatching says "we looked, and there is too little to draw" without
 * reading as data.
 */
export function paintMask(grid, scale = 4) {
  const w = grid.nx * scale, h = grid.ny * scale;
  const px = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const row = Math.floor(y / scale) * grid.nx;
    for (let x = 0; x < w; x++) {
      if (grid.status[row + Math.floor(x / scale)] !== 1) continue;
      const k = (y * w + x) * 4;
      const line = (x + y) % 7 === 0;
      px[k] = MASK_GREY[0]; px[k + 1] = MASK_GREY[1]; px[k + 2] = MASK_GREY[2];
      px[k + 3] = line ? 150 : 22;
    }
  }
  return { px, width: w, height: h };
}

/** The pixel scale for the mask: about 1,600 pixels on the long side, at least 2 per cell. */
export function maskScale(grid) {
  return Math.max(2, Math.min(8, Math.floor(1600 / Math.max(grid.nx, grid.ny, 1))));
}

// Web Mercator, the projection the surface grid is regular in.
const R = 6378137;
const mercY = (lat) => R * Math.log(Math.tan(Math.PI / 4 + (Math.max(-85.05, Math.min(85.05, lat)) * Math.PI) / 360));

/** The image's corners for MapLibre: top-left, top-right, bottom-right, bottom-left. */
export const gridCorners = (g) => [[g.west, g.north], [g.east, g.north], [g.east, g.south], [g.west, g.south]];

/** The grid cell under a point: { i, j, k, value, status } or null outside the grid. */
export function gridCellAt(grid, lon, lat) {
  if (!grid || lon < grid.west || lon > grid.east || lat < grid.south || lat > grid.north) return null;
  const j = Math.min(grid.nx - 1, Math.floor(((lon - grid.west) / (grid.east - grid.west)) * grid.nx));
  const yN = mercY(grid.north), yS = mercY(grid.south);
  const i = Math.min(grid.ny - 1, Math.floor(((yN - mercY(lat)) / (yN - yS)) * grid.ny));
  const k = i * grid.nx + j;
  return { i, j, k, value: grid.values[k], status: grid.status[k] };
}

const km = (lon1, lat1, lon2, lat2) => {
  const d = Math.PI / 180;
  const a = Math.sin(((lat2 - lat1) * d) / 2) ** 2 +
    Math.cos(lat1 * d) * Math.cos(lat2 * d) * Math.sin(((lon2 - lon1) * d) / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(a));
};

/** The wells within `radius` km of a point, nearest first, each with its distance. */
export function wellsNear(wells, lon, lat, radius) {
  return (wells || []).map((w) => ({ ...w, km: km(lon, lat, w.lon, w.lat) }))
    .filter((w) => w.km <= radius).sort((a, b) => a.km - b.km);
}

export function fmtM(v, { sign = false } = {}) {
  if (v === null || v === undefined || !Number.isFinite(Number(v))) return "no value";
  const n = Number(v);
  if (Number.isInteger(n) && Math.abs(n) >= 1) return sign && n > 0 ? `+${n}` : String(n);
  const s = Math.abs(n) >= 100 ? n.toFixed(0) : Math.abs(n) >= 10 ? n.toFixed(1) : n.toFixed(2);
  return sign && n > 0 ? `+${s}` : s;
}

/** What a change is called: rising, falling or about the same (within 5 cm). */
export function changeWord(v) {
  if (!Number.isFinite(Number(v))) return "";
  if (v > 0.05) return "rising";
  if (v < -0.05) return "falling";
  return "about the same";
}

/** The legend row's few words. */
export function rowSummary(res, { busy = false, error = null } = {}) {
  if (busy) return "reading…";
  if (error) return "could not draw";
  if (!res) return "England's wells";
  if (res.error) return "nothing to draw";
  if (!res.grid) return "too few wells here";
  return res.mode === "change" ? `change, ${res.label}` : res.label;
}

/** The density mask in a few words, for the key's hatched swatch. */
export function maskLine(res) {
  return `Hatched: fewer than ${res.min_wells} wells within ${res.radius_km} km, too few to draw.`;
}

/** The key's lines: what the colours are, the range, the rule, any note. */
export function keyLines(res) {
  if (!res) return { head: "", range: null, lines: [] };
  if (res.error) return { head: res.error, range: null, lines: [] };
  const lines = [];
  if (res.notes && res.notes.length) lines.push(...res.notes);
  if (!res.grid) lines.push(res.summary);
  const head = res.mode === "change"
    ? `Change in level, ${res.label}`
    : `Head, ${res.label}, m above Ordnance Datum`;
  const range = res.range ? { lo: fmtM(res.range[0], { sign: res.mode === "change" }), hi: fmtM(res.range[1], { sign: res.mode === "change" }) } : null;
  if (res.grid) {
    const n = res.n_wells_in_view;
    const step = res.contours && res.contours.step ? ` · lines every ${fmtM(res.contours.step)} m` : "";
    lines.push(`${n} well${n === 1 ? "" : "s"} in view${res.mode === "change" ? " with both months" : ""}${step}`);
  }
  return { head, range, lines, mask: maskLine(res) };
}

/**
 * The map card for a point on the surface: { title, sub, status, figure, note }. `cell` from gridCellAt,
 * `near` from wellsNear (within the mask radius).
 */
export function surfaceCard(res, cell, near) {
  const change = res.mode === "change";
  const what = change ? "Groundwater change" : "Groundwater head";
  const n = near.length;
  const nearest = near[0];
  const nearLine = nearest ? `Nearest well: ${nearest.name || nearest.id}, ${nearest.km.toFixed(1)} km, ` +
    `${fmtM(nearest.value, { sign: change })} m${change ? "" : " AOD"}.` : "";
  if (!cell || cell.status !== 2) {
    return {
      what, title: n ? "Too few wells here" : "No well near",
      sub: res.label,
      status: { text: n ? `${n} well${n === 1 ? "" : "s"} within ${res.radius_km} km; the surface needs ${res.min_wells}.`
        : `No well within ${res.radius_km} km, so nothing is drawn.`, color: null },
      figure: null, note: nearLine,
    };
  }
  const v = cell.value;
  return {
    what,
    title: change ? `${fmtM(v, { sign: true })} m, ${changeWord(v)}` : `${fmtM(v)} m above Ordnance Datum`,
    sub: `${res.label} · interpolated from ${n} well${n === 1 ? "" : "s"} within ${res.radius_km} km`,
    status: { text: change ? `Change since ${res.label.split(" to ")[0]}, interpolated between wells.`
      : "Head, interpolated between wells: a picture of the measurements, not a model.", color: null },
    figure: { value: fmtM(v, { sign: change }), unit: change ? "m" : "m AOD", label: change ? "change" : "head" },
    note: nearLine,
  };
}

/** The map card for one well: its value this month and how it was chosen. */
export function wellCard(res, w) {
  const change = res.mode === "change";
  const how = w.how === "nearest" ? `the reading of ${w.date}, nearest the middle of the month` : "the mean of the month's readings";
  return {
    what: "Groundwater well",
    title: w.name || w.id,
    sub: `Environment Agency · ${w.lat.toFixed(3)}, ${w.lon.toFixed(3)}`,
    status: { text: change ? `${fmtM(w.level_before)} to ${fmtM(w.level)} m AOD, ${changeWord(w.value)}.`
      : `Head ${fmtM(w.value)} m above Ordnance Datum in ${res.label}: ${how}.`, color: null },
    figure: { value: fmtM(w.value, { sign: change }), unit: change ? "m" : "m AOD", label: change ? "change" : "head" },
  };
}

// ── your MODFLOW heads ──────────────────────────────────────────────────────

/** Which dropped files are which: { hds, grid, sidecars, other }. A head file is told by its extension. */
export function sortModelFiles(files) {
  const out = { hds: null, grid: null, sidecars: [], wel: null, cbc: null, other: [] };
  for (const f of files || []) {
    const name = String(f.name || "").toLowerCase();
    if (/\.(hds|hed|bhd|head|ddn)$/.test(name)) out.hds = out.hds || f;
    else if (/\.(disv?\.grb|grb|disv?)$/.test(name)) out.grid = out.grid || f;
    else if (/\.(nam|tdis)$/.test(name)) out.sidecars.push(f);
    else if (/\.wel6?$/.test(name)) out.wel = out.wel || f;   // the wells, for the views (#624)
    else if (/\.(cbc|bud|cbb)$/.test(name)) out.cbc = out.cbc || f;   // the cell-by-cell budget (#626)
    else out.other.push(f);
  }
  return out;
}

/** Whether a drop holds MODFLOW files (a head file, a grid, a name file). The map has one drop zone for model
 * results: these go to the MODFLOW reader here, everything else to the mesh viewer (mesh-view.js, #555). */
export const MODFLOW_FILE = /\.(hds|hed|bhd|head|ddn|disv?\.grb|grb|disv?|nam|tdis|wel6?|cbc|bud|cbb)$/i;
/** The extensions the file picker offers for MODFLOW (Tools, Open model results). */
export const MODFLOW_ACCEPT = ".hds,.hed,.bhd,.head,.ddn,.dis,.disv,.grb,.nam,.tdis,.wel,.cbc,.bud,.cbb";
export const isModflowDrop = (files) => [...(files || [])].some((f) => MODFLOW_FILE.test(String((f && f.name) || "")));

/** The georeference form's fields, read and checked: { origin, rotation, crs, cell_size, start_date } or an error. */
export function readPlaceForm(v, { needCell = false } = {}) {
  const num = (x) => (x === "" || x === null || x === undefined ? null : Number(x));
  const out = {};
  const x = num(v.x), y = num(v.y);
  if ((x === null) !== (y === null) || (x !== null && (!Number.isFinite(x) || !Number.isFinite(y)))) {
    return { error: "Give both the x and the y of the lower-left corner." };
  }
  if (x !== null) out.origin = [x, y];
  const rot = num(v.rotation);
  if (rot !== null) {
    if (!Number.isFinite(rot)) return { error: "The rotation is a number of degrees." };
    out.rotation = rot;
  }
  const crs = String(v.crs || "").trim();
  if (crs) {
    if (!/^(epsg:)?\d{4,6}$/i.test(crs) && !/^\+proj=/.test(crs)) return { error: "The CRS is an EPSG code, such as 27700." };
    out.crs = /^\d+$/.test(crs) ? `EPSG:${crs}` : crs.toUpperCase().startsWith("EPSG:") ? crs.toUpperCase() : crs;
  }
  const cell = num(v.cell);
  if (cell !== null) {
    if (!(cell > 0)) return { error: "The cell size is a positive length." };
    out.cell_size = cell;
  } else if (needCell) {
    return { error: "Without a grid file the cell size is needed." };
  }
  const start = String(v.start || "").trim();
  if (start) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(start)) return { error: "The start date is YYYY-MM-DD." };
    out.start_date = start;
  }
  return out;
}

/** GeoJSON polygons of the model cells from flat rings and offsets, each with its index as the feature id. */
export function cellsGeoJSON(rings, offsets) {
  const features = [];
  for (let c = 0; c + 1 < offsets.length; c++) {
    const ring = [];
    for (let k = offsets[c]; k < offsets[c + 1]; k++) ring.push([rings[2 * k], rings[2 * k + 1]]);
    ring.push(ring[0]);
    features.push({ type: "Feature", id: c, properties: { c }, geometry: { type: "Polygon", coordinates: [ring] } });
  }
  return { type: "FeatureCollection", features };
}

/** The outer corners of a structured grid as a closed line, from the cell rings (4 corners per cell, row-major,
 * each ring top-left, top-right, bottom-right, bottom-left); null for a DISV grid. */
export function modelOutline(grid, rings) {
  if (!grid || grid.type !== "dis") return null;
  const nr = grid.drawn_nrow, nc = grid.drawn_ncol;
  const at = (cell, corner) => [rings[2 * (cell * 4 + corner)], rings[2 * (cell * 4 + corner) + 1]];
  const tl = at(0, 0), tr = at(nc - 1, 1), br = at(nr * nc - 1, 2), bl = at((nr - 1) * nc, 3);
  return [tl, tr, br, bl, tl];
}

/** One frame's heads (time t, layer index l) out of the flat Float32Array (time x layer x cell). */
export function frameOf(heads, shape, t, l) {
  const [, nl, nc] = shape;
  const start = (t * nl + l) * nc;
  return heads.subarray(start, start + nc);
}

/** The time step to show for a map date: the last model time on or before it (the first when before them all). */
export function stepForDate(dates, date) {
  if (!dates || !dates.length || !date) return 0;
  let best = 0;
  for (let i = 0; i < dates.length; i++) if (String(dates[i]).slice(0, 10) <= date) best = i;
  return best;
}

/** The model's length unit as the card and the key write it: "m", "ft", "cm", or "" when the grid does not say
 * (heads are shown in the model's own unit, never converted; the key's note says when it is not named). */
export function lengthUnit(meta) {
  const u = String((meta && meta.grid && meta.grid.length_units) || "").toLowerCase();
  if (["meters", "metres", "m"].includes(u)) return "m";
  if (["feet", "ft"].includes(u)) return "ft";
  if (["centimeters", "cm"].includes(u)) return "cm";
  return "";
}

/** "Drawdown" for a drawdown file, else "Head", from the record text MODFLOW writes. */
export const headWord = (meta) => (/DRAWDOWN/i.test(String((meta && meta.text) || "")) ? "Drawdown" : "Head");

/** How a model time reads: its date when the model has a start date, else "t = 31 days". */
export function timeLabel(model, t) {
  if (model.dates && model.dates[t]) return model.dates[t];
  const v = model.times[t];
  return `t = ${Number.isInteger(v) ? v : Number(v).toPrecision(4)} ${model.time_units || "days"}`;
}

// ── a model with no georeference: placed approximately, never blocked ──────────────────────────────────

/** The label a model placed at the map's centre carries on its legend row and card. */
export const APPROX_LABEL = "approximate placement";

/** Whether the model on the map was placed approximately (its files did not say where it is). */
export const isApproximate = (meta) => Boolean(meta && meta.grid && meta.grid.approximate);

/** The one-click choice, read and checked: { approximate_at: [lon, lat], cell_size? } or an error. A model with no
 * grid file needs its cell size even to be drawn approximately (the head file gives only rows and columns). */
export function readApproxChoice(centre, v = {}, { needCell = false } = {}) {
  const lon = Array.isArray(centre) ? Number(centre[0]) : NaN, lat = Array.isArray(centre) ? Number(centre[1]) : NaN;
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return { error: "The map has no centre yet." };
  // the reader takes latitudes within 85 degrees, like the map itself; a globe turned to a pole still places the model
  const out = { approximate_at: [Math.round(lon * 1e6) / 1e6, Math.max(-84, Math.min(84, Math.round(lat * 1e6) / 1e6))] };
  const raw = String(v.cell ?? "").trim();
  if (raw) {
    const cell = Number(raw);
    if (!(cell > 0)) return { error: "The cell size is a positive length." };
    out.cell_size = cell;
  } else if (needCell) {
    return { error: "Without a grid file the cell size is needed, in the model's length unit." };
  }
  return out;
}

/** What the approximate choice says about the model's size: in its own length unit, or read as metres. */
export function approxWords(out) {
  const u = String((out && out.length_units) || "").toLowerCase();
  const unit = ["meters", "metres", "m"].includes(u) ? "metres" : ["feet", "ft"].includes(u) ? "feet"
    : ["centimeters", "cm"].includes(u) ? "centimetres" : "";
  return unit ? `North up, at its true size in ${unit}.` : "North up, at its true size, reading its lengths as metres (its files do not name a unit).";
}
