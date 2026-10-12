// What if, on the map (#565), the pure part: no DOM, no map, so node can test it.
// Every number here comes from Python (aquascope.whatif_map.whatif: GR4J for the basin, the change carried
// downstream by area weighting); this module only decides how it is coloured, worded and split by the swipe.

// The change, in classes of percent: drier is brown, wetter is teal (ColorBrewer BrBG, safe for the common
// kinds of colour blindness), and within ±3 % a quiet grey. "Before" is the rivers' own blue (BEFORE).
export const NEUTRAL = "#9aa7b2";
// Left of the before/after swipe the rivers are drawn as they are modelled today: the network's own blue.
export const BEFORE = "#3478bd";
export const CLASSES = [
  { max: -50, color: "#8c510a", label: "−50% or less" },
  { max: -25, color: "#bf812d", label: "−50 to −25%" },
  { max: -10, color: "#dfc27d", label: "−25 to −10%" },
  { max: -3, color: "#f0d9a6", label: "−10 to −3%" },
  { max: 3, color: NEUTRAL, label: "within ±3%" },
  { max: 10, color: "#a6dbd2", label: "+3 to +10%" },
  { max: 25, color: "#80cdc1", label: "+10 to +25%" },
  { max: 50, color: "#35978f", label: "+25 to +50%" },
  { max: Infinity, color: "#01665e", label: "+50% or more" },
];

// What the reach colours stand for: the change in one of three flows.
export const MEASURES = {
  mean: { key: "mean_change_pct", range: "mean_range", label: "mean flow", outlet: "mean" },
  low: { key: "q95_change_pct", range: "q95_range", label: "low flow (Q95)", outlet: "low" },
  flood: { key: "flood_change_pct", range: "flood_range", label: "flood", outlet: "flood" },
};

// The sliders' ends (aquascope.whatif_map.LIMITS holds the package's own; these are the useful part of them).
export const SLIDERS = {
  dp_pct: { min: -50, max: 50, step: 5, unit: "%" },
  dt_c: { min: -2, max: 6, step: 0.5, unit: "°C" },
  demand_cms: { min: 0, max: null, step: null, unit: "m³/s" },
};

const finite = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));

/** A signed percentage with a true minus sign: "−12%", "+4%", "0%". */
export function pct(v, digits = 0) {
  if (!finite(v)) return "n/a";
  const n = Number(Number(v).toFixed(digits));
  if (n === 0) return "0%";
  return `${n < 0 ? "−" : "+"}${Math.abs(n).toFixed(digits)}%`;
}

/** "−15 to −9%" from a range, or "" without one. */
export function rangeWords(r, digits = 0) {
  if (!Array.isArray(r) || r.length !== 2 || !finite(r[0]) || !finite(r[1])) return "";
  const lo = pct(r[0], digits), hi = pct(r[1], digits);
  if (lo === hi) return "";   // a range that rounds to one number says nothing more than the number
  return `${lo.replace("%", "")} to ${hi}`;
}

/** The class a change falls in (its index in CLASSES). */
export function classOf(v) {
  if (!finite(v)) return 4;
  const n = Number(v);
  for (let i = 0; i < CLASSES.length; i++) if (n <= CLASSES[i].max) return i;
  return CLASSES.length - 1;
}

export const colorOf = (v) => CLASSES[classOf(v)].color;

/**
 * The line colour for the stream tiles: each painted reach in its class colour, the ones left of the swipe
 * (`before`, a Set of river ids) in the rivers' own blue (BEFORE), everything else transparent (the filter hides it anyway).
 */
export function lineColorExpr(reaches, measure = "mean", before = null) {
  const key = (MEASURES[measure] || MEASURES.mean).key;
  const groups = new Map();
  for (const r of reaches || []) {
    const id = Number(r.river_id);
    const color = before && before.has(id) ? BEFORE : colorOf(r[key]);
    if (!groups.has(color)) groups.set(color, []);
    groups.get(color).push(id);
  }
  const branches = [];
  for (const [color, ids] of groups) branches.push(ids, color);
  return branches.length ? ["match", ["to-number", ["get", "riverId"]], ...branches, "rgba(0,0,0,0)"] : "rgba(0,0,0,0)";
}

/** Only these reaches of the stream tiles; nothing at all when there are none. */
export function lineFilterExpr(ids) {
  const list = [...new Set((ids || []).map(Number).filter(Number.isFinite))];
  return list.length ? ["match", ["to-number", ["get", "riverId"]], list, true, false] : ["boolean", false];
}

/**
 * The reaches of the basin itself, every one with the outlet's change (the model is lumped: one change for the
 * whole basin), as rows lineColorExpr can paint.
 */
export function basinRows(result) {
  const ids = (result && result.basin && result.basin.ids) || [];
  const first = (result && result.reaches && result.reaches[0]) || {};
  const outlet = Number(first.river_id);
  return ids.filter((id) => Number(id) !== outlet).map((id) => ({
    river_id: Number(id), mean_change_pct: first.mean_change_pct, q95_change_pct: first.q95_change_pct,
    flood_change_pct: first.flood_change_pct,
  }));
}

/**
 * Which painted reaches lie left of the swipe: `where` maps a river id to its longitude (from the drawn line,
 * else the reach's own lon); `split` is the longitude under the handle. Returns a Set of river ids.
 */
export function beforeSet(ids, where, split) {
  const out = new Set();
  if (!finite(split)) return out;
  for (const id of ids || []) {
    const lon = where(Number(id));
    if (finite(lon) && Number(lon) < Number(split)) out.add(Number(id));
  }
  return out;
}

/** The scenario from the sliders, in the words the package uses (aquascope.whatif_map.scenario_label). */
export function scenarioWords({ dp_pct: dp = 0, dt_c: dt = 0, demand_cms: dm = 0 } = {}) {
  const g = (v) => String(Number(Math.abs(Number(v)).toFixed(1)));
  const bits = [];
  if (Number(dp)) bits.push(`${g(dp)}% ${Number(dp) < 0 ? "less" : "more"} rain`);
  if (Number(dt)) bits.push(`${g(dt)} °C ${Number(dt) < 0 ? "cooler" : "warmer"}`);
  if (Number(dm)) bits.push(`${g(dm)} m³/s taken`);
  return bits.join(", ") || "no change";
}

/** A demand slider that fits the river: up to about half its mean flow, in round steps. */
export function demandScale(meanFlow) {
  const q = finite(meanFlow) && Number(meanFlow) > 0 ? Number(meanFlow) : 20;
  const top = q / 2;
  const mag = 10 ** Math.floor(Math.log10(top));
  const max = Math.max(1, Math.ceil(top / mag) * mag);
  const step = max <= 2 ? 0.1 : max <= 20 ? 0.5 : max <= 200 ? 5 : 50;
  return { max, step };
}

/** The values the sliders take from a checked scenario, clamped to their ends. */
export function sliderValues(sc, demandMax = null) {
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, Number(v) || 0));
  return {
    dp_pct: clamp(sc && sc.dp_pct, SLIDERS.dp_pct.min, SLIDERS.dp_pct.max),
    dt_c: clamp(sc && sc.dt_c, SLIDERS.dt_c.min, SLIDERS.dt_c.max),
    demand_cms: clamp(sc && sc.demand_cms, 0, finite(demandMax) ? Number(demandMax) : Infinity),
  };
}

/** The result's three flows at the outlet as short lines: "Mean flow −38% (−39 to −37%)". */
export function outletLines(result) {
  const o = (result && result.outlet) || {};
  return ["mean", "low", "flood"].map((k) => {
    const row = o[k] || {};
    const r = rangeWords(row.range);
    // The panel is small: "low flow (Q95)", the full definition is in the package's label and the CLI.
    const label = k === "low" ? MEASURES.low.label : (row.label || MEASURES[k].label);
    return { key: k, label, change: row.change_pct, text: `${pct(row.change_pct)}${r ? ` (${r})` : ""}`,
      color: colorOf(row.change_pct) };
  });
}

/** What a painted reach's map card says: the three changes, how much of the water is the basin's, the confluence. */
export function reachCard(result, rid) {
  const reaches = (result && result.reaches) || [];
  const id = Number(rid);
  let r = reaches.find((x) => Number(x.river_id) === id);
  let inBasin = false;
  if (!r) {
    const ids = ((result && result.basin && result.basin.ids) || []).map(Number);
    if (!ids.includes(id)) return null;
    r = { ...reaches[0], river_id: id, share: 1, confluence: null };
    inBasin = true;
  }
  const outletId = reaches.length ? Number(reaches[0].river_id) : null;
  const flood = result.outlet && result.outlet.flood ? result.outlet.flood.label : "flood";
  const withRange = (v, rg) => `${pct(v)}${rangeWords(rg) ? ` (${rangeWords(rg)})` : ""}`;
  const lines = [`Low flow (Q95) ${withRange(r.q95_change_pct, r.q95_range)}`,
    `${flood[0].toUpperCase()}${flood.slice(1)} ${withRange(r.flood_change_pct, r.flood_range)}`];
  let where = "";
  if (inBasin) where = " In the basin modelled: GR4J is lumped, one change for the whole basin.";
  else if (r.confluence) where = ` Major confluence ${r.confluence.n}: ${Math.round(r.confluence.tributary_km2).toLocaleString("en-US")} km² join here.`;
  const share = Math.round(Number(r.share) * 100);
  return {
    title: `River reach ${id}`,
    status: `Mean flow ${withRange(r.mean_change_pct, r.mean_range)}`,
    color: colorOf(r.mean_change_pct),
    figure: inBasin || id === outletId
      ? { value: Math.round(r.area_km2).toLocaleString("en-US"), unit: " km²", label: id === outletId ? "drain to the outlet" : "basin modelled" }
      : { value: String(share), unit: "%", label: "of the water is the basin's" },
    note: `${lines.join(" · ")}.${where}`,
  };
}

/** The log line for a run: "What if 20% less rain, 2 °C warmer at river reach 230267171". */
export function logLabel(result) {
  const sc = (result && result.scenario) || {};
  const rid = result && result.reach ? result.reach.river_id : "";
  return `What if ${sc.label || "?"}${rid ? ` at river reach ${rid}` : ""}`;
}

/** The ids the map paints for a result: the way down, then the basin. */
export function paintedIds(result) {
  const down = ((result && result.reaches) || []).map((r) => Number(r.river_id));
  return [...down, ...basinRows(result).map((r) => r.river_id)];
}

/**
 * What to frame after a run: the basin (its box from the routing tables) and the river down to the first major
 * confluence, where the change is still large enough to see; the rest of the way is a pan away.
 */
export function resultBounds(result) {
  const pts = [];
  const reaches = (result && result.reaches) || [];
  const first = reaches.findIndex((r) => r.confluence);
  const near = first >= 0 ? reaches.slice(0, first + 1) : reaches;
  for (const r of near) if (finite(r.lon) && finite(r.lat)) pts.push([Number(r.lon), Number(r.lat)]);
  const bb = result && result.basin && result.basin.bbox;
  if (Array.isArray(bb) && bb.length === 4 && bb.every(finite)) pts.push([bb[0], bb[1]], [bb[2], bb[3]]);
  const c = result && result.basin && result.basin.centre;
  if (Array.isArray(c) && finite(c[0]) && finite(c[1])) pts.push([Number(c[1]), Number(c[0])]);
  const at = result && result.reach;
  if (at && finite(at.lon) && finite(at.lat)) pts.push([Number(at.lon), Number(at.lat)]);
  if (!pts.length) return null;
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (const [x, y] of pts) { w = Math.min(w, x); e = Math.max(e, x); s = Math.min(s, y); n = Math.max(n, y); }
  const pad = Math.max(0.05, 0.12 * Math.max(e - w, n - s));
  return [w - pad, s - pad, e + pad, n + pad];
}
