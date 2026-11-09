// Draw a reservoir (#662), the pure part: no DOM, no map, so node can test it. The storage, the spillway's
// outflow and the routing come from Python (aquascope.sketch.reservoir: level-pool routing inside Route it, the
// storage from the terrain inside the area drawn). This module checks the inputs, draws the lake and the dam, picks
// the flood from what is on the map, and says the numbers the package wrote.

import { flowWords, hoursWords } from "./route-core.js?v=__BUILD__";
import { checkRing } from "./storm-core.js?v=__BUILD__";

/** What the panel starts with: illustrative, for the reader to set. */
export const RESERVOIR_START = { crest_m: 10, top_m: 15, width_m: 30, return_period: 100, duration_h: 24 };

const finite = (v) => v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v));

/**
 * The worker's arguments, checked: the dam's levels and spillway, and the flood: { kind: "rp" | "peak" | "wave" },
 * with the wave (`wave`: Route it's or a storm's run on the map) giving where the flood comes from.
 */
export function reservoirArgs({ crest_m, top_m, width_m, kind = "rp", return_period, peak_cms, duration_h, wave = null }) {
  const c = Number(crest_m), t = Number(top_m), w = Number(width_m);
  if (!(c > 0 && c <= 300)) throw new Error("the crest must be above the lowest ground and at most 300 m above it");
  if (!(t > c && t <= 320)) throw new Error("the top of the dam must be above the crest");
  if (!(w > 0 && w <= 2000)) throw new Error("the spillway's width must be above 0 and at most 2,000 m");
  const out = { crest_m: c, top_m: t, width_m: w };
  if (kind === "wave") {
    const f = waveFlood(wave);
    if (!f) throw new Error("there is no flood wave or storm on the map to send through the reservoir");
    return { ...out, ...f };
  }
  const d = Number(duration_h);
  if (!(d > 0 && d <= 720)) throw new Error("the duration must be between 0 and 720 hours");
  out.duration_h = d;
  if (kind === "peak") {
    const q = Number(peak_cms);
    if (!(q > 0 && q <= 1e6)) throw new Error("give the peak flow in m³/s");
    out.peak_cms = q;
  } else {
    const rp = Number(return_period);
    if (!(rp >= 2 && rp <= 500)) throw new Error("the return period must be between 2 and 500 years");
    out.return_period = rp;
  }
  return out;
}

/**
 * The flood a wave on the map stands for, as route_reservoir takes it: a storm (its area and rain), or Route it's
 * flood from its start reach (its peak, or its T-year flow, over its hours). Null when there is none, or when the
 * wave is a replay's hydrograph (#659), which is not sent again.
 */
export function waveFlood(wave) {
  const r = wave && wave.result;
  if (!r || !r.ok) return null;
  // A run through a reservoir already: the flood that made it (the package keeps it), not its outflow.
  if (r.reservoir) return r.reservoir.flood ? { ...r.reservoir.flood } : null;
  if (r.storm && r.polygon) {
    return { storm: { polygon: r.polygon, rain_mm: r.storm.rain_mm, duration_h: r.storm.duration_h,
      runoff_coefficient: r.storm.runoff_coefficient } };
  }
  const i = r.inflow;
  if (!i || i.shape === "series" || !finite(r.river_id)) return null;
  const base = { from_river_id: Number(r.river_id), duration_h: Number(i.duration_h) };
  return finite(i.return_period) ? { ...base, return_period: Number(i.return_period) } : { ...base, peak_cms: Number(i.peak_cms) };
}

/** "the 100-year flow", "800 m³/s over 24 h from reach 230...", "the storm of 50 mm in 6 h": the wave's words. */
export function waveWords(wave) {
  const f = waveFlood(wave);
  if (!f) return "";
  if (f.storm) return `the storm on the map (${Number(f.storm.rain_mm)} mm in ${hoursWords(f.storm.duration_h)})`;
  const what = finite(f.return_period) ? `the ${f.return_period}-year flood` : `${flowWords(f.peak_cms)} m³/s`;
  return finite(f.from_river_id) ? `${what} routed on the map, from reach ${f.from_river_id}` : `${what} at the dam, as before`;
}

/** The lake and the dam on the map. */
export function lakeFeatures(lake, { live = false } = {}) {
  const feats = [];
  if (lake && lake.polygon && lake.polygon.length >= 3) {
    const ring = [...checkRing(lake.polygon), checkRing(lake.polygon)[0]];
    feats.push({ type: "Feature", properties: { kind: "lake", live: live ? 1 : 0 },
      geometry: { type: "Polygon", coordinates: [ring] } });
  }
  const d = lake && lake.dam;
  if (d && finite(d.lon) && finite(d.lat)) {
    feats.push({ type: "Feature", properties: { kind: "dam", label: "Dam" },
      geometry: { type: "Point", coordinates: [Number(d.lon), Number(d.lat)] } });
  }
  return { type: "FeatureCollection", features: feats };
}

/** The card's two hydrographs at the dam, for storm-core's hydroChart: inflow and outflow, one scale. */
export function damHydrographs(res) {
  const r = res && res.reservoir;
  if (!r) return [];
  return [
    { role: "inflow", cms: r.inflow_cms, dt_h: r.dt_h, peak_cms: r.peak_in_cms, peak_h: r.peak_in_h },
    { role: "outflow", cms: r.outflow_cms, dt_h: r.dt_h, peak_cms: r.peak_out_cms, peak_h: r.peak_out_h },
  ];
}

/** The result's first line, from the package's numbers. */
export function reservoirHeadline(res) {
  const r = res && res.reservoir;
  if (!r) return "";
  return `The reservoir takes the flood's peak at the dam from ${flowWords(r.peak_in_cms)} to ${flowWords(r.peak_out_cms)} m³/s, ` +
    `${Number(r.reduction_pct)} % lower and ${hoursWords(r.delay_h)} later.`;
}

/** The card's numbers: [{ label, value }], each written by the package. */
export function reservoirFacts(res) {
  const r = res && res.reservoir;
  if (!r) return [];
  const rows = [
    { label: "The flood", value: r.what },
    { label: "Highest water", value: `${r.max_depth_m} m above the lowest ground${r.overtopped ? ", over the dam" : ""}` },
    { label: "Held at most", value: volumeM3(r.flood_storage_m3) + " above the crest" },
    { label: "The dam", value: `crest ${r.crest_m} m, top ${r.top_m} m, spillway ${r.width_m} m wide` },
  ];
  const w = r.without || {};
  if (finite(w.furthest_peak_cms) && finite(w.furthest_peak_with_cms)) {
    rows.push({ label: "Furthest down", value: `${flowWords(w.furthest_peak_with_cms)} m³/s, without it ${flowWords(w.furthest_peak_cms)} m³/s` });
  }
  return rows;
}

function volumeM3(v) {
  if (!finite(v)) return "n/a";
  const n = Number(v);
  return n >= 1e6 ? `${(n / 1e6).toFixed(1).replace(/\.0$/, "")} million m³` : `${Math.round(n).toLocaleString("en-US")} m³`;
}

/** The action log's line for the lake drawn or moved. */
export function lakeLogLabel(lake, { moved = false } = {}) {
  return `${moved ? "Move" : "Draw"} a reservoir on reach ${lake.dam ? lake.dam.river_id : "?"}`;
}
