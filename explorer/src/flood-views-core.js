// The flood views' pure part (#621, #622): no DOM, so node can test it (explorer/tests/flood-views-core.test.mjs).
//
// The numbers come from the package (aquascope.flood_section and aquascope.flood_exposure, run in the mesh
// worker); this file only decides how they are drawn and said: the section's and the hydrograph's SVG, the
// words under them, the counts' wording and the map filter that shows a feature at one step.

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const r1 = (x) => Math.round(x * 10) / 10;

/** Round numbers for an axis: about `n` ticks from lo to hi. */
export function niceTicks(lo, hi, n = 5) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (hi <= lo) return [lo];
  const raw = (hi - lo) / Math.max(1, n);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

/** The decimals an axis needs so its ticks read apart: 0.05 steps -> 2, 0.5 -> 1, 2.5 -> 1, 2 -> 0. */
export function tickDigits(ticks) {
  if (!ticks || ticks.length < 2) return 1;
  const step = Math.abs(ticks[1] - ticks[0]);
  if (!(step > 0)) return 1;
  if (step >= 1) return Number.isInteger(Number(step.toFixed(6))) ? 0 : 1;
  const d = Math.ceil(-Math.log10(step) - 1e-9);
  return Math.min(4, Number.isInteger(Number((step * 10 ** d).toFixed(6))) ? d : d + 1);
}

/** A number for an axis or a line: no more digits than it needs. */
export function num(x, digits) {
  if (x === null || x === undefined || !Number.isFinite(Number(x))) return "–";
  const v = Number(x);
  const a = Math.abs(v);
  const d = digits ?? (a >= 1000 ? 0 : a >= 100 ? 1 : 2);
  return Number(v.toFixed(d)).toLocaleString("en-GB", { maximumFractionDigits: d, minimumFractionDigits: 0 });
}

/** A distance: "850 m", "1.67 km". */
export function lengthWords(m) {
  if (!Number.isFinite(m)) return "–";
  return m >= 1000 ? `${num(m / 1000, m >= 10000 ? 1 : 2)} km` : `${num(m, 0)} m`;
}

/** "2026-03-15T14:00:00" -> "14:00", or "15 Mar 14:00" with the day. */
export function clock(iso, { day = false } = {}) {
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}))?/.exec(String(iso || ""));
  if (!m) return String(iso || "");
  const hm = m[4] ? `${m[4]}:${m[5]}` : "";
  const d = `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]}`;
  return day || !hm ? `${d}${hm ? ` ${hm}` : ""}` : hm;
}

/** Hours as words: "45 min", "13 h", "2 d 6 h". */
export function hoursWords(h) {
  if (!Number.isFinite(h)) return "–";
  if (h < 1) return `${Math.round(h * 60)} min`;
  if (h < 48) return `${num(h, h < 10 ? 1 : 0)} h`;
  const d = Math.floor(h / 24), rest = Math.round(h - 24 * d);
  return rest ? `${d} d ${rest} h` : `${d} d`;
}

// ── shared bits of the charts ───────────────────────────────────────────────

function runs(values) {
  // index runs where the value is a number
  const out = [];
  let start = -1;
  for (let i = 0; i <= values.length; i++) {
    const ok = i < values.length && values[i] !== null && values[i] !== undefined && Number.isFinite(values[i]);
    if (ok && start < 0) start = i;
    if (!ok && start >= 0) { out.push([start, i - 1]); start = -1; }
  }
  return out;
}

const pt = (x, y) => `${r1(x)},${r1(y)}`;

// ── the section ─────────────────────────────────────────────────────────────

/** The section's frame: where each point and each elevation land in an SVG of w x h. */
export function sectionFrame(sec, { w = 640, h = 220 } = {}) {
  const pad = { l: 50, r: 14, t: 18, b: 32 };
  const d = sec.distance_m;
  const total = d[d.length - 1] || 1;
  const vals = [];
  for (const arr of [sec.ground || [], sec.max_level || []]) for (const v of arr) if (Number.isFinite(v)) vals.push(v);
  for (const row of sec.levels || []) for (const v of row) if (Number.isFinite(v)) vals.push(v);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (!vals.length) { lo = 0; hi = 1; }
  const span = Math.max(hi - lo, 0.5);
  lo -= span * 0.12;
  hi += span * 0.1;
  const x = (m) => pad.l + (m / total) * (w - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
  return { pad, w, h, total, lo, hi, x, y };
}

/** The water levels on show: one step's, or the highest of the run (view "max"). */
export function sectionLevels(sec, { step = 0, view = "step" } = {}) {
  if (view === "max" || !sec.levels || !sec.levels.length) return sec.max_level || [];
  return sec.levels[Math.max(0, Math.min(sec.levels.length - 1, step))] || [];
}

/**
 * The cross-section as SVG: the ground (filled), the water at `step` (or the highest of the run with view
 * "max") between the ground and its surface, the highest water of the run dashed, distance and elevation axes,
 * and A and A′ at the ends. `hover` is the index of a point to mark.
 */
export function sectionSvg(sec, { step = 0, view = "step", w = 640, h = 220, hover = -1, unit = "m" } = {}) {
  const f = sectionFrame(sec, { w, h });
  const d = sec.distance_m;
  const g = sec.ground;
  const lv = sectionLevels(sec, { step, view });
  const bottom = h - f.pad.b;
  const parts = [];
  // grid and axes
  const yt = niceTicks(f.lo, f.hi, Math.max(3, Math.round((h - 50) / 38)));
  const xt = niceTicks(0, f.total, Math.max(3, Math.round((w - 70) / 90)));
  const km = f.total >= 2000;
  const yd = tickDigits(yt);
  parts.push(...yt.map((v) => `<line class="fv-grid" x1="${f.pad.l}" x2="${w - f.pad.r}" y1="${r1(f.y(v))}" y2="${r1(f.y(v))}"/>` +
    `<text class="fv-tick" x="${f.pad.l - 6}" y="${r1(f.y(v) + 3.5)}" text-anchor="end">${esc(num(v, yd))}</text>`));
  parts.push(...xt.map((v) => `<line class="fv-tickmark" x1="${r1(f.x(v))}" x2="${r1(f.x(v))}" y1="${bottom}" y2="${bottom + 4}"/>` +
    `<text class="fv-tick" x="${r1(f.x(v))}" y="${bottom + 15}" text-anchor="middle">${esc(km ? num(v / 1000, tickDigits(xt.map((t) => t / 1000))) : num(v, 0))}</text>`));
  // the ground
  if (g) {
    for (const [a, b] of runs(g)) {
      const top = [];
      for (let i = a; i <= b; i++) top.push(pt(f.x(d[i]), f.y(g[i])));
      parts.push(`<path class="fv-ground" d="M${pt(f.x(d[a]), bottom)}L${top.join("L")}L${pt(f.x(d[b]), bottom)}Z"/>`);
      parts.push(`<path class="fv-ground-line" d="M${top.join("L")}"/>`);
    }
  }
  // the water: between the ground (or the frame's foot) and the surface
  for (const [a, b] of runs(lv)) {
    const top = [], foot = [];
    for (let i = a; i <= b; i++) {
      top.push(pt(f.x(d[i]), f.y(lv[i])));
      foot.push(pt(f.x(d[i]), g && Number.isFinite(g[i]) ? f.y(g[i]) : bottom));
    }
    parts.push(`<path class="fv-water" d="M${top.join("L")}L${foot.reverse().join("L")}Z"/>`);
    parts.push(`<path class="fv-surface" d="M${top.join("L")}"/>`);
  }
  // the highest water of the run, dashed (left out when it is what is drawn)
  if (view !== "max" && sec.max_level) {
    for (const [a, b] of runs(sec.max_level)) {
      const line = [];
      for (let i = a; i <= b; i++) line.push(pt(f.x(d[i]), f.y(sec.max_level[i])));
      parts.push(`<path class="fv-max" d="M${line.join("L")}"/>`);
    }
  }
  // the axes, the titles, A and A′
  parts.push(`<line class="fv-axis" x1="${f.pad.l}" x2="${w - f.pad.r}" y1="${bottom}" y2="${bottom}"/>`);
  parts.push(`<line class="fv-axis" x1="${f.pad.l}" x2="${f.pad.l}" y1="${f.pad.t}" y2="${bottom}"/>`);
  parts.push(`<text class="fv-title" x="${f.pad.l - 6}" y="${f.pad.t - 6}" text-anchor="end">${esc(unit)}</text>`);
  parts.push(`<text class="fv-title" x="${w - f.pad.r}" y="${h - 3}" text-anchor="end">distance, ${km ? "km" : "m"}</text>`);
  parts.push(`<text class="fv-end" x="${f.pad.l + 4}" y="${f.pad.t + 10}">A</text>`);
  parts.push(`<text class="fv-end" x="${w - f.pad.r - 4}" y="${f.pad.t + 10}" text-anchor="end">A′</text>`);
  if (hover >= 0 && hover < d.length) {
    const x = r1(f.x(d[hover]));
    parts.push(`<line class="fv-hover" x1="${x}" x2="${x}" y1="${f.pad.t}" y2="${bottom}"/>`);
    if (Number.isFinite(lv[hover])) parts.push(`<circle class="fv-dot-water" cx="${x}" cy="${r1(f.y(lv[hover]))}" r="3.4"/>`);
    if (g && Number.isFinite(g[hover])) parts.push(`<circle class="fv-dot-ground" cx="${x}" cy="${r1(f.y(g[hover]))}" r="3"/>`);
  }
  return `<svg class="fv-chart" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-hidden="true">${parts.join("")}</svg>`;
}

/** The point of the section under an x position in the SVG, or -1. */
export function sectionIndexAt(sec, px, { w = 640, h = 220 } = {}) {
  const f = sectionFrame(sec, { w, h });
  if (px < f.pad.l - 4 || px > w - f.pad.r + 4) return -1;
  const m = ((px - f.pad.l) / (w - f.pad.l - f.pad.r)) * f.total;
  const d = sec.distance_m;
  let best = 0;
  for (let i = 1; i < d.length; i++) if (Math.abs(d[i] - m) < Math.abs(d[best] - m)) best = i;
  return best;
}

/** The readout at one point of the section: "420 m · ground 262.30 · water 264.10 · 1.80 m deep". */
export function sectionReadout(sec, i, { step = 0, view = "step", unit = "m" } = {}) {
  if (i < 0 || i >= sec.distance_m.length) return "";
  const lv = sectionLevels(sec, { step, view });
  const g = sec.ground ? sec.ground[i] : null;
  const parts = [lengthWords(sec.distance_m[i])];
  if (sec.cell && sec.cell[i] < 0) return `${parts[0]} · off the mesh`;
  if (Number.isFinite(g)) parts.push(`ground ${num(g, 2)}`);
  if (Number.isFinite(lv[i])) {
    parts.push(`water ${num(lv[i], 2)}`);
    if (Number.isFinite(g)) parts.push(`${num(lv[i] - g, 2)} ${unit} deep`);
  } else parts.push("dry");
  return parts.join(" · ");
}

/** The section's heading words: "A–A′ · 1.67 km · at 15 Mar 14:00" (or "highest water"). */
export function sectionHeading(sec, { step = 0, view = "step" } = {}) {
  const when = view === "max" || !sec.times || !sec.times.length ? "highest water of the run" : `at ${clock(sec.times[step], { day: true })}`;
  return `${lengthWords(sec.facts.length_m)} · ${when}`;
}

/** One line under the section: what the ground is, what is dashed, and the facts that matter. */
export function sectionNote(sec, { unit = "m" } = {}) {
  const f = sec.facts || {};
  const bits = [];
  if (Number.isFinite(f.widest_wet_m)) bits.push(`widest water ${lengthWords(f.widest_wet_m)} at ${clock(sec.times[f.widest_step])}`);
  if (Number.isFinite(f.deepest)) bits.push(`deepest ${num(f.deepest, 2)} ${unit}, ${lengthWords(f.deepest_at_m)} along, at ${clock(sec.times[f.deepest_step])}`);
  const ground = sec.ground_is ? `Ground: ${sec.ground_is}.` : "";
  return [bits.length ? `${bits.join("; ")}.` : "", ground, "Dashed: the highest water of the run.", ...(sec.notes || [])]
    .filter(Boolean).join(" ");
}

// ── the hydrograph ──────────────────────────────────────────────────────────

function timeTicks(hours, times, n) {
  const total = hours[hours.length - 1] || 1;
  const steps = [0.25, 0.5, 1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720, 1440, 2880, 8760];
  const step = steps.find((s) => total / s <= n) || steps[steps.length - 1];
  const t0 = /T(\d{2}):(\d{2})/.exec(times[0] || "");
  const startH = t0 ? Number(t0[1]) + Number(t0[2]) / 60 : 0;
  const first = step >= 1 && step <= 24 ? ((step - (startH % step)) % step) : 0;
  const out = [];
  for (let hh = first; hh <= total + 1e-9; hh += step) out.push(hh);
  return { at: out, days: step >= 24 || total > 48 };
}

function timeLabel(times, hoursAt, days) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(times[0] || "");
  if (!m) return `${num(hoursAt, 0)} h`;
  const t = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]) + hoursAt * 3600e3);
  const hm = `${String(t.getUTCHours()).padStart(2, "0")}:${String(t.getUTCMinutes()).padStart(2, "0")}`;
  return days ? `${t.getUTCDate()} ${MONTHS[t.getUTCMonth()]}` : hm;
}

/**
 * The hydrograph as SVG: the depth over the run (left axis) and the water level (right axis, the same curve
 * lifted by the ground), the arrival, the peak and the time under water marked, and the step on show.
 */
export function hydrographSvg(hy, { w = 304, h = 150, step = -1, unit = "m" } = {}) {
  const hours = hy.hours;
  const total = hours[hours.length - 1] || 1;
  const hasDepth = Array.isArray(hy.depth);
  const series = hasDepth ? hy.depth : hy.level;
  const vals = series.filter((v) => Number.isFinite(v));
  const pad = { l: 34, r: hasDepth && Number.isFinite(hy.ground) ? 40 : 10, t: 20, b: 22 };
  let lo = hasDepth ? 0 : Math.min(...vals), hi = vals.length ? Math.max(...vals) : 1;
  if (!vals.length) { lo = 0; hi = 1; }
  if (!hasDepth) { const s = Math.max(hi - lo, 0.2); lo -= s * 0.1; hi += s * 0.1; } else hi = Math.max(hi * 1.15, hy.threshold * 4, 0.1);
  const x = (t) => pad.l + (t / total) * (w - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
  const bottom = h - pad.b;
  const parts = [];
  const yt = niceTicks(lo, hi, 4);
  const yd = tickDigits(yt);
  for (const v of yt) {
    parts.push(`<line class="fv-grid" x1="${pad.l}" x2="${w - pad.r}" y1="${r1(y(v))}" y2="${r1(y(v))}"/>`);
    parts.push(`<text class="fv-tick" x="${pad.l - 5}" y="${r1(y(v) + 3.5)}" text-anchor="end">${esc(num(v, yd))}</text>`);
    if (hasDepth && Number.isFinite(hy.ground)) {
      parts.push(`<text class="fv-tick fv-right" x="${w - pad.r + 5}" y="${r1(y(v) + 3.5)}">${esc(num(v + hy.ground, Math.max(1, yd)))}</text>`);
    }
  }
  const tt = timeTicks(hours, hy.times, Math.max(3, Math.round((w - 80) / 55)));
  for (const t of tt.at) {
    parts.push(`<line class="fv-tickmark" x1="${r1(x(t))}" x2="${r1(x(t))}" y1="${bottom}" y2="${bottom + 3}"/>`);
    parts.push(`<text class="fv-tick" x="${r1(x(t))}" y="${bottom + 13}" text-anchor="middle">${esc(timeLabel(hy.times, t, tt.days))}</text>`);
  }
  // the time under water: a band behind the curve, from the arrival to the recession (or the run's end)
  if (hy.arrival) {
    const a = x(hy.arrival.hours), b = x(hy.recession ? hy.recession.hours : total);
    parts.push(`<rect class="fv-wetband" x="${r1(a)}" y="${pad.t}" width="${r1(Math.max(1, b - a))}" height="${bottom - pad.t}"/>`);
    parts.push(`<line class="fv-bracket" x1="${r1(a)}" x2="${r1(b)}" y1="${pad.t - 7}" y2="${pad.t - 7}"/>`);
    parts.push(`<line class="fv-bracket" x1="${r1(a)}" x2="${r1(a)}" y1="${pad.t - 10}" y2="${pad.t - 4}"/>`);
    parts.push(`<line class="fv-bracket" x1="${r1(b)}" x2="${r1(b)}" y1="${pad.t - 10}" y2="${pad.t - 4}"/>`);
    const label = `${hoursWords(hy.duration_h)}${hy.still_wet ? "+" : ""} under water`;
    const mid = Math.min(Math.max((a + b) / 2, pad.l + 40), w - pad.r - 40);
    parts.push(`<text class="fv-mark" x="${r1(mid)}" y="${pad.t - 10}" text-anchor="middle">${esc(label)}</text>`);
  }
  // the curve
  for (const [a, b] of runs(series)) {
    const top = [];
    for (let i = a; i <= b; i++) top.push(pt(x(hours[i]), y(series[i])));
    if (hasDepth) parts.push(`<path class="fv-water" d="M${pt(x(hours[a]), bottom)}L${top.join("L")}L${pt(x(hours[b]), bottom)}Z"/>`);
    parts.push(`<path class="fv-surface" d="M${top.join("L")}"/>`);
  }
  if (hasDepth && hy.threshold > (hi - lo) * 0.03) {
    parts.push(`<line class="fv-thr" x1="${pad.l}" x2="${w - pad.r}" y1="${r1(y(hy.threshold))}" y2="${r1(y(hy.threshold))}"/>`);
  }
  if (hy.arrival && hy.arrival.step > 0) {
    const ax = r1(x(hy.arrival.hours));
    parts.push(`<line class="fv-arrive" x1="${ax}" x2="${ax}" y1="${pad.t}" y2="${bottom}"/>`);
  }
  if (hy.peak) {
    const v = hasDepth ? hy.peak.depth : hy.peak.level;
    const px = x(hy.peak.hours), py = y(v);
    parts.push(`<circle class="fv-peak" cx="${r1(px)}" cy="${r1(py)}" r="3.6"/>`);
    const right = px > w - pad.r - 60;
    parts.push(`<text class="fv-mark" x="${r1(px + (right ? -6 : 6))}" y="${r1(py + 4)}" text-anchor="${right ? "end" : "start"}">${esc(`${num(v, 2)} ${unit}`)}</text>`);
  }
  if (step >= 0 && step < hours.length) {
    const sx = r1(x(hours[step]));
    parts.push(`<line class="fv-now" x1="${sx}" x2="${sx}" y1="${pad.t}" y2="${bottom}"/>`);
  }
  parts.push(`<line class="fv-axis" x1="${pad.l}" x2="${w - pad.r}" y1="${bottom}" y2="${bottom}"/>`);
  parts.push(`<text class="fv-title" x="${pad.l - 5}" y="${pad.t - 10}" text-anchor="end">${hasDepth ? "depth" : unit}</text>`);
  if (hasDepth && Number.isFinite(hy.ground)) parts.push(`<text class="fv-title" x="${w - pad.r + 5}" y="${pad.t - 10}">level</text>`);
  return `<svg class="fv-chart" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">${parts.join("")}</svg>`;
}

/** The words under the hydrograph: arrival, peak and time under water, each with its time. */
export function hydroKey(hy, { unit = "m" } = {}) {
  if (!hy.wet) return hy.note || "Dry over the whole run.";
  if (!Array.isArray(hy.depth)) return `Highest water ${num(hy.peak.level, 2)} ${unit} at ${clock(hy.peak.time, { day: true })}. ${hy.note || ""}`.trim();
  const arrive = hy.arrival.step === 0 ? "Wet from the run's start" : `Arrives ${clock(hy.arrival.time, { day: true })}`;
  const peak = `peak ${num(hy.peak.depth, 2)} ${unit} deep (level ${num(hy.peak.level, 2)}) at ${clock(hy.peak.time, { day: true })}`;
  const wet = hy.still_wet ? `still wet at the end, ${hoursWords(hy.duration_h)} so far` :
    `under water ${hoursWords(hy.duration_h)}, receding by ${clock(hy.recession.time, { day: true })}`;
  return `${arrive} · ${peak} · ${wet}.`;
}

/** The hydrograph's accessible summary. */
export function hydroAlt(hy, { unit = "m" } = {}) {
  return `Depth and water level at this cell over the run. ${hydroKey(hy, { unit })}`;
}

// ── what gets flooded ───────────────────────────────────────────────────────

/** The map filter for a feature flooded at `step` ("max": at any step). Its "w" holds one character per step. */
export function wetFilter(step, view = "step") {
  if (view === "max") return ["in", "1", ["get", "w"]];
  return ["==", ["slice", ["get", "w"], Math.max(0, step), Math.max(0, step) + 1], "1"];
}

/** Whether a feature's wet string says flooded at `step` (or ever, for "max"). */
export const wetAt = (w, step, view = "step") => (view === "max" ? String(w || "").includes("1") : String(w || "")[step] === "1");

/** People, rounded to what a population grid can say: "about 16,000", "about 340", "a few". */
export function peopleWords(x) {
  if (!Number.isFinite(x) || x < 0.5) return "nobody";
  if (x < 5) return "a few people";
  const mag = x < 100 ? 10 : x < 1000 ? 10 : x < 10000 ? 100 : 10 ** (Math.floor(Math.log10(x)) - 1);
  return `about ${(Math.round(x / mag) * mag).toLocaleString("en-GB")} people`;
}

const int = (n) => Math.round(Number(n) || 0).toLocaleString("en-GB");
const kmWords = (km) => (km < 1 ? `${num(km * 1000, 0)} m` : `${num(km, km < 10 ? 1 : 0)} km`);

/** The counts at one step, or over the whole run ("max": everything that is ever flooded). */
export function countsAt(res, step = 0, view = "step") {
  const out = {};
  if (res.buildings) out.buildings = view === "max" ? res.buildings.ever : res.buildings.per_step[step] ?? 0;
  if (res.roads) out.roadsKm = view === "max" ? res.roads.ever_km : res.roads.per_step_km[step] ?? 0;
  if (res.population) out.people = view === "max" ? res.population.ever : res.population.per_step[step] ?? 0;
  return out;
}

/** The legend's rows: [{ key, value, sub }], at the step on show and against the peak. */
export function exposureRows(res, step = 0, view = "step") {
  const c = countsAt(res, step, view);
  const rows = [];
  const peak = (p, words) => (p && p.step !== null && p.step !== undefined ? `peak ${words} at ${clock(res.times[p.step])}` : "");
  if (res.buildings) {
    rows.push({ key: "buildings", value: `${int(c.buildings)} building${c.buildings === 1 ? "" : "s"}`,
      sub: view === "max" ? `flooded at some point, of ${int(res.buildings.on_mesh)} on the model` :
        `of ${int(res.buildings.on_mesh)} · ${peak(res.buildings.peak, int(res.buildings.peak.value))}` });
  }
  if (res.roads) {
    rows.push({ key: "roads", value: `${kmWords(c.roadsKm)} of road`,
      sub: view === "max" ? `under water at some point, of ${kmWords(res.roads.on_mesh_km)} on the model` :
        peak(res.roads.peak, kmWords(res.roads.peak.value)) });
  }
  if (res.population) {
    rows.push({ key: "people", value: peopleWords(c.people).replace(/^about /, "~"),
      sub: view === "max" ? "live where the water reaches at some point" :
        `in the flooded area · ${peak(res.population.peak, peopleWords(res.population.peak.value).replace(/^about /, "~").replace(/ people$/, ""))}` });
  }
  return rows;
}

/** The legend row's few words: "2,249 buildings · 129 km road". */
export function exposureSummary(res, step = 0, view = "step") {
  const c = countsAt(res, step, view);
  const bits = [];
  if (c.buildings !== undefined) bits.push(`${int(c.buildings)} bldg`);
  if (c.roadsKm !== undefined) bits.push(`${kmWords(c.roadsKm)} road`);
  return bits.join(" · ");
}

/** A small step chart of one count over the run, the step on show marked. */
export function countSvg(values, { w = 240, h = 34, step = -1 } = {}) {
  const n = values.length;
  if (!n) return "";
  const hi = Math.max(...values, 1e-9);
  const x = (i) => 2 + (i / Math.max(1, n - 1)) * (w - 4);
  const y = (v) => h - 2 - (v / hi) * (h - 6);
  const line = values.map((v, i) => pt(x(i), y(v))).join("L");
  let out = `<path class="fv-count-area" d="M${pt(x(0), h - 2)}L${line}L${pt(x(n - 1), h - 2)}Z"/><path class="fv-count" d="M${line}"/>`;
  if (step >= 0 && step < n) out += `<line class="fv-now" x1="${r1(x(step))}" x2="${r1(x(step))}" y1="1" y2="${h - 1}"/>`;
  return `<svg class="fv-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">${out}</svg>`;
}

/** The credit line of each source read: "Overture Maps buildings · ODbL 1.0 · © OpenStreetMap contributors, ...". */
export function sourceLines(res) {
  const seen = new Map();
  for (const [k, s] of Object.entries(res.sources || {})) {
    const key = `${s.credit}|${s.licence}`;
    if (!seen.has(key)) seen.set(key, { names: [], licence: s.licence, credit: s.credit, link: s.link, release: s.release });
    seen.get(key).names.push(k);
  }
  return [...seen.values()].map((s) => ({ ...s, text: `${s.names.join(" and ")}: ${s.credit}${s.release ? `, release ${s.release}` : ""}, ${s.licence}` }));
}

/** A clicked building's card words. */
export function buildingWords(f, times, { unit = "m", step = 0, view = "step" } = {}) {
  const w = String(f.w || "");
  const first = w.indexOf("1"), last = w.lastIndexOf("1");
  const kind = f.c ? `${String(f.c).replace(/_/g, " ")}` : "building";
  if (first < 0) return { title: `A ${kind}`, status: "Stays dry over the whole run.", note: "" };
  const now = view === "max" ? "" : wetAt(w, step) ? `Flooded at ${clock(times[step])}. ` : `Dry at ${clock(times[step])}. `;
  const span = first === last ? `under water at ${clock(times[first], { day: true })}` :
    `under water from ${clock(times[first], { day: true })} to ${clock(times[last], { day: true })}`;
  const deep = Number.isFinite(f.d) ? `; up to ${num(f.d, 2)} ${unit} deep in a cell under it` : "";
  return { title: `A ${kind}`, status: `${now}${span[0].toUpperCase()}${span.slice(1)}${deep}.`,
    note: first === 0 ? "Already wet at the run's start: a building in the channel or a pond, or a footprint that is off." : "" };
}
