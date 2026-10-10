// The map card (#548): a click answers on the map. A small card anchored to
// what was clicked (a gauge, a river reach, a place) says what it is, how it
// is doing today against normal, shows one sparkline and one number, and
// offers Details (the full panel, on the right tab), Trace to the sea, Watch
// and Study. The panel with its tabs is folded away until Details asks for it.
//
// It fills progressively: the name and position at once, then the daily status
// snapshot, then the record or the forecast as the worker answers. Every number
// comes from the package (the snapshot written by aquascope.nownext, the record
// from aquascope.explore, the forecast from aquascope.nownext); this module asks
// for them, and map-card-core.js decides how they are said and drawn.
//
// Other layers open a card with their own content through openCard() (also
// actions.openMapCard): a flood cell, a warning reach. See openCard below.

import { $, VAR_LABEL, actions, escapeHtml, haversineKm, sourceStyle, state, stationKey } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { selectTab, togglePanel } from "./shell.js?v=__BUILD__";
import { call, callLight } from "./worker-client.js?v=__BUILD__";
import { ensureNowStatus } from "./now-map.js?v=__BUILD__";
import { statusClass } from "./now-core.js?v=__BUILD__";
import { snapLine } from "./river-core.js?v=__BUILD__";
import { takeOfferedReach, traceRiver } from "./river.js?v=__BUILD__";
import { shapeSvg } from "./shapes.js?v=__BUILD__";
import {
  cardNumber, dayMonth, forecastPeak, lastDays, latestValue, placeCard, prettyUnit, snapshotSentence, sparkPaths,
} from "./map-card-core.js?v=__BUILD__";

const STATUS_VARIABLES = new Set(["discharge", "water_level", "groundwater_level"]);
const SNAPSHOT_CREDIT = "AquaScope daily status snapshot";
const GEOGLOWS_SHORT = "GEOGLOWS v2 forecast, CC BY 4.0, modelled";
const SHEET_QUERY = "(max-width: 640px)";
const NEAR_GAUGE_KM = 50;   // past this the nearest gauge says nothing about the place, so the card leaves it out
const reducedMotion = () => Boolean(globalThis.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
const sheetMode = () => !state.mapOk || Boolean(globalThis.matchMedia && matchMedia(SHEET_QUERY).matches);

// The card on screen: its spec (what it says) and how it was opened.
let spec = null;
let shown = false;
let restoreFocus = null;
let frame = 0;
let listening = false;
// The selection the card stands for ("gauge:<key>" or "point:<lat>,<lon>"), kept while the panel is open so
// folding the panel brings the card back, and a counter so a late answer for a place left behind is dropped.
let model = null;
let run = 0;

const el = () => $("map-card");

// ── the public API ──────────────────────────────────────────────────────────

/**
 * Open a card on the map and return a handle `{ update(patch), close(), id }`.
 *
 * spec: {
 *   id,                       a key for what the card is about (opening the same id again updates it)
 *   lngLat: [lon, lat],       where it points; lift: pixels the marker reaches above it (default 12)
 *   what, whatIcon,           the kicker above the title ("Gauge", "Flood cell") and an optional SVG
 *   title, sub,               the name and one muted line under it
 *   status: { text, color } | { pending: "Reading…" } | null     the sentence, with a class colour
 *   spark: { v, band: {lo, hi}, label, fromZero } | { pending, label } | { empty: "why" } | null
 *                             (fromZero: false scales a forecast to its own range; a record starts at zero)
 *   figure: { value, unit, label } | { pending } | null           the one key number
 *   note,                     an extra line (for example what a trace found)
 *   credit,                   where the numbers come from, with the licence
 *   details: () => void,      the Details button (omitted: no button)
 *   buttons: [{ id, label, title, onClick, disabled, pressed, hidden }]   the others, in order
 * }
 */
export function openCard(next, { focus = true } = {}) {
  const same = spec && shown && spec.id === next.id;
  if (next.lift === undefined) next.lift = 12;
  spec = next;
  if (!same) restoreFocus = document.activeElement && document.activeElement !== document.body ? document.activeElement : null;
  show({ focus: focus && !same });
  return handleFor(spec.id);
}

export function updateCard(id, patch) {
  let hit = false;
  // The selection's card keeps filling while it is folded away behind the open panel.
  if (model && model.id === id) { Object.assign(model.spec, patch); hit = true; }
  if (spec && spec.id === id && (!model || spec !== model.spec)) { Object.assign(spec, patch); hit = true; }
  if (hit && shown && spec && spec.id === id) render();
  return hit;
}

export function closeCard({ restore = true, keepModel = false } = {}) {
  const card = el();
  const was = shown;
  shown = false;
  if (card) {
    card.hidden = true;
    card.classList.remove("on");
  }
  if (!keepModel) model = null;
  if (was && restore && restoreFocus && document.contains(restoreFocus)) {
    try { restoreFocus.focus(); } catch { /* nothing to restore to */ }
  }
  if (!was) return;
  restoreFocus = null;
}

export const cardOpen = () => shown;

function handleFor(id) {
  return {
    id,
    update: (patch) => updateCard(id, patch),
    close: () => { if (spec && spec.id === id) closeCard(); },
  };
}

// ── drawing ─────────────────────────────────────────────────────────────────

function show({ focus }) {
  const card = el();
  if (!card || !spec) return;
  listen();
  hint(false);
  const wasShown = shown;
  shown = true;
  card.hidden = false;
  render();
  if (!wasShown) {
    card.classList.remove("on");
    void card.offsetWidth;   // restart the entrance
    card.classList.add("on");
    announce(`${spec.what ? `${spec.what}: ` : ""}${spec.title || ""}`);
  }
  place();
  if (sheetMode()) keepAnchorAboveSheet();
  if (focus) card.focus({ preventScroll: true });
}

function statusHtml(s) {
  if (!s) return "";
  if (s.pending) return `<p class="mc-status mc-pending"><span class="mc-skel" aria-hidden="true"></span>${escapeHtml(s.pending)}</p>`;
  const dot = s.color ? `<i class="mc-dot" style="background:${escapeHtml(s.color)}" aria-hidden="true"></i>` : "";
  return `<p class="mc-status${s.muted ? " muted" : ""}" id="mc-status">${dot}<span>${escapeHtml(s.text || "")}</span></p>`;
}

function sparkHtml(sp) {
  if (!sp) return "";
  const label = sp.label ? `<span class="mc-spark-label">${escapeHtml(sp.label)}</span>` : "";
  if (sp.pending) return `<div class="mc-spark mc-wait" aria-hidden="true"><span class="mc-skel wide"></span></div>${label}`;
  if (sp.empty) return `<p class="mc-spark-empty muted">${escapeHtml(sp.empty)}</p>`;
  return `<div class="mc-spark" role="img" aria-label="${escapeHtml(sp.alt || sp.label || "sparkline")}"></div>${label}`;
}

function figureHtml(f) {
  if (!f) return "";
  if (f.pending) return `<div class="mc-figure" aria-hidden="true"><span class="mc-skel short"></span></div>`;
  return `<div class="mc-figure"><span class="mc-num">${escapeHtml(f.value)}<small>${escapeHtml(f.unit || "")}</small></span>` +
    `<span class="mc-fig-label">${escapeHtml(f.label || "")}</span></div>`;
}

function render() {
  const card = el();
  if (!card || !spec) return;
  const active = document.activeElement && card.contains(document.activeElement) ? document.activeElement.dataset.act : null;
  const s = spec;
  const buttons = (s.buttons || []).filter((b) => !b.hidden);
  const middle = (s.spark || s.figure)
    ? `<div class="mc-row"><div class="mc-spark-wrap">${sparkHtml(s.spark)}</div>${figureHtml(s.figure)}</div>` : "";
  card.innerHTML = `
    <div class="mc-head">
      <p class="mc-what">${s.whatIcon || ""}<span>${escapeHtml(s.what || "")}</span></p>
      <button type="button" class="mc-close" data-act="close" aria-label="Close the card" title="Close (Esc)">×</button>
    </div>
    <h2 class="mc-title" id="mc-title">${escapeHtml(s.title || "")}</h2>
    ${s.sub ? `<p class="mc-sub">${escapeHtml(s.sub)}</p>` : ""}
    ${statusHtml(s.status)}
    ${middle}
    ${s.note ? `<p class="mc-note" role="status">${escapeHtml(s.note)}</p>` : ""}
    <div class="mc-actions">
      ${s.details ? `<button type="button" class="btn small primary" data-act="details" title="The full panel: the record, floods, flows, the model and more">Details</button>` : ""}
      ${buttons.map((b) => `<button type="button" class="btn small" data-act="${escapeHtml(b.id)}"${b.disabled ? " disabled" : ""}` +
        `${b.pressed === undefined ? "" : ` aria-pressed="${b.pressed ? "true" : "false"}"`}` +
        `${b.title ? ` title="${escapeHtml(b.title)}"` : ""}>${escapeHtml(b.label)}</button>`).join("")}
    </div>
    ${s.credit ? `<p class="mc-credit">${escapeHtml(s.credit)}</p>` : ""}
    <span class="mc-tail" aria-hidden="true"></span>`;
  if (s.status && !s.status.pending) card.setAttribute("aria-describedby", "mc-status");
  else card.removeAttribute("aria-describedby");
  card.querySelector('[data-act="close"]').addEventListener("click", () => closeCard());
  const det = card.querySelector('[data-act="details"]');
  if (det) det.addEventListener("click", () => s.details());
  for (const b of buttons) {
    const btn = card.querySelector(`[data-act="${CSS.escape(b.id)}"]`);
    if (btn && b.onClick) btn.addEventListener("click", () => b.onClick(btn));
  }
  drawSpark();
  if (active) {
    const again = card.querySelector(`[data-act="${CSS.escape(active)}"]`);
    if (again && !again.disabled) again.focus({ preventScroll: true });
    else card.focus({ preventScroll: true });
  }
  place();
}

function drawSpark() {
  const sp = spec && spec.spark;
  const box = el() && el().querySelector(".mc-spark:not(.mc-wait)");
  if (!sp || !box || !sp.v) return;
  const w = Math.max(120, Math.round(box.clientWidth || 180));
  const h = 40;
  const p = sparkPaths({ v: sp.v, band: sp.band || null }, { w, h, pad: 4, fromZero: sp.fromZero !== false });
  if (!p) { box.outerHTML = `<p class="mc-spark-empty muted">Too few values to draw.</p>`; return; }
  box.innerHTML = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">` +
    (p.band ? `<path class="mc-band" d="${p.band}"/>` : "") +
    (p.area ? `<path class="mc-area" d="${p.area}"/>` : "") +
    `<path class="mc-line" d="${p.line}"/>` +
    (p.end ? `<circle class="mc-end" cx="${p.end.x}" cy="${p.end.y}" r="2.6"/>` : "") +
    "</svg>";
}

// ── anchoring ───────────────────────────────────────────────────────────────

function listen() {
  if (listening || !state.mapOk || !map) return;
  listening = true;
  const schedule = () => {
    if (!shown || frame) return;
    frame = requestAnimationFrame(() => { frame = 0; place(); });
  };
  map.on("move", schedule);
  map.on("resize", schedule);
  window.addEventListener("resize", schedule);
}

// What the panel and the drawer cover on the right, so the card is not placed underneath them.
function coveredRight(frameRect) {
  let right = 0;
  for (const id of ["panel", "drawer"]) {
    const node = $(id);
    if (!node || node.hidden || node.offsetParent === null) continue;
    if (id === "panel" && document.body.classList.contains("panel-collapsed")) continue;
    const r = node.getBoundingClientRect();
    if (r.width && r.width < frameRect.width * 0.75) right = Math.max(right, frameRect.right - r.left);
  }
  return right;
}

function place() {
  const card = el();
  if (!card || !shown || !spec) return;
  const sheet = sheetMode();
  card.classList.toggle("sheet", sheet);
  if (sheet) {
    card.style.left = "";
    card.style.top = "";
    return;
  }
  const wrap = card.offsetParent || card.parentElement;
  if (!wrap || !map || !spec.lngLat) return;
  const box = wrap.getBoundingClientRect();
  const canvas = map.getContainer().getBoundingClientRect();
  let pt;
  try { pt = map.project(spec.lngLat); } catch { return; }
  let hidden = false;
  try { hidden = Boolean(map.transform && map.transform.isLocationOccluded && map.transform.isLocationOccluded({ lng: spec.lngLat[0], lat: spec.lngLat[1] })); } catch { /* older MapLibre */ }
  const ax = pt.x + canvas.left - box.left;
  const ay = pt.y + canvas.top - box.top;
  const at = placeCard({
    ax, ay, cw: card.offsetWidth, ch: card.offsetHeight, W: box.width, H: box.height,
    lift: spec.lift, reserveRight: coveredRight(box),
  });
  card.style.left = `${at.left}px`;
  card.style.top = `${at.top}px`;
  card.dataset.side = at.side;
  card.classList.toggle("detached", !at.inside || hidden);
  card.style.setProperty("--tail", `${at.tail}px`);
}

// On a phone the card is a sheet over the bottom of the map: move the map so what was clicked stays in sight.
function keepAnchorAboveSheet() {
  const card = el();
  if (!state.mapOk || !map || !spec || !spec.lngLat || !card) return;
  requestAnimationFrame(() => {
    try {
      const canvas = map.getContainer();
      const pt = map.project(spec.lngLat);
      const free = canvas.clientHeight - card.offsetHeight;
      if (pt.y > free - 28 || pt.y < 24) {
        map.panBy([0, pt.y - free / 2], { duration: reducedMotion() ? 0 : 350 });
      }
    } catch { /* the map is not ready */ }
  });
}

// ── the selection's card: a gauge ───────────────────────────────────────────

const distanceWords = (m) => (m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`);
const coords = (lat, lon) => `${Number(lat).toFixed(3)}°, ${Number(lon).toFixed(3)}°`;

function openDetails(surface, tab) {
  closeCard({ restore: false, keepModel: true });
  togglePanel(false);
  const root = $(surface);
  if (root && tab) selectTab(root, tab);
  const head = root && root.querySelector("h2");
  if (head) {
    head.tabIndex = -1;
    head.focus({ preventScroll: true });
  }
}

function watchButton(panelBtnId, what) {
  const src = $(panelBtnId);
  const on = Boolean(src && src.getAttribute("aria-pressed") === "true");
  return {
    id: "watch", label: on ? "★ Watching" : "☆ Watch", pressed: on, hidden: !src || src.hidden,
    title: on ? `Stop watching this ${what}` : `Watch this ${what}: what changed shows the next time you open the Explorer`,
    onClick: () => {
      const b = $(panelBtnId);
      if (b) b.click();
      refreshWatch();
    },
  };
}

function refreshWatch() {
  if (!model) return;
  const btns = (model.spec.buttons || []).map((b) => (b.id === "watch"
    ? watchButton(model.kind === "gauge" ? "btn-watch-st" : "btn-watch-pt", model.kind === "gauge" ? "gauge" : "river reach")
    : b));
  updateCard(model.id, { buttons: btns });
}

function traceButton(t, { ready, busy = false }) {
  return {
    id: "trace", label: busy ? "Tracing…" : "Trace to sea", disabled: !ready || busy,
    title: ready ? "Follow the river down to the sea, with the gauges and dams on the way"
      : "Waiting for the river this sits on",
    onClick: () => startTrace(t),
  };
}

function setButton(id, next) {
  if (!model) return;
  updateCard(model.id, { buttons: (model.spec.buttons || []).map((b) => (b.id === id ? next : b)) });
}

async function startTrace(t) {
  const my = model;
  if (!my) return;
  setButton("trace", traceButton(t, { ready: true, busy: true }));
  try {
    const res = await traceRiver(t);
    if (model !== my) return;
    const first = res && res.message ? String(res.message).split(/(?<=\.)\s/)[0] : "";
    updateCard(my.id, { note: first || "Traced on the map." });
  } catch (err) {
    if (model !== my) return;
    updateCard(my.id, { note: `Could not trace this river: ${err.message}` });
  } finally {
    if (model === my) setButton("trace", traceButton(t, { ready: true }));
  }
}

const studyButton = () => ({
  id: "study", label: "Study", title: "A complete study at this place, planned with you, with a report",
  onClick: () => actions.openStudy({ fresh: true }),
});

function gaugeModel(r) {
  const key = stationKey(r);
  const st = sourceStyle(r.source);
  const vars = r.variables || [];
  const variable = vars.includes("discharge") ? "discharge" : vars[0] || null;
  const period = r.period_start ? `${String(r.period_start).slice(0, 4)}–${r.period_end ? String(r.period_end).slice(0, 4) : "now"}` : "";
  const sub = [VAR_LABEL[variable] || variable, period, coords(r.lat, r.lon)].filter(Boolean).join(" · ");
  return {
    id: `gauge:${key}`, kind: "gauge", key, variable, filled: false,
    spec: {
      id: `gauge:${key}`, lngLat: [r.lon, r.lat], lift: 10,
      what: st.label, whatIcon: shapeSvg(st.shape, st.color, 11),
      title: r.name || r.station_id, sub,
      status: STATUS_VARIABLES.has(variable) ? { pending: "Placing today in the record…" } : null,
      spark: { pending: true, label: "last 12 months" },
      figure: { pending: true },
      credit: `Record: ${st.label}, the agency's own data.`,
      details: () => openDetails("panel-station", "overview"),
      buttons: [traceButton("st", { ready: false }), watchButton("btn-watch-st", "gauge"), studyButton()],
    },
  };
}

// The snapshot first (a fetch and a DuckDB read, no Python), then the record when the panel's analysis
// lands, then, if the snapshot did not have this gauge, today against normal from the record.
function fillGauge(m) {
  if (m.filled) return;
  m.filled = true;
  const my = ++run;
  const live = () => my === run && model === m;
  if (STATUS_VARIABLES.has(m.variable)) {
    void ensureNowStatus().then(() => {
      if (!live()) return;
      const row = state.nowStatus && state.nowStatus.get(m.key);
      if (!row || !statusClass(row.cls)) return;
      m.snapshot = row;
      const sentence = snapshotSentence(row, { variable: (state.nowMeta && state.nowMeta.variable) || "discharge" });
      const patch = { status: { text: sentence, color: statusClass(row.cls).color },
        credit: `Today vs normal: ${SNAPSHOT_CREDIT}. Record: ${sourceStyle(m.key.split("/")[0]).label}.` };
      // The one number is the newest value there is: the snapshot's, unless the record reaches later.
      if (Number.isFinite(row.value) && !(m.figureDate && m.figureDate > String(row.date))) {
        patch.figure = { value: cardNumber(row.value), unit: " m³/s", label: `on ${dayMonth(row.date, { short: true })}` };
        m.figureFrom = "snapshot";
        m.figureDate = String(row.date);
      }
      updateCard(m.id, patch);
    });
  }
  // The analysis may already be there (the card reopened after the panel was folded).
  if (state.result && state.selected && stationKey(state.selected) === m.key) onAnalysis(m, { result: state.result });
  else if (m.analysis) onAnalysis(m, m.analysis);
}

function onAnalysis(m, detail) {
  if (model !== m) return;
  const res = detail.result;
  if (!res || res.error || !res.n || !res.series) {
    const why = detail.message || (res && res.error) || "No observations came back for this gauge.";
    const patch = { spark: { empty: why }, figure: m.figureFrom ? m.spec.figure : null };
    if (m.spec.status && m.spec.status.pending) patch.status = null;
    updateCard(m.id, patch);
    return;
  }
  const unit = prettyUnit(res.unit);
  const year = lastDays(res.series, 365);
  const last = latestValue(res.series);
  const patch = {
    spark: { v: year.v, label: `last 12 months, to ${dayMonth(year.t[year.t.length - 1], { short: true, year: true })}`,
      alt: `Daily ${VAR_LABEL[res.variable] || res.variable} over the last 12 months of the record` },
  };
  if (last && !(m.figureDate && m.figureDate >= String(last.date))) {
    patch.figure = { value: cardNumber(last.value), unit: ` ${unit}`, label: `latest, ${dayMonth(last.date, { short: true, year: true })}` };
    m.figureFrom = "record";
    m.figureDate = String(last.date);
  }
  updateCard(m.id, patch);
  if (!m.snapshot && STATUS_VARIABLES.has(res.variable)) void statusFromRecord(m);
  else if (!STATUS_VARIABLES.has(res.variable) && m.spec.status && m.spec.status.pending) updateCard(m.id, { status: null });
}

async function statusFromRecord(m) {
  try {
    const st = await call("now", { op: "status", args: { source: m.key.split("/")[0], station_id: m.key.slice(m.key.indexOf("/") + 1) } });
    if (model !== m || m.snapshot) return;
    const cls = statusClass(st.class);
    const agency = sourceStyle(m.key.split("/")[0]).label;
    if (st.sentence) {
      updateCard(m.id, { status: { text: st.sentence, color: cls ? cls.color : null, muted: !cls },
        credit: `Record and today vs normal: ${agency}, the agency's own data, ranked by AquaScope.` });
    } else {
      updateCard(m.id, { status: { text: st.error || "Today against normal is not available here.", muted: true } });
    }
  } catch {
    if (model === m) updateCard(m.id, { status: { text: "Today against normal did not load; Details has it.", muted: true } });
  }
}

// ── the selection's card: a place, or the river it sits on ──────────────────

function pointModel(p) {
  const id = `point:${p.lat},${p.lon}`;
  return {
    id, kind: "point", lat: p.lat, lon: p.lon, filled: false,
    spec: {
      id, lngLat: [p.lon, p.lat], lift: 40,
      what: "Place", title: coords(p.lat, p.lon), sub: "",
      status: { pending: "Finding the river here…" },
      spark: { pending: true, label: "next 15 days, modelled" },
      figure: { pending: true },
      credit: "",
      details: () => openDetails("panel-point", "overview"),
      buttons: [traceButton("pt", { ready: false }), watchButton("btn-watch-pt", "river reach"), studyButton()],
    },
  };
}

function fillPoint(m) {
  if (m.filled) return;
  m.filled = true;
  if (m.snap) onSnap(m, m.snap);
}

function nearestGauge(lat, lon) {
  let best = null;
  for (const r of state.stations) {
    if (state.hidden.has(r.source) || !Number.isFinite(r.lat) || !Number.isFinite(r.lon)) continue;
    const d = haversineKm(lat, lon, r.lat, r.lon);
    if (!best || d < best.d) best = { d, r };
  }
  return best;
}

function onSnap(m, snap) {
  if (model !== m) return;
  if (!snap || snap.error) {
    updateCard(m.id, { status: { text: "The river network did not answer; Details has the climate and the catchment.", muted: true },
      spark: null, figure: null });
    return;
  }
  if (!snap.snapped) {
    // No stream within the snap's reach of the click (easy at a world zoom): offer the river the snap found,
    // in one click, rather than send the reader to the panel for it.
    const which = snap.larger ? "larger" : snap.nearest ? "nearest" : null;
    const near = nearestGauge(m.lat, m.lon);
    const take = which ? {
      id: "reach", label: which === "larger" ? "Use the larger river" : "Use the nearest river",
      title: `River reach ${snap[which].river_id}, ${distanceWords(snap[which].distance_m)} away`,
      onClick: () => {
        const reach = takeOfferedReach("pt", which);
        if (!reach || model !== m) return;
        onSnap(m, { snapped: true, river_id: reach.river_id, snap_lat: reach.lat, snap_lon: reach.lon,
          strahler_order: reach.strahler_order });
      },
    } : null;
    updateCard(m.id, {
      status: { text: snapLine(snap), muted: true },
      spark: null,
      // The nearest gauge, when it is near enough to matter.
      figure: near && near.d <= NEAR_GAUGE_KM ? { value: near.d < 10 ? near.d.toFixed(1) : String(Math.round(near.d)), unit: " km",
        label: `to the nearest gauge, ${near.r.name || near.r.station_id}` } : null,
      buttons: [
        ...(take ? [take] : []),
        ...(m.spec.buttons || []).filter((b) => b.id !== "trace" && b.id !== "reach"),
      ],
    });
    return;
  }
  const order = snap.strahler_order ? `, stream order ${snap.strahler_order}` : "";
  updateCard(m.id, {
    what: "River", title: `River reach ${snap.river_id}`, sub: `${coords(m.lat, m.lon)}${order}`,
    lngLat: Number.isFinite(snap.snap_lat) ? [snap.snap_lon, snap.snap_lat] : m.spec.lngLat,
    status: { pending: "Reading the 15-day forecast…" },
    credit: GEOGLOWS_SHORT,
    details: () => openDetails("panel-point", "now"),
    // A reach taken from the offer above gets its Trace button back.
    buttons: [traceButton("pt", { ready: Boolean(m.reach) }),
      ...(m.spec.buttons || []).filter((b) => b.id !== "trace" && b.id !== "reach")],
  });
  void loadForecast(m, snap);
}

// The quick forecast from a light worker (the next 15 days only), then the simulated record's status from
// the main worker; the same two steps the Now tab takes.
async function loadForecast(m, snap) {
  const args = { lat: snap.snap_lat, lon: snap.snap_lon, river_id: snap.river_id };
  let quick = null;
  try {
    quick = await callLight("now", { op: "forecast", args: { ...args, history: false, glofas: false } }, { priority: 1 });
  } catch { /* the full call below can still answer */ }
  if (model !== m) return;
  const g = quick && quick.geoglows;
  if (g && !g.error && Array.isArray(g.mean)) {
    const peak = forecastPeak(g);
    updateCard(m.id, {
      spark: { v: g.mean, band: { lo: g.p25 || [], hi: g.p75 || [] }, label: "next 15 days, modelled", fromZero: false,
        alt: "GEOGLOWS ensemble mean flow for the next 15 days, with the middle half of the ensemble shaded" },
      figure: peak ? { value: cardNumber(peak.value), unit: " m³/s", label: `15-day peak, ${dayMonth(peak.date, { short: true })}` } : null,
      status: { pending: "Comparing with this river's 86 simulated years…" },
    });
  } else {
    updateCard(m.id, { spark: { empty: (g && g.error) || "The forecast did not answer this time." }, figure: null });
  }
  try {
    const full = await call("now", { op: "forecast", args: { ...args, glofas: false,
      known_geoglows: g && !g.error ? g : null } });
    if (model !== m) return;
    const st = full.status;
    const cls = st && statusClass(st.class);
    if (st && st.sentence) {
      updateCard(m.id, { status: { text: st.sentence, color: cls ? cls.color : null, muted: !cls },
        credit: `${GEOGLOWS_SHORT}; normal from its simulated record since 1940.` });
    } else {
      updateCard(m.id, { status: { text: full.sentence || "No status for this reach.", muted: true } });
    }
  } catch {
    if (model === m) updateCard(m.id, { status: { text: "The simulated record did not answer; Details has the forecast.", muted: true } });
  }
}

// ── wiring ──────────────────────────────────────────────────────────────────

const panelOpen = () => !document.body.classList.contains("panel-collapsed");

// The card for what is selected now, shown when the panel is folded away.
function syncToSelection({ focus = true } = {}) {
  let next = null;
  if (state.selected) next = `gauge:${stationKey(state.selected)}`;
  else if (state.point) next = `point:${state.point.lat},${state.point.lon}`;
  if (!next) { closeCard({ restore: false }); return; }
  if (!model || model.id !== next) {
    model = state.selected ? gaugeModel(state.selected) : pointModel(state.point);
  }
  if (panelOpen()) { closeCard({ restore: false, keepModel: true }); return; }
  openCard(model.spec, { focus });   // the same object: the fills below patch it in place
  if (model.kind === "gauge") fillGauge(model); else fillPoint(model);
}

function hint(on) {
  const h = $("map-hint");
  if (h) h.hidden = !on;
}

export function initMapCard() {
  const card = el();
  if (!card) return;
  document.addEventListener("aq:surface", (e) => {
    const id = e.detail && e.detail.id;
    if (id === "panel-station" || id === "panel-point") {
      hint(false);
      syncToSelection();
    } else {
      closeCard({ restore: false });
      hint(false);
    }
  });
  // Details opens the panel and the card steps aside; folding the panel brings the card back.
  document.addEventListener("aq:panel", (e) => {
    if (e.detail && e.detail.open) { closeCard({ restore: false, keepModel: true }); hint(false); }
    else if ((state.selected && !$("panel-station").hidden) || (state.point && !$("panel-point").hidden)) {
      syncToSelection({ focus: false });
    }
  });
  $("drawer").addEventListener("drawermode", (e) => { if (e.detail && e.detail.open) closeCard({ restore: false, keepModel: true }); });
  $("panel-station").addEventListener("analysis", (e) => {
    const d = e.detail || {};
    if (model && model.kind === "gauge" && model.key === d.key) {
      model.analysis = d;
      if (model.filled) onAnalysis(model, d);
    }
  });
  for (const [t, id] of [["st", "panel-station"], ["pt", "panel-point"]]) {
    const root = $(id);
    root.addEventListener("riversnap", (e) => {
      const snap = e.detail || null;
      if (t !== "pt" || !model || model.kind !== "point") return;
      if (snap && snap.lat !== undefined && (snap.lat !== model.lat || snap.lon !== model.lon)) return;
      model.snap = snap || { error: true };
      if (model.filled) onSnap(model, model.snap);
    });
    root.addEventListener("reachchange", (e) => {
      if (!model || (t === "st") !== (model.kind === "gauge")) return;
      const reach = e.detail;
      model.reach = reach || null;
      if (!reach) return;
      setButton("trace", traceButton(t, { ready: true }));
      refreshWatch();
    });
  }
  // Escape closes the card from inside it, and from the map (where the click that opened it left focus).
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !shown) return;
    const a = document.activeElement;
    // ... and from the map's own buttons (the panel handle keeps focus after folding the panel).
    const fromMap = !a || a === document.body || (a.closest && a.closest("#map, .map-tool"));
    if (!(card.contains(a) || fromMap)) return;
    e.preventDefault();
    closeCard();
  });
  actions.openMapCard = openCard;
  actions.closeMapCard = closeCard;
  // Nothing chosen yet and the panel folded: one line says what to do.
  hint(!state.selected && !state.point && !panelOpen());
  if (state.mapOk && map) map.once("click", () => hint(false));
}
