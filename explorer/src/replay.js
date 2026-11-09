// Replay an event (#659): a past flood told as a story on the map, the camera flying between stops.
//
// The package assembles the event in the worker (aquascope.replay.event_record: IMERG over the basin through
// ClimateSERV, the gauges against GEOGLOWS, what the forecast archive said 1, 3 and 7 days before the peak, the
// observed flow routed down the river with Route it) and writes the story (aquascope.replay.story): the stops, each
// a camera, a date on the time bar, the layers and a note filled by code. This module only plays it: it flies the
// camera, moves the time bar, turns the layers on, draws the basin and the gauge, hands the wave to Route it's
// drawing, runs Flood it on the GPU at the peak flow and counts what is under it (the two notes the page fills from
// its own run, with the package's templates), and exports the story as a GIF, a WebM video or a Markdown report.
// A link (#rp=...) plays it again. Every replay is an entry in the action log.

import { $, actions, escapeHtml, setTime, state } from "./core.js?v=__BUILD__";
import { captureMap, map, whenSettled } from "./map.js?v=__BUILD__";
import { call, onReplayProgress } from "./worker-client.js?v=__BUILD__";
import { logAction, registerSlotWriter, setMapLayer, slotValue } from "./map-actions.js?v=__BUILD__";
import { clearRouted, showRouted } from "./route-map.js?v=__BUILD__";
import { drawExposureOf, exposureOfShown, floodIt } from "./floodsim-map.js?v=__BUILD__";
import { announce } from "./a11y.js?v=__BUILD__";
import { gifSize } from "./timeline.js?v=__BUILD__";
import {
  cameraFor, checkStory, decodeReplay, dwellMs, encodeReplay, exposureFill, fillTemplate, floodFill, forecastBar,
  pieceRows, scrubSegments, sparkline,
} from "./replay-core.js?v=__BUILD__";

const GIFENC = "https://cdn.jsdelivr.net/npm/gifenc@1.0.3/dist/gifenc.esm.js";
const BASIN = "replay-basin";
const DOCS = "https://rekin226.github.io/aquascope/replay/";

let cur = null;          // { args, record, story, stops, i, playing, flood, expo, wave, layersOn }
let run = 0;             // a new replay or a close cancels what the old one was doing
let timer = null;
let marker = null;
let busy = null;         // "gif" | "video" while exporting

const panel = () => $("replay");
const inPanel = (sel) => panel().querySelector(sel);
function showBody(html) {
  $("rp-stop").hidden = true;
  $("rp-body").hidden = false;
  $("rp-body").innerHTML = html;
}
const narrow = () => (globalThis.innerWidth || 1024) <= 640;
const still = () => Boolean(globalThis.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
const day = (iso) => {
  const d = new Date(`${String(iso).slice(0, 10)}T12:00:00Z`);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
};

// ── opening: from Floods past, from a reach, from a link, from Ask the map ─────

/** Floods past's floods near a point in some months, as buttons (the Floods past card). */
export async function candidatesHtml({ lat, lon, start, end }) {
  const res = await call("replay", { op: "candidates", args: { lat, lon, start, end, radius_km: 30 } });
  if (!res || res.ok === false || !(res.candidates || []).length) return { html: "", list: [] };
  const list = res.candidates.slice(0, 3);
  const html = `<div class="fp-replay"><p class="fp-pop-note">Replay a flood here:</p>` + list.map((c, k) =>
    `<button type="button" class="btn tiny fp-rp" data-k="${k}">${escapeHtml(c.start === c.end ? day(c.start) : `${day(c.start)} to ${day(c.end)}`)}` +
    ` <span class="muted">${c.reports.toLocaleString("en-GB")} report${c.reports === 1 ? "" : "s"}</span></button>`).join("") + "</div>";
  return { html, list };
}

/** The reach card's Replay: pick a date (the map's, by default) and the floods Floods past has there that month. */
export function openReplayPicker(t) {
  stop();
  const p = panel();
  p.hidden = false;
  p.classList.remove("playing");
  const d = (state.date || new Date().toISOString().slice(0, 10)).slice(0, 10);
  $("rp-title").textContent = t.name || `River reach ${t.river_id}`;
  showBody(`<form class="rp-pick" id="rp-pick">
      <label>The flood's date <input type="date" id="rp-date" value="${escapeHtml(d)}" max="${new Date().toISOString().slice(0, 10)}" min="2000-06-01"></label>
      <button type="submit" class="btn tiny primary">Replay</button></form>
    <p class="rp-hint muted">The days around it: the rain, the gauge against GEOGLOWS, what the forecast said, the wave and the flood.</p>
    <div id="rp-cands" class="rp-cands"></div>`);
  $("rp-ctrl").hidden = true;
  inPanel("#rp-pick").addEventListener("submit", (e) => {
    e.preventDefault();
    const v = inPanel("#rp-date").value;
    if (v) startReplay({ river_id: t.river_id, lat: t.lat, lon: t.lon, start: v, end: v });
  });
  const ym = d.slice(0, 7);
  call("replay", { op: "candidates", args: { lat: t.lat, lon: t.lon, start: ym, end: ym, radius_km: 30 } }).then((res) => {
    const box = inPanel("#rp-cands");
    if (!box || !res || !(res.candidates || []).length) return;
    box.innerHTML = `<p class="muted">Floods in the news near here, ${escapeHtml(day(`${ym}-15`).replace(/^\d+ /, ""))}:</p>` +
      res.candidates.slice(0, 3).map((c, k) => `<button type="button" class="btn tiny" data-k="${k}">${escapeHtml(day(c.start))}` +
        `${c.end !== c.start ? ` to ${escapeHtml(day(c.end))}` : ""} <span class="muted">${c.reports} reports</span></button>`).join("");
    box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      const c = res.candidates[Number(b.dataset.k)];
      startReplay({ river_id: t.river_id, lat: t.lat, lon: t.lon, start: c.start, end: c.end, reports: c.reports });
    }));
  }).catch(() => {});
}

/**
 * Assemble a flood and play it. ev = { river_id?, lat, lon, start, end, reports?, event? } (a Floods past candidate
 * is an event: its centre picks the river). Logged in the action log; resolves to the story, or null.
 */
export async function startReplay(ev, { by = "rules", label = "", command = "" } = {}) {
  const my = ++run;
  stop();
  const before = slotValue("replay");
  if (actions.closeMapCard) actions.closeMapCard({ restore: false });   // the story owns the map now
  const p = panel();
  p.hidden = false;
  p.classList.remove("playing");
  $("rp-ctrl").hidden = true;
  $("rp-title").textContent = label || "Assembling the flood…";
  const args = ev.event ? { event: ev.event, river_id: ev.river_id ?? null }
    : { river_id: ev.river_id ?? null, lat: ev.lat ?? null, lon: ev.lon ?? null, start: ev.start, end: ev.end || ev.start };
  if (ev.reports && !ev.event) args.event = { start: ev.start, end: ev.end || ev.start, lat: ev.lat, lon: ev.lon, reports: ev.reports };
  if (args.event && ev.river_id) { args.lat = ev.lat; args.lon = ev.lon; }
  const lines = [];
  const say = (text) => { lines.push(text); renderBuild(null, lines); };
  renderBuild(null, ["Starting the engine (once), then finding the river…"]);
  // The camera goes to the flood while the data is read, so the wait is spent looking at the place.
  const lat = Number(ev.lat ?? (ev.event && ev.event.lat)), lon = Number(ev.lon ?? (ev.event && ev.event.lon));
  if (state.mapOk && map && Number.isFinite(lat) && Number.isFinite(lon)) {
    const cam = { center: [lon, lat], zoom: narrow() ? 7.2 : 8, pitch: 0, bearing: 0 };
    if (still()) map.jumpTo(cam); else map.flyTo({ ...cam, duration: 2500, essential: true });
  }
  const off = onReplayProgress((text) => { if (my === run) say(text); });
  try {
    let record = await call("replay", { op: "record", args });
    if (my !== run) return null;
    if (!record || record.ok === false) throw new Error((record && record.error) || "no answer");
    renderBuild(record, lines);
    // The rain is a ClimateSERV request: ask again every few seconds, up to two minutes.
    for (let k = 0; k < 40 && record.pieces.rain && record.pieces.rain.pending; k++) {
      await new Promise((r) => setTimeout(r, 3000));
      if (my !== run) return null;
      const rain = await call("replay", { op: "rain", args: { piece: record.pieces.rain } });
      if (rain && rain.ok !== false) record.pieces.rain = rain;
      renderBuild(record, lines);
    }
    if (record.pieces.rain && record.pieces.rain.pending) {
      record.pieces.rain = { available: false, piece: "rain", note: "ClimateSERV did not finish in two minutes.", licence: record.pieces.rain.licence };
    }
    const made = await call("replay", { op: "story", args: { record } });
    if (my !== run) return null;
    if (!made || made.ok === false) throw new Error((made && made.error) || "no story");
    const stops = checkStory(made.story);
    cur = { args: ev, record, story: made.story, report: made.report, stops, i: -1, playing: false, flood: null,
      expo: null, wave: false, layersOn: new Set() };
    logAction({ action: { type: "replay", ...replayAction(ev) }, before, after: { ev, story: made.story, record },
      label: label || `Replay: ${made.story.event.name}`, by, detail: `${stops.length} stops`, command });
    showPlayer();
    announce(`${made.story.title}: ${stops.length} stops. Playing.`);
    play();
    return made.story;
  } catch (err) {
    if (my === run) {
      $("rp-title").textContent = "Could not replay this flood";
      showBody(`<p class="rp-err">${escapeHtml(err.message)}</p>`);
    }
    return null;
  } finally {
    off();
  }
}

const replayAction = (ev) => ({ ...(ev.river_id ? { river_id: ev.river_id } : {}), start: ev.start, end: ev.end });

function renderBuild(record, lines) {
  const rows = pieceRows(record || { pieces: {} });
  const mark = { ok: "✓", missing: "–", pending: "…", waiting: "·" };
  showBody(`<ul class="rp-build">${rows.map((r) => `<li class="${r.state}"><i aria-hidden="true">${mark[r.state]}</i>` +
    `<span>${escapeHtml(r.label)}${r.text ? ` <small>${escapeHtml(r.text)}</small>` : ""}</span></li>`).join("")}</ul>` +
    (lines.length ? `<p class="rp-progress" role="status"><span class="spinner" aria-hidden="true"></span>${escapeHtml(lines.at(-1))}</p>` : ""));
}

// ── the player ──────────────────────────────────────────────────────────────

function showPlayer() {
  const p = panel();
  p.classList.add("playing");
  document.body.classList.add("rp-on");
  $("rp-title").textContent = cur.story.event.name;
  $("rp-ctrl").hidden = false;
  $("rp-body").innerHTML = "";
  $("rp-body").hidden = true;
  $("rp-stop").hidden = false;
  const segs = scrubSegments(cur.stops);
  $("rp-scrub").innerHTML = segs.map((s) => `<button type="button" class="rp-seg" data-i="${s.i}" style="flex-grow:${(s.width * 100).toFixed(2)}" ` +
    `title="${escapeHtml(cur.stops[s.i].title)}" aria-label="${escapeHtml(`Stop ${s.i + 1}: ${cur.stops[s.i].title}`)}"><span></span></button>`).join("");
  $("rp-scrub").querySelectorAll(".rp-seg").forEach((b) => b.addEventListener("click", () => { pause(); goTo(Number(b.dataset.i)); }));
  $("rp-src").innerHTML = `Every number is written by code from the data. <a href="${DOCS}" target="_blank" rel="noopener">How</a> · ` +
    escapeHtml(cur.story.sources.filter((s) => s.available).map((s) => s.licence.split(";")[0].split(":")[0]).slice(0, 4).join(" · "));
}

function clearTimer() { if (timer) { clearTimeout(timer); timer = null; } }

function play() {
  if (!cur) return;
  cur.playing = true;
  syncControls();
  if (cur.i < 0 || cur.i >= cur.stops.length - 1) goTo(cur.i < 0 ? 0 : 0);
  else goTo(cur.i + 1);
}

function pause() {
  if (!cur) return;
  cur.playing = false;
  clearTimer();
  syncControls();
}

function syncControls() {
  const b = $("rp-play");
  if (!b || !cur) return;
  b.textContent = cur.playing ? "❚❚" : "▶";
  b.setAttribute("aria-label", cur.playing ? "Pause" : "Play");
  b.setAttribute("aria-pressed", String(cur.playing));
  $("rp-count").textContent = `${Math.max(0, cur.i) + 1} / ${cur.stops.length}`;
  $("rp-scrub").querySelectorAll(".rp-seg").forEach((el) => {
    const i = Number(el.dataset.i);
    el.classList.toggle("done", i < cur.i);
    el.classList.toggle("on", i === cur.i);
  });
}

function scheduleNext(stopObj) {
  clearTimer();
  if (!cur || !cur.playing) return;
  if (cur.i >= cur.stops.length - 1) { cur.playing = false; syncControls(); return; }
  timer = setTimeout(() => { if (cur && cur.playing) goTo(cur.i + 1); }, dwellMs(stopObj, { still: still() }));
}

/** Show stop i: the camera, the date, the layers, the marks and the note. */
export async function goTo(i, { fly = true } = {}) {
  if (!cur) return;
  const my = run;
  const st = cur.stops[Math.max(0, Math.min(cur.stops.length - 1, i))];
  cur.i = cur.stops.indexOf(st);
  clearTimer();
  syncControls();
  $("rp-stitle").textContent = st.title;
  setNote(st);
  applyLayers(st);
  if (st.date) setTime({ date: st.date }, { source: "replay" });
  drawMarks(st);
  if (st.kind !== "wave" && cur.wave) { try { clearRouted(); } catch { /* gone */ } cur.wave = false; }
  floodVisible(st.kind === "flood" || st.kind === "exposure");
  frame(st, { fly });
  if (st.kind === "wave") showWave();
  if (st.kind === "flood" || st.kind === "exposure") {
    await ensureFlood(st);
    if (my !== run || !cur) return;
    setNote(cur.stops[cur.i]);
    if (cur.stops[cur.i] === st) frame(st, { fly });   // Flood it framed its own run; the story's camera again
  }
  if (my !== run || !cur || cur.stops[cur.i] !== st) return;
  scheduleNext(st);
}

// The note card sits at the bottom: the camera keeps the subject above it.
const camPadding = () => ({ top: 70, bottom: narrow() ? Math.round((globalThis.innerHeight || 800) * 0.36) : 200, left: 40, right: 40 });

// Where most of the flooded buildings are: the exposure stop looks there once they are counted.
function expoCentre() {
  const pts = ((cur && cur.expo && cur.expo.newly_flooded) || []).map((b) => b.ring && b.ring[0]).filter(Boolean);
  if (pts.length < 3) return null;
  const med = (k) => pts.map((q) => q[k]).sort((a, b) => a - b)[Math.floor(pts.length / 2)];
  return [med(0), med(1)];
}

function frame(st, { fly = true } = {}) {
  if (!state.mapOk || !map) return;
  const cam = { ...cameraFor(st, { narrow: narrow() }), padding: camPadding() };
  if (st.kind === "exposure" && expoCentre()) cam.center = expoCentre();
  if (!fly || still()) map.jumpTo(cam);
  else map.flyTo({ ...cam, duration: 2600, essential: true });
}

// Flood it's depth and the flooded buildings belong to the flood stops: the stops before and after leave them out.
function floodVisible(on) {
  if (!state.mapOk || !map) return;
  for (const l of (map.getStyle() && map.getStyle().layers) || []) {
    if (/^floodsim-(depth|expo)/.test(l.id)) map.setLayoutProperty(l.id, "visibility", on ? "visible" : "none");
  }
}

function setNote(st) {
  let note = st.note;
  if (st.kind === "flood" && cur.flood && cur.flood.stats) {
    note = fillTemplate(st.template, floodFill(cur.flood.stats, st.fill || {}));
  } else if (st.kind === "flood" && cur.flood && cur.flood.error) {
    note = `Flood it could not run here: ${cur.flood.error}`;
  } else if (st.kind === "exposure" && st.needs === "exposure") {
    const fill = exposureFill(cur.expo);
    note = fill ? fillTemplate(st.template, fill) : (cur.expo && cur.expo.error ? `What is under the water could not be counted: ${cur.expo.error}` : st.note);
  }
  st.shown = note;
  $("rp-note").textContent = note;
  const chart = $("rp-chart");
  if (st.series) chart.innerHTML = sparkline(st.series, { marker: st.date });
  else if (st.kind === "forecast" && st.run) {
    const g = (cur.record.pieces.gauges.gauges || [])[0];
    chart.innerHTML = forecastBar(st.run, g && g.obs_peak ? g.obs_peak.cms : null);
  } else chart.innerHTML = "";
  chart.hidden = !chart.innerHTML;
}

// The layers a stop names, by Ask the map's ids; the ones this replay turned on go off when no stop wants them.
const LAYER_IDS = new Set(["precip", "floods_past", "rivers"]);
function applyLayers(st) {
  const want = new Set((st.layers || []).filter((l) => LAYER_IDS.has(l)));
  for (const id of want) {
    try { setMapLayer(id, true); cur.layersOn.add(id); } catch { /* not on this page */ }
  }
  for (const id of [...cur.layersOn]) {
    if (!want.has(id) && id === "precip") { try { setMapLayer(id, false); } catch { /* gone */ } cur.layersOn.delete(id); }
  }
  showBasin((st.layers || []).includes("basin") ? st.basin : null, st.outline);
}

// The upstream basin the rain was averaged over: its cells faintly filled, its outline (the edges of one cell only).
function showBasin(geom, outline = null) {
  if (!state.mapOk || !map) return;
  const feats = [];
  if (geom) feats.push({ type: "Feature", properties: { k: "fill" }, geometry: geom });
  if (geom && outline) feats.push({ type: "Feature", properties: { k: "line" }, geometry: outline });
  const data = { type: "FeatureCollection", features: feats };
  if (!map.getSource(BASIN)) {
    map.addSource(BASIN, { type: "geojson", data });
    map.addLayer({ id: `${BASIN}-fill`, type: "fill", source: BASIN, filter: ["==", ["get", "k"], "fill"],
      paint: { "fill-color": "#0b3d91", "fill-opacity": 0.12, "fill-antialias": false } });
    map.addLayer({ id: `${BASIN}-line`, type: "line", source: BASIN, filter: ["==", ["get", "k"], "line"],
      paint: { "line-color": "#0b3d91", "line-width": 2.2, "line-opacity": 0.9 } });
  } else map.getSource(BASIN).setData(data);
}

function drawMarks(st) {
  if (marker) { marker.remove(); marker = null; }
  if (!state.mapOk || !map || !st.focus || ["wave", "flood", "exposure", "end"].includes(st.kind)) return;
  const el = document.createElement("div");
  el.className = `rp-mark rp-${st.kind}`;
  const g = (cur.record.pieces.gauges.gauges || [])[0];
  const text = st.kind === "gauge" && g ? g.name : st.kind === "forecast" ? `${st.run && st.run.lead_days} d ahead` : "";
  el.innerHTML = `<i aria-hidden="true"></i>${text ? `<span>${escapeHtml(text)}</span>` : ""}`;
  marker = new maplibregl.Marker({ element: el, anchor: "left", offset: [-7, 0] }).setLngLat([st.focus.lon, st.focus.lat]).addTo(map);
}

function showWave() {
  const w = cur.record.pieces.wave;
  if (!w || !w.available || cur.wave) return;
  cur.wave = true;
  // The hours on the map count from the peak passing the gauge, as the note does, not from the series' first day.
  const t0 = (w.reaches && w.reaches[0] && w.reaches[0].peak_h) || 0;
  const shift = (r) => ({ ...r, peak_h: r.peak_h === null || r.peak_h === undefined ? r.peak_h : Math.round((r.peak_h - t0) * 10) / 10 });
  const result = { ok: true, river_id: w.river_id, reaches: (w.reaches || []).map(shift), frames: w.frames, geometry: w.geometry, hours: w.hours,
    inflow: w.inflow || { peak_cms: w.inflow_peak_cms, source: w.inflow_source, shape: "series", duration_h: w.hours },
    places: (w.route_places || []).map(shift).filter((pl) => pl.peak_h === null || pl.peak_h >= 0), furthest: w.furthest,
    check: { summary: w.check } };
  try { showRouted(result, { river_id: w.river_id, lat: cur.story.event.lat, lon: cur.story.event.lon }, { label: "Replay: the wave" }); }
  catch (err) { console.warn("replay wave:", err); }
}

async function ensureFlood(st) {
  const fp = cur.record.pieces.flood || {};
  if (cur.flood || !fp.pending || !st.run) { if (st.kind === "exposure") await ensureExposure(); return; }
  const t = { river_id: st.run.river_id, lat: st.run.lat, lon: st.run.lon, name: cur.story.event.name };
  $("rp-note").textContent = `Flood it is running at ${Math.round(st.run.flow_cms).toLocaleString("en-GB")} m³/s on your GPU…`;
  const pending = floodIt(t, { choice: { flow_cms: st.run.flow_cms }, by: "rules" });
  const fi = $("floodit");
  if (fi) fi.hidden = true;   // the replay's note says what Flood it says; one panel on the map
  cur.flood = { pending: true };
  try {
    const r = await pending;
    if (!cur) return;
    cur.flood = r && r.stats ? { stats: r.stats, engine: r.engine } : { error: "the run did not finish" };
  } catch (err) { if (cur) cur.flood = { error: err.message }; }
  if (fi) fi.hidden = true;
  if (st.kind === "exposure") await ensureExposure();
  else void ensureExposure();
}

async function ensureExposure() {
  if (!cur || cur.expo || !cur.flood || !cur.flood.stats) return;
  cur.expo = { pending: true };
  try { cur.expo = (await exposureOfShown()) || { error: "no run on the map" }; }
  catch (err) { if (cur) cur.expo = { error: err.message }; }
  if (cur && cur.expo && cur.expo.newly_flooded) { try { drawExposureOf(cur.expo); } catch { /* the count stands */ } }
  if (cur && cur.stops[cur.i] && cur.stops[cur.i].kind === "exposure") frame(cur.stops[cur.i]);
  if (cur && cur.stops[cur.i] && cur.stops[cur.i].kind === "exposure") setNote(cur.stops[cur.i]);
}

export function stop() {
  clearTimer();
  if (marker) { marker.remove(); marker = null; }
  if (cur && cur.layersOn.has("precip")) { try { setMapLayer("precip", false); } catch { /* gone */ } }
  showBasin(null);
  document.body.classList.remove("rp-on");
  cur = null;
}

function close() {
  run++;
  const before = slotValue("replay");
  stop();
  panel().hidden = true;
  if (before) logAction({ action: { type: "replay" }, before, after: null, label: "Replay closed" });
}

// ── what the page computed itself, for the report ───────────────────────────

function pageFlood() {
  if (!cur || !cur.flood || !cur.flood.stats) return null;
  const fp = cur.record.pieces.flood || {};
  const s = cur.flood.stats;
  const out = { flow_cms: fp.flow_cms, flow_source: fp.flow_source, wet_km2: Number(s.wet_km2.toFixed(3)),
    max_depth_m: Number((s.max_m ?? 0).toFixed(2)), engine: cur.flood.engine };
  const e = cur.expo;
  if (e && e.buildings) {
    out.exposure = { available: true, buildings: e.buildings.after, roads_km: e.roads_km ? e.roads_km.after : null,
      people: e.people ? e.people.after : null, source: "Overture Maps buildings and roads; GHS-POP R2023A",
      licence: "Overture Maps, ODbL 1.0; GHS-POP R2023A, CC BY 4.0" };
  }
  return out;
}

async function downloadReport() {
  if (!cur) return;
  let md = cur.report;
  const fl = pageFlood();
  if (fl) {
    const made = await call("replay", { op: "story", args: { record: cur.record, flood: fl } });
    if (made && made.report) md = made.report;
  }
  save(`${fileStem()}.md`, new Blob([md], { type: "text/markdown" }));
}

function fileStem() {
  const e = cur.story.event;
  return `aquascope-replay-${e.river_id}-${e.peak_date || e.start}`;
}

function save(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

// The link names the reach the story was made on (so it does not depend on the river picker), the window, the place
// and, for a Floods past event, its reports: a fresh page replays the same story.
function shareLink() {
  if (!cur) return "";
  const e = cur.story.event, a = cur.args || {}, fe = a.event || {};
  const pick = (...v) => v.find((x) => x !== undefined && x !== null && x !== "");
  const ev = { river_id: pick(a.river_id, e.river_id), start: pick(fe.start, a.start, e.start), end: pick(fe.end, a.end, e.end),
    lat: pick(fe.lat, a.lat, e.lat), lon: pick(fe.lon, a.lon, e.lon), reports: pick(fe.reports) };
  return `${location.origin}${location.pathname}#rp=${encodeReplay(ev)}`;
}

// ── export: a GIF (one frame per stop) or a WebM video of the story playing ────

function wrap(ctx, text, maxW) {
  const words = String(text).split(/\s+/);
  const lines = [];
  let line = "";
  for (const w of words) {
    const t = line ? `${line} ${w}` : w;
    if (ctx.measureText(t).width > maxW && line) { lines.push(line); line = w; } else line = t;
  }
  if (line) lines.push(line);
  return lines;
}

/** The note on the frame, at the bottom, and the credit: a GIF or a video travels without the page. */
function stampFrame(ctx, w, h, st) {
  const pad = Math.round(w / 48);
  // The stop's place, as the page marks it (an HTML marker is not in the map's canvas).
  if (st.focus && !["wave", "flood", "exposure", "end"].includes(st.kind) && state.mapOk && map) {
    const q = map.project([st.focus.lon, st.focus.lat]);
    const c = map.getCanvas();
    const x = (q.x / c.clientWidth) * w, y = (q.y / c.clientHeight) * h;
    ctx.beginPath(); ctx.arc(x, y, Math.max(5, w / 110), 0, Math.PI * 2);
    ctx.fillStyle = st.kind === "forecast" ? "#c4581c" : "#0b6bb8"; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = "#ffffff"; ctx.stroke();
    const g = (cur.record.pieces.gauges.gauges || [])[0];
    const label = st.kind === "gauge" && g ? g.name : st.kind === "forecast" && st.run ? `${st.run.lead_days} d ahead` : "";
    if (label) {
      ctx.font = `600 ${Math.max(11, Math.round(w / 60))}px system-ui, -apple-system, sans-serif`;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(255, 255, 255, .92)"; ctx.fillRect(x + 10, y - 9, tw + 10, 18);
      ctx.fillStyle = "#0d1b26"; ctx.textBaseline = "middle"; ctx.fillText(label, x + 15, y);
    }
  }
  const big = Math.max(12, Math.round(w / 44));
  const small = Math.max(10, Math.round(big * 0.8));
  ctx.font = `${small}px system-ui, -apple-system, sans-serif`;
  const lines = wrap(ctx, st.shown || st.note, w - pad * 4).slice(0, 4);
  const boxH = big + 6 + lines.length * (small + 4) + pad;
  ctx.fillStyle = "rgba(10, 22, 32, .78)";
  ctx.fillRect(pad, h - boxH - pad, w - pad * 2, boxH);
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "top";
  ctx.font = `600 ${big}px system-ui, -apple-system, sans-serif`;
  ctx.fillText(`${st.title}${st.date ? `, ${day(st.date)}` : ""}`, pad * 2, h - boxH - pad + pad * 0.6);
  ctx.font = `${small}px system-ui, -apple-system, sans-serif`;
  lines.forEach((l, k) => ctx.fillText(l, pad * 2, h - boxH - pad + pad * 0.6 + big + 6 + k * (small + 4)));
  const credit = "AquaScope Explorer · IMERG (NASA) · GEOGLOWS v2 · Floods past · OpenMapTiles";
  ctx.font = `${Math.max(9, small - 2)}px system-ui, -apple-system, sans-serif`;
  const cw = ctx.measureText(credit).width;
  ctx.fillStyle = "rgba(10, 22, 32, .6)";
  ctx.fillRect(w - cw - pad * 2, pad, cw + pad * 1.4, small + pad * 0.6);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(credit, w - cw - pad * 1.3, pad * 1.3);
}

async function exportGif() {
  if (!cur || busy) return;
  busy = "gif";
  pause();
  const status = $("rp-export");
  const { GIFEncoder, quantize, applyPalette } = await import(GIFENC);
  const gif = GIFEncoder();
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  let size = null;
  try {
    for (let i = 0; i < cur.stops.length; i++) {
      status.textContent = `Making the GIF: stop ${i + 1} of ${cur.stops.length}…`;
      await goTo(i, { fly: false });
      pause();
      await whenSettled(9000);
      const st = cur.stops[i];
      const ok = await captureMap((src) => {
        if (!size) { size = gifSize(src.width, src.height, 720); canvas.width = size.width; canvas.height = size.height; }
        ctx.drawImage(src, 0, 0, size.width, size.height);
      });
      if (!ok) throw new Error("the map could not be read");
      stampFrame(ctx, size.width, size.height, st);
      const { data } = ctx.getImageData(0, 0, size.width, size.height);
      const palette = quantize(data, 256);
      gif.writeFrame(applyPalette(data, palette), size.width, size.height, { palette, delay: Math.min(6000, dwellMs(st)) });
      await new Promise((r) => setTimeout(r, 0));
    }
    gif.finish();
    save(`${fileStem()}.gif`, new Blob([gif.bytes()], { type: "image/gif" }));
    status.textContent = "GIF saved.";
  } catch (err) {
    status.textContent = `Could not make the GIF: ${err.message}`;
  } finally { busy = null; }
}

function videoType() {
  if (typeof MediaRecorder === "undefined") return null;
  for (const t of ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"]) {
    if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) return t;
  }
  return null;
}

async function exportVideo() {
  if (!cur || busy) return;
  const type = videoType();
  const status = $("rp-export");
  if (!type) { status.textContent = "This browser cannot record a video (no MediaRecorder); the GIF works everywhere."; return; }
  busy = "video";
  pause();
  const src = map.getCanvas();
  const size = gifSize(src.width, src.height, 1280);
  const canvas = document.createElement("canvas");
  canvas.width = size.width; canvas.height = size.height;
  const ctx = canvas.getContext("2d");
  const onRender = () => {
    try { ctx.drawImage(src, 0, 0, size.width, size.height); const st = cur && cur.stops[cur.i]; if (st) stampFrame(ctx, size.width, size.height, st); }
    catch { /* a frame is skipped */ }
  };
  map.on("render", onRender);
  const rec = new MediaRecorder(canvas.captureStream(30), { mimeType: type, videoBitsPerSecond: 4_000_000 });
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  const done = new Promise((r) => { rec.onstop = r; });
  rec.start(500);
  try {
    for (let i = 0; i < cur.stops.length; i++) {
      status.textContent = `Recording: stop ${i + 1} of ${cur.stops.length}…`;
      const flight = goTo(i);
      pause();
      await flight;
      await new Promise((r) => setTimeout(r, 2700));
      await whenSettled(6000);
      await new Promise((r) => setTimeout(r, Math.min(6000, dwellMs(cur.stops[i]) - 1500)));
      map.triggerRepaint();
    }
  } finally {
    rec.stop();
    await done;
    map.off("render", onRender);
    busy = null;
  }
  const ext = type.startsWith("video/mp4") ? "mp4" : "webm";
  save(`${fileStem()}.${ext}`, new Blob(chunks, { type: type.split(";")[0] }));
  status.textContent = "Video saved.";
}

// ── boot ────────────────────────────────────────────────────────────────────

export function initReplay() {
  const p = panel();
  if (!p) return;
  $("rp-close").addEventListener("click", close);
  $("rp-play").addEventListener("click", () => (cur && cur.playing ? pause() : play()));
  $("rp-prev").addEventListener("click", () => { if (cur) { pause(); goTo(Math.max(0, cur.i - 1)); } });
  $("rp-next").addEventListener("click", () => { if (cur) { pause(); goTo(Math.min(cur.stops.length - 1, cur.i + 1)); } });
  $("rp-more").addEventListener("click", () => {
    const m = $("rp-menu");
    m.hidden = !m.hidden;
    $("rp-more").setAttribute("aria-expanded", String(!m.hidden));
  });
  $("rp-menu").addEventListener("click", async (e) => {
    const act = e.target.closest("button") && e.target.closest("button").dataset.act;
    if (!act) return;
    if (act === "gif") await exportGif();
    else if (act === "video") await exportVideo();
    else if (act === "report") await downloadReport();
    else if (act === "link") {
      const url = shareLink();
      try { await navigator.clipboard.writeText(url); $("rp-export").textContent = "Link copied."; }
      catch { $("rp-export").textContent = url; }
    }
  });
  document.addEventListener("keydown", (e) => {
    if (!cur || p.hidden || /^(INPUT|TEXTAREA|SELECT)$/.test((e.target && e.target.tagName) || "")) return;
    if (e.key === "ArrowRight") { pause(); goTo(cur.i + 1); }
    else if (e.key === "ArrowLeft") { pause(); goTo(Math.max(0, cur.i - 1)); }
  });
  registerSlotWriter("replay", (value) => {
    if (!value) { run++; stop(); p.hidden = true; return; }
    const stops = checkStory(value.story);
    run++;
    cur = { args: value.ev, record: value.record, story: value.story, stops, i: -1, playing: false, flood: null, expo: null,
      wave: false, layersOn: new Set() };
    p.hidden = false;
    showPlayer();
    goTo(0);
  });
  actions.openReplay = openReplayPicker;
  actions.startReplay = startReplay;
  actions.replayCandidates = candidatesHtml;
  globalThis.__aq = globalThis.__aq || {};
  globalThis.__aq.replay = { start: startReplay, goTo: (i) => goTo(i, { fly: false }), current: () => cur,
    pause, play, link: shareLink, gif: exportGif, report: downloadReport, pageFlood };
  const want = decodeReplay(location.hash);
  if (want) {
    const go = () => startReplay(want);
    if (state.workerReady) go(); else setTimeout(go, 1500);
  }
}
