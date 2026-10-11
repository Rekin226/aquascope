// Watch (#521): a ☆ Watch on a gauge, a river reach and a drawn area, a
// "Since you were here" panel on the next visit, a threshold per gauge, and a
// "Follow (Atom)" link to the gauge's feed. The list lives in this browser
// (watch-core.js, every storage call wrapped); what changed is worked out by
// aquascope.watch.watch_digest in the worker, one item at a time, with the
// Archive's daily status snapshot and newest forecast issue read here with
// DuckDB. This module asks and lays out; it computes nothing.

import { CONFIG } from "../config.js?v=__BUILD__";
import { $, actions, escapeHtml, state, stationKey } from "./core.js?v=__BUILD__";
import { duck } from "./catalog.js?v=__BUILD__";
import { fitBoundsTo } from "./map.js?v=__BUILD__";
import { setStatusEl, showSurface } from "./shell.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import {
  RETURN_PERIODS, WATCH_KEY, applyDigest, digestRequest, findItem, isWatched, makeItem, readWatch, removeItem,
  setThreshold, shouldGreet, thresholdChoice, toggleItem, writeWatch,
} from "./watch-core.js?v=__BUILD__";

const KIND_LABEL = { gauge: "gauge", reach: "river reach", area: "area" };
const today = () => new Date().toISOString().slice(0, 10);
const prettyUnit = (u) => ({ "m3/s": "m³/s", "ft3/s": "ft³/s" }[u] || u || "");

let returnTo = "panel-empty";
let run = 0;
let lastResults = null;
// The reach the open point (and the open gauge) snapped to, from river.js's "reachchange" event.
const reaches = { pt: null, st: null };

// ── the buttons ─────────────────────────────────────────────────────────────

function setStar(btn, on, what) {
  if (!btn) return;
  btn.setAttribute("aria-pressed", on ? "true" : "false");
  btn.textContent = on ? "★ Watching" : "☆ Watch";
  btn.title = on ? `Stop watching this ${what}` : `Watch this ${what}: what changed shows the next time you open the Explorer`;
}

function gaugeItem() {
  const r = state.selected;
  if (!r) return null;
  const reach = reaches.st;
  return makeItem("gauge", { source: r.source, station_id: r.station_id, name: r.name || r.station_id, lat: r.lat,
    lon: r.lon, river_id: reach ? reach.river_id : null });
}

function reachItem() {
  const reach = reaches.pt;
  if (!reach || reach.river_id === undefined || reach.river_id === null) return null;
  return makeItem("reach", { river_id: reach.river_id, lat: reach.lat, lon: reach.lon });
}

export function syncWatchButtons() {
  const g = state.selected ? `${state.selected.source}/${state.selected.station_id}` : null;
  const on = Boolean(g) && isWatched(g);
  setStar($("btn-watch-st"), on, "gauge");
  renderThreshold(on ? findItem(g) : null);
  const reach = reachItem();
  const pt = $("btn-watch-pt");
  if (pt) {
    pt.hidden = !reach;
    if (reach) setStar(pt, isWatched(reach.id), "river reach");
  }
  syncCount();
  void syncFeedLink();
}

function syncCount() {
  const n = readWatch().items.length;
  const badge = $("watched-count");
  if (badge) { badge.textContent = n ? String(n) : ""; badge.hidden = !n; }
}

// The threshold row under the gauge's actions, shown only while the gauge is watched.
function renderThreshold(item) {
  const row = $("st-watch-row");
  if (!row) return;
  row.hidden = !item;
  if (!item) return;
  const sel = $("st-watch-threshold");
  const flow = !state.result || !state.result.variable || state.result.variable === "discharge";
  const unit = prettyUnit(state.result && state.result.unit) || (flow ? "m³/s" : "");
  sel.replaceChildren();
  if (flow) {
    for (const t of RETURN_PERIODS) sel.add(new Option(`the ${t}-year flow${t === 2 ? " (default)" : ""}`, `rp:${t}`));
  } else {
    sel.add(new Option("no threshold", "rp:2"));
  }
  sel.add(new Option(`a value (${unit || "the record's unit"})`, "value"));
  const choice = thresholdChoice(item.threshold);
  sel.value = flow || choice === "value" ? choice : "rp:2";
  const input = $("st-watch-value");
  input.hidden = sel.value !== "value";
  input.value = item.threshold && item.threshold.value !== undefined ? String(item.threshold.value) : "";
  $("st-watch-unit").textContent = sel.value === "value" ? unit : "";
}

function saveThreshold() {
  const g = state.selected && stationKey(state.selected);
  if (!g || !isWatched(g)) return;
  const v = $("st-watch-threshold").value;
  if (v === "value") {
    const x = $("st-watch-value").value;
    setThreshold(g, x === "" ? null : { value: Number(x) });
  } else {
    const t = Number(v.slice(3));
    setThreshold(g, t === 2 ? null : { return_period: t });
  }
  renderThreshold(findItem(g));
}

// "Follow (Atom)": shown when the daily feed job has a feed for this gauge (feeds/index.json).
let feedIndex = null;
function loadFeedIndex() {
  if (!feedIndex) {
    feedIndex = fetch(`${CONFIG.feedsBase}index.json`, { cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : { feeds: {} }))
      .catch(() => { feedIndex = null; return { feeds: {} }; });
  }
  return feedIndex;
}

async function syncFeedLink() {
  const a = $("st-feed");
  if (!a) return;
  a.hidden = true;
  const r = state.selected;
  if (!r) return;
  const key = stationKey(r);
  const idx = await loadFeedIndex();
  if (!state.selected || stationKey(state.selected) !== key) return;
  const path = idx && idx.feeds && idx.feeds[key];
  if (path) { a.href = `${CONFIG.feedsBase}${path}`; a.hidden = false; }
}

// ── the area button (layer-ui.js puts it in the drawn box's result) ─────────

export function areaWatchButton(bbox) {
  const b = [bbox.west, bbox.south, bbox.east, bbox.north];
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn tiny";
  const item = makeItem("area", { bbox: b });
  setStar(btn, isWatched(item.id), "area");
  btn.addEventListener("click", () => {
    toggleItem(item);
    setStar(btn, isWatched(item.id), "area");
    syncCount();
  });
  return btn;
}

// ── the panel ───────────────────────────────────────────────────────────────

const say = (text, kind = "info") => setStatusEl($("watch-status"), text, kind);

function rowFor(item) {
  const li = document.createElement("li");
  li.className = "watch-item";
  li.dataset.id = item.id;
  li.innerHTML =
    `<div class="watch-head"><button type="button" class="watch-open link" title="Open it">${escapeHtml(item.name || item.id)}</button>` +
    `<span class="muted watch-kind">${KIND_LABEL[item.kind] || ""}</span>` +
    `<button type="button" class="btn tiny watch-remove" aria-label="Stop watching ${escapeHtml(item.name || item.id)}">Remove</button></div>` +
    `<p class="watch-line muted">Waiting…</p>`;
  li.querySelector(".watch-open").addEventListener("click", () => openItem(item));
  li.querySelector(".watch-remove").addEventListener("click", () => {
    removeItem(item.id);
    li.remove();
    $("watch-empty").hidden = readWatch().items.length > 0;
    syncWatchButtons();
  });
  return li;
}

function lineOf(id) {
  const li = $("watch-list").querySelector(`[data-id="${CSS.escape(id)}"]`);
  return li ? { li, p: li.querySelector(".watch-line") } : null;
}

function renderList(st) {
  const ul = $("watch-list");
  ul.replaceChildren(...st.items.map(rowFor));
  $("watch-empty").hidden = st.items.length > 0;
}

function showResult(r) {
  const row = lineOf(r.id);
  if (!row) return;
  row.p.textContent = r.line || r.error || "";
  row.li.classList.toggle("has-alert", Boolean(r.alerts && r.alerts.length));
  row.li.classList.toggle("changed", Boolean(r.changed));
}

function openItem(item) {
  if (item.kind === "gauge") {
    if (!state.byKey.has(item.id)) { say("This gauge is not in the current catalog, so it cannot be opened.", "warn"); return; }
    actions.selectStation(item.id, { fly: true });
  } else if (item.kind === "reach") {
    if (Number.isFinite(item.lat) && Number.isFinite(item.lon)) actions.selectPoint(item.lat, item.lon, { fly: true });
    else say("This reach has no position saved, so it cannot be opened.", "warn");
  } else if (item.kind === "area") {
    const [w, s, e, n] = item.bbox;
    fitBoundsTo([w, s, e, n]);
    actions.showArea({ west: w, south: s, east: e, north: n });
  }
}

// The Archive's daily rows for the watched gauges and areas (empty until forecast-archive.yml publishes them).
async function archiveRows(items) {
  const gauges = items.filter((i) => i.kind === "gauge").map((i) => i.id);
  const areas = items.some((i) => i.kind === "area");
  if (!gauges.length && !areas) return { snapshot: [], issued: [] };
  const plain = (row) => {
    const o = row.toJSON();
    for (const k of Object.keys(o)) if (typeof o[k] === "bigint") o[k] = Number(o[k]);
    return o;
  };
  const list = gauges.map((g) => `'${g.replace(/'/g, "''")}'`).join(",") || "''";
  let snapshot = [], issued = [];
  try {
    const { conn } = await duck();
    const where = areas ? "" : `WHERE source || '/' || station_id IN (${list})`;
    const t = await conn.query(`SELECT source, station_id, CAST(value_date AS VARCHAR) AS value_date, value, percentile,
      "class", n_years FROM read_parquet('${CONFIG.forecastsBase}status/latest.parquet') ${where}`);
    snapshot = t.toArray().map(plain);
    for (const r of snapshot) {
      const st = state.byKey.get(`${r.source}/${r.station_id}`);
      if (st) { r.lat = st.lat; r.lon = st.lon; }
    }
  } catch (err) { console.info("watch: no status snapshot yet", err && err.message); }
  if (gauges.length) {
    try {
      const man = await (await fetch(`${CONFIG.forecastsBase}manifest.json`, { cache: "no-cache" })).json();
      const issues = (man.issues || []).filter((i) => i.file);
      if (issues.length) {
        const file = issues[issues.length - 1].file.replace(/^forecasts\//, "");
        const { conn } = await duck();
        const t = await conn.query(`SELECT * FROM read_parquet('${CONFIG.forecastsBase}${file}')
          WHERE model = 'geoglows' AND source || '/' || station_id IN (${list})`);
        issued = t.toArray().map(plain).map((r) => {
          for (const k of ["issue_date", "valid_date", "init_date"]) if (r[k] !== null && r[k] !== undefined) r[k] = String(r[k]).slice(0, 10);
          return r;
        });
      }
    } catch (err) { console.info("watch: no forecast issue yet", err && err.message); }
  }
  return { snapshot, issued };
}

function rowsFor(item, rows) {
  if (item.kind === "gauge") return rows.snapshot.filter((r) => `${r.source}/${r.station_id}` === item.id);
  if (item.kind === "area") return rows.snapshot.filter((r) => Number.isFinite(r.lat) && Number.isFinite(r.lon));
  return [];
}

async function runDigest() {
  const my = ++run;
  const st = readWatch();
  renderList(st);
  lastResults = [];
  if (!st.items.length) { $("watch-summary").textContent = ""; say(""); return; }
  const day = today();
  const req = digestRequest(st, day);
  $("watch-summary").textContent = state.workerReady ? "Checking each place…" : "Loading Python in your browser (about 30 MB, once), then checking each place…";
  const rows = await archiveRows(req.items);
  if (my !== run) return;
  for (const item of req.items) {
    const row = lineOf(item.id);
    if (row) row.p.textContent = "Checking…";
    try {
      const res = await call("watch", {
        op: "digest", items: [item], last_seen: { [item.id]: req.last_seen[item.id] }, today: day,
        snapshot: rowsFor(item, rows),
        issued: item.kind === "gauge" ? rows.issued.filter((r) => `${r.source}/${r.station_id}` === item.id) : [],
      });
      if (my !== run) return;
      const r = res.items && res.items[0];
      if (r) {
        if (!lastResults.length) $("watch-summary").textContent = "Checking each place…";
        lastResults.push(r);
        showResult(r);
        writeWatch(applyDigest(readWatch(), [r], { complete: false, today: day }));
      } else if (row) {
        row.p.textContent = (res.errors && res.errors[0] && res.errors[0].error) || res.error || "Could not be checked.";
      }
    } catch (err) {
      if (my !== run) return;
      if (row) row.p.textContent = `Could not be checked this time (${err.message}).`;
    }
  }
  try {
    const sum = await call("watch", { op: "summary", items: lastResults, today: day });
    if (my !== run) return;
    $("watch-summary").textContent = sum.summary || "";
  } catch { $("watch-summary").textContent = ""; }
  // Today becomes the last visit only when every place answered, so a place that could not be checked keeps
  // its window for the next try.
  writeWatch(applyDigest(readWatch(), [], { complete: lastResults.filter((r) => !r.error).length === req.items.length, today: day }));
}

export function openWatch({ greet = false } = {}) {
  if (!$("panel-watch").hidden) return;
  for (const id of ["panel-station", "panel-point", "panel-workbench", "panel-places", "panel-empty"]) {
    const el = $(id);
    if (el && !el.hidden) { returnTo = id; break; }
  }
  $("watch-title").textContent = greet ? "Since you were here" : "Watched";
  $("watch-close").textContent = greet ? "Dismiss" : "Back";
  say("");
  showSurface("panel-watch");
  if (greet || !lastResults) {
    void runDigest();
  } else {
    renderList(readWatch());
    lastResults.forEach(showResult);
  }
}

function closeWatch() {
  showSurface(returnTo && $(returnTo) ? returnTo : "panel-empty");
}

// On load: when something is watched and today's digest has not been seen, and the link opens nothing else.
export function greetOnLoad(url = {}) {
  if (url.station || url.point || url.study || url.mode) return;
  if (shouldGreet(readWatch(), today())) openWatch({ greet: true });
}

export function initWatch() {
  $("btn-watched").addEventListener("click", () => openWatch());
  $("watch-close").addEventListener("click", closeWatch);
  $("watch-refresh").addEventListener("click", () => { void runDigest(); });
  $("btn-watch-st").addEventListener("click", () => {
    const item = gaugeItem();
    if (!item) return;
    toggleItem(item);
    syncWatchButtons();
  });
  $("btn-watch-pt").addEventListener("click", () => {
    const item = reachItem();
    if (!item) return;
    toggleItem(item);
    syncWatchButtons();
  });
  $("st-watch-threshold").addEventListener("change", () => {
    const value = $("st-watch-threshold").value === "value";
    $("st-watch-value").hidden = !value;
    if (value) $("st-watch-value").focus();
    saveThreshold();
  });
  $("st-watch-value").addEventListener("change", saveThreshold);
  for (const [t, id] of [["pt", "panel-point"], ["st", "panel-station"]]) {
    $(id).addEventListener("reachchange", (e) => {
      reaches[t] = e.detail || null;
      syncWatchButtons();
    });
  }
  // Another tab changed the list: follow it.
  globalThis.addEventListener?.("storage", (e) => {
    if (e.key !== WATCH_KEY) return;
    syncWatchButtons();
  });
  syncCount();
}
