// Place context (#520): the Context tab of a clicked point or a gauge, and the
// Context lines of a drawn box. The engine is aquascope.context, one layer per
// call on the light workers (worker-client.js), several at once; the card
// fills line by line as each lands. This module only lays the answers out,
// draws one small chart and puts flood events on the map.

import { $, EMPTY_FC, escapeHtml, state } from "./core.js?v=__BUILD__";
import { plot } from "./charts.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { setCard } from "./shell.js?v=__BUILD__";
import { Cancelled, callLightCancelable, lightPoolSize } from "./worker-client.js?v=__BUILD__";
import {
  CONTEXT_LAYERS, boxProblem, contextLine, credits, fetchOrder, floodPoints, normaliseBox, pickChart,
} from "./context-view.js?v=__BUILD__";

const SOURCE = "context-floods";
const LAYER = "context-floods";
let areaRun = 0;
let areaCalls = [];

// The Context card of a point ("pt") and of a gauge ("st"): the same card, the same calls, at the place.
const CARDS = {
  pt: { card: "pt-context", lines: "pt-context-lines", credits: "pt-context-credits", chart: "pt-context-chart",
    chartTitle: "pt-context-chart-title", plot: "plot-context" },
  st: { card: "st-context", lines: "st-context-lines", credits: "st-context-credits", chart: "st-context-chart",
    chartTitle: "st-context-chart-title", plot: "plot-st-context" },
};
const loaded = { pt: null, st: null };      // the place whose context is loaded (or loading), per card
const inFlight = { pt: [], st: [] };        // its calls, cancelled when the reader moves on

// Layers already answered this session, by place: going back to a place does not read them again.
const answered = new Map();
const ANSWERED_MAX = 140;
function remember(key, res) {
  answered.delete(key);
  answered.set(key, res);
  while (answered.size > ANSWERED_MAX) answered.delete(answered.keys().next().value);
}

// ── flood events on the map ─────────────────────────────────────────────────

let shown = EMPTY_FC;
let watching = false;

function drawFloods(fc) {
  shown = fc || EMPTY_FC;
  if (!state.mapOk || !map) return;
  if (!map.getSource(SOURCE)) {
    if (!shown.features.length) return;
    map.addSource(SOURCE, { type: "geojson", data: shown });
    map.addLayer({
      id: LAYER, type: "circle", source: SOURCE,
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 2.5, 10, 6],
        "circle-color": "#ef6c00", "circle-opacity": 0.75,
        "circle-stroke-color": "#ffffff", "circle-stroke-width": 0.8,
      },
    });
  } else {
    map.getSource(SOURCE).setData(shown);
  }
  if (!watching) {
    watching = true;
    // A basemap change replaces the style; put the events back.
    map.on("styledata", () => {
      if (shown.features.length && !map.getSource(SOURCE)) {
        try { drawFloods(shown); } catch (err) { console.info("context floods:", err && err.message); }
      }
    });
  }
}

export function clearContextFloods() { drawFloods(EMPTY_FC); }

// ── shared rendering ────────────────────────────────────────────────────────

function lineHtml(line) {
  const cls = `ctx-line ${line.state}`;
  return `<li class="${cls}"><span class="ctx-label">${escapeHtml(line.label)}</span>` +
    `<span class="ctx-text">${line.state === "loading" ? '<span class="spinner" aria-hidden="true"></span>' : ""}` +
    `${escapeHtml(line.text)}</span></li>`;
}

// Short names on the line, the full attribution one tap away (and in the tooltip).
function creditHtml(results) {
  const list = credits(results);
  if (!list.length) return "";
  const names = list.map((c) => `<a href="${escapeHtml(c.homepage || "#")}" target="_blank" rel="noopener" ` +
    `title="${escapeHtml(c.attribution)}">${escapeHtml(c.short)}</a> (${escapeHtml(c.licence)})`).join(" · ");
  const full = list.map((c) => `<li>${escapeHtml(c.attribution)}, ${escapeHtml(c.licence)}</li>`).join("");
  return `Data: ${names}<details class="ctx-attrib"><summary>Attribution</summary><ul>${full}</ul></details>`;
}

// Ask for every layer at once (the light workers take them as they come free, quickest first) and redraw
// as each lands. `stillWanted()` false means the reader moved on: what arrives is dropped. `ask(id)` returns
// { promise, cancel }; the cancels go into `calls` so a place the reader has left stops holding a worker.
async function fill(results, ask, render, stillWanted, calls = [], cacheKey = null) {
  const jobs = fetchOrder(lightPoolSize()).map(async (id) => {
    const key = cacheKey ? `${cacheKey}|${id}` : null;
    if (key && answered.has(key)) {
      results[id] = answered.get(key);
    } else {
      if (!stillWanted()) return;
      const job = ask(id);
      calls.push(job.cancel);
      try {
        results[id] = await job.promise;
        if (key && results[id] && !results[id].error && results[id].ok !== false) remember(key, results[id]);
      } catch (err) {
        if (err instanceof Cancelled) return;
        results[id] = { error: err && err.message ? err.message : String(err) };
      }
    }
    if (stillWanted()) render();
  });
  await Promise.all(jobs);
}

// ── the Context card of a point or a gauge ──────────────────────────────────

function cancelCalls(t) {
  for (const cancel of inFlight[t].splice(0)) { try { cancel(); } catch { /* answered already */ } }
}

export function resetPlaceContext(t) {
  loaded[t] = null;
  cancelCalls(t);
  const card = $(CARDS[t].card);
  if (card) card.hidden = true;
  clearContextFloods();
}

export const resetPointContext = () => resetPlaceContext("pt");

// `isCurrent()` says whether the place is still the one on screen (the point, or the selected gauge).
export async function loadPlaceContext(t, lat, lon, isCurrent) {
  const ids = CARDS[t];
  const key = `${lat},${lon}`;
  if (loaded[t] === key) return;
  cancelCalls(t);
  loaded[t] = key;
  const card = $(ids.card);
  const list = $(ids.lines);
  const foot = $(ids.credits);
  const chartWrap = $(ids.chart);
  if (!card || !list) return;
  const results = {};
  const render = () => {
    list.innerHTML = CONTEXT_LAYERS.map(({ id }) => lineHtml(contextLine(id, results[id]))).join("");
    foot.innerHTML = creditHtml(results);
    const chart = pickChart(results);
    chartWrap.hidden = !chart;
    if (chart) {
      $(ids.chartTitle).textContent = chart.what[0].toUpperCase() + chart.what.slice(1);
      plot(ids.plot, [{ x: chart.x, y: chart.y, type: "bar", name: chart.unit, marker: { color: "#1565c0" } }],
        { height: 150, margin: { l: 36, r: 8, t: 4, b: 28 }, yaxis: { title: { text: chart.unit } }, bargap: 0.25 },
        `context-${lat}-${lon}`);
    }
    if (results.flood_history) drawFloods(floodPoints(results.flood_history));
  };
  setCard(card, "ready");
  render();
  await fill(results, (id) => callLightCancelable("context", { op: "point", name: id, lat, lon }), render,
    () => loaded[t] === key && isCurrent(), inFlight[t], `pt:${key}`);
}

export function loadPointContext(lat, lon) {
  return loadPlaceContext("pt", lat, lon, () => Boolean(state.point && state.point.lat === lat && state.point.lon === lon));
}

// ── a drawn box ─────────────────────────────────────────────────────────────

function cancelAreaCalls() {
  for (const cancel of areaCalls.splice(0)) { try { cancel(); } catch { /* answered already */ } }
}

export async function openAreaContext(drawn, host) {
  const my = ++areaRun;
  cancelAreaCalls();
  const bbox = normaliseBox(drawn);
  const box = document.createElement("div");
  box.className = "area-context";
  host.querySelectorAll(".area-context").forEach((el) => el.remove());
  host.appendChild(box);
  const problem = boxProblem(bbox);
  if (problem) {
    box.innerHTML = `<p class="muted">${escapeHtml(problem)}</p>`;
    return;
  }
  const results = {};
  const render = () => {
    box.innerHTML = `<ul class="ctx-lines">${CONTEXT_LAYERS.map(({ id }) => lineHtml(contextLine(id, results[id]))).join("")}</ul>` +
      `<div class="ctx-credits muted">${creditHtml(results)}</div>`;
    if (results.flood_history) drawFloods(floodPoints(results.flood_history));
  };
  render();
  const b = [bbox.west, bbox.south, bbox.east, bbox.north];
  await fill(results, (id) => callLightCancelable("context", { op: "area", name: id, bbox: b }), render,
    () => my === areaRun && box.isConnected, areaCalls);
}

export function cancelAreaContext() {
  areaRun++;
  cancelAreaCalls();
  clearContextFloods();
}
