// Search over the catalog: accent-folded, ranked, keyboard-navigable.
// "Rhone" finds "Rhône", "loire blois" finds "La Loire à Blois", and a hidden
// source is still findable (selecting it un-hides that source instead of
// silently returning nothing, which is what the old substring scan did).

import { $, actions, escapeHtml, foldText, sourceStyle, state, stationKey } from "./core.js?v=__BUILD__";
import { shapeSvg } from "./shapes.js?v=__BUILD__";
import { groupStationSites } from "./sites.js?v=__BUILD__";
import { availabilityLabel } from "./availability.js?v=__BUILD__";
import { photonSearchUrl, placesFirst, rankPlaces } from "./place-search-core.js?v=__BUILD__";

export function searchStations(query, limit = 25) {
  const whole = foldText(query).trim();
  if (whole.length < 2) return [];
  const tokens = whole.split(/\s+/).filter(Boolean);
  const hits = [];
  for (const row of state.stations) {
    const name = foldText(row.name || ""), id = foldText(row.station_id);
    if (!tokens.every((token) => name.includes(token) || id.includes(token))) continue;
    let score = id === whole ? 1000 : 0;
    for (const token of tokens) {
      const at = name.indexOf(token);
      if (at === 0) score += 60;
      else if (at > 0) score += /[\s,('-]/.test(name[at - 1]) ? 40 : 20;
      if (id.startsWith(token)) score += 30;
      else if (id.includes(token)) score += 10;
    }
    hits.push({ row, score: score - Math.min(20, name.length / 12) });
  }
  hits.sort((a, b) => b.score - a.score);
  return groupStationSites(hits.map((hit) => hit.row), state.stations).slice(0, Math.max(0, limit));
}


// Places, not just gauges. Photon (komoot) is keyless and its terms allow
// autocomplete, which OSM's own Nominatim does not; picking a place drops the
// map there and opens it. A river or a lake is asked for separately and ranked
// first (place-search-core.js, #591): typing "Jamuna" finds the river, and
// picking it opens the River tab.
let placeAbort = null;

async function photon(url, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) return [];
  const data = await res.json();
  return data.features || [];
}

async function findPlaces(query, limit = 5) {
  if (placeAbort) placeAbort.abort();
  placeAbort = new AbortController();
  const { signal } = placeAbort;
  try {
    // Both at once; either may fail alone and the other still answers.
    const [any, water] = await Promise.all([
      photon(photonSearchUrl(query, { limit: 6 }), signal).catch((e) => { if (e.name === "AbortError") throw e; return []; }),
      photon(photonSearchUrl(query, { water: true, limit: 6 }), signal).catch((e) => { if (e.name === "AbortError") throw e; return []; }),
    ]);
    return rankPlaces([...any, ...water], query, limit);
  } catch (err) {
    if (err.name !== "AbortError") console.info("place search unavailable:", err.message);
    return [];
  }
}

export function initSearch() {
  const input = $("search"), box = $("search-results");
  let t, hits = [], places = [], active = -1, run = 0, loading = false, error = "", placesPending = false;
  let query = "", first = false;   // first: the places go above the gauges (placesFirst, #591)

  // The rows in the order they are shown: [kind, index into hits or places].
  const rows = () => {
    const g = hits.map((_, i) => ["gauge", i]);
    const p = places.map((_, i) => ["place", i]);
    return first ? [...p, ...g] : [...g, ...p];
  };

  const close = () => {
    clearTimeout(t);
    run++;
    box.hidden = true;
    active = -1;
    places = [];
    placesPending = false;
    first = false;
    input.setAttribute("aria-expanded", "false");
  };

  const gaugeRow = (r, at) => {
    const d = document.createElement("div");
    const st = sourceStyle(r.source);
    const filtered = state.hidden.has(r.source) ? ` <span class="muted">(source hidden)</span>` : "";
    d.className = "hit";
    d.innerHTML = `${shapeSvg(st.shape, st.color)}<span class="hit-name">${escapeHtml(r.name || r.station_id)}</span>` +
      `<span class="muted hit-id">${escapeHtml(r.station_id)}${r.record_count > 1 ? ` · ${r.record_count} records at this site` : ""}<br>${escapeHtml(availabilityLabel(r.source))}</span>${filtered}`;
    d.addEventListener("mousedown", (e) => { e.preventDefault(); pick(at); });
    return d;
  };

  const placeRow = (p, at) => {
    const d = document.createElement("div");
    d.className = "hit place";
    if (p.river) d.title = "Opens the River tab: the simulated flow of the river here";
    d.innerHTML = `<i class="pin"></i><span class="hit-name">${escapeHtml(p.name)}</span>` +
      `<span class="muted hit-id">${escapeHtml([p.water ? p.kind : "", p.detail].filter(Boolean).join(", ") || p.kind)}</span>`;
    d.addEventListener("mousedown", (e) => { e.preventDefault(); pick(at); });
    return d;
  };

  const paint = () => {
    box.innerHTML = "";
    if (!hits.length && !places.length) {
      const empty = loading ? "Searching gauges..." : placesPending ? "No gauge by that name; looking for places and rivers..." : "no match";
      box.innerHTML = `<div class="hit muted">${escapeHtml(error || empty)}</div>`;
    }
    let section = null;
    rows().forEach(([kind, i], at) => {
      if (kind === "place" && section !== "place") {
        const head = document.createElement("div");
        head.className = "hit-head muted";
        head.textContent = "Places and rivers (OpenStreetMap, via Photon)";
        box.appendChild(head);
      } else if (kind === "gauge" && section === "place") {
        const head = document.createElement("div");
        head.className = "hit-head muted";
        head.textContent = "Gauges";
        box.appendChild(head);
      }
      section = kind;
      const d = kind === "gauge" ? gaugeRow(hits[i], at) : placeRow(places[i], at);
      d.id = `hit-${at}`;
      d.setAttribute("role", "option");
      d.setAttribute("aria-selected", at === active ? "true" : "false");
      if (at === active) d.classList.add("active");
      box.appendChild(d);
    });
    box.hidden = false;
    input.setAttribute("aria-expanded", "true");
  };

  const choosePlace = (p) => {
    close();
    input.value = "";
    // A river lands on the River tab with the panel open; a town on the map with its card (#548). The label
    // says which of them it is (#637): the River card's title takes `river` only, never a lake's or a town's name.
    actions.selectPoint(p.lat, p.lon, { fly: true, tab: p.river ? "river" : null,
      label: { name: p.name, detail: p.detail, water: p.water, from: "search",
        river: p.river ? { name: p.name, detail: p.detail, water: true, river: true } : null } });
  };

  const chooseGauge = (r) => {
    close();
    input.value = "";
    if (state.hidden.has(r.source)) {          // do not hand back an invisible result
      state.hidden.delete(r.source);
      actions.refreshMapData();
    }
    actions.selectStation(stationKey(r), { fly: true });
  };

  const pick = (at) => {
    const row = rows()[at];
    if (!row) return;
    if (row[0] === "gauge") { if (hits[row[1]]) chooseGauge(hits[row[1]]); } else if (places[row[1]]) choosePlace(places[row[1]]);
  };

  input.addEventListener("input", () => {
    clearTimeout(t);
    const my = ++run;
    t = setTimeout(async () => {
      if (my !== run) return;
      query = input.value;
      if (foldText(query).trim().length < 2) { close(); return; }
      hits = [];
      active = -1;
      loading = true;
      error = "";
      places = [];
      first = false;
      paint();
      // Place lookup can finish while the Python search engine starts.
      placesPending = foldText(query).trim().length >= 3;
      if (placesPending) {
        void findPlaces(query.trim()).then((found) => {
          if (my !== run) return;
          placesPending = false;
          places = found;
          // "Rhine" is the river, not a gauge near Rhinelander: a place named as asked goes first when no
          // gauge carries the word itself.
          first = placesFirst(query, hits.map((r) => r.name || ""), places[0]);
          if (places.length && (first || active < 0)) active = first ? 0 : hits.length;
          paint();
        });
      }
      try {
        const found = searchStations(query);
        if (my !== run) return;
        hits = found;
        active = hits.length ? 0 : -1;
      } catch (err) {
        if (my !== run) return;
        error = `Could not search gauges: ${err.message}`;
      }
      loading = false;
      paint();
    }, 140);
  });

  input.addEventListener("keydown", (e) => {
    const total = hits.length + places.length;
    if (box.hidden && (e.key === "ArrowDown" || e.key === "ArrowUp")) { if (total) paint(); return; }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!total) return;
      active = (active + (e.key === "ArrowDown" ? 1 : -1) + total) % total;
      paint();
      const el = $(`hit-${active}`);
      if (el) el.scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter") {
      if (box.hidden || active < 0) return;
      e.preventDefault();
      pick(active);
    } else if (e.key === "Escape") {
      close();
      input.blur();
    }
  });

  input.addEventListener("blur", () => setTimeout(close, 120));
  document.addEventListener("click", (e) => { if (!box.contains(e.target) && e.target !== input) close(); });
}
