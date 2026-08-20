// Search over the catalog: accent-folded, ranked, keyboard-navigable.
// "Rhone" finds "Rhône", "loire blois" finds "La Loire à Blois", and a hidden
// source is still findable (selecting it un-hides that source instead of
// silently returning nothing, which is what the old substring scan did).

import { $, actions, escapeHtml, foldText, sourceStyle, state, stationKey } from "./core.js?v=__BUILD__";

let index = null;

function buildIndex() {
  index = state.stations.map((r) => ({
    r,
    name: foldText(r.name || ""),
    id: foldText(r.station_id),
  }));
}

// Score: id exact > name starts with > word starts with > contains. Every
// query token must appear somewhere, so "loire blois" beats "loire".
function score(entry, tokens, whole) {
  let total = 0;
  if (entry.id === whole) return 1000;
  for (const tok of tokens) {
    const inName = entry.name.indexOf(tok);
    const inId = entry.id.indexOf(tok);
    if (inName < 0 && inId < 0) return -1;
    if (inName === 0) total += 60;
    else if (inName > 0 && /[\s,('-]/.test(entry.name[inName - 1])) total += 40;
    else if (inName > 0) total += 20;
    if (inId === 0) total += 30;
    else if (inId > 0) total += 10;
  }
  total -= Math.min(20, entry.name.length / 12);   // prefer the shorter of two matches
  return total;
}

export function searchStations(query, limit = 25) {
  const whole = foldText(query).trim();
  if (whole.length < 2) return [];
  if (!index || index.length !== state.stations.length) buildIndex();
  const tokens = whole.split(/\s+/).filter(Boolean);
  const hits = [];
  for (const entry of index) {
    const s = score(entry, tokens, whole);
    if (s > 0) hits.push([s, entry.r]);
  }
  hits.sort((a, b) => b[0] - a[0]);
  return hits.slice(0, limit).map(([, r]) => r);
}

export function initSearch() {
  const input = $("search"), box = $("search-results");
  let t, hits = [], active = -1;

  const close = () => { box.hidden = true; active = -1; input.setAttribute("aria-expanded", "false"); };

  const paint = () => {
    box.innerHTML = "";
    if (!hits.length) {
      box.innerHTML = `<div class="hit muted">no match</div>`;
    } else {
      hits.forEach((r, i) => {
        const d = document.createElement("div");
        d.className = `hit${i === active ? " active" : ""}`;
        d.id = `hit-${i}`;
        d.setAttribute("role", "option");
        d.setAttribute("aria-selected", i === active ? "true" : "false");
        const st = sourceStyle(r.source);
        const filtered = state.hidden.has(r.source) ? ` <span class="muted">(source hidden)</span>` : "";
        d.innerHTML = `<i style="background:${st.color}"></i><span class="hit-name">${escapeHtml(r.name || r.station_id)}</span>` +
          `<span class="muted hit-id">${escapeHtml(r.station_id)}</span>${filtered}`;
        d.addEventListener("mousedown", (e) => { e.preventDefault(); choose(i); });
        box.appendChild(d);
      });
    }
    box.hidden = false;
    input.setAttribute("aria-expanded", "true");
  };

  const choose = (i) => {
    const r = hits[i];
    if (!r) return;
    close();
    input.value = "";
    if (state.hidden.has(r.source)) {          // do not hand back an invisible result
      state.hidden.delete(r.source);
      actions.refreshMapData();
    }
    actions.selectStation(stationKey(r), { fly: true });
  };

  input.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(() => {
      hits = searchStations(input.value);
      active = hits.length ? 0 : -1;
      if (foldText(input.value).trim().length < 2) { close(); return; }
      paint();
    }, 110);
  });

  input.addEventListener("keydown", (e) => {
    if (box.hidden && (e.key === "ArrowDown" || e.key === "ArrowUp")) { if (hits.length) paint(); return; }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!hits.length) return;
      active = (active + (e.key === "ArrowDown" ? 1 : -1) + hits.length) % hits.length;
      paint();
      const el = $(`hit-${active}`);
      if (el) el.scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter") {
      if (!box.hidden && active >= 0) { e.preventDefault(); choose(active); }
    } else if (e.key === "Escape") {
      close();
      input.blur();
    }
  });

  input.addEventListener("blur", () => setTimeout(close, 120));
  document.addEventListener("click", (e) => { if (!box.contains(e.target) && e.target !== input) close(); });
}
