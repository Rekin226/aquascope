// The left rail: which sources are on the map, the overlays, and the counts.
// It used to be a row of chips in the header that vanished below 860 px, so a
// phone could neither filter sources nor turn the basins on.

import { $, escapeHtml, sourceStyle, state } from "./core.js?v=__BUILD__";
import { sourceCounts } from "./catalog.js?v=__BUILD__";
import { refreshMapData } from "./map.js?v=__BUILD__";
import { setBasinsVisible } from "./basins.js?v=__BUILD__";
import { writeUrl } from "./url.js?v=__BUILD__";

export function buildRail() {
  const counts = sourceCounts();
  const list = $("rail-sources");
  list.innerHTML = "";
  for (const src of Object.keys(counts).sort((a, b) => counts[b] - counts[a])) {
    const st = sourceStyle(src);
    const id = `src-${src}`;
    const row = document.createElement("label");
    row.className = "rail-row";
    row.innerHTML = `<input type="checkbox" id="${id}" ${state.hidden.has(src) ? "" : "checked"}>` +
      `<i style="background:${st.color}"></i>` +
      `<span class="rail-label">${escapeHtml(st.label)}</span>` +
      `<span class="rail-count">${counts[src].toLocaleString()}</span>`;
    row.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) state.hidden.delete(src); else state.hidden.add(src);
      refreshMapData();
      updateCount();
      writeUrl();
    });
    list.appendChild(row);
  }
  const basins = $("toggle-basins");
  basins.checked = state.basinsOn;
  basins.addEventListener("change", (e) => {
    setBasinsVisible(e.target.checked);
    writeUrl();
  });
  updateCount();
}

export function updateCount() {
  const visible = state.stations.filter((r) => !state.hidden.has(r.source)).length;
  const el = $("count");
  el.textContent = `${visible.toLocaleString()} of ${state.stations.length.toLocaleString()} gauges shown`;
}

// Reflect a filter that arrived from the URL back into the checkboxes.
export function syncRail() {
  for (const input of document.querySelectorAll("#rail-sources input[type=checkbox]")) {
    const src = input.id.replace(/^src-/, "");
    input.checked = !state.hidden.has(src);
  }
  const basins = $("toggle-basins");
  if (basins) basins.checked = state.basinsOn;
  updateCount();
}
