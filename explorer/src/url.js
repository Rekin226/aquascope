// The URL is the state. The hash carries the selection, the active tab, the
// map view and the source filter, so Back works, a pasted link reproduces the
// view, and "Copy link" hands someone else exactly what you are looking at.
//
// #s=<source>/<id>&tab=floods&v=8.1/51.41/-0.31&hide=usgs,uk_ea&basins=1
// #p=<lat>,<lon>&tab=climate&v=...
// #s=<source>/<id>&study=1            (the Study drawer, open at that place)
// #study=kingston-flood                (a recorded study, by its id)
// #study=z1.<token>                    (a shared plan, see study-link.js; ?study_url= carries a study.yaml)
//
// The legacy forms (#s=key, #p=lat,lon, #solve=...) still parse, so old links
// keep working: a Solve link opens Study at the same place.

import { LAYER_DEFAULTS, actions, state, trace } from "./core.js?v=__BUILD__";

let applying = false;      // ignore our own hashchange
let lastWritten = "";

export function readUrl(hash = location.hash) {
  const raw = String(hash || "").replace(/^#/, "");
  if (!raw) return {};
  const q = new URLSearchParams(raw);
  const out = {};
  if (q.has("s")) out.station = q.get("s");
  if (q.has("p")) {
    const m = String(q.get("p")).match(/^(-?[\d.]+),(-?[\d.]+)$/);
    if (m) out.point = { lat: Number(m[1]), lon: Number(m[2]) };
  }
  if (q.has("tab")) out.tab = q.get("tab");
  // The Study drawer, open at the selection. A Solve link from before opens it too; #study=<id> with an
  // id from the recorded studies' index opens that recording.
  if (q.has("study") || q.has("solve")) {
    out.study = true;
    const v = q.get("study");
    if (v && v !== "1" && /^[A-Za-z0-9_-]+$/.test(v)) out.studyId = v;
    if (v && /^[zj]1\.[A-Za-z0-9_-]+$/.test(v)) out.studyLink = v;   // a shared plan (study-link.js)
  }
  if (q.has("v")) {
    const m = String(q.get("v")).match(/^([\d.]+)\/(-?[\d.]+)\/(-?[\d.]+)$/);
    if (m) out.view = { zoom: Number(m[1]), lat: Number(m[2]), lon: Number(m[3]) };
  }
  if (q.has("hide")) out.hidden = String(q.get("hide")).split(",").filter(Boolean);
  if (q.has("basins")) out.basins = q.get("basins") === "1";
  // layers (#232)
  if (q.has("m")) out.mode = q.get("m");
  if (q.has("b")) out.basemap = q.get("b");
  if (q.has("o")) out.overlays = String(q.get("o")).split(",").filter(Boolean);
  if (q.has("d")) out.date = q.get("d");
  if (q.has("t")) out.terrain = q.get("t") === "1";
  if (q.has("hs")) out.hillshade = q.get("hs") === "1";
  if (q.has("gl")) out.globe = q.get("gl") === "1";
  if (q.has("gs")) out.gaugeStyle = q.get("gs");
  if (q.has("hm")) out.heat = q.get("hm") === "1";
  return out;
}

function currentHash({ view } = {}) {
  const q = new URLSearchParams();
  if (state.mode === "workbench") q.set("m", "workbench");
  else if (state.selected) q.set("s", `${state.selected.source}/${state.selected.station_id}`);
  else if (state.point) q.set("p", `${state.point.lat},${state.point.lon}`);
  if (state.activeTab) q.set("tab", state.activeTab);
  if (state.drawerOpen && state.drawerMode === "study") q.set("study", state.study.recorded || "1");
  if (state.drawerOpen && state.drawerMode === "study" && state.study.link) q.set("study", state.study.link);   // a shared study stays in the address
  if (view) q.set("v", `${view.zoom.toFixed(2)}/${view.lat.toFixed(4)}/${view.lon.toFixed(4)}`);
  if (state.hidden.size) q.set("hide", [...state.hidden].join(","));
  if (state.basinsOn) q.set("basins", "1");
  if (state.basemap && state.basemap !== LAYER_DEFAULTS.basemap) q.set("b", state.basemap);
  if (state.overlays && state.overlays.size) q.set("o", [...state.overlays].join(","));
  if (state.date && (state.overlays.size || state.basemap === "daily")) q.set("d", state.date);
  // Only what differs from LAYER_DEFAULTS, written as 0 or 1 so a link can turn
  // a default-on layer off as well as a default-off layer on.
  const flag = (key, param) => {
    if (Boolean(state[key]) !== Boolean(LAYER_DEFAULTS[key])) q.set(param, state[key] ? "1" : "0");
  };
  flag("terrain", "t");
  flag("hillshade", "hs");
  flag("globe", "gl");
  if (state.gaugeStyle && state.gaugeStyle !== LAYER_DEFAULTS.gaugeStyle) q.set("gs", state.gaugeStyle);
  flag("heat", "hm");
  return `#${q.toString().replace(/%2F/gi, "/").replace(/%2C/gi, ",")}`;
}

// `push` for a new selection (so Back returns to the previous one), `replace`
// for view, tab and filter changes, which should not fill the history.
// A push only creates an entry when the hash actually changes; asking to push
// something that is already the current URL replaces instead, so Back never
// lands on a duplicate that looks like nothing happened.
export function writeUrl({ push = false, view = null } = {}) {
  const hash = currentHash({ view: view || state.view });
  if (!push && hash === lastWritten) return;
  lastWritten = hash;
  applying = true;
  try {
    if (push && hash !== location.hash) history.pushState(null, "", hash);
    else history.replaceState(null, "", hash);
  } finally {
    setTimeout(() => { applying = false; }, 0);
  }
}

export function canonicalUrl() {
  // On a Hugging Face Space the page runs in an iframe on *.static.hf.space;
  // that URL works but the shareable one is the Space itself. CONFIG-free:
  // derive it from the host so a self-hosted copy keeps its own link.
  const hash = currentHash({ view: state.view });
  const host = location.hostname;
  const m = host.match(/^(.+)\.static\.hf\.space$/);
  if (m) {
    const slug = m[1].replace(/^([^-]+)-(.*)$/, "$1/$2");
    return `https://huggingface.co/spaces/${slug}${hash}`;
  }
  return `${location.origin}${location.pathname}${hash}`;
}

export function initUrl() {
  window.addEventListener("hashchange", () => {
    if (applying) return;
    trace("hashchange");
    actions.applyUrl(readUrl(), { fromHistory: true });
  });
  window.addEventListener("popstate", () => {
    if (applying) return;
    trace("popstate");
    actions.applyUrl(readUrl(), { fromHistory: true });
  });
}
