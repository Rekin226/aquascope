// The pace of the map's ambient motion (#591), in the browser: measures how fast the motion's own steps
// really come, notices whether WebGL is drawn in software, and keeps the time of the last touch. The
// decisions are motion-core.js's; this module only feeds it. See motion-core.js for why.

import { state } from "./core.js?v=__BUILD__";
import { map } from "./map.js?v=__BUILD__";
import { MOTION, motionNote, motionPace, slowFromIntervals, softwareRenderer } from "./motion-core.js?v=__BUILD__";

const WARMUP_MS = 5000;   // the first seconds of a motion share the main thread with the page's own start
let lastTouch = now();
let software = false;
let slow = false;
let hooked = false;
const listeners = new Set();

function now() {
  return globalThis.performance ? performance.now() : Date.now();
}

// The renderer behind the map's own context (no second context: on a slow device making one costs seconds).
function rendererName() {
  try {
    const canvas = map && map.getCanvas();
    const gl = canvas && (canvas.getContext("webgl2") || canvas.getContext("webgl"));
    if (!gl) return "";
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER) || "");
  } catch {
    return "";
  }
}

function markSlow() {
  if (slow) return;
  slow = true;
  state.motionSlow = true;
  for (const fn of listeners) { try { fn(); } catch { /* a listener's own problem */ } }
}

function touch() {
  lastTouch = now();
}

// Once, after the map exists: the renderer check and the listeners that count as someone using the page.
export function initMotion() {
  if (hooked) return;
  hooked = true;
  software = softwareRenderer(rendererName());
  if (software) markSlow();
  const opts = { passive: true, capture: true };
  for (const type of ["pointerdown", "wheel", "keydown", "touchstart"]) document.addEventListener(type, touch, opts);
  let moved = 0;
  document.addEventListener("pointermove", () => {   // a hover counts, at most once a second
    const t = now();
    if (t - moved > 1000) { moved = t; touch(); }
  }, opts);
  if (map) map.on("movestart", touch);
}

// Called when the pace changes (slow found), for the legend.
export function onMotionChange(fn) {
  listeners.add(fn);
}

export function motionStatus() {
  return { slow, software, note: motionNote({ slow, software }) };
}

// A pacer for a motion that wants `fps` steps a second. Call due(t) on every animation frame with the
// frame's timestamp: true when a step should be drawn now. skip() when a frame is passed over for another
// reason (the map waiting for idle), so that wait is not counted against the device.
export function pacer(fps) {
  let last = 0;
  let started = 0;
  const intervals = [];
  return {
    due(t) {
      const pace = motionPace({ fps, slow, idleMs: t - lastTouch, hidden: Boolean(globalThis.document && document.hidden) });
      if (pace.hold) { last = 0; return false; }
      if (last && t - last < 1000 / pace.fps) return false;
      if (!started) started = t;
      if (last && !slow && t - started > WARMUP_MS) {
        intervals.push(t - last);
        if (intervals.length > MOTION.sample * 2) intervals.shift();
        if (slowFromIntervals(intervals, fps)) markSlow();
      }
      last = t;
      return true;
    },
    skip() { last = 0; },
  };
}
