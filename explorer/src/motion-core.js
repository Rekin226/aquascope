// How fast the map's ambient motion runs (#591): the rivers' flow (#545) and the pulsing gauges under
// Floods ahead (#546). Each step of either is a style change, so the map draws a whole frame for it. On a
// GPU that costs a few milliseconds. On a slow device it does not: in Chrome's software renderer
// (SwiftShader, what a machine without a usable GPU gets) a frame of the globe took a few hundred
// milliseconds, the 16 steps a second never caught up, and the page's main thread stayed busy for as long
// as it was open.
//
// So the motion watches how fast its own steps really come. A device that cannot keep up (or one that says
// its WebGL is drawn in software) gets a slower step, and the motion holds still while nobody touches the
// page; it starts again on the next move, click, scroll or key. A fast device is not changed.
//
// Pure, for node:test. motion.js measures and listens; the layers ask motionPace() what to do.

export const MOTION = {
  sample: 16,        // steps measured before deciding
  slowFactor: 2.5,   // slow: the median step came this many times later than asked
  slowFps: 2,        // the step rate on a slow device while someone is using the page
  idleHoldMs: 20000, // and how long after the last touch it holds still
};

// WebGL renderer names that mean the frames are drawn on the CPU.
const SOFTWARE = /swiftshader|llvmpipe|softpipe|lavapipe|software|basic render driver/i;

export function softwareRenderer(name) {
  return Boolean(name) && SOFTWARE.test(String(name));
}

export function median(values) {
  const v = (values || []).filter((x) => Number.isFinite(x)).slice().sort((a, b) => a - b);
  if (!v.length) return null;
  const mid = Math.floor(v.length / 2);
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

// Is a device slow, from the intervals between the steps it actually ran at a rate of `fps`?
// Too few samples say nothing yet.
export function slowFromIntervals(intervals, fps, { sample = MOTION.sample, slowFactor = MOTION.slowFactor } = {}) {
  if (!(fps > 0) || !intervals || intervals.length < sample) return false;
  const m = median(intervals.slice(-sample));
  return m !== null && m > slowFactor * (1000 / fps);
}

// What a motion asking for `fps` should do now: { fps, hold, why }. `hold` means draw nothing new.
export function motionPace({ fps, slow = false, idleMs = 0, hidden = false }, opts = MOTION) {
  if (hidden) return { fps: 0, hold: true, why: "hidden" };
  if (!slow) return { fps, hold: false, why: "" };
  if (idleMs >= opts.idleHoldMs) return { fps: 0, hold: true, why: "idle" };
  return { fps: Math.min(fps, opts.slowFps), hold: false, why: "slow" };
}

// One line for the Rivers row in "On the map", or "" when the motion runs as asked.
export function motionNote({ slow = false, software = false } = {}) {
  if (!slow) return "";
  return software
    ? "This browser draws the map without the graphics card, so the flow moves slowly and holds still while the page is not in use."
    : "This device draws the map slowly, so the flow moves slowly and holds still while the page is not in use.";
}
