import test from "node:test";
import assert from "node:assert/strict";
import { MOTION, median, motionNote, motionPace, slowFromIntervals, softwareRenderer } from "../src/motion-core.js";

test("software renderers are recognised by their WebGL names", () => {
  assert.equal(softwareRenderer("ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)"), true);
  assert.equal(softwareRenderer("llvmpipe (LLVM 15.0.7, 256 bits)"), true);
  assert.equal(softwareRenderer("Microsoft Basic Render Driver"), true);
  assert.equal(softwareRenderer("ANGLE (Apple, ANGLE Metal Renderer: Apple M2, Unspecified Version)"), false);
  assert.equal(softwareRenderer(""), false);
  assert.equal(softwareRenderer(null), false);
});

test("median ignores what is not a number", () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 3, 2]), 2.5);
  assert.equal(median([NaN, 5]), 5);
  assert.equal(median([]), null);
  assert.equal(median(null), null);
});

test("a device is slow when its steps come far later than asked, and only once there are enough", () => {
  const asked = 1000 / 16;   // 62.5 ms at 16 steps a second
  const fast = Array(MOTION.sample).fill(asked + 5);
  const slow = Array(MOTION.sample).fill(400);
  assert.equal(slowFromIntervals(fast, 16), false);
  assert.equal(slowFromIntervals(slow, 16), true);
  assert.equal(slowFromIntervals(slow.slice(0, MOTION.sample - 1), 16), false, "too few to say");
  // One stall in a fast run is not a slow device.
  assert.equal(slowFromIntervals([...fast.slice(1), 3000], 16), false);
  assert.equal(slowFromIntervals(slow, 0), false);
});

test("the pace: a fast device runs as asked, a slow one slower, then still while nobody uses the page", () => {
  assert.deepEqual(motionPace({ fps: 16 }), { fps: 16, hold: false, why: "" });
  assert.deepEqual(motionPace({ fps: 16, slow: true, idleMs: 1000 }), { fps: MOTION.slowFps, hold: false, why: "slow" });
  assert.deepEqual(motionPace({ fps: 1, slow: true, idleMs: 0 }), { fps: 1, hold: false, why: "slow" });
  assert.deepEqual(motionPace({ fps: 16, slow: true, idleMs: MOTION.idleHoldMs }), { fps: 0, hold: true, why: "idle" });
  assert.deepEqual(motionPace({ fps: 16, idleMs: 10 * MOTION.idleHoldMs }), { fps: 16, hold: false, why: "" },
    "a fast device is never held for being idle");
  assert.deepEqual(motionPace({ fps: 16, hidden: true }), { fps: 0, hold: true, why: "hidden" });
});

test("the legend says why the flow is slow, and nothing when it is not", () => {
  assert.equal(motionNote({}), "");
  assert.match(motionNote({ slow: true, software: true }), /without the graphics card/);
  assert.match(motionNote({ slow: true }), /draws the map slowly/);
});
