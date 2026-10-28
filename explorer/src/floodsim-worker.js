// Flood it (#658), the fallback without WebGPU: the same local-inertial scheme (floodsim-core.js createCpuSim and
// runLoop) in a module worker, so the page stays responsive. It gets { z, n, nx, ny, dx, edges, sources, tableDtS,
// hours, steady } and posts { frame } as each frame is kept (packed, with its full depth grid), then { done: {
// steps, wall_s, volume, hMax, steadyAt, checks } }, or { error }.

import { createCpuSim, runLoop } from "./floodsim-core.js?v=__BUILD__";

self.onmessage = async (e) => {
  const job = e.data;
  try {
    const sim = createCpuSim({ z: job.z, n: job.n, nx: job.nx, ny: job.ny, dx: job.dx, edges: job.edges },
      { sources: job.sources, tableDtS: job.tableDtS });
    const started = performance.now();
    const engine = { advance: (t) => sim.advance(t), depth: () => new Float32Array(sim.h), faces: () => sim.faces() };
    const res = await runLoop(engine, { nx: job.nx, ny: job.ny, dx: job.dx, hours: job.hours, steady: job.steady,
      onFrame: (f, h) => { self.postMessage({ frame: f, h }); } });
    self.postMessage({ done: { steps: sim.state.steps, wall_s: (performance.now() - started) / 1000, volume: sim.volume(),
      hMax: new Float32Array(sim.hMax), steadyAt: res.steadyAt, checks: res.checks, t: res.t } });
  } catch (err) {
    self.postMessage({ error: String((err && err.message) || err) });
  }
};
