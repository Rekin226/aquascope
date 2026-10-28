// Flood it (#658) on the GPU: the local-inertial scheme of floodsim-core.js (FLOODSIM_WGSL) as WebGPU compute
// shaders, every step on the device: the time step, the faces with their edge codes, the limiter, the depths and
// the inflows, with no read-back between steps. The page reads the clock (and a frame when one is due) once a
// batch.
//
// gpuAvailable() says whether this browser can; createGpuSim() returns null when it cannot, and the caller falls
// back to the worker (floodsim-worker.js), the same scheme in JS on a coarser grid.

import { ALPHA, FLOODSIM_WGSL, G, H_DRY, N_FLOODPLAIN, THETA, edgeCodes, mergeSources } from "./floodsim-core.js?v=__BUILD__";

export const gpuAvailable = () => typeof navigator !== "undefined" && Boolean(navigator.gpu);

/** The adapter's description, for the report ("apple", "swiftshader", ...), or null. */
export async function gpuInfo() {
  if (!gpuAvailable()) return null;
  try {
    const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!a) return null;
    const i = a.info || (a.requestAdapterInfo ? await a.requestAdapterInfo() : {});
    return { vendor: i.vendor || "", architecture: i.architecture || "", description: i.description || "",
      fallback: Boolean(a.isFallbackAdapter || i.isFallbackAdapter) };
  } catch {
    return null;
  }
}

/**
 * grid: { z (array ny*nx), n (array or number), nx, ny, dx, edges }; opts: { sources: [[row, col, flow or table]],
 * tableDtS, theta, alpha, dtMax, adapter (requestAdapter's options) }.
 * Returns { advance(until, stepsPerBatch) -> {t, steps}, readDepth(), readMaxDepth(), readFaces(), destroy() } or null.
 */
export async function createGpuSim(grid, opts = {}) {
  if (!gpuAvailable()) return null;
  const adapter = await navigator.gpu.requestAdapter(opts.adapter || { powerPreference: "high-performance" });
  if (!adapter) return null;
  const { nx, ny, dx } = grid;
  const N = nx * ny;
  const nq = (nx + 1) * ny + (ny + 1) * nx;
  const sources = mergeSources(opts.sources || []);
  const nsrc = sources.length;
  const nt = Math.max(1, ...sources.map((s) => s[2].length));
  const limits = adapter.limits || {};
  const need = Math.max(nq, 2 * N + 2 * nx + 2 * ny + nsrc * (nt + 1)) * 4;
  const device = await adapter.requestDevice(need > 134217728 ? { requiredLimits: {
    maxStorageBufferBindingSize: Math.min(limits.maxStorageBufferBindingSize || need, need),
    maxBufferSize: Math.min(limits.maxBufferSize || need, need) } } : undefined);

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const buf = (bytes) => device.createBuffer({ size: Math.max(16, bytes), usage });
  // zn: the bed, Manning's n, the edge codes (n, s, w, e; -1 free), the inflow cells, the flow tables.
  const codes = edgeCodes(nx, ny, grid.edges);
  const zn = new Float32Array(2 * N + 2 * nx + 2 * ny + nsrc + nsrc * nt);
  zn.set(Float32Array.from(grid.z));
  if (typeof grid.n === "number" || grid.n === undefined) zn.fill(grid.n ?? N_FLOODPLAIN, N, 2 * N);
  else zn.set(Float32Array.from(grid.n), N);
  let o = 2 * N;
  for (const side of ["n", "s", "w", "e"]) for (const v of codes[side]) zn[o++] = Number.isNaN(v) ? -1 : v;
  sources.forEach(([r, c], k) => { zn[o + k] = r * nx + c; });
  o += nsrc;
  sources.forEach(([, , tab], k) => { for (let j = 0; j < nt; j++) zn[o + k * nt + j] = tab[Math.min(j, tab.length - 1)]; });
  const bZn = buf(zn.byteLength);
  device.queue.writeBuffer(bZn, 0, zn);
  const bH = buf(N * 4), bHmax = buf(N * 4), bLim = buf(N * 4);
  const bQ = [buf(nq * 4), buf(nq * 4)];
  const bSt = buf(32);
  device.queue.writeBuffer(bSt, 0, new Uint32Array(8));
  const params = new ArrayBuffer(48);
  new Uint32Array(params, 0, 4).set([nx, ny, nsrc, nt]);
  new Float32Array(params, 16, 8).set([dx, opts.theta ?? THETA, opts.alpha ?? ALPHA, opts.dtMax ?? 10, G, H_DRY, opts.tableDtS ?? 3600, 0]);
  const bP = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(bP, 0, params);

  const module = device.createShaderModule({ code: FLOODSIM_WGSL });
  const info = module.getCompilationInfo ? await module.getCompilationInfo() : { messages: [] };
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length) throw new Error(`WGSL: ${errors.map((m) => `${m.lineNum}: ${m.message}`).join("; ")}`);
  const S = GPUShaderStage.COMPUTE;
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: S, buffer: { type: "uniform" } },
    { binding: 1, visibility: S, buffer: { type: "storage" } },
    { binding: 2, visibility: S, buffer: { type: "read-only-storage" } },
    { binding: 3, visibility: S, buffer: { type: "storage" } },
    { binding: 4, visibility: S, buffer: { type: "storage" } },
    { binding: 5, visibility: S, buffer: { type: "storage" } },
    { binding: 6, visibility: S, buffer: { type: "read-only-storage" } },
    { binding: 7, visibility: S, buffer: { type: "storage" } },
  ] });
  const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipe = (entryPoint) => device.createComputePipeline({ layout: pipeLayout, compute: { module, entryPoint } });
  const P = { dt: pipe("stepDt"), faces: pipe("faces"), limit: pipe("limit"), scale: pipe("scale"), depth: pipe("depth"),
    inflow: pipe("inflow") };
  const group = (a, b) => device.createBindGroup({ layout, entries: [
    { binding: 0, resource: { buffer: bP } }, { binding: 1, resource: { buffer: bSt } },
    { binding: 2, resource: { buffer: bZn } }, { binding: 3, resource: { buffer: bH } },
    { binding: 4, resource: { buffer: bHmax } }, { binding: 5, resource: { buffer: bLim } },
    { binding: 6, resource: { buffer: bQ[a] } }, { binding: 7, resource: { buffer: bQ[b] } },
  ] });
  const groups = [group(0, 1), group(1, 0)];
  const fx = Math.ceil((nx + 1) / 16), fy = Math.ceil((ny + 1) / 16);
  const cx = Math.ceil(nx / 16), cy = Math.ceil(ny / 16);
  const sx = Math.ceil(nsrc / 64);
  let parity = 0;

  const readBuf = device.createBuffer({ size: Math.max(N, nq) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const readSt = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  async function readState() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(bSt, 0, readSt, 0, 32);
    device.queue.submit([enc.finish()]);
    await readSt.mapAsync(GPUMapMode.READ);
    const raw = readSt.getMappedRange().slice(0);
    readSt.unmap();
    const f = new Float32Array(raw), u = new Uint32Array(raw);
    return { dt: f[0], t: f[1], tEnd: f[2], steps: u[4] };
  }

  async function read(src, count) {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, readBuf, 0, count * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ, 0, count * 4);
    const out = new Float32Array(readBuf.getMappedRange(0, count * 4).slice(0));
    readBuf.unmap();
    return out;
  }

  /** Run on the device until simulated time `until` (s), `batch` steps per submit; returns { t, steps }. */
  async function advance(until, batch = 200) {
    device.queue.writeBuffer(bSt, 8, new Float32Array([until]));
    for (;;) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let i = 0; i < batch; i++) {
        pass.setBindGroup(0, groups[parity]);
        pass.setPipeline(P.dt); pass.dispatchWorkgroups(1);
        pass.setPipeline(P.faces); pass.dispatchWorkgroups(fx, fy);
        pass.setPipeline(P.limit); pass.dispatchWorkgroups(cx, cy);
        pass.setPipeline(P.scale); pass.dispatchWorkgroups(fx, fy);
        pass.setPipeline(P.depth); pass.dispatchWorkgroups(cx, cy);
        if (nsrc) { pass.setPipeline(P.inflow); pass.dispatchWorkgroups(sx); }
        parity ^= 1;
      }
      pass.end();
      device.queue.submit([enc.finish()]);
      const s = await readState();
      // Steps past the end have dt = 0 and change nothing; the shader counts only the ones that moved the clock.
      if (s.t >= until - 1e-3 || s.dt <= 0) return { t: s.t, steps: s.steps };
    }
  }

  /** The face discharges after the last step: { qx (ny*(nx+1)), qy ((ny+1)*nx) }. */
  async function readFaces() {
    const q = await read(bQ[parity], nq);
    return { qx: q.subarray(0, (nx + 1) * ny), qy: q.subarray((nx + 1) * ny) };
  }

  return {
    advance, readDepth: () => read(bH, N), readMaxDepth: () => read(bHmax, N), readFaces, readState,
    adapter, destroy: () => device.destroy(),
  };
}
