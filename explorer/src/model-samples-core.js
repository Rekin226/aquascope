// Try a sample model (#591), the pure part: no DOM, no map, so node can test it.
//
// Two samples, each fetched by the page from a host that allows it and then read like a file you dropped:
// - MODFLOW: FloPy's transient Freyberg example, straight from GitHub (raw.githubusercontent.com answers any
//   origin), pinned to one FloPy commit. Its .nam carries FloPy's georeference (xul, yul, rotation, proj4_str,
//   start_datetime), so it lands on the map without a question. Public domain (USGS) and CC0 1.0, FloPy's LICENSE.md.
// - HEC-RAS 2D: one plan result from HydroShare (CC BY 4.0). HydroShare's storage sends no CORS header, so the page
//   reads the Archive's copy (samples/hec-ras/, made by aquascope.archive.samples) and its manifest says what it is.
//   Until that copy is published the button says so.
// Sizes and SHA-256 were measured on the files; the page checks the hash of what it fetched.

export const ARCHIVE = "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/";
export const FLOPY_COMMIT = "f29652c46a3c0d69c591f7dcacb500fd1efb8082";
const FLOPY_DIR = "examples/data/freyberg_multilayer_transient";
const FLOPY_RAW = `https://raw.githubusercontent.com/modflowpy/flopy/${FLOPY_COMMIT}/${FLOPY_DIR}/`;

export const SAMPLES = [
  {
    id: "modflow-freyberg",
    kind: "modflow",
    title: "MODFLOW: the Freyberg model",
    about: "FloPy's transient Freyberg example: 3 layers of 40 x 20 cells, 1,097 daily heads from January 2015. " +
      "Its name file says where it is, and its well file where it pumps.",
    files: [
      { name: "freyberg.nam", url: `${FLOPY_RAW}freyberg.nam`, bytes: 573,
        sha256: "9e3747ce6d6229caec55a9357285a96cb4608dae11d90dd165a23e0bb394a2bd" },
      { name: "freyberg.dis", url: `${FLOPY_RAW}freyberg.dis`, bytes: 97882,
        sha256: "1ef61a467a219c036e58902ce11297e06b4eeb5f2f9d2ea40245b421a248a471" },
      { name: "freyberg.hds", url: `${FLOPY_RAW}freyberg.hds`, bytes: 10676004,
        sha256: "0b3e911ef35f625d2d046e05a20bc1300341b41028220c5b25ace6f5a267ceef" },
      // its six pumping wells, for the drawdown, pathlines and capture zones (#624)
      { name: "freyberg.wel", url: `${FLOPY_RAW}freyberg.wel`, bytes: 327112,
        sha256: "dd322655eadff3f618f0835c9277af30720197bd48328aae2d6772f26eef2686" },
    ],
    credit: "The Freyberg model, from FloPy's examples (USGS; public domain in the US, CC0 1.0 elsewhere).",
    link: `https://github.com/modflowpy/flopy/tree/${FLOPY_COMMIT}/${FLOPY_DIR}`,
    licence: "Public domain (USGS), CC0 1.0",
  },
  {
    id: "hec-ras-portsmouth",
    kind: "mesh",
    title: "HEC-RAS 2D: compound flooding in Portsmouth, Virginia",
    about: "A HEC-RAS 6.5 plan result: 6,272 cells, rain and coastal water level, 109 steps every 15 minutes " +
      "on 30 June and 1 July 2021.",
    manifest: `${ARCHIVE}samples/hec-ras/manifest.json`,
    credit: "Zandsalimi, Z., GNN4CF: Dataset for Boundary-Aware Graph Neural Networks for Compound Flood " +
      "Forecasting, HydroShare, CC BY 4.0.",
    link: "https://www.hydroshare.org/resource/32aac3af553c4e1fbb997ec58d9de9b6/",
    licence: "CC BY 4.0",
  },
];

export const sampleById = (id) => SAMPLES.find((s) => s.id === id) || null;

/** "10.8 MB", "98 kB", "573 bytes": decimal units, as the docs and the Archive's pages give sizes. */
export function sizeWords(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n < 0) return "";
  if (n < 1000) return `${n} bytes`;
  if (n < 1e6) return `${Math.round(n / 1e3)} kB`;
  return `${(n / 1e6).toFixed(1)} MB`;
}

export const totalBytes = (files) => (files || []).reduce((a, f) => a + (Number(f.bytes) || 0), 0);

/**
 * The files a published sample manifest names, as the page fetches them: [{ name, url, bytes, sha256 }].
 * Only files under samples/ in the Archive, with a plain name, a size and a SHA-256, are taken; anything else in
 * the manifest is ignored, and a manifest with none gives an empty list.
 */
export function manifestFiles(manifest, archive = ARCHIVE) {
  const out = [];
  for (const f of (manifest && Array.isArray(manifest.files) ? manifest.files : [])) {
    const path = String((f && f.path) || "");
    const name = String((f && f.name) || "");
    if (!/^samples\/[a-z0-9-]+\/[A-Za-z0-9._-]+$/.test(path) || !path.endsWith(`/${name}`)) continue;
    if (!(Number(f.bytes) > 0) || !/^[0-9a-f]{64}$/.test(String(f.sha256 || ""))) continue;
    out.push({ name, url: `${archive}${path}`, bytes: Number(f.bytes), sha256: String(f.sha256) });
  }
  return out;
}

/**
 * The sample's button, from its state: { label, disabled, note }. `state` is "ready", "checking", "missing" (the
 * Archive's copy is not published yet), "offline" (the manifest could not be read), "loading" or "failed".
 */
export function sampleButton(sample, state, { files = sample.files, done = 0 } = {}) {
  const total = totalBytes(files);
  const size = total ? `, ${sizeWords(total)}` : "";
  if (state === "checking") return { label: `${sample.title}`, disabled: true, note: "Checking the Archive…" };
  if (state === "missing") {
    return { label: `${sample.title}`, disabled: true,
      note: "Not published yet: the Archive's copy goes up with the next sample mirror run." };
  }
  if (state === "offline") return { label: `${sample.title}`, disabled: true, note: "Could not reach the Archive just now." };
  if (state === "loading") {
    return { label: `${sample.title}`, disabled: true,
      note: total ? `Fetching: ${sizeWords(done)} of ${sizeWords(total)}` : "Fetching…" };
  }
  return { label: `${sample.title}`, disabled: false, note: `Open it${size}` };
}

/** Lower-case hex of an ArrayBuffer digest. */
export function hex(buf) {
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
