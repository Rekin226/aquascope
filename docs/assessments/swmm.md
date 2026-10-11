# SWMM in the browser: an assessment (#560)

**Question.** Urban drainage modellers use EPA SWMM (and PCSWMM, which writes the same files).
Can the Explorer read a SWMM model (`.inp`) and its binary results (`.out`) in the browser,
with Pyodide, so their results can be shown on the globe?

**Answer: yes.** Both files read in Pyodide 314. Of the two libraries named in the issue,
`swmm-api` reads both files but refuses results written by SWMM 5.1. `swmmio` cannot read
the `.out` at all. So AquaScope has its own small reader, `aquascope.io.swmm`. It needs only
numpy, reads SWMM 5.0 to 5.2 results, and gives the same values as SWMM's own reader.

Tested on 2026-10-11.

## What was tested

- **Runtime**: Pyodide 314.0.7 (npm `pyodide@314.0.7`) under Node 22.22.2, with the
  packages the Explorer's worker loads (micropip, numpy, scipy, pandas, pydantic, httpx),
  or only micropip and numpy, as the mesh worker does.
- **Models**: EPA's Example 1, 2 and 3 `.inp` files and a SWMM 5.1.0 `Example1.out`, from
  the test data of [pyswmm/swmm-python](https://github.com/pyswmm/swmm-python/tree/dev/swmm-toolkit/tests/data)
  (`swmm-toolkit/tests/data`, CC0 1.0, prepared by the US Government). The same
  `Example1.out` is in [USEPA/Stormwater-Management-Model](https://github.com/USEPA/Stormwater-Management-Model/tree/develop/tests/outfile/data)
  (public domain). None of these files are committed here.
- **Results made locally**: the three examples run with `swmm-toolkit` 0.17.0 (SWMM 5.2.4)
  on macOS arm64, plus Example 1 run for 30 days at a 1-minute report step, which writes a
  56.7 MB `.out` (43,920 report steps).

## Results

| Library | Install in Pyodide 314 | `.inp` | `.out` |
|---|---|---|---|
| `swmm-api` 0.4.74 (MIT) | works: 1.6 to 2.3 s, its wheel (305 KB) plus tqdm and packaging (174 KB); import 1.3 s | reads all three examples (Example 1: 28 sections in 0.9 ms) | reads SWMM 5.2 files (the 56.7 MB file in 0.05 s); **refuses SWMM 5.1 files**, natively too, because it expects 15 system variables and 5.1 writes 14 |
| `swmmio` 0.8.6 (MIT) | **fails**: it requires `pyswmm`, which requires `swmm-toolkit`, compiled C with no WebAssembly wheel. With `deps=False` plus pyyaml, networkx, pyproj, pillow, requests, geojson and pyshp (about 6.6 MB) it imports | Example 2 and 3 read; Example 1 fails (`KeyError: 'SU2'`), natively too | **no reader**: swmmio reads results from the text report (`.rpt`); the binary file goes through pyswmm |
| `swmm-toolkit` 0.17.0, `pyswmm` 2.2.0 | **fails**: no pure Python or WebAssembly wheel | not applicable | not applicable |
| `aquascope.io.swmm` (this PR) | part of the aquascope wheel; needs only numpy; import 0.22 s, pandas not loaded | reads all three examples | reads SWMM 5.1 and 5.2 files |

Times for `aquascope.io.swmm` in Pyodide 314.0.7 (Node), from `swmm_summary`, which reads the
network, scans node flooding, link flow and link capacity, and ranks them:

| Results file | Size | Report steps | Summary |
|---|---|---|---|
| Example 1, SWMM 5.1.0 | 44 KB | 36 | 4 ms |
| Example 1, SWMM 5.2.4 | 47.5 KB | 36 | 1 ms |
| Example 3, SWMM 5.2.4 | 267 KB | 1,800 | 1 ms |
| Example 1, 30 days at 1 minute | 56.7 MB | 43,920 | 21 ms |

Placing a model in longitude and latitude needs pyproj (3.8 MB, loaded on demand, as the
mesh and MODFLOW viewers already do); a placed summary of Example 1 took 0.18 s. The wasm
heap grew from 43 MB to 100 MB with the 56.7 MB file in Pyodide's in-memory file system.
These times are from one run on an Apple silicon Mac; a phone will be slower.

**The values are exact.** On all five results files, node flooding, node depth and link flow
for every element and report step match `swmm-toolkit`'s own output reader (largest
difference 0.0), as do the element names and the report times.

**Flooding from report steps.** SWMM reports a node's flooding as a rate at each report
step; hours flooded and volume are summed over those steps. At Example 1's hourly report
step that gives node 10 flooded 3.0 hours, a peak of 4.52 CFS and 571 m³, where SWMM's own
report (summed over every routing step) says 2.80 hours, 4.54 CFS and 0.135 million gallons
(511 m³). At a 1-minute report step the reader gives 2.8 hours, 4.54 CFS at 04:01 and
511 m³, the same as the report. The summary says so in a note.

## Why our own reader

- `swmmio` cannot read the results in the browser, and its dependencies cost about 6.6 MB.
- `swmm-api` works, but it pulls pandas into a worker that otherwise does not need it, and it
  refuses SWMM 5.1 results, which many published models still have.
- The binary layout is small and documented by SWMM's source (`src/solver/output.c`,
  public domain): a header, the names, the properties, the variable counts, then one
  fixed-size record per report step. Reading the counts from the file is what makes 5.1 and
  5.2 both work. The reader reads one variable at a time in blocks of about 32 MB, so a
  large file is never read into memory whole.

## What is in this PR

- `aquascope.io.swmm`: `read_swmm_inp(path, crs=)` (nodes and links with coordinates and,
  once placed, lon/lat; link lines run through their vertices), `read_swmm_out(path)` (the
  report times, `values(kind, variable, ids)` and `peaks(...)`), and `swmm_summary(inp, out,
  crs=, top=, element=)`. Every number in the summary is computed from the files.
- `aquascope swmm model.inp model.out [--crs EPSG] [--top N] [--element ID] [--json]`.
- The MCP tool `swmm_summary(inp, out, crs, top, element)`.
- Tests on small synthetic files written by the test itself, including a SWMM 5.1-style file.

No Explorer UI yet.

## Things to know

- **SWMM coordinates have no coordinate system.** `[MAP] Units` is none, feet, metres or
  degrees; EPA's examples are in arbitrary units. A model in degrees is placed as it is; any
  other needs its EPSG code, as the mesh viewer asks.
- On macOS arm64, `swmm-toolkit` 0.17.0's bundled `libomp.dylib` has an invalid signature,
  and Python is killed on import. Re-signing the library ad hoc in a scratch environment
  (`codesign --force -s - libomp.dylib`) fixed it for this test. It does not affect the
  browser or CI.

## Follow-up: SWMM on the map

The reader gives the map view everything it needs: points for nodes, lines for links, and
per element and step the node flooding and link flow.

1. Drop a `.inp` with its `.out` on the globe (the same drop zone as the mesh and MODFLOW
   viewers, routed by file extension); one question for the EPSG code when the model is not
   in degrees.
2. Conduits as lines, width by peak flow and colour by the largest fraction full; nodes as
   small dots, the flooding ones larger and red by peak flooding rate.
3. The time bar on the model's own clock: per step, flooding nodes pulse with their rate and
   link widths follow the flow (two float32 arrays per step from `values`).
4. The map card on a node or link with its series (`swmm_summary(element=...)`), and a
   legend row named after the file.
5. Ask the map: "show the flooding nodes"; Review: nodes that flood for long, conduits full
   for long, and the run's continuity errors from the `.rpt` when it is dropped too.
