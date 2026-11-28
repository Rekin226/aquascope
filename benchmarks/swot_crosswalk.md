# GEOGLOWS to SWORD in New Zealand

The crosswalk for GEOGLOWS VPU 514 assigns a SWORD v17b river reach to **1,078 of 15,691** GEOGLOWS stream segments (6.87%). For GEOGLOWS Strahler order 5 or higher, 623 of 1,315 segments match (47.38%). This second denominator selects larger streams independently of SWORD. SWORD targets wide rivers and its inventory is much sparser than the GEOGLOWS network. Neither rate measures positional accuracy.

| Outcome | All GEOGLOWS segments | Strahler order ≥5 |
|---|---:|---:|
| Matched | 1,078 | 623 |
| No ordinary SWORD river within 500 m of any part of the line | 13,742 | 604 |
| Nearby, insufficient overlap along the GEOGLOWS line | 677 | 69 |
| Geometry passed, drainage-area check failed | 180 | 10 |
| Geometry passed, no usable drainage area | 0 | 0 |
| Two or more plausible matches too close to distinguish | 14 | 9 |
| Total | 15,691 | 1,315 |

**Method.** The input includes every stream in VPU 514 and the overlapping Oceania HydroBASINS region 57 of SWORD v17b. Of the 575 SWORD reaches, 311 have ordinary river type 1. The other records are lakes, dams, reaches with unreliable topology or ghost reaches. Project both layers to New Zealand Transverse Mercator (EPSG:2193). For each GEOGLOWS segment, collect all ordinary SWORD lines within 500 m. Measure the distance to each candidate at 10%, 30%, 50%, 70% and 90% of the GEOGLOWS length. At least four distances must be ≤500 m. This reduces matches caused by a tributary touching a larger river only at its mouth.

Compare the upstream area at the GEOGLOWS downstream end (`DSContArea`, m² divided by 1,000,000) with SWORD's maximum upstream drainage area (`facc`, km²). Both must be finite and positive and the ratio must lie between 0.5 and 2, inclusive. Among candidates meeting both checks, take the smallest mean of the five distances. Leave the assignment empty if the runner-up is within 100 m of that mean. Several GEOGLOWS segments can match one longer SWORD reach. The 1,078 assignments cover 276 distinct SWORD river reaches. The 10th, 50th and 90th percentiles of the GEOGLOWS-to-SWORD area ratio among those assignments are 0.748, 0.930 and 1.096.

These are screening thresholds for this VPU. Holding the distance rules fixed, tightening the area ratio to 1/1.5–1.5 gives 1,011 assignments. Widening it to 1/3–3 gives 1,134. Adjacent reaches around a confluence can have similar areas. This experiment does not establish global coverage, resolve braided channels or establish the availability or quality of SWOT observations at a given overpass.

## Independent station checks

Select the nearest GEOGLOWS segment to the published station coordinate before looking at any assignment. Independently select the nearest ordinary SWORD line to the same coordinate. Gauge areas and names play no part in either the crosswalk or these selections. The three examples check a sampled river, an area discrepancy and a coverage gap.

| Station | Gauge area (km²) | GEOGLOWS ID, distance, area | SWORD result, distance, area |
|---|---:|---|---|
| Grey at Dobson | 3,830 | 570053653, 83 m, 3,689 km² | Assigned ID 57203000041 is the nearest Grey River line, 22 m, 3,620 km² |
| Wairau at Tuamarina | 3,430 | 570062821, 213 m, 3,230 km² | Assigned ID 57205200021 is the nearest Wairau River line, 206 m, 2,377 km² |
| Buller at Te Kuha | 6,350 | 570072056, 99 m, 6,048 km² | Unmatched. The nearest SWORD river is on the Grey, 49.9 km from the station |

At Wairau the two networks pass the factor-of-two screen, yet SWORD's area is 31% below the station area. A match alone does not establish agreement with a gauged catchment.

The Grey and Buller coordinates are the NZ Map Grid values in Hicks et al. (2011), Table 1, transformed from EPSG:27200 to EPSG:4326. The station areas above are from NIWA's 2020 report. Hicks et al. give different delineated areas of 3,396.2 and 5,745.0 km² respectively. The Wairau coordinates and area are from GRDC station 5870110. Its coordinates have only two decimal places, which limits interpretation of individual distances. [Hicks et al.](https://niwa.co.nz/sites/default/files/Hicks_etal_2011_sediment_yields_from_NZ_rivers.pdf) · [NIWA station areas](https://niwa.co.nz/sites/default/files/2020250CH.pdf) · [GRDC Report 20](https://doi.bafg.de/BfG/1998/GRDC_Report_20.pdf)

## Reproduce

Install the spatial reader with `python -m pip install -e ".[basins]" "fsspec[http]"`. The distance query needs GEOS 3.10 or later. The following Python downloads the 116 MB GEOGLOWS file and reads just the New Zealand reach layer from the global SWORD ZIP using HTTP byte ranges. Run it from the repository root.

```python
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import fsspec

inputs = Path("build/swot-inputs")
inputs.mkdir(parents=True, exist_ok=True)
streams = inputs / "streams_514.gpkg"
url = "https://geoglows-v2.s3.us-west-2.amazonaws.com/hydrography/vpu=514/streams_514.gpkg"
urllib.request.urlretrieve(url, streams)
assert hashlib.sha256(streams.read_bytes()).hexdigest() == "7ff9a8269d991fa1a66fdf0a8cbb49e880a9a2d60a64c32416a6d8a368a468a8"

url = "https://zenodo.org/records/15299138/files/SWORD_v17b_shp.zip"
with fsspec.open(url, "rb", block_size=131072) as remote, zipfile.ZipFile(remote) as archive:
    for suffix in ["shp", "shx", "dbf", "prj", "cpg"]:
        name = f"oc_sword_reaches_hb57_v17b.{suffix}"
        (inputs / name).write_bytes(archive.read("shp/OC/" + name))
```

```shell
python -m benchmarks.swot_crosswalk --streams build/swot-inputs/streams_514.gpkg --sword build/swot-inputs/oc_sword_reaches_hb57_v17b.shp --out build/swot-514
```

`crosswalk.csv` contains one row per GEOGLOWS ID, its stream order and area, the outcome and the assigned SWORD ID when present. `candidates.csv` retains all 2,439 nearby pairs, including rejected and ambiguous alternatives, with their five-point coverage count, mean distance and area ratio. `summary.json` reports the denominators, counts, ratio percentiles and station checks. The run used GeoPandas 1.1.4, pyogrio 0.13.0, Shapely 2.1.2 and PROJ via pyproj 3.7.1.

GEOGLOWS v2 uses TDX-Hydro river geometry from the US National Geospatial-Intelligence Agency (CC BY-SA 4.0). SWORD v17b is distributed by Altenau et al. under CC BY 4.0. Both inputs are read locally. The repository contains the analysis and the summary. [GEOGLOWS hydrography and fields](https://training.geoglows.org/rfs/datasets/hydrography/) · [TDX-Hydro terms](https://earth-info.nga.mil/php/download.php?file=tdx-hydro-license) · [SWORD v17b data, fields and licence](https://zenodo.org/records/15299138)
