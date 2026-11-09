# The Explorer in real browsers (#591)

**Question.** Do the Explorer's main journeys work in the browsers people use, not only in
the headless Chrome the pull requests are checked in?

**Answer: yes.** In Chrome on a Mac all 13 journeys now work, after the fixes below. On
2026-10-25 the maintainer reported that the Explorer also works in Safari and Firefox on
the desktop, and on phones (Safari on an iPhone, Chrome on Android), with no problem to
fix.

Chrome on macOS with a GPU was checked by the maintainer on 2026-10-12 (11 journeys) and
2026-10-25 (the two model journeys), on the live site
<https://rekin226-aquascope-explorer.static.hf.space/>.

## Chrome on a Mac

| Journey | Link (after the site's address) | Result |
|---|---|---|
| The globe opens | (none) | pass |
| Rivers flow | `#v=5/50/10` | pass |
| Click a gauge | `#s=usgs/USGS-01646500` | pass |
| Click a river | `#p=24.35,89.8&tab=river` | problem, below |
| Trace to the sea | `#p=24.35,89.8&tab=river`, then Trace to the sea | pass |
| Floods ahead and flood depth | (none) | pass |
| The time bar (play, peak to map, swipe, GIF) | (none) | pass |
| Ask the map | (none) | pass |
| Explain this view and Scout | (none) | pass |
| Drop a HEC-RAS result | (none) | not checked on 2026-10-12, below; pass on 2026-10-25 |
| Groundwater and a MODFLOW model | (none) | problem, fixed here; pass on 2026-10-25 |
| The bulletin | (none) | pass |
| Phone layout | (none) | pass, in a narrow Chrome window on the Mac, not on a phone |

### Click a river: the instructions confused

The link opens a point beside the Jamuna in Bangladesh. The checklist asked the tester to
look for "the snap line" saying the click snapped to the main channel, 86 years of
simulated flow and the return-period table. The maintainer could not tell what to look for,
and asked whether "Jamuna in Bangladesh" can be found with the Explorer's search instead of
a link. Fixed in #609: typing a river or place name such as "Jamuna" in the search now
offers it and flies there, and a link with `tab=` opens the card's panel on that tab.

### Groundwater and a MODFLOW model: asked for a georeference first (fixed)

After dropping `freyberg.hds` and `freyberg.dis` (FloPy's `freyberg_multilayer_transient`
example), the page asked for the lower-left x and y in the CRS's units, the rotation and a
CRS before it showed anything. The checklist had not said to drop `freyberg.nam`, which
carries the georeference (`xul`, `yul`, `rotation`, `proj4_str`). But a model without a
georeference must still be viewable. Fixed:

- A `.nam` with FloPy's georeference comment is read when it is dropped or picked with the
  other files. Checked with the Freyberg `.nam`: the model lands near Austin, Texas, UTM
  zone 14N, rotated 15 degrees, with dates from 2 January 2015, without a question.
- When the files do not say where the model is, nothing blocks. The first choice, one
  click, is **Place it at the centre of the map (approximate)**: north up, at its true size
  in its own length unit (metres when the files name none). The legend row and the cell
  card say "approximate placement". The exact form (origin, rotation, CRS) is the second
  choice, and stays reachable from the legend row.
- **Tools**, **Open model results** now offers **MODFLOW: the Freyberg model**: the three
  files, 10.8 MB, fetched by the page from FloPy's GitHub repository (pinned to one commit,
  public domain), checked against their SHA-256 and read like a drop.

### Drop a HEC-RAS result: not checked

The tester could not get the sample file. The HydroShare resource lists dozens of plans
with near-identical names, and a browser page cannot fetch the one #592 used: HydroShare
redirects to a signed Google Cloud Storage address that sends no
`Access-Control-Allow-Origin`. **Tools**, **Open model results** now offers **HEC-RAS 2D:
compound flooding in Portsmouth, Virginia**, read from the Archive's byte-for-byte copy
(`samples/hec-ras/`, CC BY 4.0, Zanko Zandsalimi, GNN4CF, HydroShare). Until the copy is
published by `mirror-samples.yml`, the button says so. The copy was published on
2026-10-12, and the maintainer checked this journey on 2026-10-25 with the sample and with
synthetic HEC-RAS and MODFLOW models. Both views worked; the maintainer judged them too
basic, which led to the richer flood and groundwater views of #627.

### Found in headless Chrome, not in the real one

In headless Chrome on the SwiftShader software renderer, the live page's main thread stayed
busy for more than 90 seconds after load (a script evaluation and a screenshot both timed
out). Chrome with a GPU was fine. #609 lightened the first load; the phones checked on
2026-10-25 did not freeze.

## Safari, Firefox and phones

| Browser | Checked | Result |
|---|---|---|
| Safari on a Mac | 2026-10-25, by the maintainer | works |
| Firefox | 2026-10-25, by the maintainer | works |
| Safari on an iPhone | 2026-10-25, by the maintainer | works |
| Chrome on Android | 2026-10-25, by the maintainer | works |

These were reported as a whole, not journey by journey, and without browser versions,
frame rates or screenshots. Nothing blocking was found, so nothing was fixed or filed for
them.

## What remains

- A journey-by-journey table, with versions and screenshots, for each of the four browsers
  above, from anyone who walks the checklist (contributions welcome).
- The frame rate during the time bar's play and the memory use, on a phone.

## How to run the checklist

1. Note the browser, its version and the device.
2. For each journey above, open the Explorer at the link (the site's address followed by
   the link part), do what it asks, and mark it **pass**, **problem** (with a short note of
   what looked wrong) or **skip**.
3. The two model journeys need no download: **Tools**, **Open model results**, then a
   sample. To test a drop instead, drop all of a model's files together (for MODFLOW the
   `.hds`, its `.dis` and its `.nam`).
4. On a phone, also use the map with one thumb for a minute: the legend should be a small
   chip until opened, cards should open as a bottom sheet, and nothing should cover more
   than about a quarter of the screen.
5. Add the results to this page: one table per browser, as above.
