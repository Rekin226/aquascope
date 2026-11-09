# Explorer live smoke: clipped station menu (#611)

Investigated against `main` at `e7465b08b0f88d11fd4e9d56ec653d4c90d48c1e`.
Issue [#611](https://github.com/Rekin226/aquascope/issues/611) is still open;
no existing PR for the issue was found. The merged river-search and slow-device
fixes (#609) do not change the station menu's positioning.

## Cause and correction

[Run 38718824332](https://github.com/Rekin226/aquascope/actions/runs/38718824332)
exported records for all three gauges, cold and warm, then timed out clicking
`#btn-to-workbench`. The canvas, and sometimes the panel toggle, received the
hit instead. This is a user-facing layout defect, not a data-retrieval failure
or a reason to force Playwright clicks.

The action row uses flex wrapping. `.more-menu` is absolutely positioned with
`right: 0` against `.more`, with a minimum width of 190px. When More wraps near
the left edge of the panel, the menu extends left of its scroll container.
The panel's `overflow: auto` clips that part of the menu. Its boxes still exist,
so visibility checks pass, but hit testing reaches the map behind the clipped
area. Increasing the menu's z-index cannot escape ancestor clipping.

Reproduced in Chromium 153 using the repository's HTML and CSS without service
requests. At 1440px, a menu button occupied x=834..1014 while the panel started
at x=967; its centre hit the map. The same browser regression fails on the
original CSS and passes with the correction.

The correction is `margin-left: auto` on `.more`: the trigger stays against the
right edge of the action row even when it wraps. The existing right-aligned
menu then fits inside the panel. Panel scrolling, stacking, map pointer events,
HTML, station handlers and the live smoke sequence remain unchanged.

## Regression check

```sh
npm install --no-save playwright@1.63.0
npx playwright install --with-deps chromium
node --test explorer/tests/browser/*.test.mjs
node --test explorer/tests/*.test.mjs
```

The offline browser fixture uses the actual page and stylesheet, isolates the
inspector and its toggle over a canvas, and blocks external requests. It checks
1280, 1440, 1100, 900, 860, 768, 390 and 320px widths plus a short 1280x420
viewport, each with optional exports, unit switching and the feed visible or
hidden (18 layouts). Every visible menu action must fit horizontally inside the
panel, receive the centre-point hit, and respond to a normal Playwright click.
The menu must close normally and the exposed canvas must still receive clicks.
The existing live smoke separately verifies CSV contents and the workbench's
row count through the real worker. No forced clicks, scripted button clicks,
disabled map pointer events or weakened data assertions are used.

Validation on 2026-10-16: the 18-layout browser regression passes (and fails
with the original stylesheet); 296 existing Node tests, 199 Explorer/CI Python
tests and 21 documentation/public-API tests pass. `ruff check aquascope/ tests/`,
`git diff --check` and `python explorer/build.py` also pass. The browser used
locally was Chromium 153.0.8010.0 with Playwright 1.62.1; CI uses the same pinned
Playwright 1.63.0 as the existing live workflow. Safari and Firefox were not tested.

The unchanged `.github/scripts/explorer_smoke.mjs` also passed all six cases
against a locally assembled preview of this working tree, using the real live
catalog and observed records. Each CSV kept the exact observation count and
the workbench reported that same row count; no uncaught browser errors occurred.

| Gauge | Observations | Cold | Warm |
| --- | ---: | --- | --- |
| Fish River, US | 37,554 | passed | passed |
| Kingston, UK | 51,977 | passed | passed |
| Seine, France | 7,543 | passed | passed |

This validates the patched preview, not a production deployment or a scheduled
Actions run. The local preview's build metadata still names the base commit
`e7465b0`; its stylesheet includes this unmerged patch.

## Verify the daily live check

1. Merge the patch and wait for **Deploy Explorer** to finish publishing. Check
   the live site's `/wheels.json`: its `build` must match the deployed commit's
   short SHA. A green smoke against an older deployment does not verify this fix.
2. Dispatch **Explorer live record-to-export** with `repetitions=1`, or run
   `node .github/scripts/explorer_smoke.mjs` locally with Playwright installed.
   The default target is the live site; `AQ_BASE_URL` can target a built preview.
3. Inspect the `explorer-live-record-to-export` artifact's `results.json`:
   all six US/UK/France cold/warm entries must be `passed`, with `export` and
   `workbench` timings and record evidence. Check that `environment.build`
   identifies the new deployment. A green job alone is not the evidence.
4. Confirm the next **scheduled** run also passes. The workflow deliberately
   publishes job-health records and updates/closes issues only for `schedule`,
   so a manual dispatch will not close #611 or reset its failure streak. Check
   `health/jobs/explorer-live-smoke.json` for the new last success and reset
   streak, and confirm #611 closes automatically.

A local layout pass cannot establish that live catalogs, browser dependencies,
agency records or the workbench worker are healthy. Keep those real checks and
their failures visible; do not close the incident on the layout test alone.
