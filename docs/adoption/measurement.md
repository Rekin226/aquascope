# Measuring useful adoption

The primary outcome is independent practitioners using a useful AquaScope result
in real work and returning when another task calls for it. Stars and forks measure
different actions; their ratio alone cannot establish usefulness or failure.

The optional usage log under Explorer's Tools menu is off by default. It retains
28 calendar days of local counters and bounded operation timings. No information
is sent automatically. Participants can inspect, voluntarily export, or delete it.
It excludes raw questions, coordinates, records, filenames, keys, URLs and visitor
identifiers. It cannot support population-level conversion, cross-device tracking,
campaign attribution or star attribution. Collect voluntary logs only under the
[pilot's consent protocol](pilot.md).

| Signal | What is actually counted | Interpretation |
| --- | --- | --- |
| Visit | Page initializations after consent | Not deduplicated people; bots not classified |
| Usable record | Observations rendered | Activation proxy; participant still judges task suitability |
| Table loaded | Imported/sample table ready | Data ready, not completed analysis |
| Study ready | Completed report reply | Review grade and failed checks before treating it as useful |
| Export handoff | Generated file passed to browser | Actual disk save and downstream use require pilot confirmation |
| Operation timing | Worker call through response/error | Cold/warm worker state; not an HTTP-cache benchmark |
| Useful days | Days with a record, table, report or export action | Local opt-in return proxy, not a user identifier or 28-day cohort rate |

## Repeatable speed and live reliability

`.github/workflows/explorer-live-smoke.yml` checks US, UK and French observed
records through a real browser to an actual CSV download daily. It keeps per-case
status, period, count, hash, errors and stage timings as workflow artifacts. A failed
region fails the job; other regions still run. Scheduled success is evidence only
after the workflow is deployed and has actually run.

Run five or more repetitions per region to produce p50/p75:

```bash
npm install --no-save playwright@1.63.0
npx playwright install chromium
AQ_REPETITIONS=5 node .github/scripts/explorer_smoke.mjs
```

Set `AQ_BASE_URL` to a built preview and `AQ_OUTPUT` to the output directory.
Cold means a fresh browser context; warm means a real page reload in the same
context. Both start a new Python worker. OS/CDN caches are uncontrolled. Report
browser/version, device, network conditions, number of successes and failures,
and the software build alongside the percentiles. Never count a failed run as a
fast success or present a single observation as p50/p75.

The first local run of this protocol found a CSV completeness defect: 37,534 Fish
River observations were reduced to 18,767 exported rows, and 51,947 Kingston
observations to 17,316. The checker now requires the downloaded row count to equal
the analyzed observation count. Earlier “passed” logs that checked only for a
nonempty CSV do not establish export completeness.

## Local baseline: 23 September 2026

Thirty sequential runs (five fresh contexts and five same-context reloads per
region) used Chromium 153.0.8010.12, Node 25.8.0, macOS, a 1280×900 viewport and
local preview build `e3b7042`. Live public agencies/archive/CDNs supplied data.
This was a local workstation run, not an isolated device/network benchmark.
OS/CDN caches and remote service load were uncontrolled.

| Record | Cold p50 / p75 to CSV | Warm p50 / p75 to CSV | Expected historical coverage |
| --- | --- | --- | --- |
| Fish River, US | 24.9s / 26.7s | 12.0s / 16.2s | 9 of 10 |
| Thames at Kingston, England | 19.5s / 22.1s | 16.8s / 18.6s | 10 of 10 |
| Seine at Paris, France | 42.3s / 45.1s | 37.2s / 41.8s | 10 of 10 |

Every run downloaded all analyzed observations and handed the same row count to
the workbench; no uncaught page errors were recorded. **One Fish River cold run
served 14,607 observations from 1986–2026**, instead of the 37,534-observation
1903–2026 record. The cold timing includes that fallback and is not a percentile
for five identical full-history workloads. The raw run status reports export and
workbench success, not restored historical coverage.

The stricter coverage assertion was added after this baseline exposed the fallback.
It now fails the daily check when these promoted records lose their known early
history, while retaining the actual period/count and export evidence. Do not call
this baseline 30 full-history successes or compare it with another dataset without
accounting for the input differences.

[Download raw timing and coverage evidence](browser-baseline-2026-09-23.json).
Later study-provenance/copy/local-retention changes do not change this baseline's
record-fetch path; the baseline still names the exact preview actually measured.
