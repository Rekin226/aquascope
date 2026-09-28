# Adoption audit implementation

Tracks the 23 September 2026 adoption audit and live-study evidence. Work starts
from integration commit `2e7d690` on `fix/adoption-audit`, in an isolated checkout.
The original checkout's uncommitted changes are preserved.

Completion requires verified behavior and external outcomes. Prepared software,
maintainer reproductions and proposed targets do not establish independent adoption.

| Requirement | Implemented evidence | Remaining acceptance |
| --- | --- | --- |
| Flood claim identity | Input hash/actual period, source, variable, aggregation, estimator, own interval/method/level, assumptions, checks and grade carried through findings, report appendix and workbook | Independent hydrologist review of reference answers |
| Flood trend and grade scope | Annual maxima retained; unavailable maxima are explicit; supporting failures separated from assumptions; regression cases include opposing trends and identical rounded estimates | Review in real project context |
| Model/gauge comparability | Flow magnitude match is only diagnostic; unverified catchment comparisons are skipped | Independent topology/upstream-area evidence before promoting any comparison |
| Observation coverage and freshness | Catalog-only labels before selection, actual analyzed period/count, complete-year eligibility, last successful archive update and retained-data status | Deployed live checks across every promoted region |
| Full data export | CSV and workbench handoff preserve original observations, dates and precision; plotting decimation is explicit | Browser regression now covers full CSV and full workbench row counts |
| Archive resilience | Isolated source deadlines, atomic writes, per-station checkpoints, merged history, retained catalogs/data/bundles, visible partial-refresh status | Two actual consecutive scheduled publications after deployment |
| Daily record-to-export checks | US/England/France browser workflow downloads CSV and requires exact observation count; one failure does not skip the other regions | Deployed; six production checks passed manually on 27 September; scheduled evidence pending |
| First-use routes | Find river data, current recorded study, own-table sample; advanced tools grouped; mobile header preserves brand/search | Five uncoached practitioner sessions; proposed four independent completions |
| Performance | Repeatable fresh-context/reload protocol, five repetitions required for percentiles | 30-run baseline recorded in adoption/measurement.md; further device/network measurements remain |
| Executable quickstart | Documentation's actual Python block runs on retained observed USGS annual peaks in a temporary directory and verifies artifacts | Live retrieval is separately tested, not a dependency of the tutorial |
| Completed-study sharing | Versioned, bounded, checksummed portable file retains inputs/results/artifact bytes; import does not rerun; explicit immutable publishing instructions | User-controlled public publication and real recipient use |
| Python/R/QGIS handoff | Export schema, unit/CRS guidance and readers documented | Python tutorial executed; practitioner R/QGIS exercise pending |
| Validation scope | Method/input/comparator/tolerance matrix; synthetic and observed evidence separated; Bulletin 17C scope qualified | Independent scientific review; no blanket certification claimed |
| Current regional examples | Three retained observed CSVs, generator, reports, figures, complete studies and provenance | Independent review pending; regenerated at `0a7dd68` |
| Software and data citation | Release DOI distinct from concept DOI; source revision and input hash retained; archive snapshot prep command includes agency rights and hashes | Zenodo account and actual dataset DOI registration; no invented DOI |
| Positioning and discovery | Explorer-first README/package/site/paper; AquaScope Hydrology titles; regional task pages | About metadata and deployed site verified on 27 September at `e070fdc` |
| Existing launch feedback | Actual Reddit requests mapped to replies/issues; focused follow-up drafts prepared | Explicit instruction required before sending outreach |
| Post-success advocacy | Optional star/problem links after export handoff; no gate | Real usefulness and downstream use are measured separately |
| Adoption measurement | Off-by-default local 28-day counters and bounded timings; inspect/export/delete; no raw inputs/identifiers/network collection | Opt-in pilot; no population conversion/retention claims |
| Pilot and ownership | Five-task comprehension protocol, ten-person cohort proposal, three real-project uses, review responsibilities and consent boundaries | Named consenting reviewers/participants/owners and actual outcomes |
| Focused release and reference adoption | Reviewable changes, evidence, release/follow-up materials | Deployment verified on 27 September; independent cases, teaching/research use and 28-day outcomes pending |

## Visual and interaction decisions

Visual thesis: a calm map workspace with clear data context and restrained blue
actions. The map stays the main visual anchor. Entry content presents three useful
routes, then actual coverage and export, then optional study tools and sharing.
Existing map fly-to, panel reveal and menu transitions provide spatial context;
no decorative animation or marketing overlay is added.

## Verification log

- Earlier export baseline at `4183f82`: **3,117 passed, 11 skipped** on Python
  3.12.13. Later validation is recorded below; skips and warnings remain disclosed.
- JavaScript contract suite: 62 passed, including privacy consent/retention,
  capability labels and minimum-repetition timing summaries; added to CI.
- Ruff and whitespace checks passed. Strict documentation build, wheel build and stdlib-only Explorer assembly passed.
  Desktop and 390-pixel mobile entry screenshots were inspected.
- Corrected browser checks: six of six US/England/France cold/warm cases downloaded
  all 37,534 / 51,947 / 7,542 observations respectively. Earlier nonempty-CSV-only
  runs failed to detect decimation and are not evidence of complete export.
- Completed-study browser import/export retained the original report, run, input
  tables and all 13 artifact payloads in a fixed test fixture. The actual Fish River
  complete study also retained its report, run, tables and all nine artifacts exactly.
  Its 37,534-row CSV survived station handoff and file re-import.
- Strict numerical benchmark reproduced on Python 3.12.13 and 3.14.6: 10 catchments,
  zero unexpected misses; 28 unmet individual checks include 25 documented data
  limitations and three accepted baseline misses. See validation_scope.md.
- A real archive deposit package was prepared from Hugging Face revision
  `f1f2fa19996aacb0abf82349b28ac5de16241fc7`: 15 source files, 90,246,895 bytes before
  compression. ZIP integrity and all 15 file sizes/SHA-256 hashes verified. It has no registered
  DOI and has not been published to Zenodo.

## Human and elapsed-time gates

Independent scientific review, practitioner usability sessions, three real project
uses, 28-day return behavior and consecutive scheduled publication runs remain open.
The pilot protocol and deposit package are prepared; account/participant details
have been requested. Do not mark these complete based on tests or synthetic users.

- Repeated browser baseline: all 30 exports/workbench handoffs preserved analyzed
  rows; 29 retained expected history. One Fish River fallback began in 1986. The
  new historical-coverage guard treats that as a failure; baseline timings disclose
  it explicitly. Operational coverage still needs the reviewed archive deployment.

- The original four-step Fish River question completed in the browser with 29
  artifacts. GEV L-moments was 463.9 m³/s with no borrowed interval; the model
  comparison was explicitly skipped for unverified catchment comparability.
  Approval-plan text was then aligned with that behavior. Flood length and
  extrapolation gates now use complete annual maxima (`ffa.n_years`), with a
  sparse-record regression, and the model-facing catalogue uses the same paths.

- Live JSON inspection found and repaired an additional identity mismatch: the
  compact flood tool now preserves variable/hash/revision fields, and deduplication
  replaces the whole selected claim rather than changing only its step label.
  Distinct input snapshots remain distinct even when rounded values agree.

- Final four-step browser study passed the stronger contract: the headline result
  id is `s3.ffa.fits.gev_lmoments.q.5`, its input hash and variable equal the flood
  tool payload's own fields, and its interval is absent rather than borrowed.

## Export and archive verification on 27 September

- The actual study bundle exposed a second export path that still used plotted
  points. Station and flood steps now retain their own full observation CSVs;
  a missing full input cannot silently become a decimated record table.
- Flood CSVs now use one row per estimator and return period, with that fit's own
  bounds, interval method and confidence level. Word reports render embedded
  Markdown tables and fenced code; wide tables repeat identifying columns in panels.
- Workbook tables over 100,000 rows continue in numbered sheets. A 100,001-row
  regression confirms no omitted rows. CSV timestamps and serialized precision
  remain authoritative when a spreadsheet application imposes numeric limits.
- The repeated four-step browser study at `4183f82` produced 30 artifacts. Steps
  s2 and s3 each retained 37,538 observations; both CSVs reproduced their respective
  input hashes exactly and both workbook sheets had 37,538 data rows. Every exported
  flood estimate and bound matched its own fit. The GEV L-moments headline remained
  463.9 m³/s, with no borrowed interval.
- The capped manual archive run [35817109325](https://github.com/Rekin226/aquascope/actions/runs/35817109325)
  completed with publication disabled. Its refresh status was partial: Poland
  water-level retrieval timed out; bundling and artifact upload still completed.
  All 13 Parquet row counts matched the manifest. All 2,185 previously nonempty
  station files remained present with nondecreasing manifest counts. This checks
  retention metadata and file presence, not bytewise equivalence of every observation.
  See [retained run evidence](adoption/archive-dry-run-2026-09-23.json).
  A capped manual run does not satisfy the two scheduled-publication gate.

- Final report corrections at `f6ce050` distinguish skipped comparisons from
  passed gates in the summary, step narrative and limitations. Unavailable GloFAS
  statistics are explicit; the primary estimate's scoped grade is preserved.
- Re-authored the retained real browser study and visually inspected all 27 Word
  pages: repeated table headers, unsplit rows, figure/caption placement, reference
  numbering and a single reproducibility appendix. This is an export-layout check,
  not independent scientific review. The original downloaded study remains retained.
- All 30 artifacts, full input tables, run results and report survived the actual
  browser import/export roundtrip. Three reference examples were regenerated from
  retained observed CSVs at `f6ce050`; independent review remains pending.
- Final local validation: Python 3.12 full suite **3,118 passed, 11 skipped**;
  Ruff, wheel build, strict documentation build and Explorer assembly passed.

## Current-main integration on 27 September

- Merged `e480bba` into the adoption branch after new base changes created conflicts.
  Kept the 0.19 release, observation-quality support, IMGW cache and USGS throttling
  behavior. Scheduled isolated workers now use main's per-source station/time
  budgets; USGS runs last. Rate-limited stations remain eligible for the next run.
- Python reports and the Explorer citation dialog use the 0.19 release DOI recorded
  in CITATION.cff (`10.5281/zenodo.22930129`), alongside the actual source revision.
- Regenerated the three observed reference studies against merged source `0a7dd68`.
  The earlier archive dry run remains evidence for its recorded source revision,
  not proof of production behavior after this merge.
- Merged-build validation: **3,200 passed, 11 skipped**, all **62 JavaScript
  checks**, Ruff, strict documentation build, wheel build and Explorer assembly.
  Follow-up gate presentation checks cover the browser footer, event summary,
  workbook and notebook so skipped checks are never presented as passed.
- The final skipped-gate presentation follow-up (`104ab2d`) passed all **242 Studio
  tests**, all **62 JavaScript checks**, lint and wheel/Explorer builds. The merged
  reference examples remain explicitly pinned to `0a7dd68`.

## Notebook replay verification on 27 September

- The old notebook replaced run results in the original workspace but retained
  old findings/artifacts and omitted its new gate list. It also bypassed Studio's
  full-observation tool wrappers. Replay now creates a fresh workspace and runs the
  full Studio pipeline, then exports to `rerun-<id>/` without overwriting the original.
- Source `f7ea193` passed all **245 Studio tests**, including execution of the actual
  generated cells, failed-agency stale-evidence prevention, and source-pinned install
  instructions. Ruff, strict documentation and wheel builds also passed.
- The real Fish River notebook executed all four steps and saved 32 artifacts.
  Original files stayed byte-identical. The native request fell back to 14,607
  observations from 1986–2026; its new 583.2 m³/s estimate, input hash and workbook
  all describe that shorter record, not the prior browser's 37,538 observations and
  463.9 m³/s estimate. Both new CSV input hashes and workbook row counts verified.
  This is successful replay with a disclosed coverage limitation, not restored
  full-history availability. See [replay evidence](adoption/notebook-replay-2026-09-27.json).

## Release verification — 27 September 2026

[PR #457](https://github.com/Rekin226/aquascope/pull/457) merged as
`e070fdc77b083e71c9ef37ec59807e2e4e276abf`. All final PR checks passed.
The [Explorer deployment](https://github.com/Rekin226/aquascope/actions/runs/36297315173)
and [documentation deployment](https://github.com/Rekin226/aquascope/actions/runs/36297315134)
succeeded; both public wheel manifests served build `e070fdc`, package 0.19.0.

The deployed Explorer passed six browser checks: fresh context and reload for
Fish River (37,538 observations), Kingston (51,947), and Seine (7,543). Every CSV
and workbench handoff retained the analyzed record, and each case met its reference
historical coverage requirement. These are manual release checks, not scheduled-run
or population adoption evidence.

The deployed four-step Fish River study produced 30 artifacts. Its GEV L-moments
estimate was 463.9 m³/s with no borrowed interval. Both full-input CSV hashes matched
their result snapshots and workbook sheets retained all 37,538 rows. The unverified
model comparison remained skipped: 11 of 12 gates passed, one skipped. Reopening
and exporting the actual complete study preserved its report, run, tables, and all
30 artifacts exactly. Desktop and mobile displays were inspected.

The release is deployed. Independent hydrologist review, consenting practitioner
sessions and real-project pilots, dataset DOI registration, two consecutive
scheduled archive publications, and 28-day adoption evidence remain follow-up
outcomes. No independent reviewers, users, or elapsed-time outcomes are inferred
from successful software tests.
