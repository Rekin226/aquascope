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
| Daily record-to-export checks | US/England/France browser workflow downloads CSV and requires exact observation count; one failure does not skip the other regions | Deploy workflow and observe scheduled runs |
| First-use routes | Find river data, current recorded study, own-table sample; advanced tools grouped; mobile header preserves brand/search | Five uncoached practitioner sessions; proposed four independent completions |
| Performance | Repeatable fresh-context/reload protocol, five repetitions required for percentiles | 30-run baseline recorded in adoption/measurement.md; further device/network measurements remain |
| Executable quickstart | Documentation's actual Python block runs on retained observed USGS annual peaks in a temporary directory and verifies artifacts | Live retrieval is separately tested, not a dependency of the tutorial |
| Completed-study sharing | Versioned, bounded, checksummed portable file retains inputs/results/artifact bytes; import does not rerun; explicit immutable publishing instructions | User-controlled public publication and real recipient use |
| Python/R/QGIS handoff | Export schema, unit/CRS guidance and readers documented | Python tutorial executed; practitioner R/QGIS exercise pending |
| Validation scope | Method/input/comparator/tolerance matrix; synthetic and observed evidence separated; Bulletin 17C scope qualified | Independent scientific review; no blanket certification claimed |
| Current regional examples | Three retained observed CSVs, generator, reports, figures, complete studies and provenance | Independent review pending; generated at `efe3a0e9f8469011f2f6ad9a8c87967dbf22072a` |
| Software and data citation | Release DOI distinct from concept DOI; source revision and input hash retained; archive snapshot prep command includes agency rights and hashes | Zenodo account and actual dataset DOI registration; no invented DOI |
| Positioning and discovery | Explorer-first README/package/site/paper; AquaScope Hydrology titles; regional task pages | About metadata updated on GitHub; deployed site verification remains |
| Existing launch feedback | Actual Reddit requests mapped to replies/issues; focused follow-up drafts prepared | Explicit instruction required before sending outreach |
| Post-success advocacy | Optional star/problem links after export handoff; no gate | Real usefulness and downstream use are measured separately |
| Adoption measurement | Off-by-default local 28-day counters and bounded timings; inspect/export/delete; no raw inputs/identifiers/network collection | Opt-in pilot; no population conversion/retention claims |
| Pilot and ownership | Five-task comprehension protocol, ten-person cohort proposal, three real-project uses, review responsibilities and consent boundaries | Named consenting reviewers/participants/owners and actual outcomes |
| Focused release and reference adoption | Reviewable changes, evidence, release/follow-up materials | Deployment, independent cases, teaching/research use and 28-day outcomes |

## Visual and interaction decisions

Visual thesis: a calm map workspace with clear data context and restrained blue
actions. The map stays the main visual anchor. Entry content presents three useful
routes, then actual coverage and export, then optional study tools and sharing.
Existing map fly-to, panel reveal and menu transitions provide spatial context;
no decorative animation or marketing overlay is added.

## Verification log

- Python 3.12.13 final full suite: **3,109 passed, 11 skipped**. Optional skips
  and runtime warnings remain disclosed. Final source revision: `efe3a0e`.
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
