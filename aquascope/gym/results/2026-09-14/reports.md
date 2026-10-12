# Report quality on the twelve recorded studies (2026-09-14)

The twelve studies the Explorer's Study drawer replays (`explorer/showcase/studies/*/`, Claude Sonnet 5,
recorded 2026-09-07) scored with `aquascope.gym.reports.score_report`: deterministic, no model, reusing the
Studio Critic's own checks over the recorded `workspace.json`. Three cases carry a hand-written reference
(`aquascope/gym/reports/*.yaml`); the other nine are scored on the six dimensions alone. The rows, per-study
`dimensions` and `evidence`, are in `reports.jsonl`; the aggregate table is `leaderboard.md`.

```bash
aquascope gym reports bench --out aquascope/gym/results/2026-09-14/reports.jsonl
aquascope gym leaderboard aquascope/gym/results/2026-09-14/reports.jsonl \
    --out aquascope/gym/results/2026-09-14/leaderboard.md
```

| study | mean | traceability | not-established | no filled holes | intervals | citations | recommendations | reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bamako-ungauged | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.00 | 1.00 | - |
| cambridge-groundwater | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.00 | 1.00 | 1.00 |
| kingston-flood | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| little-falls-water-quality | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.00 | 1.00 | - |
| nairobi-drought | 1.00 | 1.00 | - | 1.00 | - | 1.00 | 1.00 | - |
| orleans-drought | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | - |
| own-table-flood | 1.00 | 1.00 | - | 1.00 | 1.00 | 1.00 | 1.00 | - |
| potomac-supply | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.00 | 1.00 | - |
| sintra-supply | 0.99 | 0.94 | 1.00 | 1.00 | - | 1.00 | 1.00 | - |
| taichung-drought | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.00 | 1.00 | - |
| toulouse-irrigation | 0.90 | 1.00 | - | 0.50 | 1.00 | 1.00 | 1.00 | 0.88 |
| wagga-flood | 1.00 | 1.00 | 1.00 | 1.00 | - | 1.00 | 1.00 | - |

`-` is a dimension the study gave nothing to judge (no interval-gated result was quoted at all, or nothing in
the run failed, so `not_established_completeness` has no item to check), left out of that study's mean rather
than counted as a pass. Overall: mean 0.99 across the twelve, cost 6.957 USD (what recording them cost on
2026-09-07; scoring again is free, no model call).

## Where it agrees with the study's own Critic pass, and where it goes further

Ten of the twelve score a clean 1.00 on every dimension: the reports they carry trace their numbers, name their
own gaps, quote intervals next to the estimates they gate, cite the methods they use, and keep their
recommendations inside what the run established. The two that do not, and the three references, are the
family's actual test:

**sintra-supply (0.99, traceability 0.94).** No gauge sits within 50 km of the point, so the study regionalises
flow signatures from 10 donor basins and converts the transferred mm/d values to m3/s by hand for the answer:
0.38 m3/s (the median) and 0.046 m3/s (the 4 ML/day demand) are arithmetic on numbers the tools did return
(0.0691 mm/d and the catchment area; 4 ML/day and the day-to-second conversion), not numbers a tool returned
directly. `traceability` reads this as six untraced claims (0.38 and 0.046, each repeated three times in the
answer, the summary and the methodology); the study's own Critic pass already caught the same gap as a failed
`numbers_come_from_tools` check (`checks: 3 of 4` in `meta.json`) and printed "these numbers are not in any
tool result: 0.38, 0.046, 0.38, 0.046, 0.38, 0.046" under its own limitations. `score_report` does not find a
new problem here; it turns the Critic's one failed boolean into a fraction (0.94) and a byte-for-byte list of
the numbers at fault, which is the point of scoring the document rather than trusting that "checks passed" is
binary.

**toulouse-irrigation (0.90, no_filled_holes 0.50).** Every number in the report is right (`traceability`
1.00): 564.4 mm gross demand, 96.5% of growing-season days met, a "mostly reliable" verdict, a p = 0.302
Mann-Kendall result on the 30-year annual mean flow correctly called "statistically insignificant" in the
answer. But the limitations section's own sentence about that same test reads "No cause is established for the
(statistically insignificant) downward trend" — a direction word a p = 0.302 test gives no basis for asserting
either way. This is not new either: the study's own Critic pass flagged the exact sentence as a "note" ("the
limitations text calls it a '(statistically insignificant) downward trend', which risks readers taking away a
directional signal that the test did not establish", with a rephrase suggested), but a "note" does not trigger
the Author's fix round the way a "fix" does, and the sentence shipped unchanged. `score_report`'s direction-
inversion check (the claim-locked reporting metric, `metrics.direction_inversions`) catches it independently by
comparing the sentence's claimed direction against the trend test's own classification ("no trend"), which is
what a deterministic score over the *document* is for: the plan was fine, the numbers were fine, and the flaw
still shipped in the prose a person reads.

## The three references

`aquascope/gym/reports/{kingston-flood,cambridge-groundwater,toulouse-irrigation}.yaml`, each a hydrologist's
assertions about what a faithful report of that recorded study must and must not say, scored against the same
recording:

* **kingston-flood (1.00 of 1.00).** The planned GloFAS cross-check (step `s4`) failed its `not_empty` gate on
  `'climate'`, and its `regionalize_signatures` fallback then failed too. All five `must_say` assertions pass
  (the GEV and Log-Pearson III Q100 figures, the 142.9-year record, the Mann-Kendall pre-test, and the failed
  step named in the limitations), and the one `must_not_say` (no fabricated GloFAS number) holds: the report
  says the cross-check "could not be obtained because the data service failed" and quotes nothing in its place.
* **cambridge-groundwater (1.00 of 1.00).** The full 48.7-year record (n = 227) shows a *significant increasing*
  trend (Sen's slope +0.1659 mAOD/yr); the brief asks whether levels are declining, and the honest answer is no.
  The report gets this right ("not a decline", "no declining trend"), flags its assumed-daily resolution in the
  limitations, and names the data (abstraction records) the cause question would need; the `must_not_say`
  assertion (no claim that levels are actually declining) holds.
* **toulouse-irrigation (0.88 of 1.00).** Five of five `must_say` assertions pass (the irrigation numbers, the
  verdict, the "planning estimate" hedge on the FAO-56/ERA5 demand, the trend's own p = 0.302). The one
  `must_not_say` — no directional trend claim for a test that found none — is violated by the same "downward
  trend" sentence `no_filled_holes` catches above; this is the one place in this run where a reference and the
  domain-general dimensions agree on the same, specific finding from two different mechanisms (a curated
  assertion, and a generic direction-word scan), which is some evidence the direction-inversion check
  generalises rather than being tuned to this one sentence.

## What this run does not establish

One hydrologist's three references on one model's twelve studies, all from a single recording session
(2026-09-07); no second model, no repeats, no adversarial candidate (every study scored here is the one
`aquascope studio-showcase record` actually produced, not a report deliberately built to fail a dimension).
`not_established_completeness` and `no_filled_holes` match a failed step's id, tool, or a quoted field from its
gate detail (or, failing that, the numbers in the gate's own message) against the report's prose: a keyword
heuristic, not a semantic one, so a limitations sentence that names a gap in genuinely different words than the
gate's own detail could under-score. The direction-inversion check is scoped to sentences that also mention a
trend by name ("trend", "Mann-Kendall", "Sen's slope"), which is deliberately conservative against false
positives (an earlier version, before that scope and before excluding "fallback" from matching the stem "fall",
flagged four honest sentences in the cambridge-groundwater report); it will miss a direction claim written far
from any such word. `interval_discipline` and `citations` are document-level checks (did an interval appear
anywhere in the report; are the methods' citations present anywhere in the reference list), not tied to which
specific number or method sits next to them. And the claim-locked reporting metric's other half, cross-run
agreement of report-visible numbers (a Jaccard index over repeated runs of the same case), is not computed at
all: it needs more than one run of the same case, which this benchmark's single recorded run of each study does
not have.
