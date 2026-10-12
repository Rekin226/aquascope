# Daily-mean flood screening: Seine at Paris-Austerlitz, France

Find the observed record, inspect its usable period, and take a reproducible
flood-screening example into your own work. This is a maintainer reproduction;
**independent hydrologist review is pending**.

[Open the gauge](https://rekin226-aquascope-explorer.static.hf.space/#s=hubeau_hydrometrie/F700000103&tab=overview) ·
[Open the recorded analysis](https://rekin226-aquascope-explorer.static.hf.space/#study=reference-seine-fr)

## What this snapshot contains

| Property | Retained evidence |
| --- | --- |
| Agency | Eaufrance / SCHAPI (Hub'Eau) |
| Station | `hubeau_hydrometrie/F700000103` |
| Observations | 7,542 daily mean discharge values, m³/s |
| Actual period | 2006-01-28 to 2026-09-21 |
| Agency licence | etalab-2.0 |
| Input CSV SHA-256 | `06735536efee67994b0c8f1941aad8c5cf2b721bc4a3cef0788e2df175be9f54` |

The recorded example compares GEV L-moments, LP3, and GEV MLE on complete-year
annual maxima. Each uncertainty interval belongs to its named estimator.
Record length, extrapolation, fit spread and maxima trend are checked separately.
The result's grade applies to this screening question and its tested checks.

**Recorded conclusion:** Screen the 100-year daily-mean flood at Seine at Paris-Austerlitz, France; identify estimator uncertainty and limitations: 100-year return level, GEV (L-moments) 1951 m3/s (not established).

Daily means can understate instantaneous flood peaks. Stationarity and independence
remain assumptions. No channel topology, regulation assessment, or model/gauge
catchment comparability was established. This is not a certified design flood or a
complete Bulletin 17C analysis.

## Keep and reproduce the evidence

Download the [observations CSV](https://rekin226-aquascope-explorer.static.hf.space/showcase/studies/reference-seine-fr/observations.csv),
[self-contained report](https://rekin226-aquascope-explorer.static.hf.space/showcase/studies/reference-seine-fr/report.html),
[complete study](https://rekin226-aquascope-explorer.static.hf.space/showcase/studies/reference-seine-fr/complete.aqstudy.json), and
[provenance](https://rekin226-aquascope-explorer.static.hf.space/showcase/studies/reference-seine-fr/provenance.json).
Use **Tools → Open saved study** for the complete file; it restores the recorded
results without rerunning the analysis. These URLs become available when this
adoption build is deployed. For immutable citation, use a named release or the
source commit recorded in the provenance, not a mutable deployment URL.

To regenerate all three examples, save the three retained CSVs under their original
names (`fish-river-us-cold.csv`, `kingston-uk-cold.csv`, `seine-fr-cold.csv`) and run
`python -m examples.adoption_reference --input-dir retained-csvs --out regenerated`.
Set `AQUASCOPE_REVISION` to the checked-out commit. The script makes no agency or
language-model calls; it regenerates statistics, checks, figures and reports.

See [validation scope](../validation_scope.md), [saving and sharing](sharing.md),
and [Python, R and QGIS handoffs](../getting_started.md#take-an-explorer-result-into-your-tools).
