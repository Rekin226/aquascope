# Daily-mean flood screening: Thames at Kingston, England

Find the observed record, inspect its usable period, and take a reproducible
flood-screening example into your own work. This is a maintainer reproduction;
**independent hydrologist review is pending**.

[Open the gauge](https://rekin226-aquascope-explorer.static.hf.space/#s=uk_ea/8496ce69-482c-406a-a2f0-ac418ef8f099&tab=overview) ·
[Open the recorded analysis](https://rekin226-aquascope-explorer.static.hf.space/#study=reference-kingston-uk)

## What this snapshot contains

| Property | Retained evidence |
| --- | --- |
| Agency | Environment Agency |
| Station | `uk_ea/8496ce69-482c-406a-a2f0-ac418ef8f099` |
| Observations | 51,947 daily mean discharge values, m³/s |
| Actual period | 1883-10-01 to 2026-09-09 |
| Agency licence | OGL-UK-3.0 |
| Input CSV SHA-256 | `9bb306f0fe804576bc5591c8f79727dfe0a5078a9c86c2592efb89bdc7b13957` |

The recorded example compares GEV L-moments, LP3, and GEV MLE on complete-year
annual maxima. Each uncertainty interval belongs to its named estimator.
Record length, extrapolation, fit spread and maxima trend are checked separately.
The result's grade applies to this screening question and its tested checks.

**Recorded conclusion:** Screen the 100-year daily-mean flood at Thames at Kingston, England; identify estimator uncertainty and limitations: 100-year return level, GEV (L-moments) 652.5 m3/s (established).

Daily means can understate instantaneous flood peaks. Stationarity and independence
remain assumptions. No channel topology, regulation assessment, or model/gauge
catchment comparability was established. This is not a certified design flood or a
complete Bulletin 17C analysis.

## Keep and reproduce the evidence

Download the [observations CSV](https://rekin226-aquascope-explorer.static.hf.space/showcase/studies/reference-kingston-uk/observations.csv),
[self-contained report](https://rekin226-aquascope-explorer.static.hf.space/showcase/studies/reference-kingston-uk/report.html),
[complete study](https://rekin226-aquascope-explorer.static.hf.space/showcase/studies/reference-kingston-uk/complete.aqstudy.json), and
[provenance](https://rekin226-aquascope-explorer.static.hf.space/showcase/studies/reference-kingston-uk/provenance.json).
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
