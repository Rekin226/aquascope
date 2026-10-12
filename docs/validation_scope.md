# What has been checked

AquaScope distinguishes reproducible software checks from independent scientific
validation and from acceptance for a particular decision. A passing CI run alone
does not establish the latter two.

| Capability | Input | Comparator and check | Evidence and limitation |
| --- | --- | --- | --- |
| GEV L-moments flood quantiles | Cached observed USGS annual peaks for ten catchments | Independent `lmoments3` quantiles; method-specific tolerances in the benchmark harness | `data/camels_benchmark/ffa_reference.json`; current results and known misses must be read together. |
| GEV MLE flood quantiles | Same observed peaks | SciPy reference; bounded-shape/fallback behavior can differ from unconstrained MLE | `benchmarks/known_misses.json` records exceptions. A constrained result is not proof that the unconstrained tail is reliable. |
| LP3 flood quantiles | Same observed peaks | SciPy Pearson III reference in log space | Validates the fitted distribution, not every Bulletin 17C data-screening or historical-data procedure. |
| Baseflow and daily signatures | Synthetic daily series calibrated toward ten published CAMELS catchments | Published attributes, independent references and integrity checks in `benchmarks/camels_benchmark.py` | Regression evidence. Synthetic series are not an observed CAMELS daily validation dataset. |
| Explorer flood screening | Annual maxima of daily means; years with at least 292 observed days | Fit-spread, record-length, extrapolation and maxima-trend checks | Daily means can understate instantaneous flood peaks. Check the actual period, missing years and method assumptions. |
| Study conclusions and exports | Fixed tool outputs including the captured Fish River values | Tests distinguish estimators, interval identities, grade scope and mean/maxima trends | Tests cover evidence propagation. Independent hydrologist review remains required for the reference examples. |
| Model/gauge comparisons | GloFAS neighborhood and gauge record | Flow-magnitude diagnostic, then independent catchment comparability required | Matching mean flow alone cannot verify upstream area or river topology; unverified comparisons are skipped. |

Run the numerical benchmark with `python -m benchmarks.camels_benchmark --strict`.
Read the generated per-method result, tolerance and status rather than quoting
one overall “validated” label. The [benchmark guide](https://github.com/Rekin226/aquascope/blob/main/benchmarks/README.md)
documents fixture origins, reference generation, and accepted limitations.

## Flood-frequency scope

GEV and Gumbel fitting are separate from Bulletin 17C. AquaScope includes LP3,
EMA and associated routines; whether a particular analysis follows Bulletin 17C
depends on its annual-peak input, low-outlier treatment, regional skew,
historical/censored observations and interval procedure. The ordinary Explorer
daily-flow workflow does not certify a complete regulatory procedure.

## Reproducing a result

Retain the input files, software version and source revision, data content hash,
actual analysis period, estimator, interval method, assumptions and check outcomes.
An interval belongs only to the fitted result identified in its evidence record.
A missing interval is reported as missing; another estimator's interval is not a substitute.

The software concept DOI identifies the evolving project. Use the release's
version DOI for a released build and also record the commit for unreleased work.
Data require a separate immutable archive revision or published snapshot DOI and
the original agency attribution. A hash identifies content; it is not a DOI or
a promise that the content remains publicly retrievable.

## Reproduction on 23 September 2026

The strict benchmark on the adoption branch (base `2e7d690`, uncommitted changes;
Python 3.12.13; matching the earlier Python 3.14.6 run) produced 10 catchment results: mean-flow normalized RMSE 0.00%
against a 25% aggregate threshold, baseflow PBIAS 0.92% against 25%, and dependable
flood-reference mean relative error 0.22% against 20%. This is a maintainer run.

There were **28 unmet individual checks**, including 25 documented data-limitation
findings and three accepted baseline misses. Strict mode reported zero unexpected
misses. These figures must be reported together; “strict passed” does not mean
every method agreed with every comparator. GEV-MLE limitations are reported
separately from the dependable L-moments/LP3 implementation gate.

Individual tolerances remain 25% relative error for calibrated synthetic signatures,
0.15 absolute BFI, two calendar months for peak month, and 20% relative error for
flood reference quantiles. The benchmark output records each result and rationale.
Tolerance or fixture changes
require an explicit scientific justification, not merely making CI green.
