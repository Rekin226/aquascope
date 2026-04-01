# CAMELS Benchmark Validation

This directory contains scripts that validate AquaScope's hydrological
computations against published catchment characteristics from the
CAMELS dataset (Catchment Attributes and MEteorology for Large-sample
Studies).

## Approach

1. **Synthetic benchmark data** — 10 years of daily streamflow are
   generated for 10 diverse US catchments using a log-normal AR(1) model
   with seasonal modulation.  The generator is seeded (`seed=42`) for
   full reproducibility.

2. **Signature computation** — AquaScope's `compute_signatures()` and
   `lyne_hollick()` functions produce hydrological signatures from
   the synthetic series.

3. **Comparison** — Computed signatures are compared against published
   CAMELS attributes with generous tolerances (±25 % relative, ±0.15
   absolute for BFI, ±2 months for peak timing).  Because the input
   data is *synthetic*, exact matches are not expected.  The goal is to
   catch **gross errors** — sign flips, off-by-order-of-magnitude bugs,
   or broken computations.

## Why synthetic data is still useful

- **Reproducible** — no external downloads or credentials needed.
- **Deterministic** — same seed always produces the same CSV files.
- **Structurally realistic** — autocorrelated, seasonal, non-negative
  discharge with realistic spread and magnitude.
- **Catches regressions** — if a refactoring accidentally breaks the
  signature computation, the tests will fail.

## Running

```bash
# Generate the synthetic CSV files (only needed once or after changes)
python data/camels_benchmark/generate_synthetic.py

# Run the validation script (prints a table and writes validation_results.csv)
python examples/validation/validate_camels.py

# Run the automated test suite
pytest tests/test_validation/ -v
```

## References

- Addor, N., Newman, A. J., Mizukami, N., and Clark, M. P. (2017).
  The CAMELS data set: catchment attributes and meteorology for
  large-sample studies.  *Hydrol. Earth Syst. Sci.*, 21, 5293–5313.
  doi:[10.5194/hess-21-5293-2017](https://doi.org/10.5194/hess-21-5293-2017)

- Newman, A. J., Clark, M. P., Sampson, K., et al. (2015).
  Development of a large-sample watershed-scale hydrometeorological
  data set for the contiguous USA.  *Hydrol. Earth Syst. Sci.*, 19,
  209–223.
  doi:[10.5194/hess-19-209-2015](https://doi.org/10.5194/hess-19-209-2015)
