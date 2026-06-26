# Uncertainty quantification

Point hydrographs are no longer enough — calibrated predictive intervals are
the expected default for credible model output. AquaScope's GR4J model emits
quantile prediction bands, and the `analysis.metrics` module provides proper
probabilistic scores to evaluate them.

## Quantile prediction with GR4J

```python
from aquascope.models.rainfall_runoff import predict_quantiles

res = predict_quantiles(
    precip, pet, observed,
    quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
    method="residual",        # or "ensemble" (parameter perturbation)
    objective="kge",
    heteroscedastic=True,     # bands widen at high flow
)
res.median            # central estimate
res.quantiles[0.05]   # lower band; bands are non-negative and monotonic
```

The deterministic `GR4J.simulate()` path is unchanged — UQ is opt-in.

## Probabilistic metrics

`analysis.metrics` adds the standard scoring rules:

- `picp(observed, lower, upper)` — prediction interval coverage probability.
- `mpiw(lower, upper, observed=None)` — mean interval width (sharpness).
- `pinball_loss(observed, predicted, quantile)` — quantile loss.
- `crps_ensemble(observed, ensemble)` and `crps_from_quantiles(observed, bands)`
  — continuous ranked probability score.

A well-calibrated central 90% interval gives PICP close to 0.90.

## Multi-basin validation

`examples/12_uq_camels_benchmark.py` runs GR4J quantile UQ across the bundled
CAMELS benchmark basins and reports per-basin and aggregate **PICP** and
**CRPS**, plus a reliability diagram (observed vs nominal coverage). On the
bundled basins the residual method achieves central-interval coverage of
~0.90 against the 0.90 nominal target, demonstrating calibration across
diverse catchments rather than a single-basin anecdote.

```bash
python examples/12_uq_camels_benchmark.py
```
