"""Multi-basin UQ benchmark validation (#78, epic #71).

Runs the GR4J quantile-UQ benchmark across a few bundled CAMELS basins and
checks that the central interval is approximately calibrated in aggregate.
This is the multi-basin demonstration that makes the UQ citable. Kept small
(few basins, short series, low maxiter) so it stays CI-runnable."""

from __future__ import annotations

import importlib.util
import math
import pathlib

EXAMPLE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples"
    / "12_uq_camels_benchmark.py"
)


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("uq_camels_benchmark", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_multi_basin_uq_is_approximately_calibrated():
    bench = _load_benchmark_module()
    results = bench.run_benchmark(
        basin_ids=["01013500", "02231000"],
        max_days=1200,
        maxiter=4,
        warmup_days=365,
    )

    assert len(results) == 2
    for r in results:
        assert math.isfinite(r["picp90"])
        assert math.isfinite(r["crps"]) and r["crps"] >= 0.0
        # every basin produced a usable central interval
        assert 0.0 <= r["picp90"] <= 1.0

    # In-sample residual bands should give central-interval coverage near the
    # nominal 0.90 when pooled across basins (calibration, not exactness).
    mean_picp = sum(r["picp90"] for r in results) / len(results)
    assert 0.80 <= mean_picp <= 0.99
