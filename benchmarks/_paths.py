"""Shared filesystem locations for the CAMELS benchmark scripts and tests.

Both ``benchmarks.camels_benchmark`` and ``benchmarks.fetch_peak_flows``
resolve the benchmark data tree from the repository root; keeping the paths
in one module means a rename moves exactly one ``import`` in each consumer
rather than a hand-carved constant.
"""

from __future__ import annotations

import pathlib

BASE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = BASE.parent
BENCHMARK_DIR = REPO_ROOT / "data" / "camels_benchmark"
DAILY_CATCHMENTS_FILE = BENCHMARK_DIR / "daily_catchments.json"
DAILY_DIR = BENCHMARK_DIR / "daily"
PEAKS_DIR = BENCHMARK_DIR / "peaks"
FFA_REFERENCE_FILE = BENCHMARK_DIR / "ffa_reference.json"
