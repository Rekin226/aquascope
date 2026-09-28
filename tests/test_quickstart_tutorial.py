"""Execute the actual reader-facing Python block, rather than a parallel example."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_getting_started_runs_from_the_committed_observed_fixture(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    source = Path("data/camels_benchmark/peaks/01013500_peaks.csv")
    local = tmp_path / source
    local.parent.mkdir(parents=True)
    local.write_bytes((ROOT / source).read_bytes())
    monkeypatch.chdir(tmp_path)
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "docs/getting_started.md").read_text(), re.S)
    assert len(blocks) == 1
    exec(compile(blocks[0], "docs/getting_started.md", "exec"), {})
    table = pd.read_csv(tmp_path / "quickstart-output/return_levels.csv")
    assert table.return_period_years.tolist() == [10, 50, 100]
    assert table.discharge_m3s.is_monotonic_increasing and table.discharge_m3s.min() > 0
    provenance = json.loads((tmp_path / "quickstart-output/provenance.json").read_text())
    assert provenance["data_sha256"] == hashlib.sha256(local.read_bytes()).hexdigest()
    assert provenance["n_peaks"] > 30 and provenance["interval"] is None
    assert (tmp_path / "quickstart-output/qq_diagnostic.png").read_bytes().startswith(b"\x89PNG")
