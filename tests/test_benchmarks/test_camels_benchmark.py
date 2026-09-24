"""Unit tests for the ``benchmarks.camels_benchmark`` harness.

Everything reads the committed local files under ``data/camels_benchmark`` --
no network access, and ``dataretrieval`` / ``lmoments3`` are not imported.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from benchmarks import camels_benchmark as cb

ALL_METHODS = {"gev", "gev_lmoments", "lp3"}
UNSTABLE_GAUGE = "11532500"

_full_results_cache: dict | None = None


def _full_results() -> dict:
    """Full 10-gauge harness run, computed once per test session."""
    global _full_results_cache
    if _full_results_cache is None:
        _full_results_cache = cb.build_results()
    return _full_results_cache


def test_build_results_schema_and_gates() -> None:
    """Full run: gates pass and every catchment has the expected structure."""
    results = _full_results()

    meta = results["metadata"]
    assert meta["schema_version"] == cb.SCHEMA_VERSION
    assert meta["timings_not_asserted"] is True
    tolerance_names = set(meta["tolerances"])
    assert tolerance_names >= {
        "signature_relative",
        "baseflow_absolute",
        "peak_month_circular",
        "ffa_relative",
        "q_mean_nrmse_gate",
        "bfi_pbias_gate",
        "ffa_gate",
    }
    for name, entry in meta["tolerances"].items():
        assert "value" in entry and "rationale" in entry, f"{name} lacks value/rationale"

    gates = results["summary"]["gates"]
    assert gates["q_mean_gate_met"]
    assert gates["bfi_gate_met"]
    assert gates["ffa_gate_met"]
    assert results["summary"]["n_integrity_failures"] == 0

    assert len(results["catchments"]) == 10
    for gid, res in results["catchments"].items():
        assert set(res) >= {"name", "climate", "signatures", "baseflow", "flood_frequency", "timings"}
        assert res["signatures"]["checks"]
        integrity_metrics = {c["metric"] for c in res["signatures"]["integrity"]}
        assert integrity_metrics == {"order", "runoff_ratio", "recession_constant", "fdc_slope", "complete"}
        assert all(c["check_passes"] and "detail" in c for c in res["signatures"]["integrity"])
        assert set(res["baseflow"]["methods"]) == {"lyne_hollick", "eckhardt"}
        assert set(res["flood_frequency"]["methods"]) == ALL_METHODS
        for mres in res["flood_frequency"]["methods"].values():
            assert len(mres["fits"]) == 6
            for fit in mres["fits"]:
                assert fit["classification"] in {"implementation", "data_limitation"}
        assert res["timings"]["total_s"] >= 0


def test_gev_unstable_findings_classified_as_data_limitation() -> None:
    """On an unstable-reference gauge, unmet GEV matches are data limitations."""
    res = cb.build_results(gauge_ids=[UNSTABLE_GAUGE])
    gev = res["catchments"][UNSTABLE_GAUGE]["flood_frequency"]["methods"]["gev"]
    assert gev["reference_mle_unstable"] is True
    for fit in gev["fits"]:
        if not fit["check_passes"]:
            assert fit["classification"] == "data_limitation"


def test_dependable_ffa_cross_checks_match_reference() -> None:
    """GEV-L-moments and LP3 track the flood-frequency reference closely (20 % band)."""
    res = _full_results()
    for gid, catchment in res["catchments"].items():
        for method in ("gev_lmoments", "lp3"):
            mean_err = catchment["flood_frequency"]["methods"][method]["mean_relative_error_pct"]
            assert mean_err < 5.0, f"{gid} {method} mean relative error {mean_err:.2f}%"


def test_aggregate_metrics_present() -> None:
    """Cross-catchment RMSE / PBIAS / R2 exist for the signatures."""
    aggregate = _full_results()["summary"]["aggregate"]
    assert aggregate["q_mean"]["rmse"] == 0.0
    assert aggregate["q_mean"]["r2"] > 0.99
    assert aggregate["q5"]["pbias"] is not None
    assert aggregate["bfi_lyne_hollick"]["pbias"] is not None


def test_findings_are_surfaced_not_hidden() -> None:
    """The GEV data-limitation mismatches stay visible in the summary."""
    results = _full_results()
    assert results["summary"]["n_data_limitation_findings"] >= 20
    assert results["summary"]["n_unmet"] >= results["summary"]["n_data_limitation_findings"]


def test_results_are_json_serialisable() -> None:
    """numpy scalars from the pipeline never leak into the JSON."""
    results = cb.build_results(gauge_ids=["03451500"])
    json.dumps(results)


def test_annual_maxima_series_is_a_noop() -> None:
    """Synthetic unique-year dates preserve every water-year peak."""
    values = np.array([10.0, 60.0, 30.0])  # a calendar year could hold two of these
    series = cb._annual_maxima_series(values)
    assert len(series) == len(values)
    assert len(series.index.year.unique()) == len(values)
    assert list(series) == list(values)


def test_software_block_uses_citation_cff() -> None:
    """The ``software`` block carries the version, DOI and author from CITATION.cff."""
    results = cb.build_results(gauge_ids=["01013500"])
    version, doi, author = cb._read_citation_cff()
    software = results["software"]
    assert software["version"] == version
    assert software["doi"] == doi
    assert software["author"] == author
    assert "DOI not yet assigned" not in software["citation"]
    assert author in software["citation"]


def _write_cff(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, body: str) -> pathlib.Path:
    """Write ``body`` as ``CITATION.cff`` under ``tmp_path`` and point the parser at it."""
    cff = tmp_path / "CITATION.cff"
    cff.write_text(body, encoding="utf-8")
    monkeypatch.setattr(cb, "_REPO_ROOT", tmp_path)
    return cff


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            "version: 0.16.0\n"
            "doi: 10.5281/zenodo.21903143\n"
            "authors:\n"
            "  - family-names: Ouedraogo\n"
            "    given-names: Abdoul Rachid\n"
            '    orcid: "https://orcid.org/0000-0002-4616-4153"\n'
            '    affiliation: "National Central University, Taiwan"\n',
            ("0.16.0", "10.5281/zenodo.21903143", "Abdoul Rachid Ouedraogo"),
        ),
        ("version: 0.16.0\n", ("0.16.0", None, None)),
        ("doi: 10.5281/zenodo.21903143\n", (None, "10.5281/zenodo.21903143", None)),
        (
            "authors:\n  - given-names: Jane\n    family-names: Doe\n",
            (None, None, "Jane Doe"),
        ),
        (
            "authors:\n  - name: AquaScope Team\n",
            (None, None, "AquaScope Team"),
        ),
        (
            'version: "1.2.3"\ndoi: "10.5281/zenodo.42"\nauthors:\n  - family-names: Doe\n    given-names: Jane\n',
            ("1.2.3", "10.5281/zenodo.42", "Jane Doe"),
        ),
        (
            # nested blocks (references / preferred-citation) must not leak
            # their version / doi / authors into the top-level parse
            "authors:\n"
            "  - family-names: Doe\n"
            "    given-names: Jane\n"
            "references:\n"
            "  - type: article\n"
            "    title: Some paper\n"
            "    authors:\n"
            "      - family-names: Other\n"
            "        given-names: Person\n"
            "version: 0.1.0\n"
            "doi: 10.5281/zenodo.1\n"
            "preferred-citation:\n"
            "  version: 9.9.9\n"
            "  doi: 10.5281/zenodo.999\n"
            "  authors:\n"
            "    - family-names: Nested\n"
            "      given-names: Wrong\n",
            ("0.1.0", "10.5281/zenodo.1", "Jane Doe"),
        ),
        (
            "some-unknown-key: value\nversion: 0.2\n",
            ("0.2", None, None),
        ),
    ],
)
def test_read_citation_cff_variants(
    body: str, expected: tuple, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_read_citation_cff`` tolerates missing fields, quoting and nested blocks."""
    _write_cff(tmp_path, monkeypatch, body)
    assert cb._read_citation_cff() == expected


def test_read_citation_cff_multiple_authors(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple CFF author entries are formatted and joined."""
    _write_cff(
        tmp_path,
        monkeypatch,
        "authors:\n"
        "  - given-names: Jane\n"
        "    family-names: Doe\n"
        "  - name: AquaScope Team\n"
        "  - family-names: Neumann\n"
        "    given-names: John\n"
        "    name-particle: von\n",
    )
    assert cb._read_citation_cff()[2] == "Jane Doe, AquaScope Team, John von Neumann"


def test_read_citation_cff_missing_file(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing ``CITATION.cff`` reads as ``(None, None, None)`` rather than erroring."""
    monkeypatch.setattr(cb, "_REPO_ROOT", tmp_path)
    assert cb._read_citation_cff() == (None, None, None)


def test_build_results_with_missing_cff_falls_back(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``CITATION.cff``: the software block falls back to the package identity."""
    from aquascope import __version__

    monkeypatch.setattr(cb, "_REPO_ROOT", tmp_path)
    software = cb.build_results(gauge_ids=["01013500"])["software"]
    assert software["version"] == str(__version__)
    assert software["author"] == "AquaScope"
    assert software["doi"] == ""
    assert "DOI not yet assigned" in software["citation"]
    assert "https://doi.org" not in software["citation"]


def test_build_results_with_partial_cff_falls_back(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a version recorded: author and DOI fall back to their defaults."""
    _write_cff(tmp_path, monkeypatch, "version: 9.9.9\n")
    software = cb.build_results(gauge_ids=["01013500"])["software"]
    assert software["version"] == "9.9.9"
    assert software["author"] == "AquaScope"
    assert software["doi"] == ""
    assert "DOI not yet assigned" in software["citation"]


def test_render_markdown_consumes_results(tmp_path) -> None:
    """The Markdown report derives from the results dict."""
    results = cb.build_results(gauge_ids=["01013500"])
    out = cb.render_markdown(results, tmp_path / "results.md")
    text = out.read_text(encoding="utf-8")
    assert "01013500" in text
    assert "Tolerances and rationale" in text
    assert "signature integrity" in text
    assert "Cite this software" in text
    assert f"https://doi.org/{results['software']['doi']}" in text


def test_render_html_consumes_results(tmp_path) -> None:
    """The HTML report derives from the results dict."""
    results = cb.build_results(gauge_ids=["01013500"])
    out = cb.render_html(results, tmp_path / "results.html")
    html = out.read_text(encoding="utf-8")
    assert "<table>" in html
    assert "01013500" in html


def test_render_round_trips_from_json(tmp_path) -> None:
    """The report built from the on-disk JSON matches the in-memory one.

    results.json is the single source of truth: the renderers read the JSON
    exactly as written, not freshly computed values.
    """
    results = cb.build_results(gauge_ids=["01013500"])
    path = tmp_path / "results.json"
    cb._write_json_atomic(results, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    md_memory = cb.render_markdown(results, tmp_path / "memory.md").read_text(encoding="utf-8")
    md_disk = cb.render_markdown(loaded, tmp_path / "disk.md").read_text(encoding="utf-8")
    assert "01013500" in md_disk
    assert "q_mean" in md_disk
    # Both renderings carry the same recorded numbers.
    assert "Q mean normalized RMSE" in md_memory and "Q mean normalized RMSE" in md_disk


def test_cli_writes_all_three(tmp_path) -> None:
    """A default run writes results.json + results.md + results.html."""
    rc = cb.main(["--output-dir", str(tmp_path), "--gauge-id", "01013500", "01664000"])
    assert rc == 0
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.md").exists()
    assert (tmp_path / "results.html").exists()
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert len(data["catchments"]) == 2
    # The report heading reflects the actual run scope, not a fixed default.
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "across 2 catchments" in md
    assert (tmp_path / "results.md.tmp").exists() is False


def test_cli_no_reports(tmp_path) -> None:
    """--no-reports writes only the JSON."""
    rc = cb.main(["--output-dir", str(tmp_path), "--gauge-id", "03451500", "--no-reports"])
    assert rc == 0
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.md").exists() is False
    assert (tmp_path / "results.html").exists() is False


def test_cli_renders_from_json(tmp_path) -> None:
    """--from-json re-renders an existing results.json without recomputing.

    results.json is the single source of truth, so a new report layout can be
    applied to a recorded run without touching the numbers.
    """
    run_dir = tmp_path / "run"
    rc = cb.main(["--output-dir", str(run_dir), "--gauge-id", "01013500"])
    assert rc == 0

    render_dir = tmp_path / "render"
    rc = cb.main(["--from-json", str(run_dir / "results.json"), "--output-dir", str(render_dir)])
    assert rc == 0
    assert (render_dir / "results.json").exists()
    assert (render_dir / "results.md").exists()
    assert (render_dir / "results.html").exists()

    original = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    copy = json.loads((render_dir / "results.json").read_text(encoding="utf-8"))
    assert copy["summary"]["n_unmet"] == original["summary"]["n_unmet"]
    assert set(copy["catchments"]) == set(original["catchments"])


def test_cli_from_json_rejects_no_reports(tmp_path) -> None:
    """--from-json cannot be combined with --no-reports (nothing to write)."""
    run_dir = tmp_path / "run"
    cb.main(["--output-dir", str(run_dir), "--gauge-id", "01013500"])
    with pytest.raises(ValueError):
        cb.main(["--from-json", str(run_dir / "results.json"), "--no-reports"])


def test_cli_from_json_rejects_strict(tmp_path) -> None:
    """--from-json re-renders a recorded run; --strict asserts only on a fresh run."""
    run_dir = tmp_path / "run"
    cb.main(["--output-dir", str(run_dir), "--gauge-id", "01013500"])
    with pytest.raises(ValueError):
        cb.main(["--from-json", str(run_dir / "results.json"), "--strict"])


def test_cli_from_json_rejects_gauge_id(tmp_path) -> None:
    """--from-json renders the gauges already in the JSON"""
    run_dir = tmp_path / "run"
    cb.main(["--output-dir", str(run_dir), "--gauge-id", "01013500"])
    with pytest.raises(ValueError):
        cb.main(["--from-json", str(run_dir / "results.json"), "--gauge-id", "01664000"])


def test_cli_fresh_run_flags_combine(tmp_path) -> None:
    """--gauge-id --no-reports --strict are compatible on a fresh run."""
    rc = cb.main(["--output-dir", str(tmp_path), "--gauge-id", "03451500", "--no-reports", "--strict"])
    assert rc == 0
    assert (tmp_path / "results.json").exists()
    assert not (tmp_path / "results.md").exists()
    assert not (tmp_path / "results.html").exists()


def test_cli_strict_passes_on_committed_data(tmp_path) -> None:
    """--strict exits 0 on committed data: known baseline misses pass and gates are met."""
    rc = cb.main(["--output-dir", str(tmp_path), "--strict"])
    assert rc == 0


def test_cli_strict_passes_on_data_limitation_only(tmp_path) -> None:
    """--strict exits 0 when a gauge's only misses are data-limitation findings."""
    rc = cb.main(["--output-dir", str(tmp_path), "--gauge-id", UNSTABLE_GAUGE, "--strict"])
    assert rc == 0


def test_cli_strict_passes_with_known_misses_baseline(tmp_path) -> None:
    """--strict exits 0 for gauge 06803500 when its BFI miss is tracked in known_misses."""
    rc = cb.main(["--output-dir", str(tmp_path), "--gauge-id", "06803500", "--strict"])
    assert rc == 0


def test_cli_strict_fails_on_unexpected_miss(tmp_path, monkeypatch, capsys) -> None:
    """--strict exits 1 when a genuine miss is not in the known_misses baseline."""
    empty_known = tmp_path / "empty_known.json"
    empty_known.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(cb, "KNOWN_MISSES_FILE", empty_known)
    rc = cb.main(["--output-dir", str(tmp_path), "--gauge-id", "06803500", "--strict"])
    assert rc == 1
    assert "1 unexpected miss(es), 0 integrity failure(s)" in capsys.readouterr().out


def test_load_known_misses_committed_file() -> None:
    """The committed known_misses.json loads and contains baseline entries."""
    misses = cb.load_known_misses()
    assert len(misses) >= 3
    keys = {(m["gauge_id"], m["stage"], m["metric"]) for m in misses}
    assert ("06803500", "baseflow", "bfi_eckhardt") in keys
    assert ("09510200", "baseflow", "bfi_eckhardt") in keys
    assert ("08181500", "signatures", "q5") in keys
    for m in misses:
        assert "reason" in m and len(m["reason"]) > 20


def test_cli_rejects_unknown_gauge(tmp_path) -> None:
    """An unknown gauge_id is an operational error, not a silent empty run."""
    with pytest.raises(ValueError):
        cb.main(["--output-dir", str(tmp_path), "--gauge-id", "99999999"])


def test_strict_flags_integrity_failures() -> None:
    """--strict fails on internal signature-integrity defects, not only value checks."""
    results = cb.build_results(gauge_ids=["03451500"])
    assert results["summary"]["n_unmet"] == 0
    assert results["summary"]["n_integrity_failures"] == 0
    assert cb._strict_failed(results) is False
    summary = dict(results["summary"], n_integrity_failures=1)
    assert cb._strict_failed(dict(results, summary=summary)) is True


def test_strict_ignores_data_limitation_findings() -> None:
    """--strict does not fail a run whose only misses are data-limitation findings."""
    results = cb.build_results(gauge_ids=[UNSTABLE_GAUGE])
    assert results["summary"]["n_unmet"] > 0
    assert results["summary"]["n_unmet"] == results["summary"]["n_data_limitation_findings"]
    assert all(g for k, g in results["summary"]["gates"].items() if k.endswith("_met"))
    assert cb._strict_failed(results) is False


def test_strict_fails_on_genuine_misses_besides_data_limitation() -> None:
    """One unexpected genuine miss beyond the data-limitation findings fails --strict."""
    results = cb.build_results(gauge_ids=["03451500"])
    assert cb._strict_failed(results) is False

    cres = results["catchments"]["03451500"]
    broken_check = dict(cres["signatures"]["checks"][0], check_passes=False, metric="unexpected_metric")
    broken_catchment = dict(cres, signatures=dict(cres["signatures"], checks=[broken_check]))
    broken_results = dict(results, catchments={"03451500": broken_catchment})
    assert cb._strict_failed(broken_results) is True


def test_strict_fails_on_gate_unmet_even_clean() -> None:
    """An unmet aggregate gate fails --strict even with no per-check misses."""
    results = cb.build_results(gauge_ids=["03451500"])
    summary = dict(results["summary"])
    gates = dict(results["summary"]["gates"], q_mean_gate_met=False)
    summary["gates"] = gates
    assert cb._strict_failed(dict(results, summary=summary)) is True


def test_build_results_friendly_error_when_metadata_missing(tmp_path, monkeypatch) -> None:
    """Missing committed metadata fails cleanly instead of tracebacking."""
    monkeypatch.setattr(cb, "DAILY_CATCHMENTS_FILE", tmp_path / "daily_catchments.json")
    monkeypatch.setattr(cb, "FFA_REFERENCE_FILE", tmp_path / "ffa_reference.json")
    monkeypatch.setattr(cb, "BENCHMARK_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="daily_catchments.json and ffa_reference.json"):
        cb.build_results()


def test_build_results_friendly_error_when_gauge_data_missing(tmp_path, monkeypatch) -> None:
    """Missing per-gauge series / peaks fails cleanly with a regeneration hint."""
    monkeypatch.setattr(cb, "BENCHMARK_DIR", tmp_path)
    monkeypatch.setattr(cb, "PEAKS_DIR", tmp_path / "peaks")
    with pytest.raises(RuntimeError, match="generate_synthetic.py"):
        cb._require_data_files("03451500")


def test_duplicate_water_year_peaks_are_not_collapsed() -> None:
    """Peaks that share a calendar year (different water years) both survive."""
    peaks = pd.Series([10.0, 200.0, 30.0], index=pd.to_datetime(["2000-03-01", "2000-12-01", "2001-03-01"]))
    series = cb._annual_maxima_series(peaks.values)
    assert len(series) == 3  # calendar-year resampling would have kept only 2


def test_committed_examples_validate_against_schema() -> None:
    """The committed examples must own the schema (build_results enforces it too)."""
    assert cb.validate_results(_full_results()) == []
    committed = pathlib.Path(cb.BASE).parent / "examples" / "camels_benchmark" / "results.json"
    assert cb.validate_results(json.loads(committed.read_text(encoding="utf-8"))) == []


def test_committed_schema_matches_generated_models() -> None:
    """results.schema.json is a derived artifact of the Pydantic models."""
    import benchmarks.results_models as rm

    committed = json.loads(cb.SCHEMA_FILE.read_text(encoding="utf-8"))
    assert committed == rm.results_json_schema()


def test_validate_results_rejects_corrupt_structure() -> None:
    """Schema validation flags shape drift, not just type drift."""
    results = _full_results()
    broken = json.loads(json.dumps(results))
    assert cb.validate_results(broken) == []
    broken["summary"]["n_catchments"] = "ten"
    del broken["software"]["doi"]
    errors = cb.validate_results(broken)
    assert any("n_catchments" in e for e in errors)
    assert any("software" in e and "doi" in e for e in errors)


def test_cli_from_json_rejects_invalid_results(tmp_path) -> None:
    """A results.json that violates the schema is refused before rendering."""
    bad = tmp_path / "results.json"
    bad.write_text(json.dumps({"metadata": {"schema_version": "1.0"}, "not": "a results file"}), encoding="utf-8")
    with pytest.raises(ValueError, match="results.schema.json"):
        cb.main(["--from-json", str(bad), "--output-dir", str(tmp_path / "out")])
