"""Unit tests for the ``benchmarks.fetch_peak_flows`` generation script.

``dataretrieval`` (optional, generation-only) is
injected as a stub so no network access occurs. ``lmoments3``-dependent tests
are skipped if the real package is unavailable.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks import fetch_peak_flows as fpf

FT3S_TO_M3S = fpf.FT3S_TO_M3S
RETURN_PERIODS = fpf.RETURN_PERIODS

try:
    import lmoments3  # noqa: F401

    HAS_LMOMENTS3 = True
except ImportError:
    HAS_LMOMENTS3 = False


# Helpers to build a fake NWIS peaks frame


def _peaks_df(
    values: list[float | str | None],
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Build a fake NWIS peaks response DataFrame indexed by datetime."""
    if years is None:
        years = list(range(2000, 2000 + len(values)))
    dates = pd.to_datetime([f"{year}-05-01" for year in years])
    frame = pd.DataFrame({"peak_va": values, "peak_dt": dates})
    return frame.set_index("peak_dt")


def _qualifier_frame(
    values: list[object],
    codes: list[object],
) -> pd.DataFrame:
    """Build a raw NWIS peaks frame with a mixed-type ``peak_cd`` column."""
    return pd.DataFrame({"peak_va": values, "peak_cd": codes})


class _FakeNWIS:
    """A ``dataretrieval.nwis`` stand-in with a configurable ``get_record``."""

    response: pd.DataFrame | None

    @classmethod
    def get_record(cls, sites: str, service: str) -> pd.DataFrame | None:
        if service != "peaks":
            raise AssertionError(f"unexpected service {service!r}")
        return cls.response


@pytest.fixture(autouse=True)
def _inject_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``dataretrieval`` so no real (network) import happens."""
    monkeypatch.setitem(sys.modules, "dataretrieval", SimpleNamespace(nwis=_FakeNWIS))


# _fetch_peaks


def test_fetch_peaks_converts_and_indexes() -> None:
    """A clean fetch returns m^3/s peaks indexed by datetime."""
    cfs_values = [100.0 + 10.0 * year_index for year_index in range(10)]
    _FakeNWIS.response = _peaks_df(cfs_values, years=list(range(1995, 2005)))
    result = fpf._fetch_peaks("99999999")
    assert list(result) == [value * FT3S_TO_M3S for value in cfs_values]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index[0].year == 1995


def test_fetch_peaks_rejects_too_few_records() -> None:
    """Fewer than 10 peaks is a hard error, raised as RuntimeError."""
    _FakeNWIS.response = _peaks_df([100.0] * 9)
    with pytest.raises(RuntimeError, match="99999999"):
        fpf._fetch_peaks("99999999")


def test_fetch_peaks_rejects_missing_response() -> None:
    """A ``None`` response is a hard error, not a silent skip."""
    _FakeNWIS.response = None
    with pytest.raises(RuntimeError, match="99999999"):
        fpf._fetch_peaks("99999999")


def test_fetch_peaks_rejects_empty_or_missing_column() -> None:
    """An empty frame or one lacking ``peak_va`` raises RuntimeError."""
    _FakeNWIS.response = pd.DataFrame({})
    with pytest.raises(RuntimeError, match="99999999"):
        fpf._fetch_peaks("99999999")
    _FakeNWIS.response = pd.DataFrame({"other": [1.0, 2.0]})
    with pytest.raises(RuntimeError, match="99999999"):
        fpf._fetch_peaks("99999999")


def test_fetch_peaks_drops_nulls_before_length_check() -> None:
    """Null peak values are removed *before* the >=10 check."""
    values = [100.0 if row_index % 2 == 0 else None for row_index in range(24)]  # 12 valid, 12 null
    _FakeNWIS.response = _peaks_df(values, years=list(range(1990, 1990 + len(values))))
    result = fpf._fetch_peaks("99999999")
    assert len(result) == 12
    assert all(value == 100.0 * FT3S_TO_M3S for value in result)


def test_fetch_peaks_rejects_when_cleaning_leaves_too_few() -> None:
    """The >=10 check applies to the *cleaned* series, not the raw fetch."""
    frame = _peaks_df([100.0] * 12)
    frame["peak_cd"] = ["8"] * 4 + ["6"] * 8
    _FakeNWIS.response = frame
    with pytest.raises(RuntimeError, match="Failed to fetch peaks for 99999999") as excinfo:
        fpf._fetch_peaks("99999999")
    assert "only 8 annual peaks" in str(excinfo.value.__cause__)


# _peak_codes / _clean_peaks (screening out invalid peak values)


def test_peak_codes_normalizes_mixed_types() -> None:
    """The messy ``peak_cd`` column (str/float, multi-code) is normalized."""
    assert fpf._peak_codes(6.0) == {"6"}
    assert fpf._peak_codes("2,O") == {"2", "O"}
    assert fpf._peak_codes(" 5 ") == {"5"}
    assert fpf._peak_codes("R") == {"R"}
    assert fpf._peak_codes(None) == set()
    assert fpf._peak_codes(np.nan) == set()
    assert fpf._peak_codes("") == set()


def test_clean_peaks_drops_censored_and_dam_failure_codes() -> None:
    """Codes 3/4/8 (dam failure / censored) are dropped, even in combos."""
    frame = _qualifier_frame(
        values=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
        codes=["1", "2", "5", "6", "7", "9", "R", "O", "3", "4", "8", "2,6", "2,8", "5,3"],
    )
    series, counts = fpf._clean_peaks(frame)
    assert list(series) == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 21.0]
    assert counts["rows_removed_qualifier"] == 5  # "3", "4", "8", "2,8", "5,3"
    assert counts["rows_kept"] == 9
    assert counts["qualifiers_seen"]["3"] == 2  # "3" and inside "5,3"
    assert counts["qualifiers_seen"]["8"] == 2  # "8" and inside "2,8"


def test_clean_peaks_coerces_strings_and_drops_non_positive_values() -> None:
    """String numerics are coerced; non-positive/sentinel values are dropped."""
    frame = _qualifier_frame(
        values=["150", "1.0e3", 200.0, 0.0, -999.0, -50.0, None, 250.0],
        codes=["6", "6", "6", "6", "6", "6", "6", "6"],
    )
    series, counts = fpf._clean_peaks(frame)
    assert list(series) == [150.0, 1000.0, 200.0, 250.0]
    assert counts["rows_removed_bad_value"] == 4
    assert counts["rows_kept"] == 4


def test_clean_peaks_tracks_qualifiers_and_counts() -> None:
    """Full provenance dict reports downloads, removals, and codes seen."""
    frame = _qualifier_frame(
        values=[100.0, 200.0, 300.0, 400.0],
        codes=["6", "5", "6", "R"],
    )
    _, counts = fpf._clean_peaks(frame)
    assert counts == {
        "rows_downloaded": 4,
        "rows_removed_qualifier": 0,
        "rows_removed_bad_value": 0,
        "rows_kept": 4,
        "qualifiers_seen": {"5": 1, "6": 2, "R": 1},
    }


def test_clean_peaks_tolerates_missing_peak_cd_column() -> None:
    """A response without ``peak_cd`` is treated as fully usable."""
    series, counts = fpf._clean_peaks(pd.DataFrame({"peak_va": [10.0, 20.0, None]}))
    assert list(series) == [10.0, 20.0]
    assert counts["rows_removed_qualifier"] == 0
    assert counts["qualifiers_seen"] == {}
    assert counts["rows_kept"] == 2


# _prepare_annual_series (one observation per USGS water year)


def test_prepare_annual_series_sorts_and_keeps_max_per_water_year() -> None:
    """Unordered rows sort; duplicates collapse to the annual maximum."""
    peaks = pd.Series(
        [150.0, 100.0, 300.0, 200.0, 250.0],
        index=pd.to_datetime(
            [
                "2004-06-01",  # WY2004
                "2003-05-01",  # WY2003
                "2003-07-15",  # WY2003 duplicate — keep the 300.0 max
                "2002-11-20",  # Nov peak belongs to the *next* water year
                "2001-10-01",  # Oct 1 is the first day of WY2002
            ]
        ),
    )
    annual, counts = fpf._prepare_annual_series(peaks)
    assert list(annual) == [250.0, 300.0, 150.0]
    assert [d.strftime("%Y-%m-%d") for d in annual.index] == [
        "2001-10-01",
        "2003-07-15",
        "2004-06-01",
    ]
    assert counts["rows_removed_duplicate"] == 2
    assert counts["water_years"] == 3


def test_prepare_annual_series_drops_undated_rows() -> None:
    """Rows without a usable date are excluded rather than crashing."""
    peaks = pd.Series(
        [10.0, 20.0, 30.0, 40.0],
        index=pd.to_datetime(["2019-04-01", None, "2020-05-01", None]),
    )
    annual, counts = fpf._prepare_annual_series(peaks)
    assert list(annual) == [10.0, 30.0]
    assert counts["rows_removed_missing_date"] == 2
    assert counts["rows_removed_duplicate"] == 0
    assert counts["water_years"] == 2


def test_fetch_peaks_attaches_clean_counts_and_screens_end_to_end() -> None:
    """A fetch returns the annual-max series plus full provenance in ``attrs``."""
    days = pd.to_datetime(
        ["2020-06-01", "2020-05-01", "2019-08-01", "2012-05-01"] + [f"{y}-05-01" for y in range(2000, 2011)]
    )
    values = [300.0, 50.0, 42.0, -999.0] + [float(60 + i) for i in range(11)]
    codes = ["6"] * 3 + ["8"] + ["R"] * 11
    _FakeNWIS.response = pd.DataFrame({"peak_va": values, "peak_cd": codes, "peak_dt": days}).set_index("peak_dt")

    series = fpf._fetch_peaks("99999999")
    counts = series.attrs["clean_counts"]
    assert counts["rows_downloaded"] == 15
    assert counts["rows_removed_qualifier"] == 1  # the -999 row carries code 8
    assert counts["rows_removed_bad_value"] == 1  # ...and is non-positive
    assert counts["rows_kept"] == 14
    assert counts["rows_removed_duplicate"] == 1  # the 2020-05-01 row
    assert counts["water_years"] == 13
    assert counts["qualifiers_seen"] == {"6": 3, "8": 1, "R": 11}

    assert len(series) == 13
    assert series.index[0].year == 2000  # sorted chronologically
    assert series.loc["2020-06-01"] == 300.0 * FT3S_TO_M3S  # WY2020 max wins
    assert series.index.is_monotonic_increasing


# _fit_gev_mle


def test_fit_gev_mle_stable_shape_reports_stable() -> None:
    """A moderate-shape sample yields a stable MLE with all 6 quantiles."""
    rng = np.random.default_rng(0)
    sample = rng.lognormal(2.0, 0.3, 200)
    quantiles, stable = fpf._fit_gev_mle(sample)
    assert stable is True
    assert sorted(quantiles) == RETURN_PERIODS
    assert all(np.isfinite(v) for v in quantiles.values())


def test_fit_gev_mle_unstable_warns_and_returns() -> None:
    """Unstable MLE still returns the raw fit, plus a RuntimeWarning."""
    rng = np.random.default_rng(0)

    heavy = np.concatenate([rng.lognormal(2.0, 0.5, 90), [500.0, 700.0, 1200.0]])
    with pytest.warns(RuntimeWarning, match="unstable"):
        quantiles, stable = fpf._fit_gev_mle(heavy)
    assert stable is False
    assert sorted(quantiles) == RETURN_PERIODS
    assert all(np.isfinite(v) for v in quantiles.values())


def test_fit_gev_mle_quantiles_increase_with_return_period() -> None:
    """Higher return periods give higher (or equal) quantiles."""
    rng = np.random.default_rng(1)
    quantiles, _ = fpf._fit_gev_mle(rng.lognormal(2.0, 0.3, 200))
    quantile_values = [quantiles[rp] for rp in RETURN_PERIODS]
    assert quantile_values == sorted(quantile_values)


# _fit_gev_lmoments


@pytest.mark.skipif(not HAS_LMOMENTS3, reason="lmoments3 not installed")
def test_fit_gev_lmoments_monotonic_and_keys() -> None:
    """L-moments fit yields all return periods with a monotone curve."""
    rng = np.random.default_rng(0)
    quantiles = fpf._fit_gev_lmoments(rng.lognormal(2.0, 0.3, 200))
    assert sorted(quantiles) == RETURN_PERIODS
    assert all(np.isfinite(v) for v in quantiles.values())
    quantile_values = [quantiles[rp] for rp in RETURN_PERIODS]
    assert quantile_values == sorted(quantile_values)
    assert all(v > 0 for v in quantile_values)


@pytest.mark.skipif(not HAS_LMOMENTS3, reason="lmoments3 not installed")
def test_fit_gev_lmoments_isf_uses_survival_probability() -> None:
    """Quantiles are computed with ``isf(1/rp)``, so Q100 > Q50 > Q2."""
    rng = np.random.default_rng(3)
    quantiles = fpf._fit_gev_lmoments(rng.lognormal(3.0, 0.4, 150))
    assert quantiles[100] > quantiles[50] > quantiles[2]


# _fit_lp3


def test_fit_lp3_matches_manual_log10_pearson3() -> None:
    """LP3 reproduces a direct log10 + pearson3 computation on positive data."""
    from scipy.stats import pearson3

    rng = np.random.default_rng(7)
    samples = np.abs(rng.standard_normal(120)) + 1.0  # strictly positive
    log_samples = np.log10(samples)
    mean = float(np.mean(log_samples))
    std = float(np.std(log_samples, ddof=1))
    skew = float(np.mean(((log_samples - mean) / std) ** 3))
    expected = {
        return_period: round(float(10 ** pearson3.ppf(1 - 1.0 / return_period, skew, loc=mean, scale=std)), 2)
        for return_period in RETURN_PERIODS
    }
    assert fpf._fit_lp3(samples) == expected


def test_fit_lp3_handles_constant_input_gracefully() -> None:
    """Zero variance yields NaN quantiles rather than a crash."""
    with pytest.warns(RuntimeWarning):
        quantiles = fpf._fit_lp3(np.full(50, 10.0))
    assert all(np.isnan(v) for v in quantiles.values())


# _write_artifacts (atomic publish — no partial cache on failure)


def _tiny_frames() -> dict[str, pd.DataFrame]:
    """Two one-row cache frames for exercising the atomic writer."""
    return {
        "11111111": pd.DataFrame({"date": ["2020-01-01"], "peak_va": [1.0]}),
        "22222222": pd.DataFrame({"date": ["2021-01-01"], "peak_va": [2.0]}),
    }


def test_write_artifacts_commits_all_outputs(tmp_path, monkeypatch) -> None:
    """A successful run publishes CSVs + JSON with no temp leftovers."""
    monkeypatch.setattr(fpf, "PEAKS_DIR", tmp_path)
    monkeypatch.setattr(fpf, "FFA_REFERENCE_FILE", tmp_path / "ffa_reference.json")

    fpf._write_artifacts(_tiny_frames(), {"meta": "fresh"})

    assert (tmp_path / "11111111_peaks.csv").read_text().startswith("date,peak_va")
    assert (tmp_path / "22222222_peaks.csv").read_text().startswith("date,peak_va")
    assert json.loads((tmp_path / "ffa_reference.json").read_text())["meta"] == "fresh"
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_artifacts_aborted_commit_leaves_prior_outputs(tmp_path, monkeypatch) -> None:
    """A mid-commit failure leaves prior artifacts byte-identical, no temps."""
    monkeypatch.setattr(fpf, "PEAKS_DIR", tmp_path)
    monkeypatch.setattr(fpf, "FFA_REFERENCE_FILE", tmp_path / "ffa_reference.json")

    (tmp_path / "11111111_peaks.csv").write_text("OLD_1\n")
    (tmp_path / "22222222_peaks.csv").write_text("OLD_2\n")
    (tmp_path / "ffa_reference.json").write_text("OLD_JSON\n")

    def boom_dump(obj, fp, **kwargs):
        raise RuntimeError("mid-commit failure")

    monkeypatch.setattr(fpf.json, "dump", boom_dump)
    with pytest.raises(RuntimeError, match="mid-commit"):
        fpf._write_artifacts(_tiny_frames(), {"meta": "fresh"})

    assert (tmp_path / "11111111_peaks.csv").read_text() == "OLD_1\n"
    assert (tmp_path / "22222222_peaks.csv").read_text() == "OLD_2\n"
    assert (tmp_path / "ffa_reference.json").read_text() == "OLD_JSON\n"
    assert list(tmp_path.glob("*.tmp")) == []


# main() — end-to-end smoke test (fit functions and fetch are mocked)


def _gbm_sample(years: list[int]) -> pd.Series:
    """A deterministic pseudo-random positive series of ``peak_va`` (m³/s)."""
    rng = np.random.default_rng(42)
    log_values = np.cumsum(0.2 * rng.standard_normal(len(years))) + 2.0
    date_index = pd.to_datetime([f"{year}-05-01" for year in years])
    return pd.Series(np.exp(log_values), index=date_index)


def _clean_counts_fixture() -> dict:
    """Deterministic provenance mirroring what ``_fetch_peaks`` attaches."""
    return {
        "rows_downloaded": 30,
        "rows_removed_qualifier": 2,
        "rows_removed_bad_value": 1,
        "rows_kept": 27,
        "qualifiers_seen": {"5": 2, "6": 28},
        "rows_removed_missing_date": 0,
        "rows_removed_duplicate": 0,
        "water_years": 30,
    }


def _quantile_dict(scale: float = 1.0) -> dict[int, float]:
    """A deterministic quantile dict keyed by return period."""
    return {return_period: round(100.0 * scale * return_period, 2) for return_period in RETURN_PERIODS}


def test_main_writes_expected_outputs(tmp_path, monkeypatch) -> None:
    """``main()`` produces the reference JSON + cached peak CSVs."""
    catchments = [
        {"gauge_id": "11111111", "name": "Catchment One"},
        {"gauge_id": "22222222", "name": "Catchment Two"},
    ]
    json.dump(catchments, open(tmp_path / "daily_catchments.json", "w"))
    years = list(range(1990, 2020))

    monkeypatch.setattr(fpf, "DAILY_CATCHMENTS_FILE", tmp_path / "daily_catchments.json")
    monkeypatch.setattr(fpf, "PEAKS_DIR", tmp_path / "peaks")
    monkeypatch.setattr(fpf, "FFA_REFERENCE_FILE", tmp_path / "ffa_reference.json")

    def fake_fetch(gauge_id: str) -> pd.Series:
        sample = _gbm_sample(years)
        sample.attrs["clean_counts"] = _clean_counts_fixture()
        return sample

    monkeypatch.setattr(fpf, "_fetch_peaks", fake_fetch)
    monkeypatch.setattr(fpf, "_fit_gev_mle", lambda arr: (_quantile_dict(), True))
    monkeypatch.setattr(fpf, "_fit_gev_lmoments", lambda arr: _quantile_dict(1.5))
    monkeypatch.setattr(fpf, "_fit_lp3", lambda arr: _quantile_dict(0.5))

    fpf.main()

    # Reference JSON is written, well-formed and complete.
    with open(tmp_path / "ffa_reference.json") as f:
        reference = json.load(f)
    assert set(reference) == {"metadata", "catchments"}
    assert set(reference["metadata"]) == {
        "generated",
        "source",
        "return_periods",
        "note",
    }
    assert reference["metadata"]["return_periods"] == RETURN_PERIODS
    assert set(reference["catchments"]) == {"11111111", "22222222"}

    for _, catchment in reference["catchments"].items():
        assert set(catchment) == {
            "name",
            "num_records",
            "start_year",
            "end_year",
            "mle_stable",
            "scipy_gev",
            "scipy_gev_lmoments",
            "scipy_lp3",
            "data_quality",
        }
        assert catchment["data_quality"] == _clean_counts_fixture()
        assert catchment["num_records"] == len(years)
        assert catchment["start_year"] == 1990
        assert catchment["end_year"] == 2019
        assert catchment["mle_stable"] is True and isinstance(catchment["mle_stable"], bool)
        for dist_key in ("scipy_gev", "scipy_gev_lmoments", "scipy_lp3"):
            quantiles = catchment[dist_key]
            assert sorted(map(int, quantiles)) == RETURN_PERIODS
            assert all(isinstance(v, (int, float)) for v in quantiles.values())
            assert all(np.isfinite(v) for v in quantiles.values())

    # Peak CSVs are cached, one per gauge, with no temp files left behind.
    assert (tmp_path / "peaks" / "11111111_peaks.csv").is_file()
    assert (tmp_path / "peaks" / "22222222_peaks.csv").is_file()
    assert list((tmp_path / "peaks").glob("*.tmp")) == []
    assert not (tmp_path / "ffa_reference.json.tmp").exists()


def test_main_writes_nothing_when_a_later_gauge_fails(tmp_path, monkeypatch) -> None:
    """A fit failure mid-loop must not publish any partial cache."""
    catchments = [
        {"gauge_id": "11111111", "name": "Catchment One"},
        {"gauge_id": "22222222", "name": "Catchment Two"},
    ]
    json.dump(catchments, open(tmp_path / "daily_catchments.json", "w"))
    years = list(range(1990, 2020))

    monkeypatch.setattr(fpf, "DAILY_CATCHMENTS_FILE", tmp_path / "daily_catchments.json")
    monkeypatch.setattr(fpf, "PEAKS_DIR", tmp_path / "peaks")
    monkeypatch.setattr(fpf, "FFA_REFERENCE_FILE", tmp_path / "ffa_reference.json")

    def fake_fetch(gauge_id: str) -> pd.Series:
        sample = _gbm_sample(years)
        sample.attrs["clean_counts"] = _clean_counts_fixture()
        return sample

    fit_calls = {"n": 0}

    def failing_fit(arr) -> tuple[dict[int, float], bool]:
        fit_calls["n"] += 1
        if fit_calls["n"] >= 2:  # second gauge's fit blows up
            raise RuntimeError("fit boom")
        return (_quantile_dict(), True)

    monkeypatch.setattr(fpf, "_fetch_peaks", fake_fetch)
    monkeypatch.setattr(fpf, "_fit_gev_mle", failing_fit)
    monkeypatch.setattr(fpf, "_fit_gev_lmoments", lambda arr: _quantile_dict(1.5))
    monkeypatch.setattr(fpf, "_fit_lp3", lambda arr: _quantile_dict(0.5))

    with pytest.raises(RuntimeError, match="fit boom"):
        fpf.main()

    assert not (tmp_path / "peaks").exists(), "no gauge CSV may be written on failure"
    assert not (tmp_path / "ffa_reference.json").exists()
