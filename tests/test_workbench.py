"""The workbench: every analysis the dashboard offers, as shareable functions.

The point of these tests is the contract the module promises: results are
strictly JSON (no NaN, no infinity, no numpy, no Timestamps), the column rules
match what each dashboard page used to do, and the analyses still produce the
numbers they produced behind Streamlit.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aquascope import workbench as wb


@pytest.fixture(scope="module")
def discharge_frame() -> pd.DataFrame:
    """Twelve years of daily flow with a seasonal cycle and storms."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2012-01-01", periods=365 * 12, freq="D")
    seasonal = 5 + 3 * np.sin(np.arange(len(idx)) / 365.25 * 2 * np.pi)
    q = np.clip(seasonal + rng.gamma(1.2, 1.4, len(idx)), 0.05, None)
    return pd.DataFrame({"sample_datetime": idx, "station_id": "DEMO-1", "discharge_cms": q})


@pytest.fixture(scope="module")
def quality_frame() -> pd.DataFrame:
    """A long-format water-quality table, the shape the EDA and WHO screens expect."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    parts = []
    for name, lo, hi in [("ph", 6.2, 8.9), ("dissolved_oxygen", 3.0, 9.0), ("nitrate", 1.0, 60.0)]:
        parts.append(pd.DataFrame({
            "sample_datetime": dates, "station_id": "S1", "parameter": name,
            "value": rng.uniform(lo, hi, len(dates)), "latitude": 25.1, "longitude": 121.4,
        }))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def level_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2015-01-01", periods=120, freq="MS")
    return pd.DataFrame({"date": idx, "water_level_m": 10 + np.cumsum(rng.normal(0, 0.3, len(idx)))})


def _strict_json(result: dict) -> str:
    """json.dumps with allow_nan=False: NaN or infinity anywhere raises."""
    return json.dumps({k: v for k, v in result.items() if k != "frame"}, allow_nan=False)


# ── the shape of a table ────────────────────────────────────────────────────


def test_profile_finds_the_columns(discharge_frame: pd.DataFrame, quality_frame: pd.DataFrame) -> None:
    p = wb.profile(discharge_frame)
    assert p.datetime_col == "sample_datetime"
    assert p.discharge_col == "discharge_cms"
    assert p.station_col == "station_id"
    assert p.has_time and not p.has_params and not p.has_geo
    assert p.span_years == pytest.approx(12, abs=0.1)

    q = wb.profile(quality_frame)
    assert q.param_col == "parameter" and q.value_col == "value"
    assert q.has_params and q.has_geo
    assert "ph" in q.parameters
    # latitude/longitude must not be mistaken for the value column
    assert q.value_col not in {"latitude", "longitude"}


def test_profile_of_an_empty_table_is_harmless() -> None:
    p = wb.profile(pd.DataFrame())
    assert p.n_records == 0 and p.numeric_cols == [] and p.datetime_col is None


def test_pick_column_keeps_the_three_dashboard_rules() -> None:
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=3),
        "value": [1.0, 2.0, 3.0],
        "gw_level": [10.0, 11.0, 12.0],
        "discharge": [4.0, 5.0, 6.0],
    })
    assert wb.pick_column(df, prefer="discharge") == "discharge"   # Hydrology Lab
    assert wb.pick_column(df, prefer="level") == "gw_level"        # Groundwater page
    assert wb.pick_column(df, prefer="value") == "value"           # everything else
    assert wb.pick_column(df, "gw_level", prefer="discharge") == "gw_level"   # explicit wins
    with pytest.raises(ValueError):
        wb.pick_column(df, "nope")


def test_datetime_indexed_uses_the_date_column(discharge_frame: pd.DataFrame) -> None:
    s = wb.datetime_indexed(discharge_frame, "discharge_cms")
    assert isinstance(s.index, pd.DatetimeIndex)
    assert len(s) == len(discharge_frame)


def test_datetime_indexed_survives_a_table_with_no_dates() -> None:
    df = pd.DataFrame({"q": [1.0, 2.0, 3.0]})
    s = wb.datetime_indexed(df, "q")
    assert list(s) == [1.0, 2.0, 3.0]


# ── serialisation contract ──────────────────────────────────────────────────


def test_jsonable_turns_nan_and_infinity_into_null() -> None:
    assert wb.jsonable(float("nan")) is None
    assert wb.jsonable(float("inf")) is None
    assert wb.jsonable(np.float64(2.5)) == 2.5
    assert wb.jsonable(np.int64(3)) == 3
    assert wb.jsonable(pd.Timestamp("2020-01-01")) == "2020-01-01T00:00:00"
    assert wb.jsonable(np.array([1, 2])) == [1, 2]


@pytest.mark.parametrize("analysis,kwargs", [
    ("flow_duration", {}),
    ("baseflow", {"method": "lyne_hollick"}),
    ("baseflow", {"method": "eckhardt", "alpha": 0.98, "bfi_max": 0.75}),
    ("baseflow", {"method": "ukih", "block_size": 5}),
    ("recession", {"min_length": 5}),
    ("signatures", {}),
    ("flood_frequency", {}),
    ("return_periods", {"n_bootstrap": 40}),
])
def test_discharge_analyses_return_strict_json(discharge_frame: pd.DataFrame, analysis: str, kwargs: dict) -> None:
    result = wb.run(analysis, discharge_frame, **kwargs)
    _strict_json(result)                      # raises on NaN, infinity or a numpy type
    assert result["methods"], f"{analysis} must carry its method and citation"
    assert result["column"] == "discharge_cms"


@pytest.mark.parametrize("analysis", ["eda", "quality", "who_screen", "insights"])
def test_table_analyses_return_strict_json(quality_frame: pd.DataFrame, analysis: str) -> None:
    _strict_json(wb.run(analysis, quality_frame))


def test_signatures_json_survives_an_infinite_ratio() -> None:
    """q5/q95 is infinite when the record has zero flows; that must not break JSON."""
    idx = pd.date_range("2015-01-01", periods=800, freq="D")
    q = np.concatenate([np.zeros(400), np.full(400, 3.0)])
    df = pd.DataFrame({"date": idx, "discharge": q})
    result = wb.signatures(df)
    _strict_json(result)
    assert result["signatures"]["zero_flow_fraction"] > 0


# ── the analyses themselves ─────────────────────────────────────────────────


def test_flow_duration_percentiles_are_ordered(discharge_frame: pd.DataFrame) -> None:
    res = wb.flow_duration(discharge_frame)
    pct = {float(k): v for k, v in res["percentiles"].items()}
    assert pct[5] > pct[50] > pct[95], "higher exceedance percentiles mean lower flows"
    assert res["n"] == len(discharge_frame)


def test_baseflow_index_is_a_fraction_and_methods_differ(discharge_frame: pd.DataFrame) -> None:
    lh = wb.baseflow(discharge_frame, method="lyne_hollick")
    ec = wb.baseflow(discharge_frame, method="eckhardt")
    for res in (lh, ec):
        assert 0.0 <= res["bfi"] <= 1.0
    assert lh["method"] == "lyne_hollick" and ec["method"] == "eckhardt"
    assert lh["bfi"] != ec["bfi"], "two filters should not agree exactly"


def test_baseflow_rejects_an_unknown_method(discharge_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        wb.baseflow(discharge_frame, method="nope")


def test_flood_frequency_return_levels_increase(discharge_frame: pd.DataFrame) -> None:
    res = wb.flood_frequency(discharge_frame)
    levels = [res["return_periods"][k] for k in sorted(res["return_periods"], key=float)]
    assert levels == sorted(levels), "a longer return period cannot mean a smaller flood"
    assert res["distribution"] == "GEV"
    assert res["n_bootstrap"] == 1000
    assert isinstance(res["n_bootstrap_discarded"], int) and res["n_bootstrap_discarded"] >= 0


def test_return_periods_carry_the_empirical_points(discharge_frame: pd.DataFrame) -> None:
    res = wb.return_periods(discharge_frame, periods=[2, 10, 100], n_bootstrap=40)
    assert res["return_periods"] == [2.0, 10.0, 100.0]
    assert len(res["empirical"]["value"]) == res["n_years"]
    for lo, level, hi in zip(res["lower_bound"], res["return_levels"], res["upper_bound"], strict=True):
        assert lo <= level <= hi


def test_return_periods_refuses_a_record_too_short_to_fit() -> None:
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=400), "discharge": np.arange(400.0)})
    with pytest.raises(ValueError, match="at least three"):
        wb.return_periods(df, n_bootstrap=10)


def test_signatures_need_a_year_of_dated_daily_flow() -> None:
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=100), "discharge": np.arange(100.0)})
    with pytest.raises(ValueError, match="at least one year"):
        wb.signatures(df)


def test_who_screen_flags_the_parameters_outside_the_guideline() -> None:
    df = pd.DataFrame({
        "parameter": ["ph"] * 5 + ["nitrate"] * 5 + ["not_a_parameter"] * 3,
        "value": [6.0, 7.0, 8.0, 9.0, 7.5] + [10.0, 20.0, 60.0, 70.0, 5.0] + [1.0, 2.0, 3.0],
    })
    res = wb.who_screen(df)
    by_name = {r["parameter"]: r for r in res["rows"]}
    assert set(by_name) == {"ph", "nitrate"}, "unknown parameters are skipped, not guessed at"
    assert by_name["ph"]["n_exceed"] == 2        # 6.0 below 6.5 and 9.0 above 8.5
    assert by_name["nitrate"]["n_exceed"] == 2   # 60 and 70 above 50
    assert by_name["ph"]["status"] == "Alert"    # 40 % is over the 10 % alert line


def test_who_screen_handles_a_one_sided_guideline() -> None:
    """Dissolved oxygen has a floor and no ceiling; the rule must not read as 'at most inf'."""
    df = pd.DataFrame({"parameter": ["dissolved_oxygen"] * 4, "value": [2.0, 4.0, 6.0, 8.0]})
    row = wb.who_screen(df)["rows"][0]
    assert row["n_exceed"] == 2 and "at least" in row["rule"]


def test_who_screen_says_so_when_there_is_nothing_to_screen() -> None:
    res = wb.who_screen(pd.DataFrame({"discharge": [1.0, 2.0]}))
    assert res["rows"] == [] and "no parameter" in res["note"].lower()


def test_insights_scores_quality_and_suggests_next_steps(quality_frame: pd.DataFrame) -> None:
    res = wb.insights(quality_frame)
    assert 0 <= res["quality_score"] <= 100
    assert len(res["suggestions"]) <= 4
    assert res["who_checked"] >= 2


def test_insights_penalises_missing_values_and_duplicates() -> None:
    clean = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100), "value": np.arange(100.0)})
    messy = pd.concat([clean, clean.head(20)], ignore_index=True)
    # Blank rows the copies do not cover, or the duplicates stop being duplicates.
    messy.loc[40:70, "value"] = np.nan
    assert wb.insights(messy)["quality_score"] < wb.insights(clean)["quality_score"]
    assert wb.insights(messy)["n_duplicates"] > 0


def test_preprocess_reports_what_it_did(quality_frame: pd.DataFrame) -> None:
    doubled = pd.concat([quality_frame, quality_frame], ignore_index=True)
    res = wb.preprocess(doubled, steps=["remove_duplicates"])
    assert res["n_before"] == 2 * len(quality_frame)
    assert res["n_after"] < res["n_before"]
    assert isinstance(res["frame"], pd.DataFrame)
    _strict_json(res)


def test_preprocess_names_a_step_it_does_not_know() -> None:
    df = pd.DataFrame({"value": [1.0, 2.0]})
    assert wb.preprocess(df, steps=["remove_duplicates", "teleport"])["unknown_steps"] == ["teleport"]


# ── groundwater and aquifer ─────────────────────────────────────────────────


def test_sgi_drought_finds_events(level_frame: pd.DataFrame) -> None:
    res = wb.sgi_drought(level_frame, threshold=-0.5)
    _strict_json(res)
    assert res["column"] == "water_level_m"
    assert res["sgi"]["n"] > 0
    for event in res["events"]:
        assert event["duration"] >= 1 and event["peak"] <= -0.5


def test_recharge_scales_with_specific_yield(level_frame: pd.DataFrame) -> None:
    low = wb.recharge(level_frame, specific_yield=0.05)
    high = wb.recharge(level_frame, specific_yield=0.20)
    assert high["value_mm_per_year"] == pytest.approx(4 * low["value_mm_per_year"], rel=1e-6)


def test_aquifer_drawdown_grows_with_pumping_and_shrinks_with_distance() -> None:
    well = {"transmissivity": 500, "storativity": 1e-3, "pumping_rate": 1000, "distance": 100, "time_days": 10}
    base = wb.aquifer_drawdown(**well)
    more_pumping = wb.aquifer_drawdown(**{**well, "pumping_rate": 2000})
    further = wb.aquifer_drawdown(**{**well, "distance": 500})
    assert more_pumping["drawdown_m"] == pytest.approx(2 * base["drawdown_m"], rel=1e-9)
    assert further["drawdown_m"] < base["drawdown_m"]
    _strict_json(base)


# ── agriculture ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def weather() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-03-01", periods=150, freq="D")
    return pd.DataFrame({
        "t_min": 15 + rng.normal(0, 2, len(idx)), "t_max": 27 + rng.normal(0, 3, len(idx)),
        "rh_min": 45 + rng.normal(0, 5, len(idx)), "rh_max": 85 + rng.normal(0, 5, len(idx)),
        "wind_speed": 2 + rng.normal(0, 0.4, len(idx)), "solar_radiation": 20 + rng.normal(0, 3, len(idx)),
        "precipitation": rng.gamma(0.6, 4, len(idx)),
    }, index=idx)


def test_reference_et_is_a_plausible_daily_rate(weather: pd.DataFrame) -> None:
    res = wb.reference_et(weather, latitude=25.0, elevation=10.0)
    _strict_json(res)
    assert 1.0 < res["mean_mm_per_day"] < 12.0
    assert res["eto"]["n"] == len(weather)


def test_irrigation_schedule_balances_rain_and_demand(weather: pd.DataFrame) -> None:
    res = wb.irrigation(weather, latitude=25.0, elevation=10.0, crop="maize", planting_date="2025-03-01")
    _strict_json(res)
    totals = res["totals_mm"]
    assert totals["gross_irrigation"] >= totals["net_irrigation"], "efficiency losses only add water"
    assert res["season_days"] > 0
    assert res["schedule"]["columns"][:3] == ["date", "stage", "kc"]


def test_dual_crop_coefficient_is_available(weather: pd.DataFrame) -> None:
    res = wb.irrigation(weather, latitude=25.0, elevation=10.0, crop="maize",
                        planting_date="2025-03-01", method="dual")
    assert "kcb" in res["schedule"]["columns"]


# ── the registry ────────────────────────────────────────────────────────────


def test_every_tool_is_callable_and_described() -> None:
    for name, spec in wb.TOOLS.items():
        assert callable(spec["func"]), name
        assert spec["needs"] in {"frame", "weather", "none"}, name
        assert spec["summary"].endswith("."), f"{name}: the summary reads as a sentence"


def test_run_rejects_an_unknown_analysis(discharge_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown analysis"):
        wb.run("nope", discharge_frame)


def test_run_says_when_an_analysis_needs_data() -> None:
    with pytest.raises(ValueError, match="needs a table"):
        wb.run("signatures", None)


def test_the_dashboard_uses_the_workbench_rather_than_its_own_copy() -> None:
    """_state used to carry a second implementation of the column rules."""
    pytest.importorskip("streamlit")
    from aquascope.dashboard import _state

    assert _state.profile is wb.profile
    assert _state.datetime_indexed is wb.datetime_indexed
    assert _state.DataProfile is wb.DataProfile


# ── loading data that is already in memory (the browser has no filesystem) ──


def test_ingest_text_reads_a_pasted_csv() -> None:
    from aquascope.ingest import ingest_text

    idx = pd.date_range("2015-01-01", periods=1200, freq="D")
    rng = np.random.default_rng(2)
    csv = pd.DataFrame({"date": idx, "flow_m3s": np.abs(rng.normal(6, 2, len(idx)))}).to_csv(index=False)

    res = ingest_text(csv, "gauge.csv")
    assert res["mapping"]["datetime_column"] == "date"
    assert res["mapping"]["value_column"] == "flow_m3s"
    assert res["mapping"]["variable"] == "discharge"
    assert res["qa"]["n_values"] == len(idx)
    assert res["analysis"]["n"] == len(idx)


def test_ingest_text_accepts_bytes_too() -> None:
    from aquascope.ingest import ingest_text

    csv = b"date,level_m\n2020-01-01,1.5\n2020-01-02,1.7\n2020-01-03,1.6\n"
    res = ingest_text(csv, "levels.csv")
    assert res["qa"]["n_values"] == 3


# ── drought indices ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def climate_frame() -> pd.DataFrame:
    """Forty years of monthly rainfall and temperature with two degrees of warming."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("1985-01-01", periods=40 * 12, freq="MS")
    phase = np.arange(len(idx)) % 12
    return pd.DataFrame({
        "date": idx,
        "rain_mm": rng.gamma(2.0, (80 + 60 * np.sin(2 * np.pi * phase / 12)) / 2.0),
        "tmean_c": 10 + 8 * np.sin(2 * np.pi * (phase - 3) / 12) + np.linspace(0, 2, len(idx)),
    })


def test_spei_from_temperature_reports_both_indices_and_their_divergence(climate_frame: pd.DataFrame) -> None:
    res = wb.run("spei", climate_frame, temperature_column="tmean_c", latitude=51.4)
    _strict_json(res)
    assert res["column"] == "rain_mm" and res["pet_method"] == "thornthwaite" and res["timescales"] == [1, 3, 12]
    assert res["headline_timescale"] == 3 and res["headline_index"] == "spei" and res["years"] == 40.0
    row = next(r for r in res["indices"] if r["timescale"] == 12)
    assert row["spi"]["class"] in ("normal", "moderately_dry", "severely_dry", "extremely_dry", "moderately_wet",
                                   "very_wet", "extremely_wet")
    assert row["divergence"]["mean_last_10y"] < 0, "warming: SPEI runs drier than SPI over the last decade"
    assert row["divergence"]["correlation"] > 0.9 and set(row["series"]) == {"index", "step", "spi", "spei"}
    assert res["current"]["spi"]["12"] is not None and res["current"]["spei"]["12"] is not None
    assert [m["name"] for m in res["methods"]][1:] == ["Standardized Precipitation-Evapotranspiration Index",
                                                        "Thornthwaite potential evapotranspiration"]


def test_spei_takes_a_ready_pet_column_or_says_what_it_needs(climate_frame: pd.DataFrame) -> None:
    df = climate_frame.assign(pet_mm=60.0)
    res = wb.spei(df, "rain_mm", pet_column="pet_mm", timescales=[3])
    assert res["pet_method"] == "given" and [r["timescale"] for r in res["indices"]] == [3]
    with pytest.raises(ValueError, match="pet_column"):
        wb.spei(climate_frame)
    with pytest.raises(ValueError, match="latitude"):
        wb.spei(climate_frame, temperature_column="tmean_c")
    with pytest.raises(ValueError, match="No column"):
        wb.spei(climate_frame, pet_column="nope")


def test_standardized_indices_without_pet_gives_spi_only(climate_frame: pd.DataFrame) -> None:
    p = climate_frame.set_index("date")["rain_mm"]
    res = wb.standardized_indices(p, timescales=(3,))
    assert res["headline_index"] == "spi" and res["indices"][0]["spei"] is None
    assert res["indices"][0]["divergence"] is None and res["current"]["spei"] == {}
    with pytest.raises(ValueError, match="two years"):
        wb.standardized_indices(p.iloc[:12])
    with pytest.raises(ValueError, match="positive"):
        wb.standardized_indices(p, timescales=(0,))
