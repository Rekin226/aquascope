"""Tests for the CCME Water Quality Index implementation."""
import pandas as pd
import pytest

from aquascope.analysis.water_quality_index import ccme_wqi, wqi_category


@pytest.mark.parametrize(
    "score, expected",
    [
        (0, "Poor"),
        (44.9, "Poor"),
        (45, "Marginal"),
        (64.9, "Marginal"),
        (65, "Fair"),
        (79.9, "Fair"),
        (80, "Good"),
        (94.9, "Good"),
        (95, "Excellent"),
        (100, "Excellent"),
    ],
)
def test_wqi_categories(score, expected):
    assert wqi_category(score) == expected


@pytest.mark.parametrize(
    "score",
    [-1, 101, float("nan"), float("inf"), float("-inf")],
)
def test_wqi_category_rejects_invalid_scores(score):
    with pytest.raises(ValueError):
        wqi_category(score)


def test_ccme_wqi_all_measurements_meet_guidelines():
    measurements = pd.DataFrame(
        {
            "parameter": ["DO", "pH", "TP", "Pb"] * 4,
            "value": [8.0, 7.5, 0.02, 0.001] * 4,
        }
    )

    guidelines = {
        "DO": {"min": 5.0},
        "pH": {"min": 6.5, "max": 9.0},
        "TP": {"max": 0.05},
        "Pb": {"max": 0.004},
    }

    result = ccme_wqi(measurements, guidelines)

    assert result.score == 100.0
    assert result.category == "Excellent"
    assert result.f1 == 0.0
    assert result.f2 == 0.0
    assert result.f3 == 0.0


def test_ccme_wqi_one_measurement_below_minimum():
    measurements = pd.DataFrame(
        {
            "parameter": ["DO", "pH", "TP", "Pb"] * 4,
            "value": [8.0, 7.5, 0.02, 0.001] * 4,
        }
    )

    guidelines = {
        "DO": {"min": 5.0},
        "pH": {"min": 6.5, "max": 9.0},
        "TP": {"max": 0.05},
        "Pb": {"max": 0.004},
    }

    measurements.loc[0, "value"] = 2.5
    result = ccme_wqi(measurements, guidelines)

    assert result.f1 == pytest.approx(25.0)
    assert result.f2 == pytest.approx(6.25)
    assert result.f3 == pytest.approx(5.88235294117647)
    assert result.score == pytest.approx(84.738878, abs=0.000001)
    assert result.category == "Good"

@pytest.mark.parametrize(
    "measurements",
    [
        pd.DataFrame({"value": [8.0]}),
        pd.DataFrame({"parameter": ["DO"]}),
    ],
)
def test_ccme_wqi_rejects_missing_columns(measurements):
    guidelines = {"DO": {"min": 5.0}}

    with pytest.raises(ValueError, match="Missing required columns"):
        ccme_wqi(measurements, guidelines)


def test_ccme_wqi_rejects_empty_table():
    measurements = pd.DataFrame(columns=["parameter", "value"])
    guidelines = {"DO": {"min": 5.0}}

    with pytest.raises(ValueError):
        ccme_wqi(measurements, guidelines)

@pytest.mark.parametrize(
    "limits",
    [
        {},
        {"minimum": 5.0},
        {"min": 5.0, "maximum": 10.0},
        {"min": 9.0, "max": 5.0},
    ],
)
def test_ccme_wqi_rejects_invalid_guideline_structure(limits):
    measurements = pd.DataFrame(
        {
            "parameter": ["DO"] * 4,
            "value": [8.0] * 4,
        }
    )

    with pytest.raises(ValueError, match="guideline"):
        ccme_wqi(measurements, {"DO": limits})

@pytest.mark.parametrize(
    "limits",
    [
        {"max": 0.0},
        {"max": -1.0},
        {"min": -1.0},
        {"max": float("nan")},
        {"min": float("nan")},
        {"max": float("inf")},
        {"min": float("inf")},
        {"max": "5.0"},
        {"min": None},
        {"max": True},
    ],
)
def test_ccme_wqi_rejects_invalid_guideline_values(limits):
    measurements = pd.DataFrame(
        {
            "parameter": ["DO"] * 4,
            "value": [8.0] * 4,
        }
    )

    with pytest.raises(ValueError, match="guideline"):
        ccme_wqi(measurements, {"DO": limits})

def test_ccme_wqi_published_devon_example():
    """Reproduce the published North Saskatchewan River, Devon, 1997 example."""
    # Source: CCME WQI User's Manual, 2017 update, Table 1, pp. 6–7.
    # https://ccme.ca/en/res/wqimanualen.pdf
    # Below-detection readings use their detection limits, as the manual directs.
    # None represents an absent measurement.
    wide = pd.DataFrame(
        {
            "DO": [11.4, 11.0, 11.5, 12.5, 10.4, 8.9, 8.5, 7.5, 9.2, 11.0, 12.1, 13.3],
            "pH": [8.0, 7.9, 7.9, 7.9, 8.1, 8.2, 8.3, 8.2, 8.2, 8.1, 8.0, 8.0],
            "TP": [0.006, 0.005, 0.006, 0.058, 0.042, 0.108, 0.017, 0.008, 0.006, 0.008, 0.006, 0.004],
            "TN": [0.160, 0.170, 0.132, 0.428, 0.250, 0.707, 0.153, 0.153, 0.130, 0.093, 0.296, 0.054],
            "FC": [4, 4, 4, 4, 4, 26, 9, 8, 12, 12, 8, 4],
            "As": [0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0006, 0.0002, 0.0002, 0.0003, 0.0002, 0.0002, 0.0002],
            "Pb": [0.0004, 0.0094, 0.0003, 0.0008, 0.0008, 0.0013, 0.0004, 0.0003, 0.0018, 0.0011, 0.0051, 0.0003],
            "Hg": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, None, 0.05, 0.05, 0.05, 0.05, 0.05],
            "2,4-D": [0.005, None, None, 0.004, None, None, None, 0.005, None, 0.005, None, None],
            "Lindane": [0.005, None, None, 0.005, None, None, None, 0.005, None, 0.005, None, None],
        }
    )

    guidelines = {
        "DO": {"min": 5.0},
        "pH": {"min": 6.5, "max": 9.0},
        "TP": {"max": 0.05},
        "TN": {"max": 1.0},
        "FC": {"max": 400.0},
        "As": {"max": 0.05},
        "Pb": {"max": 0.004},
        "Hg": {"max": 0.1},
        "2,4-D": {"max": 4.0},
        "Lindane": {"max": 0.01},
    }

    measurements = wide.melt(var_name="parameter", value_name="value")
    result = ccme_wqi(measurements, guidelines)

    assert measurements["value"].count() == 103
    assert result.f1 == pytest.approx(20.0)
    assert result.f2 == pytest.approx(100.0 * 4 / 103)
    assert result.f3 == pytest.approx(2.7797442069)
    assert round(result.score) == 88
    assert result.category == "Good"

@pytest.mark.parametrize(
    "value",
    [float("inf"), float("-inf"), "NaN"],
)
def test_ccme_wqi_rejects_nonfinite_measurements(value):
    measurements = pd.DataFrame(
        {
            "parameter": ["DO"],
            "value": [value],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        ccme_wqi(measurements, {"DO": {"min": 5.0}})


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_ccme_wqi_rejects_nonpositive_value_below_minimum(value):
    measurements = pd.DataFrame(
        {
            "parameter": ["DO"],
            "value": [value],
        }
    )

    with pytest.raises(ValueError, match="positive"):
        ccme_wqi(measurements, {"DO": {"min": 5.0}})


def test_ccme_wqi_accepts_zero_under_maximum():
    measurements = pd.DataFrame(
        {
            "parameter": ["TP"],
            "value": [0.0],
        }
    )

    result = ccme_wqi(measurements, {"TP": {"max": 0.05}})

    assert result.score == 100.0
    assert result.category == "Excellent"

def test_ccme_wqi_handles_mixed_and_two_sided_guidelines():
    measurements = pd.DataFrame(
        {
            "parameter": ["DO", "pH", "TP", "Pb"] * 4,
            "value": [8.0, 7.5, 0.02, 0.001] * 4,
        }
    )

    guidelines = {
        "DO": {"min": 5.0},
        "pH": {"min": 6.5, "max": 9.0},
        "TP": {"max": 0.05},
        "Pb": {"max": 0.004},
    }

    # Four failures, each with an excursion of 1.
    measurements.loc[0, "value"] = 2.5   # DO below its minimum
    measurements.loc[1, "value"] = 3.25  # pH below its minimum
    measurements.loc[2, "value"] = 0.10  # TP above its maximum
    measurements.loc[5, "value"] = 18.0  # pH above its maximum

    result = ccme_wqi(measurements, guidelines)

    assert result.f1 == pytest.approx(75.0)
    assert result.f2 == pytest.approx(25.0)
    assert result.f3 == pytest.approx(20.0)
    assert result.score == pytest.approx(52.917129, abs=0.000001)
    assert result.category == "Marginal"
