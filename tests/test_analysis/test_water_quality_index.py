"""Tests for the CCME Water Quality Index module."""

import pandas as pd
import pytest

from aquascope.analysis.water_quality_index import (
    CCMEWQIResult,
    ccme_wqi,
    wqi_category,
)


def _make_measurements(params: dict[str, list[float]]) -> pd.DataFrame:
    """Create a tidy measurements DataFrame from a param->values dict."""
    rows = []
    for param, values in params.items():
        for val in values:
            rows.append({"parameter": param, "value": val})
    return pd.DataFrame(rows)


# CCME (2001) User's Manual Appendix II worked example.
# 3 parameters, 4 tests each, guideline = maximum objective.
_CCME_EXAMPLE_DATA = {
    "pH": [7.0, 6.0, 5.5, 6.5],       # guideline 6.5 -> 2 failures
    "DO": [8.0, 5.0, 6.0, 7.0],       # guideline 6.5 -> 2 failures
    "Turbidity": [5.0, 3.0, 4.0, 2.0],  # guideline 4.0 -> 1 failure
}
_CCME_EXAMPLE_GUIDELINES = {"pH": 6.5, "DO": 6.5, "Turbidity": 4.0}


class TestCCMEWQI:
    def test_returns_result(self):
        df = _make_measurements(_CCME_EXAMPLE_DATA)
        result = ccme_wqi(df, _CCME_EXAMPLE_GUIDELINES, objective="minimum")
        assert isinstance(result, CCMEWQIResult)

    def test_wqi_in_valid_range(self):
        df = _make_measurements(_CCME_EXAMPLE_DATA)
        result = ccme_wqi(df, _CCME_EXAMPLE_GUIDELINES, objective="minimum")
        assert 0.0 <= result.wqi <= 100.0

    def test_perfect_compliance_gives_100(self):
        """All values well within guidelines -> WQI = 100."""
        data = {"pH": [7.0, 7.5, 8.0], "DO": [9.0, 8.5, 9.5]}
        guidelines = {"pH": 6.5, "DO": 6.5}
        df = _make_measurements(data)
        result = ccme_wqi(df, guidelines, objective="minimum")
        assert result.wqi == pytest.approx(100.0)
        assert result.n_failed_tests == 0
        assert result.n_failed_parameters == 0

    def test_complete_failure_gives_low_score(self):
        """All values exceed guidelines -> very low WQI."""
        data = {"pH": [5.0, 4.5, 4.0], "DO": [3.0, 2.5, 2.0]}
        guidelines = {"pH": 6.5, "DO": 6.5}
        df = _make_measurements(data)
        result = ccme_wqi(df, guidelines, objective="minimum")
        assert result.wqi < 20.0
        assert result.n_failed_parameters == 2

    def test_scope_fraction_correct(self):
        """F1 = 100 * n_failed_params / n_params."""
        data = {"pH": [8.0], "DO": [4.0], "Turbidity": [3.0]}
        guidelines = {"pH": 6.5, "DO": 6.5, "Turbidity": 10.0}
        df = _make_measurements(data)
        # maximum objective: pH fails (8.0 > 6.5); DO and Turbidity pass
        result = ccme_wqi(df, guidelines, objective="maximum")
        assert result.n_failed_parameters == 1
        assert result.scope == pytest.approx(100.0 / 3.0, abs=0.1)

    def test_frequency_fraction_correct(self):
        """F2 = 100 * n_failed_tests / n_tests."""
        data = {"pH": [5.0, 7.0, 7.0, 7.0]}  # 1 of 4 fails
        guidelines = {"pH": 6.5}
        df = _make_measurements(data)
        result = ccme_wqi(df, guidelines, objective="minimum")
        assert result.n_failed_tests == 1
        assert result.n_tests == 4
        assert result.frequency == pytest.approx(25.0)

    def test_maximum_objective(self):
        """For maximum objective, exceedance = value > threshold."""
        data = {"Turbidity": [1.0, 2.0, 10.0, 1.5]}  # 1 failure
        guidelines = {"Turbidity": 5.0}
        df = _make_measurements(data)
        result = ccme_wqi(df, guidelines, objective="maximum")
        assert result.n_failed_tests == 1

    def test_parameters_not_in_guidelines_ignored(self):
        """Parameters without guidelines are silently dropped."""
        data = {"pH": [7.0], "Unknown": [999.0]}
        guidelines = {"pH": 6.5}
        df = _make_measurements(data)
        result = ccme_wqi(df, guidelines, objective="minimum")
        assert result.n_parameters == 1

    def test_category_assigned(self):
        """Category string must be one of the 5 CCME bands."""
        df = _make_measurements(_CCME_EXAMPLE_DATA)
        result = ccme_wqi(df, _CCME_EXAMPLE_GUIDELINES, objective="minimum")
        assert result.category in {
            "Excellent", "Good", "Fair", "Marginal", "Poor"
        }

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ccme_wqi(pd.DataFrame(), {"pH": 6.5})

    def test_empty_guidelines_raises(self):
        df = _make_measurements({"pH": [7.0]})
        with pytest.raises(ValueError, match="empty"):
            ccme_wqi(df, {})

    def test_invalid_objective_raises(self):
        df = _make_measurements({"pH": [7.0]})
        with pytest.raises(ValueError, match="objective"):
            ccme_wqi(df, {"pH": 6.5}, objective="invalid")

    def test_no_matching_parameters_raises(self):
        df = _make_measurements({"pH": [7.0]})
        with pytest.raises(ValueError):
            ccme_wqi(df, {"Turbidity": 5.0})


class TestWQICategory:
    def test_excellent(self):
        assert wqi_category(98.0) == "Excellent"

    def test_good(self):
        assert wqi_category(85.0) == "Good"

    def test_fair(self):
        assert wqi_category(70.0) == "Fair"

    def test_marginal(self):
        assert wqi_category(55.0) == "Marginal"

    def test_poor(self):
        assert wqi_category(30.0) == "Poor"

    def test_boundary_95_is_excellent(self):
        assert wqi_category(95.0) == "Excellent"

    def test_boundary_80_is_good(self):
        assert wqi_category(80.0) == "Good"
