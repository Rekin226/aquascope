"""Tests for shared formatting utilities."""

from __future__ import annotations

import math

import pytest

from aquascope.utils.formatting import format_p_value


class TestFormatPValue:
    """Test suite verifying standard p-value formatting conventions."""

    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            (0, "< 0.001"),
            (0.0, "< 0.001"),
            (1e-20, "< 0.001"),
            (1e-12, "< 0.001"),
            (0.0005, "< 0.001"),
            (0.0009999, "< 0.001"),
        ],
    )
    def test_values_below_threshold_return_inequality(self, val: float, expected: str):
        """Values strictly below 0.001 should display '< 0.001' rather than misleading '0.0000'."""
        assert format_p_value(val) == expected

    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            (0.001, "0.001"),
            (0.0012, "0.001"),
            (0.042, "0.042"),
            (0.0423, "0.042"),
            (0.05, "0.050"),
            (0.0501, "0.050"),
            (0.1, "0.100"),
            (0.999, "0.999"),
            (1.0, "1.000"),
        ],
    )
    def test_values_at_or_above_threshold_format_three_decimals(self, val: float, expected: str):
        """Values at or above 0.001 should round to exactly three decimal places."""
        assert format_p_value(val) == expected

    @pytest.mark.parametrize(
        "val",
        [
            None,
            float("nan"),
            float("inf"),
            float("-inf"),
            math.nan,
            "not-a-number",
        ],
    )
    def test_non_finite_and_none_return_em_dash(self, val: float | None):
        """None, NaN, infinite, or unparseable inputs should return an em dash '—'."""
        assert format_p_value(val) == "—"
