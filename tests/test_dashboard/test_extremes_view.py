"""Tests for the Extremes dashboard view, verifying KS p-value formatting."""

from __future__ import annotations

import pytest

from aquascope.utils.formatting import format_p_value


def test_extremes_view_ks_pvalue_metric_formatting():
    """Verify that the KS p-value metric in extremes.py uses the shared format_p_value helper."""
    pytest.importorskip("streamlit")
    from aquascope.dashboard.views import extremes

    assert extremes.format_p_value is format_p_value
    assert extremes.format_p_value(1e-12) == "< 0.001"
    assert extremes.format_p_value(0.045) == "0.045"
    assert extremes.format_p_value(None) == "—"
