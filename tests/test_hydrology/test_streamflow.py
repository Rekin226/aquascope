import numpy as np
import pandas as pd
import pytest

from aquascope.hydrology.rating_curve import RatingCurveResult
from aquascope.hydrology.streamflow import stage_to_runoff


def _mock_rating(a: float, b: float) -> RatingCurveResult:
    """Helper to mock a RatingCurveResult with required dummy parameters."""
    return RatingCurveResult(
        a=a,
        b=b,
        h0=0.0,
        r_squared=0.99,
        rmse=0.1,
        n_points=10,
        residuals=np.array([]),
        stage_range=(0.0, 10.0),
    )


def test_stage_to_runoff_conversion():
    # Worked Example Analysis:
    # Given a rating curve where Q = a * (stage - h0)^b -> 2.0 * (stage - 0.0)^1.0
    # For stage = 5.0m -> Q = 10.0 m3/s
    # Given catchment area = 100.0 km2
    # Runoff mm/day = (10.0 * 86400) / (100.0 * 1_000_000) * 1000 = 8.64 mm/day
    rating = _mock_rating(a=2.0, b=1.0)
    stages = np.array([5.0])
    area = 100.0

    result = stage_to_runoff(rating, stages, catchment_area_km2=area)
    assert result.iloc[0] == pytest.approx(8.64)


def test_stage_to_runoff_preserves_pandas_index():
    rating = _mock_rating(a=1.5, b=1.0)
    dates = pd.date_range("2026-01-01", periods=3)
    stages = pd.Series([2.0, 3.0, 4.0], index=dates)
    area = 50.0

    result = stage_to_runoff(rating, stages, catchment_area_km2=area)
    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, dates)


def test_stage_to_runoff_invalid_area():
    rating = _mock_rating(a=1.0, b=1.0)
    stages = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="catchment_area_km2 must be positive"):
        stage_to_runoff(rating, stages, catchment_area_km2=0.0)

    with pytest.raises(ValueError, match="catchment_area_km2 must be positive"):
        stage_to_runoff(rating, stages, catchment_area_km2=-15.5)
