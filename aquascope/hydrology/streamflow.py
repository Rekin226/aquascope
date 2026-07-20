"""Stage -> discharge -> area-normalized runoff chain.

Bridges :mod:`aquascope.hydrology.rating_curve` (stage -> discharge) with
:func:`aquascope.schemas.water_data.discharge_cms_to_runoff_mm_day`
(discharge -> mm/day), so a station with only stage records and a fitted
rating curve can be converted straight to daily runoff, one of the
prerequisites for Caravan-compatible export (#100).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aquascope.hydrology.rating_curve import RatingCurveResult, predict_discharge
from aquascope.schemas.water_data import discharge_cms_to_runoff_mm_day

logger = logging.getLogger(__name__)


def stage_to_runoff(
    rating: RatingCurveResult,
    stage: np.ndarray | pd.Series,
    catchment_area_km2: float,
) -> pd.Series:
    """Convert a stage time series to area-normalized daily runoff (mm/day).

    Parameters
    ----------
    rating : RatingCurveResult
        A fitted rating curve, e.g. from :func:`fit_rating_curve` or
        :func:`fit_segmented_rating_curve`.
    stage : array-like or pandas.Series
        Stage (water level) values. If a ``Series`` with a datetime index
        is passed, the index is preserved on the output.
    catchment_area_km2 : float
        Upstream drainage area in km2. Must be positive.

    Returns
    -------
    pandas.Series
        Runoff in mm/day, same length and index as the input stage series.

    Raises
    ------
    ValueError
        If ``catchment_area_km2`` is not positive.
    """
    if catchment_area_km2 <= 0:
        raise ValueError(f"catchment_area_km2 must be positive, got {catchment_area_km2}")

    discharge_cms = predict_discharge(rating, stage)

    runoff = np.array(
        [discharge_cms_to_runoff_mm_day(float(q), catchment_area_km2) for q in discharge_cms]
    )

    if isinstance(stage, pd.Series):
        return pd.Series(runoff, index=stage.index, name="runoff_mm_day")
    return pd.Series(runoff, name="runoff_mm_day")
