"""Baseflow separation using digital filter methods.

Implements two widely-used recursive digital filters:

- **Lyne–Hollick** (1979) — single-parameter filter, the most common
  automated baseflow separation technique.
- **Eckhardt** (2005) — two-parameter filter that accounts for the
  maximum baseflow index (BFI_max).

Both filters operate on a daily discharge series and return a DataFrame
with ``total``, ``baseflow``, and ``quickflow`` columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BaseflowResult:
    """Result of baseflow separation.

    Attributes
    ----------
    df:
        DataFrame with ``total``, ``baseflow``, ``quickflow`` columns
        and the original DatetimeIndex.
    bfi:
        Baseflow Index — ratio of total baseflow to total discharge.
    method:
        Name of the filter used.
    """

    df: pd.DataFrame
    bfi: float
    method: str


def lyne_hollick(
    discharge: pd.Series,
    *,
    alpha: float = 0.925,
    n_passes: int = 3,
) -> BaseflowResult:
    """Separate baseflow using the Lyne–Hollick recursive digital filter.

    Parameters
    ----------
    discharge:
        Daily discharge series with a DatetimeIndex.
    alpha:
        Filter parameter (0 < α < 1).  Higher values yield less baseflow.
        Default 0.925 is the value recommended by Nathan & McMahon (1990).
    n_passes:
        Number of forward/backward filter passes (typically 3).

    Returns
    -------
    A :class:`BaseflowResult` with separated components and BFI.
    """
    q = discharge.dropna().values.astype(float).copy()
    n = len(q)
    if n == 0:
        empty = pd.DataFrame({"total": [], "baseflow": [], "quickflow": []})
        return BaseflowResult(df=empty, bfi=0.0, method="lyne_hollick")

    qf = np.zeros(n)

    for pass_num in range(n_passes):
        if pass_num % 2 == 0:
            rng = range(1, n)  # forward
        else:
            rng = range(n - 2, -1, -1)  # backward

        for i in rng:
            qf[i] = alpha * qf[i - 1 if pass_num % 2 == 0 else i + 1] + ((1 + alpha) / 2) * (
                q[i] - q[i - 1 if pass_num % 2 == 0 else i + 1]
            )
            qf[i] = max(0.0, qf[i])
            qf[i] = min(qf[i], q[i])

    baseflow = q - qf
    baseflow = np.clip(baseflow, 0, q)

    idx = discharge.dropna().index
    result_df = pd.DataFrame(
        {"total": q, "baseflow": baseflow, "quickflow": qf},
        index=idx,
    )

    bfi = float(baseflow.sum() / q.sum()) if q.sum() > 0 else 0.0
    logger.info("Lyne-Hollick: BFI=%.3f, alpha=%.3f, %d passes", bfi, alpha, n_passes)
    return BaseflowResult(df=result_df, bfi=bfi, method="lyne_hollick")


def eckhardt(
    discharge: pd.Series,
    *,
    alpha: float = 0.98,
    bfi_max: float = 0.80,
) -> BaseflowResult:
    """Separate baseflow using the Eckhardt two-parameter digital filter.

    Parameters
    ----------
    discharge:
        Daily discharge series with a DatetimeIndex.
    alpha:
        Recession constant (typically 0.95–0.99).
    bfi_max:
        Maximum baseflow index — depends on aquifer type:
        - 0.80 for perennial streams with porous aquifers
        - 0.50 for ephemeral streams with porous aquifers
        - 0.25 for perennial streams with hard-rock aquifers

    Returns
    -------
    A :class:`BaseflowResult` with separated components and BFI.
    """
    q = discharge.dropna().values.astype(float).copy()
    n = len(q)
    if n == 0:
        empty = pd.DataFrame({"total": [], "baseflow": [], "quickflow": []})
        return BaseflowResult(df=empty, bfi=0.0, method="eckhardt")

    bf = np.zeros(n)
    bf[0] = min(q[0], q[0] * bfi_max)

    for i in range(1, n):
        numerator = (1 - bfi_max) * alpha * bf[i - 1] + (1 - alpha) * bfi_max * q[i]
        denominator = 1 - alpha * bfi_max
        bf[i] = numerator / denominator
        bf[i] = min(bf[i], q[i])
        bf[i] = max(bf[i], 0.0)

    quickflow = q - bf

    idx = discharge.dropna().index
    result_df = pd.DataFrame(
        {"total": q, "baseflow": bf, "quickflow": quickflow},
        index=idx,
    )

    bfi = float(bf.sum() / q.sum()) if q.sum() > 0 else 0.0
    logger.info("Eckhardt: BFI=%.3f, alpha=%.3f, BFI_max=%.2f", bfi, alpha, bfi_max)
    return BaseflowResult(df=result_df, bfi=bfi, method="eckhardt")


def ukih(
    discharge: pd.Series,
    *,
    block_size: int = 5,
) -> BaseflowResult:
    """Separate baseflow using the UKIH smoothed-minima (sliding-interval) method.

    The record is divided into non-overlapping blocks of ``block_size``
    days. The minimum discharge in each block is a candidate turning
    point if 0.9 times its value is less than both neighbouring block
    minima. Baseflow is then linearly interpolated between turning
    points and capped at the observed discharge.

    Parameters
    ----------
    discharge:
        Daily discharge series with a DatetimeIndex.
    block_size:
        Number of days per non-overlapping block (N). Default 5, as
        recommended by the Institute of Hydrology (1980).

    Returns
    -------
    A :class:`BaseflowResult` with separated components and BFI.

    References
    ----------
    Institute of Hydrology (1980). Low Flow Studies Report No. 1.
    Wallingford, UK.
    """
    q = discharge.dropna().values.astype(float).copy()
    idx = discharge.dropna().index
    n = len(q)

    if n == 0:
        empty = pd.DataFrame({"total": [], "baseflow": [], "quickflow": []})
        return BaseflowResult(df=empty, bfi=0.0, method="ukih")

    # Step 1: divide into non-overlapping blocks; record each block's
    # minimum value and its position in the full series.
    block_min_positions: list[int] = []
    block_min_values: list[float] = []
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = q[start:end]
        local_pos = start + int(np.argmin(block))
        block_min_positions.append(local_pos)
        block_min_values.append(float(block[np.argmin(block)]))

    # Step 2: a block minimum Q_i is a turning point if 0.9 * Q_i is
    # less than both neighbouring block minima. Endpoints check only
    # their single available neighbour.
    turning_positions: list[int] = []
    turning_values: list[float] = []

    m = len(block_min_values)
    for i in range(m):
        qi = block_min_values[i]
        is_turning = True
        if i > 0 and not (0.9 * qi < block_min_values[i - 1]):
            is_turning = False
        if i < m - 1 and not (0.9 * qi < block_min_values[i + 1]):
            is_turning = False
        if is_turning:
            turning_positions.append(block_min_positions[i])
            turning_values.append(qi)

    # Step 3: linearly interpolate baseflow between turning points,
    # flat-extending before the first and after the last. Fall back
    # to a flat line at the overall minimum if no turning points exist.
    if turning_positions:
        positions = np.arange(n)
        baseflow = np.interp(positions, turning_positions, turning_values)
    else:
        baseflow = np.full(n, float(np.min(q)))

    # Step 4: cap baseflow at observed total flow and ensure non-negative.
    baseflow = np.clip(baseflow, 0.0, q)

    quickflow = q - baseflow

    result_df = pd.DataFrame(
        {"total": q, "baseflow": baseflow, "quickflow": quickflow},
        index=idx,
    )

    bfi = float(baseflow.sum() / q.sum()) if q.sum() > 0 else 0.0
    logger.info(
        "UKIH: BFI=%.3f, block_size=%d, %d turning points",
        bfi,
        block_size,
        len(turning_positions),
    )
    return BaseflowResult(df=result_df, bfi=bfi, method="ukih")
