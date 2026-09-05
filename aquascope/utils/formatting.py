"""Shared formatting helpers for CLI, dashboard, and reporting."""

from __future__ import annotations

import math


def format_p_value(p: float | None) -> str:
    """Format a p-value following the standard reporting convention.

    P-values strictly less than 0.001 are formatted as ``"< 0.001"`` to prevent
    misleading display of non-zero statistical significance as zero (e.g. ``"0.0000"``).
    Values at or above 0.001 are rounded to three decimal places.
    None, NaN, or non-finite values return ``"—"``.

    Parameters
    ----------
    p:
        The p-value to format.

    Returns
    -------
    str
        Formatted p-value string (e.g., ``"< 0.001"``, ``"0.042"``, ``"1.000"``, or ``"—"``).
    """
    if p is None:
        return "—"
    try:
        val = float(p)
    except (ValueError, TypeError):
        return "—"
    if not math.isfinite(val):
        return "—"
    if val < 0.001:
        return "< 0.001"
    return f"{val:.3f}"
