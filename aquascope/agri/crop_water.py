"""Crop water requirements using FAO-56 crop coefficients.

Provides crop coefficient lookup, crop evapotranspiration calculation,
and full irrigation scheduling based on the single- and dual-crop-coefficient
approaches from FAO Irrigation and Drainage Paper 56.

References
----------
Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998).
    Crop evapotranspiration: Guidelines for computing crop water requirements.
    FAO Irrigation and Drainage Paper 56. Rome: FAO.
    ISBN 92-5-104219-5
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FAO-56 Table 12 – single crop coefficients (Kc)
# Keys: initial, mid, late
# ---------------------------------------------------------------------------

KC_TABLE: dict[str, dict[str, float]] = {
    "wheat_winter": {"initial": 0.4, "mid": 1.15, "late": 0.25},
    "maize": {"initial": 0.3, "mid": 1.20, "late": 0.60},
    "rice_paddy": {"initial": 1.05, "mid": 1.20, "late": 0.90},
    "soybean": {"initial": 0.4, "mid": 1.15, "late": 0.50},
    "cotton": {"initial": 0.35, "mid": 1.15, "late": 0.70},
    "sugarcane": {"initial": 0.40, "mid": 1.25, "late": 0.75},
    "tomato": {"initial": 0.60, "mid": 1.15, "late": 0.80},
    "potato": {"initial": 0.50, "mid": 1.15, "late": 0.75},
    "grape": {"initial": 0.30, "mid": 0.85, "late": 0.45},
    "citrus": {"initial": 0.65, "mid": 0.60, "late": 0.65},
    "olive": {"initial": 0.65, "mid": 0.70, "late": 0.65},
    "sunflower": {"initial": 0.35, "mid": 1.05, "late": 0.35},
    "barley": {"initial": 0.30, "mid": 1.15, "late": 0.25},
    "alfalfa": {"initial": 0.40, "mid": 0.95, "late": 0.90},
    "onion": {"initial": 0.70, "mid": 1.05, "late": 0.75},
    "cabbage": {"initial": 0.70, "mid": 1.05, "late": 0.95},
    "pepper": {"initial": 0.60, "mid": 1.05, "late": 0.90},
    "banana": {"initial": 0.50, "mid": 1.10, "late": 1.00},
    "coffee": {"initial": 0.90, "mid": 0.95, "late": 0.95},
    "tea": {"initial": 0.95, "mid": 1.00, "late": 1.00},
}

# ---------------------------------------------------------------------------
# FAO-56 Table 17 – basal crop coefficients (Kcb) for dual approach
# Keys: initial, mid, late
# ---------------------------------------------------------------------------

KCB_TABLE: dict[str, dict[str, float]] = {
    "wheat_winter": {"initial": 0.15, "mid": 1.10, "late": 0.25},
    "maize": {"initial": 0.15, "mid": 1.15, "late": 0.50},
    "rice_paddy": {"initial": 1.00, "mid": 1.15, "late": 0.70},
    "soybean": {"initial": 0.15, "mid": 1.10, "late": 0.30},
    "cotton": {"initial": 0.15, "mid": 1.10, "late": 0.50},
    "sugarcane": {"initial": 0.15, "mid": 1.20, "late": 0.70},
    "tomato": {"initial": 0.15, "mid": 1.10, "late": 0.70},
    "potato": {"initial": 0.15, "mid": 1.10, "late": 0.65},
    "grape": {"initial": 0.15, "mid": 0.80, "late": 0.40},
    "citrus": {"initial": 0.60, "mid": 0.55, "late": 0.60},
    "olive": {"initial": 0.55, "mid": 0.65, "late": 0.55},
    "sunflower": {"initial": 0.15, "mid": 1.00, "late": 0.30},
    "barley": {"initial": 0.15, "mid": 1.10, "late": 0.15},
    "alfalfa": {"initial": 0.30, "mid": 0.90, "late": 0.85},
    "onion": {"initial": 0.15, "mid": 1.00, "late": 0.70},
    "cabbage": {"initial": 0.15, "mid": 1.00, "late": 0.90},
    "pepper": {"initial": 0.15, "mid": 1.00, "late": 0.80},
    "banana": {"initial": 0.15, "mid": 1.05, "late": 0.90},
    "coffee": {"initial": 0.85, "mid": 0.90, "late": 0.90},
    "tea": {"initial": 0.90, "mid": 0.95, "late": 0.95},
}

# ---------------------------------------------------------------------------
# Default stage durations (days) per crop
# ---------------------------------------------------------------------------

DEFAULT_STAGE_LENGTHS: dict[str, dict[str, int]] = {
    "wheat_winter": {"initial": 30, "development": 140, "mid": 40, "late": 30},
    "maize": {"initial": 20, "development": 35, "mid": 40, "late": 30},
    "rice_paddy": {"initial": 30, "development": 30, "mid": 60, "late": 30},
    "soybean": {"initial": 20, "development": 30, "mid": 60, "late": 25},
    "cotton": {"initial": 30, "development": 50, "mid": 55, "late": 45},
    "sugarcane": {"initial": 35, "development": 60, "mid": 190, "late": 120},
    "tomato": {"initial": 30, "development": 40, "mid": 40, "late": 25},
    "potato": {"initial": 25, "development": 30, "mid": 45, "late": 30},
    "grape": {"initial": 20, "development": 40, "mid": 120, "late": 60},
    "citrus": {"initial": 60, "development": 90, "mid": 120, "late": 95},
    "olive": {"initial": 30, "development": 90, "mid": 60, "late": 90},
    "sunflower": {"initial": 25, "development": 35, "mid": 45, "late": 25},
    "barley": {"initial": 15, "development": 25, "mid": 50, "late": 30},
    "alfalfa": {"initial": 10, "development": 30, "mid": 25, "late": 10},
    "onion": {"initial": 15, "development": 25, "mid": 70, "late": 40},
    "cabbage": {"initial": 20, "development": 25, "mid": 60, "late": 15},
    "pepper": {"initial": 25, "development": 35, "mid": 40, "late": 20},
    "banana": {"initial": 120, "development": 60, "mid": 180, "late": 5},
    "coffee": {"initial": 60, "development": 90, "mid": 120, "late": 60},
    "tea": {"initial": 60, "development": 90, "mid": 120, "late": 60},
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_kc(crop: str, stage: str | None = None) -> float | dict[str, float]:
    """Get single crop coefficient(s) for a crop from FAO-56 Table 12.

    Parameters
    ----------
    crop : str
        Crop name (must be a key in ``KC_TABLE``).
    stage : str | None
        Growth stage: ``"initial"``, ``"mid"``, or ``"late"``.
        If *None*, returns all coefficients as a dict.

    Returns
    -------
    float | dict[str, float]
        Kc value for the given stage, or a dict of all stages.

    Raises
    ------
    ValueError
        If *crop* or *stage* is unknown.

    References
    ----------
    Allen et al. (1998), Table 12. ISBN 92-5-104219-5.
    """
    if crop not in KC_TABLE:
        raise ValueError(f"Unknown crop '{crop}'. Available: {sorted(KC_TABLE)}")
    if stage is None:
        return dict(KC_TABLE[crop])
    if stage not in KC_TABLE[crop]:
        raise ValueError(f"Unknown stage '{stage}' for crop '{crop}'. Available: {sorted(KC_TABLE[crop])}")
    return KC_TABLE[crop][stage]


def get_kcb(crop: str, stage: str | None = None) -> float | dict[str, float]:
    """Get basal crop coefficient(s) for a crop from FAO-56 Table 17.

    Parameters
    ----------
    crop : str
        Crop name (must be a key in ``KCB_TABLE``).
    stage : str | None
        Growth stage: ``"initial"``, ``"mid"``, or ``"late"``.
        If *None*, returns all coefficients as a dict.

    Returns
    -------
    float | dict[str, float]
        Kcb value for the given stage, or a dict of all stages.

    Raises
    ------
    ValueError
        If *crop* or *stage* is unknown.

    References
    ----------
    Allen et al. (1998), Table 17. ISBN 92-5-104219-5.
    """
    if crop not in KCB_TABLE:
        raise ValueError(f"Unknown crop '{crop}'. Available: {sorted(KCB_TABLE)}")
    if stage is None:
        return dict(KCB_TABLE[crop])
    if stage not in KCB_TABLE[crop]:
        raise ValueError(f"Unknown stage '{stage}' for crop '{crop}'. Available: {sorted(KCB_TABLE[crop])}")
    return KCB_TABLE[crop][stage]


def compute_ke(
    kcb: float,
    kc_max: float = 1.20,
    few: float = 1.0,
    kr: float = 1.0,
) -> float:
    """Compute soil-evaporation coefficient Ke (FAO-56 Eq. 71).

    The soil-evaporation coefficient represents evaporation from the exposed
    and wetted soil fraction. It is bounded so that ``Kcb + Ke ≤ Kc_max``.

    FAO-56 Eq. 71::

        Ke = min(Kr × (Kc_max − Kcb), few × Kc_max)

    Parameters
    ----------
    kcb : float
        Basal crop coefficient for the current day/stage.
    kc_max : float
        Maximum Kc following rain or irrigation (default 1.20, FAO-56 §7.3).
        Typically 1.05–1.30 depending on wind and humidity.
    few : float
        Fraction of soil that is both exposed and wetted (0–1).
        ``few = min(1 − fc, fw)`` where ``fc`` is canopy cover fraction
        and ``fw`` is wetted soil fraction.  Default 1.0 (bare soil / full
        wetting), which gives the upper-bound Ke.
    kr : float
        Soil evaporation reduction coefficient (0–1).  ``kr = 1`` when the
        topsoil is wet (stage 1 evaporation); decreases as the soil dries
        (stage 2).  Default 1.0 (soil surface is wet after rain/irrigation).

    Returns
    -------
    float
        Ke value (dimensionless), clipped to ``[0, kc_max − kcb]``.

    Notes
    -----
    For daily scheduling without explicit soil-water tracking, pass the
    defaults (``kr=1``, ``few=1``) to obtain a conservative upper-bound Ke
    that matches FAO-56 worked examples.  For more accurate site-specific
    estimates, supply ``kr`` and ``few`` derived from your soil-water balance.

    References
    ----------
    Allen et al. (1998), FAO-56 Eq. 71, §7.3. ISBN 92-5-104219-5.
    """
    ke = min(kr * (kc_max - kcb), few * kc_max)
    return max(ke, 0.0)


def crop_et(eto: float, kc: float) -> float:
    """Crop evapotranspiration.

    FAO-56 Eq. 58::

        ETc = Kc × ET₀

    Parameters
    ----------
    eto : float
        Reference evapotranspiration in mm/day.
    kc : float
        Crop coefficient (single Kc, or Kcb + Ke for dual method).

    Returns
    -------
    float
        Crop ET in mm/day.

    References
    ----------
    Allen et al. (1998), Eq. 58. ISBN 92-5-104219-5.
    """
    return kc * eto


def effective_rainfall(precipitation: float, method: str = "usda") -> float:
    """Estimate effective rainfall from total precipitation.

    Parameters
    ----------
    precipitation : float
        Daily precipitation in mm.
    method : str
        Estimation method: ``'usda'``, ``'fao'``, or ``'fixed_fraction'``.

    Returns
    -------
    float
        Effective rainfall in mm.

    Raises
    ------
    ValueError
        If *method* is unrecognised.

    References
    ----------
    USDA SCS (1970). Irrigation Water Requirements. Technical Release 21.
    Allen et al. (1998), FAO-56. ISBN 92-5-104219-5.
    """
    if precipitation <= 0:
        return 0.0

    if method == "usda":
        p = precipitation
        return max(p * (125.0 - 0.2 * p) / 125.0, 0.0) if p < 250 else p * 0.1 + 100.0
    if method == "fao":
        if precipitation <= 5.0:
            return 0.0
        return precipitation * 0.8 - 5.0 if precipitation <= 70.0 else precipitation * 0.6
    if method == "fixed_fraction":
        return precipitation * 0.7
    raise ValueError(f"Unknown method '{method}'. Use 'usda', 'fao', or 'fixed_fraction'.")


def _interpolate_kc(
    day_in_season: int,
    stage_lengths: dict[str, int],
    kc_values: dict[str, float],
) -> tuple[str, float]:
    """Return (stage_name, interpolated Kc) for a given day in the season."""
    ini_len = stage_lengths["initial"]
    dev_len = stage_lengths["development"]
    mid_len = stage_lengths["mid"]
    late_len = stage_lengths["late"]

    kc_ini = kc_values["initial"]
    kc_mid = kc_values["mid"]
    kc_late = kc_values["late"]

    if day_in_season < ini_len:
        return "initial", kc_ini
    if day_in_season < ini_len + dev_len:
        frac = (day_in_season - ini_len) / dev_len
        return "development", kc_ini + frac * (kc_mid - kc_ini)
    if day_in_season < ini_len + dev_len + mid_len:
        return "mid", kc_mid
    if day_in_season < ini_len + dev_len + mid_len + late_len:
        frac = (day_in_season - ini_len - dev_len - mid_len) / late_len
        return "late", kc_mid + frac * (kc_late - kc_mid)
    return "late", kc_late


def crop_water_requirement(
    eto_series: pd.Series,
    crop: str,
    planting_date: date,
    stage_lengths: dict[str, int] | None = None,
    method: str = "single",
    kc_max: float = 1.20,
    few: float = 1.0,
    kr: float = 1.0,
) -> pd.DataFrame:
    """Compute daily crop water requirement over the growing season.

    Supports both the FAO-56 single crop coefficient (Kc) and dual crop
    coefficient (Kcb + Ke) approaches.

    Parameters
    ----------
    eto_series : pandas.Series
        Daily reference ET (mm/day) with a ``DatetimeIndex``.
    crop : str
        Crop name (key in ``KC_TABLE`` / ``KCB_TABLE``).
    planting_date : date
        Planting or sowing date.
    stage_lengths : dict[str, int] | None
        Days per stage.  Defaults to ``DEFAULT_STAGE_LENGTHS[crop]``.
    method : str
        ``"single"`` (default, FAO-56 Ch. 6) or ``"dual"`` (FAO-56 Ch. 7).
        The single method uses Kc directly; the dual method computes
        ETc = (Kcb + Ke) × ET₀ where Kcb is the basal crop coefficient
        and Ke is the soil-evaporation coefficient.
    kc_max : float
        Maximum Kc after rain/irrigation (dual method only, default 1.20).
        See ``compute_ke`` for details.
    few : float
        Exposed-and-wetted soil fraction (dual method only, default 1.0).
        See ``compute_ke`` for details.
    kr : float
        Soil evaporation reduction coefficient (dual method only, default 1.0).
        See ``compute_ke`` for details.

    Returns
    -------
    pandas.DataFrame
        Single method columns: ``date``, ``stage``, ``kc``, ``eto``, ``etc``.
        Dual method adds: ``kcb``, ``ke``, ``kc_dual``.

    Raises
    ------
    ValueError
        If *method* is not ``"single"`` or ``"dual"``.

    References
    ----------
    Allen et al. (1998), FAO-56 Ch. 6 (single), Ch. 7 (dual).
    ISBN 92-5-104219-5.
    """
    import pandas as pd

    if method not in ("single", "dual"):
        raise ValueError(f"Unknown method '{method}'. Use 'single' or 'dual'.")

    kc_values = get_kc(crop)
    if not isinstance(kc_values, dict):  # pragma: no cover
        raise TypeError("Expected dict from get_kc when stage is None")

    lengths = stage_lengths or DEFAULT_STAGE_LENGTHS.get(crop)
    if lengths is None:
        raise ValueError(
            f"No default stage lengths for '{crop}'. Provide stage_lengths explicitly."
        )

    if method == "dual":
        kcb_values = get_kcb(crop)
        if not isinstance(kcb_values, dict):  # pragma: no cover
            raise TypeError("Expected dict from get_kcb when stage is None")

    total_days = sum(lengths.values())
    rows: list[dict] = []

    for day_offset in range(total_days):
        current_date = planting_date + timedelta(days=day_offset)
        stage, kc = _interpolate_kc(day_offset, lengths, kc_values)

        date_key = pd.Timestamp(current_date)
        if date_key in eto_series.index:
            eto_val = float(eto_series.loc[date_key])
        else:
            eto_val = float(eto_series.mean())

        if method == "single":
            etc_val = crop_et(eto_val, kc)
            rows.append({
                "date": current_date,
                "stage": stage,
                "kc": round(kc, 3),
                "eto": round(eto_val, 2),
                "etc": round(etc_val, 2),
            })
        else:
            # Dual method: ETc = (Kcb + Ke) × ET₀
            _, kcb = _interpolate_kc(day_offset, lengths, kcb_values)
            ke = compute_ke(kcb, kc_max=kc_max, few=few, kr=kr)
            kc_dual = kcb + ke
            etc_val = crop_et(eto_val, kc_dual)
            rows.append({
                "date": current_date,
                "stage": stage,
                "kc": round(kc, 3),
                "kcb": round(kcb, 3),
                "ke": round(ke, 3),
                "kc_dual": round(kc_dual, 3),
                "eto": round(eto_val, 2),
                "etc": round(etc_val, 2),
            })

        logger.debug(
            "crop_water_requirement: day=%d crop=%s stage=%s method=%s etc=%.2f",
            day_offset, crop, stage, method, etc_val,
        )

    return pd.DataFrame(rows)


def irrigation_schedule(
    eto_series: pd.Series,
    precip_series: pd.Series,
    crop: str,
    planting_date: date,
    efficiency: float = 0.7,
    stage_lengths: dict[str, int] | None = None,
    method: str = "single",
    kc_max: float = 1.20,
    few: float = 1.0,
    kr: float = 1.0,
) -> pd.DataFrame:
    """Full irrigation scheduling over the growing season.

    Parameters
    ----------
    eto_series : pandas.Series
        Daily reference ET (mm/day) with a ``DatetimeIndex``.
    precip_series : pandas.Series
        Daily precipitation (mm) with a ``DatetimeIndex``.
    crop : str
        Crop name.
    planting_date : date
        Planting date.
    efficiency : float
        Irrigation system efficiency (0–1).
    stage_lengths : dict[str, int] | None
        Days per stage.
    method : str
        ``"single"`` (default) or ``"dual"``.  Passed through to
        ``crop_water_requirement``.
    kc_max : float
        Maximum Kc after rain/irrigation (dual method only, default 1.20).
    few : float
        Exposed-and-wetted soil fraction (dual method only, default 1.0).
    kr : float
        Soil evaporation reduction coefficient (dual method only, default 1.0).

    Returns
    -------
    pandas.DataFrame
        All columns from ``crop_water_requirement`` plus
        ``effective_rain``, ``net_irrigation``, ``gross_irrigation``.

    References
    ----------
    Allen et al. (1998), FAO-56 Ch. 7. ISBN 92-5-104219-5.
    """
    import pandas as pd

    cwr = crop_water_requirement(
        eto_series,
        crop,
        planting_date,
        stage_lengths,
        method=method,
        kc_max=kc_max,
        few=few,
        kr=kr,
    )

    eff_rain_list: list[float] = []
    net_irr_list: list[float] = []
    gross_irr_list: list[float] = []

    for _, row in cwr.iterrows():
        current_date = row["date"]
        date_key = pd.Timestamp(current_date)
        if date_key in precip_series.index:
            precip = float(precip_series.loc[date_key])
        else:
            precip = 0.0

        eff_rain = effective_rainfall(precip)
        net_irr = max(row["etc"] - eff_rain, 0.0)
        gross_irr = net_irr / efficiency if efficiency > 0 else net_irr

        eff_rain_list.append(round(eff_rain, 2))
        net_irr_list.append(round(net_irr, 2))
        gross_irr_list.append(round(gross_irr, 2))

    cwr["effective_rain"] = eff_rain_list
    cwr["net_irrigation"] = net_irr_list
    cwr["gross_irrigation"] = gross_irr_list

    return cwr
