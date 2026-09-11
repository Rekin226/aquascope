"""Crop water requirements using FAO-56 crop coefficients.

Provides crop coefficient lookup, crop evapotranspiration calculation,
and full irrigation scheduling based on the single- and dual-crop-coefficient
approaches from FAO Irrigation and Drainage Paper 56.

References
----------
Pereira, L. S., Allen, R. G., Paredes, P., López-Urrea, R., Raes, D.,
    Smith, M., Kilic, A., & Salman, M. (2025).
    Crop evapotranspiration: Guidelines for computing crop water requirements.
    Second edition, revised 2025. FAO Irrigation and Drainage Paper No. 56 Rev.1.
    Rome: FAO. doi:10.4060/cd6621en

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
# FAO-56 Rev.1 Table 6.1, 6.2, 6.3 – single crop coefficients (Kc)
# Keys: initial, mid, late
# ---------------------------------------------------------------------------

KC_TABLE: dict[str, dict[str, float]] = {
    # FAO-56 Rev.1 (2025) Table 6.3 (woody crops) & Table 6.2 (winter wheat) default standard classes
    "wheat_winter": {"initial": 0.35, "mid": 1.15, "late": 0.30},
    "grape": {"initial": 0.30, "mid": 0.70, "late": 0.45},
    "citrus": {"initial": 0.70, "mid": 0.65, "late": 0.70},
    "olive": {"initial": 0.65, "mid": 0.65, "late": 0.65},
    # FAO-56 Rev.1 (2025) Table 6.1 (vegetables) & Table 6.2 (field crops)
    "alfalfa": {"initial": 0.50, "mid": 1.20, "late": 1.15},
    "banana": {"initial": 0.50, "mid": 1.05, "late": 1.00},
    "barley": {"initial": 0.30, "mid": 1.10, "late": 0.25},
    "cabbage": {"initial": 0.70, "mid": 1.05, "late": 1.00},
    "cassava": {"initial": 0.30, "mid": 1.00, "late": 0.60},
    "chickpea": {"initial": 0.40, "mid": 1.05, "late": 0.35},
    "coffee": {"initial": 0.90, "mid": 1.00, "late": 1.00},
    "cotton": {"initial": 0.40, "mid": 1.10, "late": 0.50},
    "groundnut": {"initial": 0.40, "mid": 1.05, "late": 0.60},
    "maize": {"initial": 0.30, "mid": 1.20, "late": 0.30},
    "millet": {"initial": 0.30, "mid": 1.10, "late": 0.35},
    "onion": {"initial": 0.70, "mid": 1.05, "late": 0.70},
    "pepper": {"initial": 0.60, "mid": 1.10, "late": 1.00},
    "potato": {"initial": 0.50, "mid": 1.10, "late": 0.40},
    "rice_paddy": {"initial": 1.05, "mid": 1.20, "late": 1.05},
    "sorghum": {"initial": 0.30, "mid": 1.05, "late": 0.45},
    "soybean": {"initial": 0.40, "mid": 1.10, "late": 0.50},
    "sugar_beet": {"initial": 0.35, "mid": 1.10, "late": 0.75},
    "sugarcane": {"initial": 0.40, "mid": 1.20, "late": 0.80},
    "sunflower": {"initial": 0.35, "mid": 1.15, "late": 0.30},
    "tea": {"initial": 1.05, "mid": 1.05, "late": 1.05},
    "tomato": {"initial": 0.60, "mid": 1.10, "late": 0.85},
}

# ---------------------------------------------------------------------------
# FAO-56 Rev.1 Table 6.3 & Table 6.2 – Multi-class woody crop and wheat Kc tables
# ---------------------------------------------------------------------------

WOODY_CROP_KC_CLASSES: dict[str, dict[str, dict[str, float]]] = {
    "olive": {
        # Rev.1 Table 6.3: Olive (Olea europaea L.)
        "young": {"initial": 0.50, "mid": 0.45, "late": 0.50},  # Young / low density (canopy cover 20-30%)
        "medium_density": {
            "initial": 0.65,
            "mid": 0.65,
            "late": 0.65,
        },  # Mature low/medium density (ground cover 35-50%, standard default)
        "high_density": {"initial": 0.70, "mid": 0.70, "late": 0.70},  # Intensive hedgerow (ground cover ~60%)
        "super_intensive": {"initial": 0.75, "mid": 0.75, "late": 0.75},  # Super-intensive (ground cover >70%)
    },
    "grape": {
        # Rev.1 Table 6.3: Grapes (Vitis vinifera L.)
        "table_high_cover": {"initial": 0.35, "mid": 0.85, "late": 0.50},  # Table grapes (overhead arbor / high cover)
        "wine_low_cover": {"initial": 0.25, "mid": 0.55, "late": 0.35},  # Wine grapes, low ground cover (25-30%)
        "wine_medium_cover": {
            "initial": 0.30,
            "mid": 0.70,
            "late": 0.45,
        },  # Wine grapes, medium ground cover (40-50%, VSP trellis, standard default)
        "wine_high_cover": {"initial": 0.30, "mid": 0.85, "late": 0.50},  # Wine grapes, high ground cover (>60%)
    },
    "citrus": {
        # Rev.1 Table 6.3: Citrus (Citrus spp.)
        "mandarin": {"initial": 0.65, "mid": 0.60, "late": 0.65},  # Clementine / Mandarin / Lime
        "lemon": {"initial": 0.70, "mid": 0.70, "late": 0.70},  # Lemon
        "orange_low_density": {
            "initial": 0.60,
            "mid": 0.55,
            "late": 0.60,
        },  # Orange / Grapefruit, low density (~30% ground cover)
        "orange_medium_density": {
            "initial": 0.70,
            "mid": 0.65,
            "late": 0.70,
        },  # Orange / Grapefruit, medium density (50-70% ground cover, standard default)
        "orange_high_density": {
            "initial": 0.80,
            "mid": 0.75,
            "late": 0.80,
        },  # Orange / Grapefruit, high density (>70% ground cover)
    },
    "wheat_winter": {
        # Rev.1 Table 6.2: Winter wheat (Triticum aestivum / T. durum)
        "standard": {
            "initial": 0.35,
            "mid": 1.15,
            "late": 0.30,
        },  # Common winter wheat, standard harvest moisture (default)
        "dry_harvest": {
            "initial": 0.35,
            "mid": 1.15,
            "late": 0.25,
        },  # Common winter wheat, dry grain harvest (<14% moisture)
        "durum": {"initial": 0.35, "mid": 1.15, "late": 0.30},  # Durum winter wheat
    },
}

# ---------------------------------------------------------------------------
# FAO-56 Table 17 / Tables 7.x – basal crop coefficients (Kcb) for dual approach
# Keys: initial, mid, late
# ---------------------------------------------------------------------------

KCB_TABLE: dict[str, dict[str, float]] = {
    # FAO-56 Rev.1 (2025) Table 7.3 (woody crops) & Table 7.2 (winter wheat) default standard classes
    "wheat_winter": {"initial": 0.15, "mid": 1.10, "late": 0.25},
    "grape": {"initial": 0.15, "mid": 0.65, "late": 0.40},
    "citrus": {"initial": 0.60, "mid": 0.55, "late": 0.60},
    "olive": {"initial": 0.55, "mid": 0.55, "late": 0.55},
    # FAO-56 Rev.1 (2025) Tables 7.1 (vegetables), 7.2 (field crops), and 7.3 (woody/mature)
    "alfalfa": {"initial": 0.30, "mid": 0.90, "late": 0.85},
    "banana": {"initial": 0.15, "mid": 1.00, "late": 0.90},
    "barley": {"initial": 0.15, "mid": 1.05, "late": 0.20},
    "cabbage": {"initial": 0.15, "mid": 1.00, "late": 0.90},
    "cassava": {"initial": 0.15, "mid": 0.95, "late": 0.50},
    "chickpea": {"initial": 0.15, "mid": 1.00, "late": 0.25},
    "coffee": {"initial": 0.85, "mid": 0.90, "late": 0.90},
    "cotton": {"initial": 0.15, "mid": 1.05, "late": 0.40},
    "groundnut": {"initial": 0.15, "mid": 1.00, "late": 0.50},
    "maize": {"initial": 0.15, "mid": 1.15, "late": 0.25},
    "millet": {"initial": 0.15, "mid": 1.05, "late": 0.20},
    "onion": {"initial": 0.15, "mid": 1.00, "late": 0.60},
    "pepper": {"initial": 0.15, "mid": 1.00, "late": 0.70},
    "potato": {"initial": 0.15, "mid": 1.05, "late": 0.30},
    "rice_paddy": {"initial": 0.15, "mid": 1.15, "late": 0.90},
    "sorghum": {"initial": 0.15, "mid": 1.00, "late": 0.35},
    "soybean": {"initial": 0.15, "mid": 1.05, "late": 0.35},
    "sugar_beet": {"initial": 0.15, "mid": 1.05, "late": 0.65},
    "sugarcane": {"initial": 0.15, "mid": 1.15, "late": 0.70},
    "sunflower": {"initial": 0.15, "mid": 1.10, "late": 0.20},
    "tea": {"initial": 0.95, "mid": 0.95, "late": 0.95},
    "tomato": {"initial": 0.15, "mid": 1.05, "late": 0.75},
}

# ---------------------------------------------------------------------------
# FAO-56 Rev.1 Table 7.3 & Table 7.2 – Multi-class woody crop and wheat Kcb tables
# ---------------------------------------------------------------------------

WOODY_CROP_KCB_CLASSES: dict[str, dict[str, dict[str, float]]] = {
    "olive": {
        "young": {"initial": 0.35, "mid": 0.35, "late": 0.35},
        "medium_density": {"initial": 0.55, "mid": 0.55, "late": 0.55},
        "high_density": {"initial": 0.65, "mid": 0.65, "late": 0.65},
        "super_intensive": {"initial": 0.70, "mid": 0.70, "late": 0.70},
    },
    "grape": {
        "table_high_cover": {"initial": 0.20, "mid": 0.80, "late": 0.45},
        "wine_low_cover": {"initial": 0.15, "mid": 0.50, "late": 0.30},
        "wine_medium_cover": {"initial": 0.15, "mid": 0.65, "late": 0.40},
        "wine_high_cover": {"initial": 0.15, "mid": 0.80, "late": 0.45},
    },
    "citrus": {
        "mandarin": {"initial": 0.55, "mid": 0.50, "late": 0.55},
        "lemon": {"initial": 0.60, "mid": 0.60, "late": 0.60},
        "orange_low_density": {"initial": 0.50, "mid": 0.45, "late": 0.50},
        "orange_medium_density": {"initial": 0.60, "mid": 0.55, "late": 0.60},
        "orange_high_density": {"initial": 0.70, "mid": 0.65, "late": 0.70},
    },
    "wheat_winter": {
        "standard": {"initial": 0.15, "mid": 1.10, "late": 0.25},
        "dry_harvest": {"initial": 0.15, "mid": 1.10, "late": 0.20},
        "durum": {"initial": 0.15, "mid": 1.10, "late": 0.25},
    },
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
    "sorghum": {"initial": 20, "development": 35, "mid": 40, "late": 30},
    "groundnut": {"initial": 25, "development": 35, "mid": 45, "late": 25},
    "sugar_beet": {"initial": 30, "development": 45, "mid": 90, "late": 15},
    # FAO-56 Table 11
    "millet": {"initial": 15, "development": 25, "mid": 40, "late": 25},
    # FAO-56 Table 11, cassava year 1. Year 2 is listed separately at 360 days.
    "cassava": {"initial": 20, "development": 40, "mid": 90, "late": 60},
    # FAO-56 Table 11 has no chickpea row. Lentil is the closest listed crop
    # (cool-season food legume, same duration class), used here as a proxy.
    "chickpea": {"initial": 20, "development": 30, "mid": 60, "late": 40},  # 150 d
}


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _resolve_crop_coefficients(
    crop: str,
    *,
    is_kcb: bool = False,
    ground_cover: float | None = None,
    density: str | None = None,
    variety: str | None = None,
    class_name: str | None = None,
) -> dict[str, float]:
    """Resolve crop coefficients (Kc or Kcb) for a given crop and optional planting parameters."""
    classes_table = WOODY_CROP_KCB_CLASSES if is_kcb else WOODY_CROP_KC_CLASSES
    base_table = KCB_TABLE if is_kcb else KC_TABLE

    if crop not in base_table:
        raise ValueError(f"Unknown crop '{crop}'. Available: {sorted(base_table)}")

    if crop not in classes_table:
        return dict(base_table[crop])

    crop_classes = classes_table[crop]

    if class_name is not None:
        if class_name not in crop_classes:
            raise ValueError(f"Unknown class_name '{class_name}' for crop '{crop}'. Available: {sorted(crop_classes)}")
        return dict(crop_classes[class_name])

    # Parameter resolution based on FAO-56 Rev.1 Table 6.3 & 6.2
    if crop == "olive":
        if density == "young" or (ground_cover is not None and ground_cover < 0.35):
            return dict(crop_classes["young"])
        if density in ("high", "high_density", "hedgerow") or (
            ground_cover is not None and 0.55 < ground_cover <= 0.70
        ):
            return dict(crop_classes["high_density"])
        if density in ("super_intensive", "intensive") or (ground_cover is not None and ground_cover > 0.70):
            return dict(crop_classes["super_intensive"])
        if density in ("medium", "medium_density", "mature") or (
            ground_cover is not None and 0.35 <= ground_cover <= 0.55
        ):
            return dict(crop_classes["medium_density"])

    elif crop == "grape":
        if variety in ("table", "table_grape", "arbor") or density == "table":
            return dict(crop_classes["table_high_cover"])
        if density in ("low", "low_cover") or (ground_cover is not None and ground_cover < 0.35):
            return dict(crop_classes["wine_low_cover"])
        if density in ("high", "high_cover") or (ground_cover is not None and ground_cover > 0.55):
            return dict(crop_classes["wine_high_cover"])
        if density in ("medium", "medium_cover", "vsp") or (ground_cover is not None and 0.35 <= ground_cover <= 0.55):
            return dict(crop_classes["wine_medium_cover"])

    elif crop == "citrus":
        if variety in ("mandarin", "clementine", "lime"):
            return dict(crop_classes["mandarin"])
        if variety == "lemon":
            return dict(crop_classes["lemon"])
        if density in ("low", "low_density") or (ground_cover is not None and ground_cover < 0.40):
            return dict(crop_classes["orange_low_density"])
        if density in ("high", "high_density") or (ground_cover is not None and ground_cover > 0.70):
            return dict(crop_classes["orange_high_density"])
        if density in ("medium", "medium_density") or (ground_cover is not None and 0.40 <= ground_cover <= 0.70):
            return dict(crop_classes["orange_medium_density"])

    elif crop == "wheat_winter":
        if variety in ("durum", "durum_wheat"):
            return dict(crop_classes["durum"])
        if variety in ("dry", "dry_harvest", "low_moisture") or density == "dry_harvest":
            return dict(crop_classes["dry_harvest"])
        if variety in ("standard", "common"):
            return dict(crop_classes["standard"])

    # Fallback to standard documented default class
    return dict(base_table[crop])


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_kc(
    crop: str,
    stage: str | None = None,
    *,
    ground_cover: float | None = None,
    density: str | None = None,
    variety: str | None = None,
    class_name: str | None = None,
) -> float | dict[str, float]:
    """Get single crop coefficient(s) for a crop from FAO-56 Table 12 / Rev.1 Tables 6.1–6.3.

    Parameters
    ----------
    crop : str
        Crop name (must be a key in ``KC_TABLE``).
    stage : str | None
        Growth stage: ``"initial"``, ``"mid"``, or ``"late"``.
        If *None*, returns all coefficients as a dict.
    ground_cover : float | None, optional
        Effective ground cover fraction (0–1), used to resolve woody-crop
        classes (e.g. olive, grape, citrus) per FAO-56 Rev.1 Table 6.3.
    density : str | None, optional
        Planting density or training class (e.g. ``"low"``, ``"medium"``,
        ``"high"``, ``"super_intensive"``).
    variety : str | None, optional
        Crop variety or sub-species (e.g. ``"table"`` / ``"wine"`` for grape;
        ``"lemon"`` / ``"mandarin"`` for citrus; ``"durum"`` for wheat_winter).
    class_name : str | None, optional
        Explicit FAO-56 Rev.1 class identifier from ``WOODY_CROP_KC_CLASSES``.

    Returns
    -------
    float | dict[str, float]
        Kc value for the given stage, or a dict of all stages.

    Raises
    ------
    ValueError
        If *crop*, *stage*, or *class_name* is unknown.

    References
    ----------
    Allen et al. (1998), Table 12; Pereira et al. (2025), FAO-56 Rev.1 Tables 6.1–6.3.
    """
    coeffs = _resolve_crop_coefficients(
        crop,
        is_kcb=False,
        ground_cover=ground_cover,
        density=density,
        variety=variety,
        class_name=class_name,
    )
    if stage is None:
        return coeffs
    if stage not in coeffs:
        raise ValueError(f"Unknown stage '{stage}' for crop '{crop}'. Available: {sorted(coeffs)}")
    return coeffs[stage]


def get_kcb(
    crop: str,
    stage: str | None = None,
    *,
    ground_cover: float | None = None,
    density: str | None = None,
    variety: str | None = None,
    class_name: str | None = None,
) -> float | dict[str, float]:
    """Get basal crop coefficient(s) for a crop from FAO-56 Table 17 / Rev.1 Tables 7.1–7.3.

    Parameters
    ----------
    crop : str
        Crop name (must be a key in ``KCB_TABLE``).
    stage : str | None
        Growth stage: ``"initial"``, ``"mid"``, or ``"late"``.
        If *None*, returns all coefficients as a dict.
    ground_cover : float | None, optional
        Effective ground cover fraction (0–1), used to resolve woody-crop
        classes per FAO-56 Rev.1 Table 7.3.
    density : str | None, optional
        Planting density class (e.g. ``"low"``, ``"medium"``, ``"high"``).
    variety : str | None, optional
        Crop variety or species (e.g. ``"table"`` / ``"wine"`` for grape;
        ``"lemon"`` / ``"mandarin"`` for citrus; ``"durum"`` for wheat_winter).
    class_name : str | None, optional
        Explicit FAO-56 Rev.1 class identifier from ``WOODY_CROP_KCB_CLASSES``.

    Returns
    -------
    float | dict[str, float]
        Kcb value for the given stage, or a dict of all stages.

    Raises
    ------
    ValueError
        If *crop*, *stage*, or *class_name* is unknown.

    References
    ----------
    Allen et al. (1998), Table 17; Pereira et al. (2025), FAO-56 Rev.1 Tables 7.1–7.3.
    """
    coeffs = _resolve_crop_coefficients(
        crop,
        is_kcb=True,
        ground_cover=ground_cover,
        density=density,
        variety=variety,
        class_name=class_name,
    )
    if stage is None:
        return coeffs
    if stage not in coeffs:
        raise ValueError(f"Unknown stage '{stage}' for crop '{crop}'. Available: {sorted(coeffs)}")
    return coeffs[stage]


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
    Allen et al. (1998); Pereira et al. (2025), FAO-56 Eq. 71, §7.3.
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
    Allen et al. (1998); Pereira et al. (2025), FAO-56 Eq. 58.
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
    Allen et al. (1998); Pereira et al. (2025), FAO-56.
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
    *,
    ground_cover: float | None = None,
    density: str | None = None,
    variety: str | None = None,
    class_name: str | None = None,
) -> pd.DataFrame:
    """Compute daily crop water requirement over the growing season.

    Supports both the FAO-56 single crop coefficient (Kc) and dual crop
    coefficient (Kcb + Ke) approaches, with optional woody-crop parameterization
    per FAO-56 Rev.1 (2025).

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
    ground_cover : float | None, optional
        Effective ground cover fraction (0–1), passed to ``get_kc`` / ``get_kcb``.
    density : str | None, optional
        Planting density or training class, passed to ``get_kc`` / ``get_kcb``.
    variety : str | None, optional
        Crop variety or sub-species, passed to ``get_kc`` / ``get_kcb``.
    class_name : str | None, optional
        Explicit class name, passed to ``get_kc`` / ``get_kcb``.

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
    Allen et al. (1998); Pereira et al. (2025), FAO-56 Ch. 6 (single), Ch. 7 (dual).
    """
    import pandas as pd

    if method not in ("single", "dual"):
        raise ValueError(f"Unknown method '{method}'. Use 'single' or 'dual'.")

    kc_values = get_kc(
        crop,
        ground_cover=ground_cover,
        density=density,
        variety=variety,
        class_name=class_name,
    )
    if not isinstance(kc_values, dict):  # pragma: no cover
        raise TypeError("Expected dict from get_kc when stage is None")

    lengths = stage_lengths or DEFAULT_STAGE_LENGTHS.get(crop)
    if lengths is None:
        raise ValueError(f"No default stage lengths for '{crop}'. Provide stage_lengths explicitly.")

    if method == "dual":
        kcb_values = get_kcb(
            crop,
            ground_cover=ground_cover,
            density=density,
            variety=variety,
            class_name=class_name,
        )
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
            rows.append(
                {
                    "date": current_date,
                    "stage": stage,
                    "kc": round(kc, 3),
                    "eto": round(eto_val, 2),
                    "etc": round(etc_val, 2),
                }
            )
        else:
            # Dual method: ETc = (Kcb + Ke) × ET₀
            _, kcb = _interpolate_kc(day_offset, lengths, kcb_values)
            ke = compute_ke(kcb, kc_max=kc_max, few=few, kr=kr)
            kc_dual = kcb + ke
            etc_val = crop_et(eto_val, kc_dual)
            rows.append(
                {
                    "date": current_date,
                    "stage": stage,
                    "kc": round(kc, 3),
                    "kcb": round(kcb, 3),
                    "ke": round(ke, 3),
                    "kc_dual": round(kc_dual, 3),
                    "eto": round(eto_val, 2),
                    "etc": round(etc_val, 2),
                }
            )

        logger.debug(
            "crop_water_requirement: day=%d crop=%s stage=%s method=%s etc=%.2f",
            day_offset,
            crop,
            stage,
            method,
            etc_val,
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
    *,
    ground_cover: float | None = None,
    density: str | None = None,
    variety: str | None = None,
    class_name: str | None = None,
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
    ground_cover : float | None, optional
        Passed to ``crop_water_requirement``.
    density : str | None, optional
        Passed to ``crop_water_requirement``.
    variety : str | None, optional
        Passed to ``crop_water_requirement``.
    class_name : str | None, optional
        Passed to ``crop_water_requirement``.

    Returns
    -------
    pandas.DataFrame
        All columns from ``crop_water_requirement`` plus
        ``effective_rain``, ``net_irrigation``, ``gross_irrigation``.

    References
    ----------
    Allen et al. (1998); Pereira et al. (2025), FAO-56 Ch. 7.
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
        ground_cover=ground_cover,
        density=density,
        variety=variety,
        class_name=class_name,
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
