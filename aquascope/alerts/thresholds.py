"""
Regulatory water-quality threshold databases.

Provides WHO, US EPA, and EU Water Framework Directive thresholds for
common water-quality parameters, plus helpers to query them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Threshold:
    """A single regulatory threshold for a water-quality parameter.

    Parameters
    ----------
    parameter:
        Canonical parameter name (e.g. ``"pH"``, ``"nitrate"``).
    limit:
        Numeric limit value.  For lower-bound parameters such as dissolved
        oxygen the convention is to store the *minimum* acceptable value
        as a positive number; callers compare ``value < limit``.
    unit:
        Measurement unit (e.g. ``"mg/L"``, ``"NTU"``).
    standard:
        Originating standard (``"WHO"``, ``"EPA"``, ``"EU_WFD"``).
    category:
        Use-case category — ``"drinking"``, ``"aquatic_life"``,
        ``"recreation"``, or ``"irrigation"``.
    description:
        Human-readable explanation of the threshold.
    """

    parameter: str
    limit: float
    unit: str
    standard: str
    category: str
    description: str


# ---------------------------------------------------------------------------
# WHO Drinking Water Guidelines (2022)
# ---------------------------------------------------------------------------

WHO_THRESHOLDS: dict[str, list[Threshold]] = {
    "pH": [
        Threshold("pH", 6.5, "pH units", "WHO", "drinking", "Minimum acceptable pH for drinking water"),
        Threshold("pH", 8.5, "pH units", "WHO", "drinking", "Maximum acceptable pH for drinking water"),
    ],
    "turbidity": [
        Threshold("turbidity", 1.0, "NTU", "WHO", "drinking", "Ideal turbidity for effective disinfection"),
        Threshold("turbidity", 4.0, "NTU", "WHO", "drinking", "Maximum acceptable turbidity"),
    ],
    "nitrate": [
        Threshold("nitrate", 50.0, "mg/L", "WHO", "drinking", "Maximum nitrate (as NO3)"),
    ],
    "fluoride": [
        Threshold("fluoride", 1.5, "mg/L", "WHO", "drinking", "Maximum fluoride concentration"),
    ],
    "arsenic": [
        Threshold("arsenic", 0.01, "mg/L", "WHO", "drinking", "Maximum arsenic concentration"),
    ],
    "lead": [
        Threshold("lead", 0.01, "mg/L", "WHO", "drinking", "Maximum lead concentration"),
    ],
    "total_dissolved_solids": [
        Threshold("total_dissolved_solids", 600.0, "mg/L", "WHO", "drinking", "Maximum TDS for good palatability"),
    ],
    "e_coli": [
        Threshold("e_coli", 0.0, "CFU/100mL", "WHO", "drinking", "Must not be detected in drinking water"),
    ],
    "total_coliform": [
        Threshold("total_coliform", 0.0, "CFU/100mL", "WHO", "drinking", "Must not be detected in drinking water"),
    ],
    "chloride": [
        Threshold("chloride", 250.0, "mg/L", "WHO", "drinking", "Maximum chloride for taste"),
    ],
    "sulfate": [
        Threshold("sulfate", 250.0, "mg/L", "WHO", "drinking", "Maximum sulfate for taste"),
    ],
    "iron": [
        Threshold("iron", 0.3, "mg/L", "WHO", "drinking", "Maximum iron for taste and appearance"),
    ],
    "manganese": [
        Threshold("manganese", 0.1, "mg/L", "WHO", "drinking", "Maximum manganese for taste and staining"),
    ],
    "copper": [
        Threshold("copper", 2.0, "mg/L", "WHO", "drinking", "Maximum copper for taste"),
    ],
    "zinc": [
        Threshold("zinc", 3.0, "mg/L", "WHO", "drinking", "Maximum zinc for taste"),
    ],
    "ammonia": [
        Threshold("ammonia", 1.5, "mg/L", "WHO", "drinking", "Maximum ammonia for taste and odour"),
    ],
    "dissolved_oxygen": [
        Threshold("dissolved_oxygen", 4.0, "mg/L", "WHO", "aquatic_life", "Minimum DO for aquatic life protection"),
    ],
}

# ---------------------------------------------------------------------------
# US EPA National Primary / Secondary Drinking Water Standards
# ---------------------------------------------------------------------------

EPA_THRESHOLDS: dict[str, list[Threshold]] = {
    "pH": [
        Threshold("pH", 6.5, "pH units", "EPA", "drinking", "Secondary standard minimum pH"),
        Threshold("pH", 8.5, "pH units", "EPA", "drinking", "Secondary standard maximum pH"),
    ],
    "turbidity": [
        Threshold("turbidity", 1.0, "NTU", "EPA", "drinking", "Treatment technique trigger level"),
        Threshold("turbidity", 4.0, "NTU", "EPA", "drinking", "Maximum single measurement (SWTR)"),
    ],
    "nitrate": [
        Threshold("nitrate", 10.0, "mg/L", "EPA", "drinking", "MCL for nitrate (as N)"),
    ],
    "fluoride": [
        Threshold("fluoride", 4.0, "mg/L", "EPA", "drinking", "Primary MCL for fluoride"),
        Threshold("fluoride", 2.0, "mg/L", "EPA", "drinking", "Secondary standard for fluoride"),
    ],
    "arsenic": [
        Threshold("arsenic", 0.01, "mg/L", "EPA", "drinking", "MCL for arsenic"),
    ],
    "lead": [
        Threshold("lead", 0.015, "mg/L", "EPA", "drinking", "Action level for lead (Lead and Copper Rule)"),
    ],
    "total_dissolved_solids": [
        Threshold("total_dissolved_solids", 500.0, "mg/L", "EPA", "drinking", "Secondary standard for TDS"),
    ],
    "e_coli": [
        Threshold("e_coli", 0.0, "CFU/100mL", "EPA", "drinking", "MCL — no E. coli in drinking water"),
    ],
    "total_coliform": [
        Threshold("total_coliform", 0.0, "CFU/100mL", "EPA", "drinking", "No more than 5% positive monthly"),
    ],
    "chloride": [
        Threshold("chloride", 250.0, "mg/L", "EPA", "drinking", "Secondary standard for chloride"),
    ],
    "sulfate": [
        Threshold("sulfate", 250.0, "mg/L", "EPA", "drinking", "Secondary standard for sulfate"),
    ],
    "iron": [
        Threshold("iron", 0.3, "mg/L", "EPA", "drinking", "Secondary standard for iron"),
    ],
    "manganese": [
        Threshold("manganese", 0.05, "mg/L", "EPA", "drinking", "Secondary standard for manganese"),
    ],
    "copper": [
        Threshold("copper", 1.3, "mg/L", "EPA", "drinking", "Action level for copper (Lead and Copper Rule)"),
    ],
    "zinc": [
        Threshold("zinc", 5.0, "mg/L", "EPA", "drinking", "Secondary standard for zinc"),
    ],
    "ammonia": [
        Threshold("ammonia", 1.5, "mg/L", "EPA", "aquatic_life", "Acute aquatic-life criterion (approximate)"),
    ],
    "dissolved_oxygen": [
        Threshold("dissolved_oxygen", 5.0, "mg/L", "EPA", "aquatic_life", "Minimum DO for warm-water aquatic life"),
    ],
}

# ---------------------------------------------------------------------------
# EU Water Framework Directive (2000/60/EC)
# ---------------------------------------------------------------------------

EU_WFD_THRESHOLDS: dict[str, list[Threshold]] = {
    "BOD": [
        Threshold("BOD", 3.0, "mg/L", "EU_WFD", "aquatic_life", "Good ecological status upper limit for BOD"),
    ],
    "phosphate": [
        Threshold("phosphate", 0.1, "mg/L", "EU_WFD", "aquatic_life", "Good status upper limit for ortho-phosphate"),
    ],
    "nitrate": [
        Threshold("nitrate", 50.0, "mg/L", "EU_WFD", "drinking", "Nitrates Directive limit (91/676/EEC)"),
    ],
    "ammonia": [
        Threshold("ammonia", 0.6, "mg/L", "EU_WFD", "aquatic_life", "Good ecological status upper limit for ammonia"),
    ],
    "dissolved_oxygen": [
        Threshold("dissolved_oxygen", 6.0, "mg/L", "EU_WFD", "aquatic_life", "Good ecological status minimum DO"),
    ],
    "pH": [
        Threshold("pH", 6.0, "pH units", "EU_WFD", "aquatic_life", "Lower limit for good ecological status"),
        Threshold("pH", 9.0, "pH units", "EU_WFD", "aquatic_life", "Upper limit for good ecological status"),
    ],
    "total_dissolved_solids": [
        Threshold(
            "total_dissolved_solids", 1000.0, "mg/L", "EU_WFD", "aquatic_life", "Guideline TDS for surface water"
        ),
    ],
    "turbidity": [
        Threshold("turbidity", 5.0, "NTU", "EU_WFD", "drinking", "Guideline turbidity for treated water"),
    ],
    "iron": [
        Threshold("iron", 0.2, "mg/L", "EU_WFD", "drinking", "Guideline iron for drinking-water abstraction"),
    ],
    "manganese": [
        Threshold(
            "manganese", 0.05, "mg/L", "EU_WFD", "drinking", "Guideline manganese for drinking-water abstraction"
        ),
    ],
    "fluoride": [
        Threshold("fluoride", 1.5, "mg/L", "EU_WFD", "drinking", "Drinking Water Directive limit for fluoride"),
    ],
    "lead": [
        Threshold("lead", 0.01, "mg/L", "EU_WFD", "drinking", "Drinking Water Directive limit for lead"),
    ],
    "copper": [
        Threshold("copper", 2.0, "mg/L", "EU_WFD", "drinking", "Drinking Water Directive limit for copper"),
    ],
}


# ---------------------------------------------------------------------------
# Combined registry
# ---------------------------------------------------------------------------

_STANDARD_MAP: dict[str, dict[str, list[Threshold]]] = {
    "WHO": WHO_THRESHOLDS,
    "EPA": EPA_THRESHOLDS,
    "EU_WFD": EU_WFD_THRESHOLDS,
}

ALL_THRESHOLDS: dict[str, list[Threshold]] = {}
for _std_thresholds in _STANDARD_MAP.values():
    for _param, _thresh_list in _std_thresholds.items():
        ALL_THRESHOLDS.setdefault(_param, []).extend(_thresh_list)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_thresholds(parameter: str, standard: str | None = None) -> list[Threshold]:
    """Return thresholds for a parameter, optionally filtered by standard.

    Parameters
    ----------
    parameter:
        Canonical parameter name (e.g. ``"nitrate"``).
    standard:
        If given, only return thresholds from this standard
        (``"WHO"``, ``"EPA"``, or ``"EU_WFD"``).

    Returns
    -------
    list[Threshold]
        Matching thresholds.  Empty list if none found.
    """
    if standard is not None:
        source = _STANDARD_MAP.get(standard, {})
        return list(source.get(parameter, []))
    return list(ALL_THRESHOLDS.get(parameter, []))


def list_parameters(standard: str | None = None) -> list[str]:
    """Return sorted list of parameter names with defined thresholds.

    Parameters
    ----------
    standard:
        If given, restrict to parameters from this standard.

    Returns
    -------
    list[str]
        Sorted parameter names.
    """
    if standard is not None:
        source = _STANDARD_MAP.get(standard, {})
        return sorted(source.keys())
    return sorted(ALL_THRESHOLDS.keys())


def list_standards() -> list[str]:
    """Return the available standard names.

    Returns
    -------
    list[str]
        ``["EPA", "EU_WFD", "WHO"]`` (alphabetically sorted).
    """
    return sorted(_STANDARD_MAP.keys())
