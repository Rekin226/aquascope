"""The water-quality parameter vocabulary and its units, shared by the WHO screen and the indices (#62).

Agencies label the same quantity many ways (``DO``, ``Dissolved oxygen (DO)``,
``00300``) and report it in several units (``mg/l``, ``ug/L``, ``% saturation``).
:func:`resolve_parameter` maps a reported name to one canonical key and
:func:`convert_value` takes a value to that key's canonical unit (mg/L, uS/cm,
deg C, NTU, pH units, CFU/100mL), or says that it cannot. No pandas here: the
module is light enough for the workbench to import at load time.
"""

from __future__ import annotations

import math
import re
from typing import Any

__all__ = [
    "CANONICAL_UNITS",
    "EQUIVALENT_WEIGHTS",
    "convert_value",
    "normalise_unit",
    "resolve_parameter",
]

# ── parameters and units ────────────────────────────────────────────────────

#: The unit every value of a parameter is converted to before comparison.
CANONICAL_UNITS: dict[str, str] = {
    "ph": "pH units",
    "dissolved_oxygen": "mg/L",
    "dissolved_oxygen_saturation": "%",
    "temperature": "deg C",
    "temperature_change": "deg C",
    "conductivity": "uS/cm",
    "turbidity": "NTU",
    "tds": "mg/L",
    "total_solids": "mg/L",
    "suspended_solids": "mg/L",
    "bod": "mg/L",
    "cod": "mg/L",
    "nitrate": "mg/L",           # as NO3
    "nitrite": "mg/L",           # as NO2
    "ammonia": "mg/L",           # as N
    "total_nitrogen": "mg/L",
    "total_phosphorus": "mg/L",  # as P
    "phosphate": "mg/L",         # as PO4
    "e_coli": "CFU/100mL",
    "fecal_coliform": "CFU/100mL",
    "total_coliform": "CFU/100mL",
    "fluoride": "mg/L",
    "chloride": "mg/L",
    "sulfate": "mg/L",
    "sodium": "mg/L",
    "potassium": "mg/L",
    "calcium": "mg/L",
    "magnesium": "mg/L",
    "bicarbonate": "mg/L",
    "carbonate": "mg/L",
    "alkalinity": "mg/L",        # as CaCO3
    "hardness": "mg/L",          # as CaCO3
    "boron": "mg/L",
    "arsenic": "mg/L",
    "lead": "mg/L",
    "mercury": "mg/L",
    "cadmium": "mg/L",
    "chromium": "mg/L",
    "copper": "mg/L",
    "zinc": "mg/L",
    "iron": "mg/L",
    "manganese": "mg/L",
    "nickel": "mg/L",
    "selenium": "mg/L",
    "uranium": "mg/L",
    "antimony": "mg/L",
    "barium": "mg/L",
}

#: Exact spellings (lower-cased) of a parameter, with the factor that takes the
#: reported value to the canonical form (nitrate reported as N becomes NO3).
_ALIASES: dict[str, tuple[str, float]] = {
    "ph": ("ph", 1.0), "00400": ("ph", 1.0),
    "do": ("dissolved_oxygen", 1.0), "dissolved oxygen": ("dissolved_oxygen", 1.0),
    "dissolved oxygen (do)": ("dissolved_oxygen", 1.0), "dissolved_oxygen": ("dissolved_oxygen", 1.0),
    "oxygen": ("dissolved_oxygen", 1.0), "00300": ("dissolved_oxygen", 1.0),
    "dissolved oxygen saturation": ("dissolved_oxygen_saturation", 1.0),
    "dissolved oxygen (do), percent saturation": ("dissolved_oxygen_saturation", 1.0),
    "do saturation": ("dissolved_oxygen_saturation", 1.0), "do%": ("dissolved_oxygen_saturation", 1.0),
    "temperature": ("temperature", 1.0), "temperature, water": ("temperature", 1.0),
    "water temperature": ("temperature", 1.0), "temp": ("temperature", 1.0), "00010": ("temperature", 1.0),
    "temperature change": ("temperature_change", 1.0), "delta t": ("temperature_change", 1.0),
    "conductivity": ("conductivity", 1.0), "specific conductance": ("conductivity", 1.0),
    "specific conductivity": ("conductivity", 1.0), "electrical conductivity": ("conductivity", 1.0),
    "ec": ("conductivity", 1.0), "00095": ("conductivity", 1.0),
    "turbidity": ("turbidity", 1.0),
    "tds": ("tds", 1.0), "total dissolved solids": ("tds", 1.0), "dissolved solids": ("tds", 1.0),
    "total solids": ("total_solids", 1.0), "ts": ("total_solids", 1.0),
    "ss": ("suspended_solids", 1.0), "tss": ("suspended_solids", 1.0),
    "suspended solids": ("suspended_solids", 1.0), "total suspended solids": ("suspended_solids", 1.0),
    "bod": ("bod", 1.0), "bod5": ("bod", 1.0), "biochemical oxygen demand": ("bod", 1.0),
    "cod": ("cod", 1.0), "chemical oxygen demand": ("cod", 1.0),
    "nitrate": ("nitrate", 1.0), "no3": ("nitrate", 1.0),
    "nitrate-n": ("nitrate", 4.4268), "no3-n": ("nitrate", 4.4268), "nitrate nitrogen": ("nitrate", 4.4268),
    "nitrate as n": ("nitrate", 4.4268),
    "nitrite": ("nitrite", 1.0), "no2": ("nitrite", 1.0), "nitrite-n": ("nitrite", 3.2845),
    "no2-n": ("nitrite", 3.2845),
    "ammonia": ("ammonia", 1.0), "nh3-n": ("ammonia", 1.0), "ammonia-n": ("ammonia", 1.0),
    "ammonia nitrogen": ("ammonia", 1.0), "nh3": ("ammonia", 0.8224), "ammonium": ("ammonia", 0.7765),
    "nh4": ("ammonia", 0.7765),
    "tn": ("total_nitrogen", 1.0), "total nitrogen": ("total_nitrogen", 1.0),
    "tp": ("total_phosphorus", 1.0), "total phosphorus": ("total_phosphorus", 1.0),
    "phosphorus": ("total_phosphorus", 1.0),
    "phosphate": ("phosphate", 1.0), "total phosphate": ("phosphate", 1.0), "po4": ("phosphate", 1.0),
    "orthophosphate": ("phosphate", 1.0),
    "e. coli": ("e_coli", 1.0), "e coli": ("e_coli", 1.0), "e_coli": ("e_coli", 1.0),
    "escherichia coli": ("e_coli", 1.0),
    "fecal coliform": ("fecal_coliform", 1.0), "faecal coliform": ("fecal_coliform", 1.0),
    "fecal coliforms": ("fecal_coliform", 1.0), "fc": ("fecal_coliform", 1.0),
    "total coliform": ("total_coliform", 1.0), "total coliforms": ("total_coliform", 1.0),
    "hco3": ("bicarbonate", 1.0), "co3": ("carbonate", 1.0), "alkalinity": ("alkalinity", 1.0),
    "alkalinity, total": ("alkalinity", 1.0), "00410": ("alkalinity", 1.0),
    "hardness": ("hardness", 1.0), "total hardness": ("hardness", 1.0),
    "na": ("sodium", 1.0), "k": ("potassium", 1.0), "ca": ("calcium", 1.0), "mg": ("magnesium", 1.0),
    "cl": ("chloride", 1.0), "so4": ("sulfate", 1.0), "sulphate": ("sulfate", 1.0), "b": ("boron", 1.0),
    "as": ("arsenic", 1.0), "pb": ("lead", 1.0), "hg": ("mercury", 1.0), "cd": ("cadmium", 1.0),
    "cr": ("chromium", 1.0), "cu": ("copper", 1.0), "zn": ("zinc", 1.0), "fe": ("iron", 1.0),
    "mn": ("manganese", 1.0), "ni": ("nickel", 1.0), "se": ("selenium", 1.0), "u": ("uranium", 1.0),
    "sb": ("antimony", 1.0), "ba": ("barium", 1.0), "f": ("fluoride", 1.0),
}
for _name in CANONICAL_UNITS:
    _ALIASES.setdefault(_name, (_name, 1.0))
    _ALIASES.setdefault(_name.replace("_", " "), (_name, 1.0))

#: Substring rules, tried in order after the exact aliases.
_ALIAS_RULES: tuple[tuple[str, str], ...] = (
    (r"escherichia|e\.?\s*coli", "e_coli"),
    (r"f(a)?ecal.*coli", "fecal_coliform"),
    (r"coliform", "total_coliform"),
    (r"saturation", "dissolved_oxygen_saturation"),
    (r"dissolved oxygen|\boxygen\b", "dissolved_oxygen"),
    (r"conductance|conductivity", "conductivity"),
    (r"temperature", "temperature"),
    (r"turbid", "turbidity"),
    (r"nitrate", "nitrate"),
    (r"nitrite", "nitrite"),
    (r"ammoni", "ammonia"),
    (r"total nitrogen|\bnitrogen\b", "total_nitrogen"),
    (r"phosphate", "phosphate"),
    (r"phosphorus", "total_phosphorus"),
    (r"biochemical|\bbod", "bod"),
    (r"chemical oxygen|\bcod\b", "cod"),
    (r"dissolved solids", "tds"),
    (r"suspended", "suspended_solids"),
    (r"total solids", "total_solids"),
    (r"bicarbonate", "bicarbonate"),
    (r"carbonate", "carbonate"),
    (r"alkalinity", "alkalinity"),
    (r"hardness", "hardness"),
    (r"chloride", "chloride"), (r"sulph?ate", "sulfate"), (r"fluoride", "fluoride"),
    (r"sodium", "sodium"), (r"potassium", "potassium"), (r"calcium", "calcium"), (r"magnesium", "magnesium"),
    (r"boron", "boron"), (r"arsenic", "arsenic"), (r"\blead\b", "lead"), (r"mercury", "mercury"),
    (r"cadmium", "cadmium"), (r"chromium", "chromium"), (r"copper", "copper"), (r"zinc", "zinc"),
    (r"\biron\b", "iron"), (r"manganese", "manganese"), (r"nickel", "nickel"), (r"selenium", "selenium"),
    (r"uranium", "uranium"), (r"antimony", "antimony"), (r"barium", "barium"),
    (r"\bph\b", "ph"),
)

#: mg/L as N to mg/L as the ion, for units that say "as N".
_N_TO_ION = {"nitrate": 4.4268, "nitrite": 3.2845}

#: Equivalent weights (mg per meq) for the ions the irrigation index uses.
EQUIVALENT_WEIGHTS: dict[str, float] = {
    "sodium": 22.99, "potassium": 39.10, "calcium": 20.04, "magnesium": 12.15,
    "bicarbonate": 61.02, "carbonate": 30.00, "chloride": 35.45, "sulfate": 48.03,
}

_MASS_FACTORS = {
    "mg/l": 1.0, "mg/L": 1.0, "ppm": 1.0, "mgl": 1.0, "mg l-1": 1.0, "mg/l as caco3": 1.0, "mg/l as p": 1.0,
    "mg/l as po4": 1.0, "mg/l as no3": 1.0, "mg/l as no2": 1.0, "mg/l as nh3": 1.0,
    "ug/l": 0.001, "µg/l": 0.001, "μg/l": 0.001, "ppb": 0.001, "ug l-1": 0.001, "ug/l as n": 0.001,
    "ng/l": 1e-6, "g/l": 1000.0, "g/m3": 1.0, "mg/dm3": 1.0,
}
_CONDUCTIVITY_FACTORS = {
    "us/cm": 1.0, "µs/cm": 1.0, "μs/cm": 1.0, "umho/cm": 1.0, "µmho/cm": 1.0, "us/cm @25c": 1.0,
    "us/cm @ 25c": 1.0, "us/cm at 25c": 1.0, "us/cm @25 c": 1.0, "usiemens/cm": 1.0,
    "ms/cm": 1000.0, "ds/m": 1000.0, "mmho/cm": 1000.0, "s/m": 10_000.0,
}
_TURBIDITY_UNITS = {"ntu", "fnu", "ntru", "fnru", "jtu"}
_PH_UNITS = {"", "ph", "ph units", "std units", "su", "none", "unitless", "standard units", "std unit", "-"}
_BACTERIA_UNITS = {"cfu/100ml", "mpn/100ml", "#/100ml", "#/dl", "/100ml", "col/100ml", "cfu/100 ml",
                   "mpn/100 ml", "count/100ml", "cfu per 100 ml", "no/100ml", "cfu/dl", "mpn/dl"}
_PERCENT_UNITS = {"%", "% saturation", "percent", "% sat", "%sat", "percent saturation"}


def normalise_unit(unit: Any) -> str:
    u = "" if unit is None or (isinstance(unit, float) and math.isnan(unit)) else str(unit)
    u = u.strip().lower().replace("μ", "u").replace("µ", "u").replace("°", "deg ")
    u = re.sub(r"\s+", " ", u)
    return u.replace("deg  ", "deg ")


def resolve_parameter(name: Any, unit: Any = None) -> tuple[str | None, float]:
    """The canonical parameter key a reported name stands for, and the factor to its canonical form.

    ``("nitrate", 4.4268)`` for ``"Nitrate"`` reported ``"mg/l as N"``;
    ``(None, 1.0)`` when the name is not recognised.
    """
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return None, 1.0
    raw = re.sub(r"\s+", " ", str(name).strip().lower())
    key, factor = _ALIASES.get(raw, (None, 1.0))
    if key is None:
        for pattern, canon in _ALIAS_RULES:
            if re.search(pattern, raw):
                key = canon
                break
    if key is None:
        return None, 1.0
    u = normalise_unit(unit)
    if key in _N_TO_ION and factor == 1.0 and (re.search(r"\bas n\b", u) or re.search(r"\bas n\b|-n\b", raw)):
        factor = _N_TO_ION[key]
    if key == "dissolved_oxygen" and u in _PERCENT_UNITS:
        key = "dissolved_oxygen_saturation"
    return key, factor


def convert_value(key: str, value: float, unit: Any) -> tuple[float | None, str | None]:
    """``(value in the canonical unit, note)``; ``(None, reason)`` when the unit cannot be converted."""
    u = normalise_unit(unit)
    canon = CANONICAL_UNITS[key]
    if key == "ph":
        return (value, None) if u in _PH_UNITS or "ph" in u else (None, u)
    if key in ("temperature", "temperature_change"):
        if u in ("", "deg c", "degc", "c", "celsius", "deg. c", "degrees c", "°c"):
            return value, ("unit assumed deg C" if u == "" else None)
        if u in ("deg f", "degf", "f", "fahrenheit", "degrees f"):
            return (value - 32.0) * 5.0 / 9.0 if key == "temperature" else value * 5.0 / 9.0, None
        return None, u
    if key == "conductivity":
        if u == "":
            return value, "unit assumed uS/cm"
        f = _CONDUCTIVITY_FACTORS.get(u)
        if f is None and u.startswith("us/cm"):
            f = 1.0
        return (value * f, None) if f is not None else (None, u)
    if key == "turbidity":
        if u == "":
            return value, "unit assumed NTU"
        return (value, None) if u in _TURBIDITY_UNITS else (None, u)
    if key == "dissolved_oxygen_saturation":
        return (value, None) if u in _PERCENT_UNITS or u == "" else (None, u)
    if key in ("e_coli", "fecal_coliform", "total_coliform"):
        if u == "":
            return value, "unit assumed CFU/100mL"
        return (value, None) if u in _BACTERIA_UNITS or "100" in u else (None, u)
    # mass concentrations
    if u == "":
        return value, f"unit assumed {canon}"
    if u in _MASS_FACTORS:
        return value * _MASS_FACTORS[u], None
    base = re.sub(r"\s+as\s+\w+.*$", "", u).strip()
    if base in _MASS_FACTORS:
        return value * _MASS_FACTORS[base], None
    if u.startswith("meq/l") and key in EQUIVALENT_WEIGHTS:
        return value * EQUIVALENT_WEIGHTS[key], None
    if u.startswith("meq/l") and key == "alkalinity":
        return value * 50.04, None
    return None, u
