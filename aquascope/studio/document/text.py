"""How a report writes numbers, units, places and lists.

Three significant figures for an estimate, thousands separators above 1,000,
a p-value to two decimals (``< 0.001`` below that), units as a reader writes
them (``m³/s``, ``km²``, ``°C``), and a list joined with "and", never a run of
semicolons. Every function takes whatever a payload holds and never raises.
"""

from __future__ import annotations

import math
import re
from typing import Any

from aquascope.viz.publication import unit_text

__all__ = ["coord", "join", "num", "p_value", "pct", "plural", "sentence", "unit", "value", "years"]


def num(x: Any, sig: int = 3) -> str:
    """``463.9`` -> ``464``, ``41.21`` -> ``41.2``, ``0.06361`` -> ``0.0636``, ``2320.4`` -> ``2,320``."""
    if x is None or isinstance(x, bool):
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "n/a"
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{round(v):,}"
    digits = sig - int(math.floor(math.log10(abs(v)))) - 1
    rounded = round(v, digits)
    if digits <= 0:
        return f"{int(rounded):,}"
    text = f"{rounded:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def unit(u: Any) -> str:
    return unit_text(str(u)) if u else ""


def value(x: Any, u: Any = None, sig: int = 3) -> str:
    """``464 m³/s``; a percentage unit hugs the number (``43 %`` stays spaced, as SI writes it)."""
    s = num(x, sig)
    uu = unit(u)
    if not uu or s == "n/a":
        return s
    if uu in ("years", "year", "yr") and s == "1":
        uu = "year"
    return f"{s} {uu}"


def span(lo: Any, hi: Any, sig: int = 3) -> str:
    """``405–490``: a range in a table cell (an en dash, the typographic range sign)."""
    if lo is None or hi is None:
        return ""
    return f"{num(lo, sig)}\u2013{num(hi, sig)}"


def assumption(text: Any) -> str:
    """A condition the Interpreter states as a fact ("The annual maxima are independent ...") as the assumption
    it is ("The estimate assumes that the annual maxima are independent ...")."""
    t = sentence(text)
    if re.match(r"^(The|Annual|Daily|Flows?|Peaks?)\b.*\b(are|is)\b", t) and not re.match(
            r"^The (estimate|answer|result) ", t) and not re.search(r"\b(can|may|could)\b", t.split(",")[0]):
        return "The estimate assumes that " + t[0].lower() + t[1:]
    return t


def band(lo: Any, hi: Any, u: Any = None, sig: int = 3) -> str:
    """``405 to 490 m³/s``."""
    if lo is None or hi is None:
        return ""
    uu = unit(u)
    return f"{num(lo, sig)} to {num(hi, sig)}" + (f" {uu}" if uu else "")


def p_value(p: Any) -> str:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if v < 0.001:
        return "< 0.001"
    return f"{v:.2f}" if v >= 0.01 else f"{v:.3f}"


def pct(x: Any, digits: int = 0) -> str:
    try:
        return f"{float(x) * 100:.{digits}f} %"
    except (TypeError, ValueError):
        return "n/a"


def years(x: Any) -> str:
    """``123 years`` from 123.2; ``1 year``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    n = int(round(v))
    return f"{n} year" if n == 1 else f"{n} years"


def coord(lat: Any, lon: Any) -> str:
    """``47.2375° N, 68.5828° W``."""
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return ""
    return f"{abs(la):.4f}° {'N' if la >= 0 else 'S'}, {abs(lo):.4f}° {'E' if lo >= 0 else 'W'}"


def join(items: list[str], word: str = "and") -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    items = [i for i in (str(x).strip() for x in items) if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


_SPACE = re.compile(r"\s+")
_DOUBLE_STOP = re.compile(r"\.\s*\.(?!\.)")


def sentence(text: Any) -> str:
    """One clean sentence: single spaces, a capital first letter, exactly one full stop, units as a reader
    writes them (``m3/s`` -> ``m³/s``)."""
    t = _SPACE.sub(" ", str(text or "")).strip()
    if not t:
        return ""
    t = re.sub(r"\b(m3/s|km2|m3|deg C)\b", lambda m: unit_text(m.group(1)), t)
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?":
        t += "."
    return _DOUBLE_STOP.sub(".", t)


def clean_units(text: Any) -> str:
    """Unit spellings fixed inside running text, nothing else touched."""
    return re.sub(r"(?<![A-Za-z])(m3/s|km2|m3|deg C)(?![A-Za-z0-9])", lambda m: unit_text(m.group(1)),
                  str(text or ""))
