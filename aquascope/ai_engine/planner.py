"""
Natural-language challenge planner — parses user descriptions into structured challenges.

Works entirely offline via keyword matching (no LLM required).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_CHALLENGE_KEYWORDS: dict[str, list[str]] = {
    "flood": [
        "flood", "flooding", "discharge", "streamflow", "river discharge",
        "inundation", "high water", "peak flow", "return period", "GEV",
    ],
    "drought": [
        "drought", "dry", "precipitation", "rainfall", "SPI",
        "water stress", "water scarcity", "desertification", "arid",
    ],
    "water_quality": [
        "quality", "contamination", "pollution", "pollutant", "turbidity",
        "nitrate", "phosphate", "dissolved oxygen", "pH", "heavy metal",
        "BOD", "COD", "E. coli", "coliform", "arsenic", "WHO",
    ],
}

_VARIABLE_KEYWORDS: dict[str, list[str]] = {
    "discharge": ["discharge", "streamflow", "flow", "runoff"],
    "water_level": ["water level", "stage", "gauge height"],
    "precipitation": ["precipitation", "rainfall", "rain"],
    "temperature": ["temperature", "temp"],
    "ph": ["pH"],
    "dissolved_oxygen": ["dissolved oxygen", "DO"],
    "turbidity": ["turbidity", "sediment"],
    "nitrate": ["nitrate", "nitrogen"],
}


@dataclass
class ChallengeSpec:
    """Structured representation of a parsed challenge request."""

    challenge_type: str = "unknown"
    variables: list[str] = field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    location_name: str | None = None
    forecast_days: int = 7
    raw_query: str = ""
    confidence: float = 0.0


class ChallengePlanner:
    """Parse natural-language descriptions into structured challenge specs.

    Example
    -------
    >>> planner = ChallengePlanner()
    >>> spec = planner.parse("Forecast flooding on the Niger River at lat 13.5, lon 2.1")
    >>> spec.challenge_type
    'flood'
    """

    def parse(self, query: str) -> ChallengeSpec:
        """Parse a natural-language query into a ``ChallengeSpec``.

        Parameters
        ----------
        query : str
            Free-text description of the challenge.

        Returns
        -------
        ChallengeSpec
            Structured challenge with type, variables, and location.
        """
        spec = ChallengeSpec(raw_query=query)
        q_lower = query.lower()

        # Detect challenge type
        type_scores: dict[str, int] = {}
        for ctype, keywords in _CHALLENGE_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw.lower() in q_lower)
            if hits:
                type_scores[ctype] = hits

        if type_scores:
            best = max(type_scores, key=type_scores.get)
            spec.challenge_type = best
            spec.confidence = min(type_scores[best] / 3.0, 1.0)

        # Detect variables
        for var, keywords in _VARIABLE_KEYWORDS.items():
            if any(kw.lower() in q_lower for kw in keywords):
                spec.variables.append(var)

        # Extract coordinates
        coord_patterns = [
            r"lat(?:itude)?\s*[=:]?\s*(-?\d+\.?\d*)\s*[,;]?\s*lon(?:gitude)?\s*[=:]?\s*(-?\d+\.?\d*)",
            r"(-?\d+\.?\d*)\s*°?\s*[NS]\s*[,;]?\s*(-?\d+\.?\d*)\s*°?\s*[EW]",
        ]
        for pattern in coord_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                spec.latitude = float(match.group(1))
                spec.longitude = float(match.group(2))
                break

        # Extract forecast horizon
        days_match = re.search(r"(\d+)\s*(?:day|days)", q_lower)
        if days_match:
            spec.forecast_days = int(days_match.group(1))

        # Extract location name (heuristic: capitalized phrases after "at/in/on/near")
        loc_match = re.search(r"(?:at|in|on|near|for)\s+(?:the\s+)?([A-Z][A-Za-z\s]+)", query)
        if loc_match:
            spec.location_name = loc_match.group(1).strip()

        logger.info(
            "ChallengePlanner: parsed '%s' → type=%s (%.0f%%), vars=%s",
            query[:60],
            spec.challenge_type,
            spec.confidence * 100,
            spec.variables,
        )
        return spec
