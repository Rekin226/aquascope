"""Domain-specific challenge workflows (flood, drought, water quality)."""

from aquascope.challenges.drought import DroughtChallenge
from aquascope.challenges.flood import FloodChallenge
from aquascope.challenges.quality import WaterQualityChallenge

__all__ = ["FloodChallenge", "DroughtChallenge", "WaterQualityChallenge"]
