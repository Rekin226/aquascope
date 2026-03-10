"""
Abstract base class for all data collectors.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from aquascope.utils.http_client import CachedHTTPClient

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """
    Every collector must implement ``fetch_raw`` and ``normalise``.

    The public entry-point is ``collect()`` which chains those two steps.
    """

    name: str = "base"

    def __init__(self, client: CachedHTTPClient | None = None):
        self.client = client or CachedHTTPClient()

    @abstractmethod
    def fetch_raw(self, **kwargs) -> Any:
        """Fetch raw data from the upstream API."""

    @abstractmethod
    def normalise(self, raw: Any) -> Sequence[BaseModel]:
        """Convert raw API response into unified Pydantic records."""

    def collect(self, **kwargs) -> Sequence[BaseModel]:
        """Fetch + normalise in one call."""
        logger.info("[%s] Starting collection …", self.name)
        raw = self.fetch_raw(**kwargs)
        records = self.normalise(raw)
        logger.info("[%s] Collected %d records.", self.name, len(records))
        return records
