"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fresh_usgs_area_cache():
    """The USGS drainage-area cache is process-wide (one lookup per station per run); tests start empty."""
    from aquascope.collectors.usgs import USGSCollector

    USGSCollector._shared_area_cache.clear()
    yield
    USGSCollector._shared_area_cache.clear()


@pytest.fixture(autouse=True)
def _fresh_imgw_frames():
    """The IMGW parsed-archive memo is process-wide (one parse per file per run); tests start empty."""
    from aquascope.collectors.poland_imgw import PolandIMGWCollector

    PolandIMGWCollector._shared_frames.clear()
    yield
    PolandIMGWCollector._shared_frames.clear()
