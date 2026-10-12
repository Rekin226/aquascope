"""Shared test fixtures."""

from __future__ import annotations

import logging
import os

import pytest


@pytest.fixture(autouse=True)
def _restore_environ():
    """Each test ends with the environment it started with.

    ``aquascope studio`` puts saved model keys into ``os.environ`` for the rest of the process
    (``keys.load_saved_keys``), which is right for a CLI run and wrong for the next test: a later test that
    expects no LLM key would find one. Restoring the whole environment covers that and any other direct write.
    """
    saved = dict(os.environ)
    yield
    if dict(os.environ) != saved:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture(autouse=True)
def _no_saved_keys(tmp_path_factory, monkeypatch):
    """No test reads the developer's own ``~/.config/aquascope/keys.json``.

    Without this, a test that runs ``aquascope studio`` loads a real saved key into the environment, so the
    suite behaves differently on a maintainer's machine than in CI. A test that wants a saved key sets
    ``AQUASCOPE_CONFIG_DIR`` itself.
    """
    monkeypatch.setenv("AQUASCOPE_CONFIG_DIR", str(tmp_path_factory.mktemp("aquascope-config")))


@pytest.fixture(autouse=True)
def _restore_logger_levels():
    """Each test ends with the logger levels it started with.

    The ``studio`` CLI verbs quiet ``aquascope``, ``aquascope.collectors`` and others to WARNING or ERROR for
    the rest of the process. In a test run that hides warnings from later ``caplog`` tests (the BOM catalog and
    USGS DEMO_KEY warnings went missing after any test that ran ``aquascope studio``).
    """
    manager = logging.Logger.manager

    def loggers() -> list[logging.Logger]:
        named = [lg for lg in list(manager.loggerDict.values()) if isinstance(lg, logging.Logger)]
        return [logging.getLogger(), *named]

    before = {lg: (lg.level, lg.disabled, lg.propagate) for lg in loggers()}
    disabled_below = manager.disable
    yield
    for lg in loggers():
        level, disabled, propagate = before.get(lg, (logging.NOTSET, False, True))
        if lg.level != level:
            lg.setLevel(level)  # setLevel also clears the per-logger level cache
        lg.disabled, lg.propagate = disabled, propagate
    if manager.disable != disabled_below:
        logging.disable(disabled_below)


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


@pytest.fixture(autouse=True)
def _no_geoglows_network(monkeypatch):
    """aquascope.rivers reads GEOGLOWS over HTTP; no test reaches it. The Scout's reach lookup then finds no
    network and leaves the reach out, and tests that want a reach patch these seams themselves."""
    from aquascope import rivers

    def offline(*args, **kwargs):
        raise RuntimeError("GEOGLOWS is not reachable in tests")

    for name in ("_fetch_range", "_fetch_json", "_fetch_text"):
        monkeypatch.setattr(rivers, name, offline)


@pytest.fixture(autouse=True)
def _no_dam_mirror_reads(monkeypatch):
    """The trace and the upstream-dams lookup read the Archive's Global Dam Watch mirror; tests never reach it.

    With this the mirror reads as not published. A test that wants dams patches ``river_path._dam_rows`` again.
    """
    from aquascope import river_path

    monkeypatch.setattr(river_path, "_dam_rows", lambda keys: None)
    river_path._COUNTRIES.clear()
    yield
    river_path._COUNTRIES.clear()


@pytest.fixture(autouse=True)
def _no_flood_history_reads(monkeypatch):
    """The Studio Scout reads the flood history for a flood question (#520); tests never reach the network for it.

    A test that wants a flood history patches ``scout._read_flood_history`` again with what it needs. The same
    goes for the dams upstream it reads for a flood or supply question (``scout._read_upstream_dams``).
    """
    from aquascope.studio.roles import scout

    monkeypatch.setattr(scout, "_read_flood_history", lambda lat, lon: {})
    monkeypatch.setattr(scout, "_read_upstream_dams", lambda lat, lon, river_id=None: {})
    yield


@pytest.fixture(autouse=True)
def _no_model_skill_reads(monkeypatch):
    """The Studio Scout reads the published model-skill table near the site (#518); tests never reach the network
    for it. A test that wants a model choice patches ``scout._read_model_skill`` itself."""
    from aquascope.studio.roles import scout

    monkeypatch.setattr(scout, "_read_model_skill", lambda lat, lon: {})
    yield


@pytest.fixture(autouse=True)
def _no_archive_listing(monkeypatch):
    """The daily jobs' publish lists its folder in the Archive to prune old dated files (#597); tests never reach
    the network for it. A test that wants a published folder patches ``retention.list_published`` itself."""
    from aquascope.archive import retention

    monkeypatch.setattr(retention, "list_published", lambda folder, repo_id, token=None: [])
    yield
