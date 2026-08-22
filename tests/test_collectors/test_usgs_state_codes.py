"""Offline checks for the NWIS state/territory sweep codes (issue #239)."""

from aquascope.collectors.usgs import NWIS_STATE_CODES


def test_american_samoa_uses_fips_alpha_not_postal():
    # NWIS uses the FIPS 5-1 alpha code "aq" for American Samoa. The postal
    # "as" is rejected by the site service (HTTP 400), so it must not be swept.
    assert "aq" in NWIS_STATE_CODES
    assert "as" not in NWIS_STATE_CODES


def test_pacific_territories_present():
    for code in ("fm", "mh", "pw"):
        assert code in NWIS_STATE_CODES


def test_codes_are_unique():
    assert len(NWIS_STATE_CODES) == len(set(NWIS_STATE_CODES))
