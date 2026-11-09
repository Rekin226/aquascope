"""Tests for AquaScope QGIS plugin helpers."""

from integrations.qgis.helpers import escape_expression, https_url


def test_escape_expression_doubles_apostrophes():
    assert escape_expression("king's river") == "king''s river"


def test_escape_expression_handles_empty_value():
    assert escape_expression("") == ""


def test_escape_expression_converts_non_string():
    assert escape_expression(123) == "123"


def test_https_url_accepts_valid_https_url():
    url = "https://example.com/station/123"
    assert https_url(url) == url


def test_https_url_rejects_http():
    assert https_url("http://example.com") == ""


def test_https_url_rejects_javascript():
    assert https_url("javascript:alert(1)") == ""


def test_https_url_rejects_missing_host():
    assert https_url("https:///station") == ""


def test_https_url_rejects_empty_value():
    assert https_url("") == ""


def test_https_url_trims_whitespace():
    assert https_url(" https://example.com ") == "https://example.com"
