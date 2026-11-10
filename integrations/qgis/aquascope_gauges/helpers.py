"""Pure helper functions for the AquaScope QGIS plugin."""

from urllib.parse import urlsplit


def escape_expression(value):
    """Escape a value for use in a QGIS expression string literal."""
    return str(value).replace("'", "''")


def https_url(value):
    """Return a valid HTTPS URL, or an empty string."""
    value = str(value or "").strip()

    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""

    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return ""

    return value
