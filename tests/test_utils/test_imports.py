"""Tests for aquascope.utils.imports.require()."""
from __future__ import annotations

import pytest

from aquascope.utils.imports import require


class TestRequire:
    """Tests for the require() lazy-import helper."""

    def test_require_existing_module(self):
        """Requiring a stdlib module returns the module object."""
        mod = require("json")
        assert hasattr(mod, "dumps")

    def test_require_missing_module(self):
        """Requiring a non-existent package raises ImportError with install hint."""
        with pytest.raises(ImportError, match="pip install"):
            require("nonexistent_pkg_xyz")

    def test_require_with_feature(self):
        """The feature name appears in the error message."""
        with pytest.raises(ImportError, match="my cool feature"):
            require("nonexistent_pkg_xyz", feature="my cool feature")
