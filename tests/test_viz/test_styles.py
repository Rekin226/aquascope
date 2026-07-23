"""Tests for aquascope.viz.styles — palette selection."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from aquascope.viz.styles import (  # noqa: E402
    OKABE_ITO_COLOURS,
    SERIES_COLOURS,
    apply_aqua_style,
)


class TestApplyAquaStylePalette:
    """Verify that apply_aqua_style sets the correct prop_cycle."""

    @staticmethod
    def _get_cycle_colours() -> list[str]:
        """Return the current axes.prop_cycle colour list."""
        return [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

    def test_default_palette_uses_series_colours(self):
        apply_aqua_style()  # explicit default
        assert self._get_cycle_colours() == SERIES_COLOURS

    def test_default_keyword_matches_implicit(self):
        apply_aqua_style(palette="default")
        assert self._get_cycle_colours() == SERIES_COLOURS

    def test_colorblind_palette_uses_okabe_ito(self):
        apply_aqua_style(palette="colorblind")
        assert self._get_cycle_colours() == OKABE_ITO_COLOURS

    def test_invalid_palette_raises(self):
        with pytest.raises(ValueError, match="Unknown palette"):
            apply_aqua_style(palette="neon")

    def test_switching_palettes(self):
        """Calling with a different palette actually changes the cycle."""
        apply_aqua_style(palette="default")
        default = self._get_cycle_colours()

        apply_aqua_style(palette="colorblind")
        colorblind = self._get_cycle_colours()

        assert default != colorblind
        assert default == SERIES_COLOURS
        assert colorblind == OKABE_ITO_COLOURS
