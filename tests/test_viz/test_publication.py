"""The publication figure style: units, probability axes, corners, output."""

from __future__ import annotations

import math

import pytest

from aquascope.viz import publication as pub


@pytest.mark.parametrize("raw, want", [("m3/s", "m³/s"), ("m3 s-1", "m³/s"), ("km2", "km²"), ("deg C", "°C"),
                                       ("mm/d", "mm/d"), ("-", ""), (None, ""), ("cfs", "ft³/s")])
def test_unit_text(raw, want):
    assert pub.unit_text(raw) == want


def test_axis_label_sentence_cases_and_adds_the_unit():
    assert pub.axis_label("annual maximum discharge", "m3/s") == "Annual maximum discharge (m³/s)"
    assert pub.axis_label("aridity index", "-") == "Aridity index"


def test_gumbel_variate_orders_return_periods():
    ys = [float(pub.gumbel_variate(t)) for t in (2, 10, 100)]
    assert ys == sorted(ys)
    assert math.isclose(float(pub.gumbel_variate(2)), -math.log(-math.log(0.5)), rel_tol=1e-9)


def test_probit_is_symmetric_about_the_median():
    assert abs(float(pub.probit(50))) < 1e-9
    assert math.isclose(float(pub.probit(10)), -float(pub.probit(90)), rel_tol=1e-9)


def test_figures_are_serif_framed_and_sized_for_the_page():
    pytest.importorskip("matplotlib")
    import matplotlib

    fig, ax = pub.new_figure("full", 3.0)
    try:
        assert tuple(round(x, 2) for x in fig.get_size_inches()) == (6.3, 3.0)
        assert matplotlib.rcParams["font.family"] == ["serif"]
        assert matplotlib.rcParams["axes.spines.top"] and matplotlib.rcParams["xtick.direction"] == "in"
        ax.plot([0, 1, 2], [0, 1, 4])
        png = pub.png_bytes(fig)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        svg = pub.svg_bytes(fig)
        assert b"<svg" in svg
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_clear_corners_avoids_the_data():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    fig, ax = pub.new_figure("half", 2.5)
    try:
        xs, ys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # a rising diagonal
        ax.plot(xs, ys)
        corners = pub.clear_corners(ax, xs, ys, k=2)
        assert set(corners) == {"upper left", "lower right"}
    finally:
        plt.close(fig)


def test_importing_the_style_does_not_import_matplotlib():
    import subprocess
    import sys

    code = "import sys, aquascope.viz.publication; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
