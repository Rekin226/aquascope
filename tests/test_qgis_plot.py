"""Tests for the AquaScope QGIS observation time-series plot."""

import importlib.util
import sys
import types
from datetime import date

import pytest


@pytest.fixture
def plot_module(monkeypatch):
    """Load the plot module with lightweight Qt mocks."""
    qgis = types.ModuleType("qgis")
    pyqt = types.ModuleType("qgis.PyQt")
    core = types.ModuleType("qgis.PyQt.QtCore")
    gui = types.ModuleType("qgis.PyQt.QtGui")
    widgets = types.ModuleType("qgis.PyQt.QtWidgets")

    class FakeWidget:
        def __init__(self, parent=None):
            self.minimum_height = None
            self.updated = False

        def setMinimumHeight(self, height):  # noqa: N802
            self.minimum_height = height

        def update(self):
            self.updated = True

    class FakeQt:
        pass

    class FakeQtClass:
        pass

    core.QPointF = FakeQtClass
    core.QRectF = FakeQtClass
    core.Qt = FakeQt
    gui.QColor = FakeQtClass
    gui.QFont = FakeQtClass
    gui.QPainter = FakeQtClass
    gui.QPainterPath = FakeQtClass
    gui.QPen = FakeQtClass
    widgets.QWidget = FakeWidget

    qgis.PyQt = pyqt
    pyqt.QtCore = core
    pyqt.QtGui = gui
    pyqt.QtWidgets = widgets

    monkeypatch.setitem(sys.modules, "qgis", qgis)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", pyqt)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", core)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtGui", gui)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtWidgets", widgets)

    module_name = "aquascope_test_plot"
    module_path = (
        "integrations/qgis/aquascope_gauges/plot.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_plot_starts_empty(plot_module):
    plot = plot_module.ObservationPlot()

    assert plot.records == []
    assert plot.title == "Daily observations"
    assert plot.unit == ""
    assert plot.minimum_height == 260


def test_set_data_sorts_records_by_date(plot_module):
    plot = plot_module.ObservationPlot()
    records = [
        {"date": date(2026, 1, 3), "value": 30.0},
        {"date": date(2026, 1, 1), "value": 10.0},
        {"date": date(2026, 1, 2), "value": 20.0},
    ]

    plot.set_data(records, title="Discharge", unit="m3/s")

    assert [record["date"] for record in plot.records] == [
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
    ]
    assert plot.title == "Discharge"
    assert plot.unit == "m3/s"
    assert plot.updated is True


def test_set_data_supports_single_record(plot_module):
    plot = plot_module.ObservationPlot()
    records = [{"date": date(2026, 1, 1), "value": 5.0}]

    plot.set_data(records)

    assert len(plot.records) == 1
    assert plot.records[0]["value"] == 5.0


def test_set_data_supports_constant_values(plot_module):
    plot = plot_module.ObservationPlot()
    records = [
        {"date": date(2026, 1, 1), "value": 5.0},
        {"date": date(2026, 1, 2), "value": 5.0},
    ]

    plot.set_data(records)

    assert len(plot.records) == 2
    assert all(record["value"] == 5.0 for record in plot.records)


def test_clear_removes_records_and_requests_repaint(plot_module):
    plot = plot_module.ObservationPlot()
    plot.set_data(
        [{"date": date(2026, 1, 1), "value": 5.0}]
    )
    plot.updated = False

    plot.clear()

    assert plot.records == []
    assert plot.updated is True
