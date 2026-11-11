"""QGIS plugin entry point for AquaScope Gauges."""


def classFactory(iface):  # noqa: N802
    """Create the plugin instance required by QGIS."""
    from .plugin import AquaScopePlugin

    return AquaScopePlugin(iface)
