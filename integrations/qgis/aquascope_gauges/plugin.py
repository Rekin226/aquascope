"""AquaScope Gauges QGIS plugin."""

from html import escape
from urllib.parse import quote

from qgis.core import (
    Qgis,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateTransform,
    QgsFeatureRequest,
    QgsMarkerSymbol,
    QgsProject,
    QgsRendererCategory,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .helpers import escape_expression, https_url

BASE_URL = "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/"

EXPLORER_URL = "https://rekin226-aquascope-explorer.static.hf.space/"

MAX_RESULTS = 100

SOURCE_STYLES = (
    ("usgs", "USGS (US)", "21,101,192,255"),
    ("uk_ea", "Environment Agency (UK)", "46,125,50,255"),
    ("hubeau_hydrometrie", "Hub'Eau (FR)", "198,40,40,255"),
    ("pegelonline", "PEGELONLINE (DE)", "239,108,0,255"),
    ("ireland_opw", "OPW (IE)", "106,27,154,255"),
    ("taiwan_cwa", "CWA (TW)", "0,131,143,255"),
    ("", "other", "84,110,122,255"),
)


class AquaScopePlugin:
    """AquaScope station catalog browser."""

    def __init__(self, iface):
        self.iface = iface
        self.layer = None
        self.dock = None
        self.search_input = None
        self.extent_checkbox = None
        self.station_list = None
        self.details = None
        self.features = []

    def initGui(self):  # noqa: N802
        """Create the plugin interface."""
        self.dock = QDockWidget(
            "AquaScope Gauges",
            self.iface.mainWindow(),
        )
        self.dock.setObjectName("AquaScopeGaugesDock")

        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Search the remote AquaScope station catalog. Data is accessed over HTTPS."))

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Station name, river, agency, or station ID")
        self.search_input.returnPressed.connect(self.search_stations)
        layout.addWidget(self.search_input)

        self.extent_checkbox = QCheckBox("Limit results to current map extent")
        layout.addWidget(self.extent_checkbox)

        search_button = QPushButton("Search stations")
        search_button.clicked.connect(self.search_stations)
        layout.addWidget(search_button)

        self.station_list = QListWidget()
        self.station_list.currentRowChanged.connect(self.show_station_details)
        layout.addWidget(self.station_list)

        layout.addWidget(QLabel("Station details"))

        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(True)
        layout.addWidget(self.details)

        self.dock.setWidget(widget)

        self.iface.addDockWidget(
            Qt.RightDockWidgetArea,
            self.dock,
        )

        self.iface.addPluginToMenu(
            "&AquaScope Gauges",
            self.dock.toggleViewAction(),
        )

        self.load_catalog()

    def unload(self):
        """Remove the plugin interface without removing the map layer."""
        if self.dock is None:
            return

        self.iface.removePluginMenu(
            "&AquaScope Gauges",
            self.dock.toggleViewAction(),
        )

        self.iface.removeDockWidget(self.dock)
        self.dock.deleteLater()
        self.dock = None

    def _style_catalog(self, layer):
        """Apply the categorized styling used by the AquaScope QLR."""
        categories = []

        for source, label, color in SOURCE_STYLES:
            symbol = QgsMarkerSymbol.createSimple(
                {
                    "name": "circle",
                    "color": color,
                    "size": "1.8",
                    "outline_color": "255,255,255,255",
                    "outline_width": "0.2",
                }
            )

            categories.append(
                QgsRendererCategory(
                    source,
                    symbol,
                    label,
                )
            )

        renderer = QgsCategorizedSymbolRenderer(
            "source",
            categories,
        )

        layer.setRenderer(renderer)
        layer.setCustomProperty("aquascope/catalog", True)

        layer.setMapTipTemplate(
            """
            <div style="font-family: sans-serif; min-width: 220px;">
                <h3>[% coalesce("name", 'Unnamed station') %]</h3>
                <b>Station ID:</b> [% coalesce("station_id", '') %]<br>
                <b>Source:</b> [% coalesce("source", '') %]<br>
                <b>Agency:</b> [% coalesce("agency", '') %]<br>
                <b>River:</b> [% coalesce("river", '') %]<br>
                <b>Country:</b> [% coalesce("country", '') %]<br>
                <b>Observation period:</b>
                [% coalesce("period_start", '') %] –
                [% coalesce("period_end", '') %]<br>

                [% if(
                    left(lower(trim(coalesce("url", ''))), 8) = 'https://',
                    '<a href="' || "url" || '">Open agency station page</a><br>',
                    ''
                ) %]

                [% if(
                    coalesce("source", '') != '' AND
                    coalesce("station_id", '') != '',
                    '<a href="https://rekin226-aquascope-explorer.static.hf.space/#s=' ||
                    "source" || '/' || "station_id" ||
                    '">Open in AquaScope Explorer</a>',
                    ''
                ) %]
            </div>
            """
        )

    def load_catalog(self):
        """Load the remote catalog, falling back to GeoJSON."""
        project = QgsProject.instance()

        if self.layer and self.layer.isValid() and project.mapLayer(self.layer.id()) is not None:
            return True

        # Reuse a catalog layer already added by this plugin.
        for existing_layer in project.mapLayers().values():
            if (
                existing_layer.customProperty(
                    "aquascope/catalog",
                    False,
                )
                and existing_layer.isValid()
            ):
                self.layer = existing_layer
                return True

        parquet_supported = False

        try:
            from osgeo import gdal

            parquet_supported = gdal.GetDriverByName("Parquet") is not None
        except ImportError:
            pass

        sources = []

        if parquet_supported:
            sources.append(
                (
                    "/vsicurl/" + BASE_URL + "stations.parquet" + "|layername=stations",
                    "Parquet",
                )
            )

        sources.append(
            (
                "/vsicurl/" + BASE_URL + "stations.geojson",
                "GeoJSON",
            )
        )

        for uri, source_name in sources:
            candidate = QgsVectorLayer(
                uri,
                "AquaScope stations",
                "ogr",
            )

            if not candidate.isValid():
                continue

            self._style_catalog(candidate)

            project.addMapLayer(candidate)
            self.layer = candidate

            self.iface.messageBar().pushMessage(
                "AquaScope",
                f"Station catalog opened using {source_name}.",
                level=Qgis.Info,
                duration=5,
            )

            return True

        QMessageBox.warning(
            self.iface.mainWindow(),
            "AquaScope Gauges",
            "Could not open the remote station catalog.\n\n"
            "Check your internet connection and confirm that "
            "your QGIS/GDAL installation supports the catalog "
            "formats.",
        )

        return False

    def search_stations(self):
        """Search the catalog and optionally filter by map extent."""
        if not self.load_catalog():
            return

        query = self.search_input.text().strip()
        request = QgsFeatureRequest()

        if query:
            value = escape_expression(query.lower())

            fields = (
                "name",
                "river",
                "agency",
                "source",
                "station_id",
            )

            available_fields = {field.name() for field in self.layer.fields()}

            clauses = []

            for field in fields:
                if field in available_fields:
                    clauses.append(f"""strpos(lower(coalesce("{field}", '')), '{value}') > 0""")

            if clauses:
                request.setFilterExpression(" OR ".join(clauses))

        if self.extent_checkbox.isChecked():
            canvas = self.iface.mapCanvas()
            extent = canvas.extent()

            transform = QgsCoordinateTransform(
                canvas.mapSettings().destinationCrs(),
                self.layer.crs(),
                QgsProject.instance(),
            )

            try:
                extent = transform.transformBoundingBox(extent)
                request.setFilterRect(extent)

            except Exception as exc:
                QMessageBox.warning(
                    self.iface.mainWindow(),
                    "AquaScope Gauges",
                    f"Could not transform the map extent: {exc}",
                )
                return

        request.setLimit(MAX_RESULTS)

        self.features = list(self.layer.getFeatures(request))

        self.station_list.clear()
        self.details.clear()

        available_fields = set(self.layer.fields().names())

        for feature in self.features:
            name = feature["name"] if "name" in available_fields and feature["name"] else "Unnamed station"
            source = feature["source"] if "source" in available_fields and feature["source"] else "Unknown source"
            station_id = (
                feature["station_id"] if "station_id" in available_fields and feature["station_id"] else "Unknown ID"
            )

            self.station_list.addItem(f"{name} | {source} | {station_id}")

        if not self.features:
            self.details.setPlainText("No stations found. Try another search.")
        else:
            self.details.setPlainText(
                f"Showing up to {MAX_RESULTS} matching stations. Select a station to view its details."
            )

    def show_station_details(self, row):
        """Display metadata and HTTPS links for a station."""
        if row < 0 or row >= len(self.features):
            return

        feature = self.features[row]

        def value(field):
            if field not in feature.fields().names():
                return ""

            raw = feature[field]

            return "" if raw is None else str(raw)

        name = value("name") or "Unnamed station"
        station_id = value("station_id")
        source = value("source")
        agency = value("agency")
        river = value("river")
        country = value("country")
        period_start = value("period_start")
        period_end = value("period_end")

        station_url = https_url(value("url"))

        explorer_url = ""

        if source and station_id:
            explorer_url = EXPLORER_URL + "#s=" + quote(source, safe="") + "/" + quote(station_id, safe="")

        lines = [
            f"<h3>{escape(name)}</h3>",
            f"<b>Station ID:</b> {escape(station_id)}",
            f"<b>Source:</b> {escape(source)}",
            f"<b>Agency:</b> {escape(agency)}",
            f"<b>River:</b> {escape(river)}",
            f"<b>Country:</b> {escape(country)}",
            (f"<b>Observation period:</b> {escape(period_start)} – {escape(period_end)}"),
        ]

        if station_url:
            lines.append('<p><a href="' + escape(station_url) + '">Open agency station page</a></p>')

        if explorer_url:
            lines.append('<p><a href="' + escape(explorer_url) + '">Open in AquaScope Explorer</a></p>')

        self.details.setHtml("<br>".join(lines))
