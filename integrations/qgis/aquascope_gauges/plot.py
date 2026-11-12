"""Lightweight Qt time-series plot for AquaScope observations."""

from qgis.PyQt.QtCore import QPointF, QRectF, Qt
from qgis.PyQt.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from qgis.PyQt.QtWidgets import QWidget


class ObservationPlot(QWidget):
    """Display daily observation records as a time-series chart."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.records = []
        self.title = "Daily observations"
        self.unit = ""

        self.setMinimumHeight(260)

    def set_data(self, records, title="Daily observations", unit=""):
        """Set observation records and refresh the chart."""
        self.records = sorted(
            records,
            key=lambda record: record["date"],
        )
        self.title = str(title)
        self.unit = str(unit)

        self.update()

    def clear(self):
        """Clear the current chart."""
        self.records = []
        self.update()

    def paintEvent(self, event):  # noqa: N802
        """Paint the time-series chart."""
        painter = QPainter(self)

        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor("#ffffff"))

            width = self.width()
            height = self.height()

            painter.setPen(QColor("#202124"))
            title_font = QFont()
            title_font.setBold(True)
            title_font.setPointSize(11)
            painter.setFont(title_font)

            painter.drawText(
                QRectF(12, 8, width - 24, 28),
                Qt.AlignLeft | Qt.AlignVCenter,
                self.title,
            )

            if not self.records:
                painter.setPen(QColor("#777777"))
                painter.drawText(
                    self.rect(),
                    Qt.AlignCenter,
                    "Fetch observations to display the time series.",
                )
                return

            left = 68
            right = 20
            top = 48
            bottom = 52

            plot_width = max(1, width - left - right)
            plot_height = max(1, height - top - bottom)

            dates = [record["date"] for record in self.records]
            values = [float(record["value"]) for record in self.records]

            min_date = min(dates)
            max_date = max(dates)
            min_value = min(values)
            max_value = max(values)

            if min_value == max_value:
                padding = max(abs(min_value) * 0.05, 1.0)
                min_value -= padding
                max_value += padding
            else:
                padding = (max_value - min_value) * 0.08
                min_value -= padding
                max_value += padding

            date_range = max(
                (max_date - min_date).days,
                1,
            )
            value_range = max_value - min_value

            def point_for(record):
                date_position = (
                    (record["date"] - min_date).days / date_range
                )
                value_position = (
                    (float(record["value"]) - min_value) / value_range
                )

                return QPointF(
                    left + date_position * plot_width,
                    top + (1 - value_position) * plot_height,
                )

            # Draw horizontal grid lines and Y-axis labels.
            label_font = QFont()
            label_font.setPointSize(8)
            painter.setFont(label_font)

            for index in range(5):
                fraction = index / 4
                y = top + fraction * plot_height
                value = max_value - fraction * value_range

                painter.setPen(QPen(QColor("#e5e7eb"), 1))
                painter.drawLine(
                    QPointF(left, y),
                    QPointF(left + plot_width, y),
                )

                painter.setPen(QColor("#555555"))
                painter.drawText(
                    QRectF(2, y - 9, left - 10, 18),
                    Qt.AlignRight | Qt.AlignVCenter,
                    f"{value:.4g}",
                )

            # Draw chart axes.
            axis_pen = QPen(QColor("#555555"), 1)
            painter.setPen(axis_pen)

            painter.drawLine(
                QPointF(left, top),
                QPointF(left, top + plot_height),
            )
            painter.drawLine(
                QPointF(left, top + plot_height),
                QPointF(left + plot_width, top + plot_height),
            )

            # Draw the observation time series.
            points = [point_for(record) for record in self.records]

            path = QPainterPath()
            path.moveTo(points[0])

            for point in points[1:]:
                path.lineTo(point)

            painter.setPen(QPen(QColor("#1976d2"), 2))
            painter.drawPath(path)

            # Draw observation points.
            painter.setBrush(QColor("#1976d2"))
            painter.setPen(QPen(QColor("#ffffff"), 1))

            for point in points:
                painter.drawEllipse(point, 3, 3)

            # Draw date labels.
            painter.setPen(QColor("#555555"))

            label_count = min(5, len(self.records))

            for index in range(label_count):
                record_index = (
                    round(index * (len(self.records) - 1)
                          / max(label_count - 1, 1))
                )
                record = self.records[record_index]
                point = points[record_index]

                painter.drawText(
                    QRectF(
                        point.x() - 42,
                        top + plot_height + 8,
                        84,
                        20,
                    ),
                    Qt.AlignCenter,
                    record["date"].isoformat(),
                )

            # Draw the Y-axis unit label.
            painter.save()
            painter.translate(15, top + plot_height / 2)
            painter.rotate(-90)
            painter.drawText(
                QRectF(-plot_height / 2, -10, plot_height, 20),
                Qt.AlignCenter,
                self.unit or "Value",
            )
            painter.restore()

        finally:
            painter.end()
