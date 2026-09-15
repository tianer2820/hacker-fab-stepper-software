from typing import Optional
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.engine import StepperEngine
from ui.bridge import QtEngineBridge


class StageMapWidget(QWidget):
    """2D visualization of stage travel boundaries, tool position, and exposed spots."""

    def __init__(
        self,
        engine: StepperEngine,
        bridge: QtEngineBridge,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header row: coordinate readout + clear history button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        self.coord_label = QLabel("Position: X 0.000, Y 0.000, Z 0.000 um")
        self.coord_label.setStyleSheet("font-size: 11px;")
        header_layout.addWidget(self.coord_label, stretch=1)

        self.btn_clear_history = QPushButton("Clear History")
        self.btn_clear_history.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        self.btn_clear_history.setToolTip("Clear all exposure history records")
        self.btn_clear_history.clicked.connect(self._on_clear_history)
        header_layout.addWidget(self.btn_clear_history)

        layout.addLayout(header_layout)

        self.canvas = StageMapCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # Connect signals
        self.bridge.stage_position_changed.connect(self._on_stage_moved)
        self.bridge.project_changed.connect(lambda _: self.canvas.update())
        self.bridge.exposure_history_changed.connect(lambda _: self.canvas.update())

    def _on_stage_moved(self, coords: tuple):
        x, y, z = coords
        self.coord_label.setText(f"Position: X {x:.3f}, Y {y:.3f}, Z {z:.3f} um")
        self.canvas.update()

    def _on_clear_history(self):
        if hasattr(self.engine.project, "clear_exposure_history"):
            self.engine.project.clear_exposure_history()


class StageMapCanvas(QFrame):
    """Drawing area for the stage map."""

    def __init__(self, parent_widget: StageMapWidget):
        super().__init__()
        self.parent_widget = parent_widget
        self.setStyleSheet("background-color: #121217; border-radius: 4px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Margins
        margin = 25
        avail_w = w - 2 * margin
        avail_h = h - 2 * margin

        if avail_w <= 0 or avail_h <= 0:
            return

        # Stage bounds in mm
        bounds = self.parent_widget.engine.stage.get_bounds()
        if bounds:
            min_x, max_x = bounds["x"]
            min_y, max_y = bounds["y"]
        else:
            min_x, max_x = -15.0, 0.0
            min_y, max_y = 0.0, 15.0

        span_x = max(1e-5, max_x - min_x)
        span_y = max(1e-5, max_y - min_y)

        # Non-distorting aspect ratio matching stage physical motion range
        scale = min(avail_w / span_x, avail_h / span_y)
        box_w = span_x * scale
        box_h = span_y * scale

        # Center the blue area in the canvas
        ox = margin + (avail_w - box_w) / 2.0
        oy = margin + (avail_h - box_h) / 2.0

        def to_screen(x: float, y: float):
            sx = ox + (x - min_x) * scale
            sy = oy + (max_y - y) * scale
            return sx, sy

        # Draw stage travel border
        painter.setPen(QPen(QColor("#334155"), 1.5))
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.drawRect(QRectF(ox, oy, box_w, box_h))

        # Draw previous exposure footprints
        chip_project = self.parent_widget.engine.project
        if chip_project and hasattr(chip_project, "exposure_history"):
            painter.setPen(QPen(QColor("#f59e0b"), 1))
            painter.setBrush(QBrush(QColor(245, 158, 11, 80)))
            pitch_x = getattr(chip_project.settings, "pitch_x", 983.0) / 1000.0
            pitch_y = getattr(chip_project.settings, "pitch_y", 512.0) / 1000.0
            tile_w = max(4.0, pitch_x * scale)
            tile_h = max(3.0, pitch_y * scale)

            for exp in chip_project.exposure_history:
                ex_x, ex_y = exp.coords[0], exp.coords[1]
                sx, sy = to_screen(ex_x, ex_y)
                painter.drawRect(QRectF(sx - tile_w / 2, sy - tile_h / 2, tile_w, tile_h))

        # Draw current stage position indicator
        cx, cy, _ = self.parent_widget.engine.stage.get_position()
        cur_sx, cur_sy = to_screen(cx, cy)

        # Target cross
        painter.setPen(QPen(QColor("#38bdf8"), 2))
        painter.drawLine(QPointF(cur_sx - 8, cur_sy), QPointF(cur_sx + 8, cur_sy))
        painter.drawLine(QPointF(cur_sx, cur_sy - 8), QPointF(cur_sx, cur_sy + 8))

        # # Target point
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.setBrush(QBrush(QColor("#38bdf8")))
        painter.drawEllipse(QPointF(cur_sx, cur_sy), 4, 4)

