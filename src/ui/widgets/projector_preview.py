from typing import Optional
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.engine import StepperEngine
from core.events import ColorMode
from ui.bridge import QtEngineBridge


class ProjectorPreviewWidget(QWidget):
    """Monitors what the projector is currently displaying in real time."""

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

        # Status mode label
        self.mode_label = QLabel("Output: Disabled (Off)")
        self.mode_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.mode_label)

        self.canvas = ProjectorCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # Connect signals
        self.bridge.projector_image_changed.connect(self._on_image_changed)
        self.bridge.projector_color_mode_changed.connect(lambda *_: self._on_image_changed())
        self.bridge.projector_on_off_changed.connect(lambda *_: self._on_image_changed())

    def _on_image_changed(self, *args):
        is_on = self.engine.projector.is_on
        color_mode = self.engine.projector.color_mode
        mode_str = "Red" if color_mode == ColorMode.RED else "UV"
        if is_on:
            self.mode_label.setText(f"Output: ON ({mode_str} Illumination Active)")
            self.mode_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #4CAF50;")
        else:
            self.mode_label.setText(f"Output: OFF ({mode_str} Illumination Ready)")
            self.mode_label.setStyleSheet("font-size: 11px; color: #888888;")
        self.canvas.update()


class ProjectorCanvas(QWidget):
    def __init__(self, parent_view: ProjectorPreviewWidget):
        super().__init__()
        self.parent_view = parent_view
        self.setStyleSheet("background-color: #000000; border-radius: 4px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))

        img = (
            self.parent_view.engine.projector._displayed_image_cache
            if self.parent_view.engine.projector.is_on
            else None
        )
        if img is not None and isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            contig = np.ascontiguousarray(img)
            if img.ndim == 2:
                qimg = QImage(contig.data, w, h, w, QImage.Format_Grayscale8)
            elif img.shape[2] == 3:
                qimg = QImage(contig.data, w, h, 3 * w, QImage.Format_RGB888)
            elif img.shape[2] == 4:
                qimg = QImage(contig.data, w, h, 4 * w, QImage.Format_RGBA8888)
            else:
                qimg = None

            if qimg is not None:
                pixmap = QPixmap.fromImage(qimg)
                scaled = pixmap.scaled(
                    self.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                x = (self.width() - scaled.width()) // 2
                y = (self.height() - scaled.height()) // 2
                painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor("#444444"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Output (Screen Off)")

