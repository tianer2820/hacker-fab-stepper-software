from typing import Optional
import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
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


        # Initial refresh
        self._on_image_changed()

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
        self.canvas.refresh_image()


class ProjectorCanvas(QWidget):
    def __init__(self, parent_view: ProjectorPreviewWidget):
        super().__init__(parent_view)
        self.parent_view = parent_view
        self.setStyleSheet("background-color: #141416; border-radius: 4px;")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._pixmap: Optional[QPixmap] = None

    def refresh_image(self):
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
                self._pixmap = QPixmap.fromImage(qimg)
            else:
                self._pixmap = None
        else:
            self._pixmap = None

        self.update()
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#141416"))

        # Determine projector display dimensions
        pw, ph = 1920, 1080
        if hasattr(self.parent_view.engine, "projector") and self.parent_view.engine.projector is not None:
            size = self.parent_view.engine.projector.projector_size()
            if size and size[0] > 0 and size[1] > 0:
                pw, ph = size

        margin = 8
        avail_w = max(1, self.width() - 2 * margin)
        avail_h = max(1, self.height() - 2 * margin)
        scale = min(avail_w / pw, avail_h / ph)
        rect_w = max(1, int(pw * scale))
        rect_h = max(1, int(ph * scale))
        rect_x = (self.width() - rect_w) // 2
        rect_y = (self.height() - rect_h) // 2
        proj_rect = QRect(rect_x, rect_y, rect_w, rect_h)

        # Fill projector area with black background
        painter.fillRect(proj_rect, QColor("#000000"))

        is_on = (
            self.parent_view.engine.projector.is_on
            if hasattr(self.parent_view.engine, "projector")
            else False
        )

        if is_on and self._pixmap is not None and not self._pixmap.isNull():
            img_draw_w = min(rect_w, max(1, int(self._pixmap.width() * scale)))
            img_draw_h = min(rect_h, max(1, int(self._pixmap.height() * scale)))
            scaled = self._pixmap.scaled(
                img_draw_w,
                img_draw_h,
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )
            painter.drawPixmap(rect_x, rect_y, scaled)
        else:
            painter.setPen(QColor("#555555"))
            painter.drawText(proj_rect, Qt.AlignCenter, "No Output (Screen Off)")

        # Draw rectangular frame representing the projector area
        frame_color = QColor("#38bdf8") if is_on else QColor("#52525b")
        painter.setPen(QPen(frame_color, 1.5))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(proj_rect)

