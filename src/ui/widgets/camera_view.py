import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from camera import CameraModule
from core.engine import StepperEngine
from ui.bridge import QtEngineBridge


class CameraViewWidget(QWidget):
    """Live camera view with crosshairs, FPS tracking, zooming, and snapshot capture."""

    def __init__(
        self,
        engine: StepperEngine,
        bridge: QtEngineBridge,
        camera: Optional[CameraModule] = None,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge
        self.camera = camera

        self.current_frame: Optional[np.ndarray] = None
        self.current_qimage: Optional[QImage] = None
        self.show_crosshairs = True
        self.is_moving_crosshair = False

        # FPS metrics
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        if self.camera and not self.camera.is_open():
            if not self.camera.open():
                print("Warning: Camera failed to open")

        self._init_ui()

        # Listen for camera frame ready event from engine bridge
        self.bridge.camera_frame_ready.connect(self._on_frame_ready)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header bar with controls
        header = QHBoxLayout()
        header.setSpacing(6)

        self.crosshair_cb = QCheckBox("Crosshair")
        self.crosshair_cb.setChecked(True)
        self.crosshair_cb.toggled.connect(self._on_crosshair_toggled)
        header.addWidget(self.crosshair_cb)

        self.move_crosshair_btn = QPushButton("Move Crosshair")
        self.move_crosshair_btn.setCheckable(True)
        self.move_crosshair_btn.setStyleSheet("font-weight: bold;")
        self.move_crosshair_btn.toggled.connect(self._on_move_crosshair_toggled)
        header.addWidget(self.move_crosshair_btn)

        self.center_crosshair_btn = QPushButton("Center")
        self.center_crosshair_btn.setToolTip("Recenter crosshair to image center")
        self.center_crosshair_btn.clicked.connect(self._on_center_crosshair_clicked)
        header.addWidget(self.center_crosshair_btn)

        header.addSpacing(6)

        # Zoom Controls
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(26)
        self.btn_zoom_out.setToolTip("Zoom Out")
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        header.addWidget(self.btn_zoom_out)

        self.btn_zoom_reset = QPushButton("100%")
        self.btn_zoom_reset.setFixedWidth(52)
        self.btn_zoom_reset.setToolTip("Reset Zoom and Pan (Fit)")
        self.btn_zoom_reset.clicked.connect(self._zoom_reset)
        header.addWidget(self.btn_zoom_reset)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(26)
        self.btn_zoom_in.setToolTip("Zoom In")
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        header.addWidget(self.btn_zoom_in)

        self.snapshot_btn = QPushButton("Snapshot")
        self.snapshot_btn.clicked.connect(self._take_snapshot)
        header.addWidget(self.snapshot_btn)

        header.addStretch()

        self.res_label = QLabel("No Camera")
        self.res_label.setStyleSheet("font-size: 11px;")
        header.addWidget(self.res_label)

        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setStyleSheet("font-size: 11px;")
        header.addWidget(self.fps_label)

        layout.addLayout(header)

        # Main viewport canvas
        self.viewport = CameraViewport(self)
        layout.addWidget(self.viewport, stretch=1)

    def _on_crosshair_toggled(self, checked: bool):
        self.show_crosshairs = checked
        self.move_crosshair_btn.setEnabled(checked)
        self.center_crosshair_btn.setEnabled(checked)
        if not checked and self.move_crosshair_btn.isChecked():
            self.move_crosshair_btn.setChecked(False)
        self.viewport.update()

    def _on_move_crosshair_toggled(self, checked: bool):
        self.is_moving_crosshair = checked
        if checked:
            self.viewport.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.viewport.setCursor(QCursor(Qt.ArrowCursor))
        self.viewport.update()

    def _on_center_crosshair_clicked(self):
        self.viewport.crosshair_u = 0.5
        self.viewport.crosshair_v = 0.5
        self.viewport.has_secondary_crosshair = False
        self.viewport.update()

    def _zoom_in(self):
        self.viewport.zoom(1.25)
        self._update_zoom_label()

    def _zoom_out(self):
        self.viewport.zoom(1.0 / 1.25)
        self._update_zoom_label()

    def _zoom_reset(self):
        self.viewport.reset_zoom()
        self._update_zoom_label()

    def _update_zoom_label(self):
        pct = int(round(self.viewport.zoom_level * 100))
        self.btn_zoom_reset.setText(f"{pct}%")

    def _take_snapshot(self):
        if self.current_frame is None:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        capture_dir = Path("stepper_captures")
        capture_dir.mkdir(exist_ok=True)
        filename = capture_dir / f"manual_snapshot_{timestamp}.png"
        cv2.imwrite(str(filename), self.current_frame)
        print(f"Saved snapshot to {filename}")
        self.bridge.status_message.emit(f"Snapshot saved: {filename.name}")

    def cleanup(self):
        try:
            self.bridge.camera_frame_ready.disconnect(self._on_frame_ready)
        except Exception:
            pass
        if self.camera and self.camera.is_open():
            try:
                self.camera.close()
            except Exception as e:
                print(f"Error closing camera: {e}")

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def _on_frame_ready(self, frame: Optional[np.ndarray]):
        if frame is None:
            return

        self.current_frame = frame

        # Convert directly to QImage without cvtColor overhead
        h, w = frame.shape[:2]
        bytes_per_line = frame.strides[0]
        if frame.ndim == 2:
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
        else:
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888).copy()

        self.current_qimage = qimg
        self.res_label.setText(f"{w}x{h}")

        # FPS calculate
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_fps_time
        if dt >= 1.0:
            self.current_fps = self.frame_count / dt
            self.fps_label.setText(f"{self.current_fps:.1f} FPS")
            self.frame_count = 0
            self.last_fps_time = now

        self.viewport.update()


class CameraViewport(QWidget):
    """Subwidget that paints the image with crosshair overlay, zooming, and panning."""

    def __init__(self, parent_view: CameraViewWidget):
        super().__init__()
        self.parent_view = parent_view
        self.setStyleSheet("background-color: #0d0d11; border-radius: 4px;")

        # Zoom and Pan parameters (UI level)
        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0

        # Normalized crosshair coordinates relative to the image [0.0 - 1.0]
        self.crosshair_u: float = 0.5
        self.crosshair_v: float = 0.5

        # Secondary crosshair (set by right-click in move-crosshair mode)
        self.secondary_crosshair_u: float = 0.0
        self.secondary_crosshair_v: float = 0.0
        self.has_secondary_crosshair: bool = False

        # Mouse interaction state
        self._is_panning: bool = False
        self._last_mouse_pos: Optional[QPointF] = None

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def zoom(self, factor: float, center_point: Optional[QPointF] = None):
        old_zoom = self.zoom_level
        new_zoom = max(0.2, min(25.0, old_zoom * factor))
        if new_zoom == old_zoom:
            return

        qimg = self.parent_view.current_qimage
        if qimg is not None and not qimg.isNull() and center_point is not None:
            # Mouse-centered zoom
            vw, vh = self.width(), self.height()
            iw, ih = qimg.width(), qimg.height()
            base_scale = min(vw / iw, vh / ih)

            old_dw = iw * base_scale * old_zoom
            old_dh = ih * base_scale * old_zoom
            old_dx = (vw - old_dw) / 2.0 + self.pan_x
            old_dy = (vh - old_dh) / 2.0 + self.pan_y

            # Normalized coordinate under the mouse
            u = (center_point.x() - old_dx) / old_dw
            v = (center_point.y() - old_dy) / old_dh

            self.zoom_level = new_zoom
            new_dw = iw * base_scale * new_zoom
            new_dh = ih * base_scale * new_zoom

            new_dx = center_point.x() - u * new_dw
            new_dy = center_point.y() - v * new_dh

            self.pan_x = new_dx - (vw - new_dw) / 2.0
            self.pan_y = new_dy - (vh - new_dh) / 2.0
        else:
            self.zoom_level = new_zoom

        self.update()

    def _get_target_rect(self) -> QRectF:
        qimg = self.parent_view.current_qimage
        vw = self.width()
        vh = self.height()
        if qimg is None or qimg.isNull():
            return QRectF(0, 0, vw, vh)

        iw = qimg.width()
        ih = qimg.height()
        base_scale = min(vw / iw, vh / ih)
        dw = iw * base_scale * self.zoom_level
        dh = ih * base_scale * self.zoom_level
        dx = (vw - dw) / 2.0 + self.pan_x
        dy = (vh - dh) / 2.0 + self.pan_y
        return QRectF(dx, dy, dw, dh)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Draw background
        painter.fillRect(self.rect(), QColor("#0d0d11"))

        qimg = self.parent_view.current_qimage
        target_rect = self._get_target_rect()

        if qimg is not None and not qimg.isNull():
            # Fast direct render into target_rect without per-frame CPU reallocation
            painter.drawImage(target_rect, qimg)
        else:
            painter.setPen(QColor("#555555"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Camera Feed Available")

        # Draw Crosshair
        if self.parent_view.show_crosshairs:
            cx = target_rect.x() + self.crosshair_u * target_rect.width()
            cy = target_rect.y() + self.crosshair_v * target_rect.height()

            pen = QPen(QColor(0, 255, 128, 180), 1.5, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(0, cy), QPointF(self.width(), cy))
            painter.drawLine(QPointF(cx, 0), QPointF(cx, self.height()))

            # Center target circle
            pen_solid = QPen(QColor(0, 255, 128, 220), 1.5)
            painter.setPen(pen_solid)
            painter.drawEllipse(QPointF(cx, cy), 15, 15)
            painter.drawEllipse(QPointF(cx, cy), 35, 35)

            # Draw secondary crosshair (orange) and distance label
            if self.has_secondary_crosshair:
                scx = target_rect.x() + self.secondary_crosshair_u * target_rect.width()
                scy = target_rect.y() + self.secondary_crosshair_v * target_rect.height()

                pen_sec = QPen(QColor(255, 165, 0, 180), 1.5, Qt.DashLine)
                painter.setPen(pen_sec)
                painter.drawLine(QPointF(0, scy), QPointF(self.width(), scy))
                painter.drawLine(QPointF(scx, 0), QPointF(scx, self.height()))

                pen_sec_solid = QPen(QColor(255, 165, 0, 220), 1.5)
                painter.setPen(pen_sec_solid)
                painter.drawEllipse(QPointF(scx, scy), 15, 15)
                painter.drawEllipse(QPointF(scx, scy), 35, 35)

                # Compute pixel distance between the two crosshairs
                qimg = self.parent_view.current_qimage
                if qimg is not None and not qimg.isNull() and target_rect.width() > 0:
                    iw = qimg.width()
                    ih = qimg.height()
                    du = (self.secondary_crosshair_u - self.crosshair_u) * iw
                    dv = (self.secondary_crosshair_v - self.crosshair_v) * ih
                    dist_px = (du ** 2 + dv ** 2) ** 0.5

                    # Draw a line connecting the two crosshairs
                    pen_line = QPen(QColor(255, 255, 255, 120), 1.0, Qt.DotLine)
                    painter.setPen(pen_line)
                    painter.drawLine(QPointF(cx, cy), QPointF(scx, scy))

                    # Distance label at the midpoint
                    mid_x = (cx + scx) / 2
                    mid_y = (cy + scy) / 2
                    label = f"{dist_px:.1f} px"
                    font = painter.font()
                    font.setPointSize(9)
                    font.setBold(True)
                    painter.setFont(font)
                    fm = painter.fontMetrics()
                    lw = fm.horizontalAdvance(label)
                    lh = fm.height()
                    pad = 4
                    bg_rect = QRectF(mid_x - lw / 2 - pad, mid_y - lh / 2 - pad, lw + pad * 2, lh + pad * 2)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(0, 0, 0, 160))
                    painter.drawRoundedRect(bg_rect, 3, 3)
                    painter.setPen(QColor(255, 220, 80))
                    painter.drawText(QRectF(mid_x - lw / 2, mid_y - lh / 2, lw, lh), Qt.AlignCenter, label)

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        if angle != 0:
            factor = 1.15 if angle > 0 else (1.0 / 1.15)
            self.zoom(factor, event.position())
            self.parent_view._update_zoom_label()
        event.accept()

    def mousePressEvent(self, event):
        if self.parent_view.is_moving_crosshair and event.button() == Qt.LeftButton:
            self._update_crosshair_from_pos(event.position())
            event.accept()
            return

        if self.parent_view.is_moving_crosshair and event.button() == Qt.RightButton:
            self._update_secondary_crosshair_from_pos(event.position())
            event.accept()
            return

        if event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._is_panning = True
            self._last_mouse_pos = event.position()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.parent_view.is_moving_crosshair and (event.buttons() & Qt.LeftButton):
            self._update_crosshair_from_pos(event.position())
            event.accept()
            return

        if self._is_panning and self._last_mouse_pos is not None:
            delta = event.position() - self._last_mouse_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self._last_mouse_pos = event.position()
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._is_panning = False
            self._last_mouse_pos = None
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if self.parent_view.is_moving_crosshair:
            self.crosshair_u = 0.5
            self.crosshair_v = 0.5
            self.update()
        else:
            self.reset_zoom()
            self.parent_view._update_zoom_label()
        event.accept()

    def _update_crosshair_from_pos(self, pos: QPointF):
        target_rect = self._get_target_rect()
        if target_rect.width() > 0 and target_rect.height() > 0:
            u = (pos.x() - target_rect.x()) / target_rect.width()
            v = (pos.y() - target_rect.y()) / target_rect.height()
            self.crosshair_u = max(0.0, min(1.0, u))
            self.crosshair_v = max(0.0, min(1.0, v))
            self.update()

    def _update_secondary_crosshair_from_pos(self, pos: QPointF):
        target_rect = self._get_target_rect()
        if target_rect.width() > 0 and target_rect.height() > 0:
            u = (pos.x() - target_rect.x()) / target_rect.width()
            v = (pos.y() - target_rect.y()) / target_rect.height()
            self.secondary_crosshair_u = max(0.0, min(1.0, u))
            self.secondary_crosshair_v = max(0.0, min(1.0, v))
            self.has_secondary_crosshair = True
            self.update()


