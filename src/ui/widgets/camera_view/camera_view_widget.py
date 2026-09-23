import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QImage
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
from .camera_viewport import CameraViewport


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
