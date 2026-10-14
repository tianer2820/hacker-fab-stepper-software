import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from operations.stage_calibration import (
    StageCalibrationPatternAndZConfig,
    StageCalibrationPatternAndZOperation,
    StageCalibrationScanConfig,
    StageCalibrationScanOperation,
    compute_calibration_grid_params,
)
from ui.bridge import QtEngineBridge


class StageCalibrationTabWidget(QScrollArea):
    """Tab 5: Two-pass stage coordinate calibration (Pass 1: Pattern & Z, Pass 2: Optical Scan)."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge
        self._active_op: Optional[object] = None

        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # 1. Instructions
        info_box = QGroupBox("Stage Calibration Procedure")
        info_layout = QVBoxLayout(info_box)
        info_lbl = QLabel(
            "<b>2-Pass Coordinate & Height Map Calibration</b><br><br>"
            "<b>Pass 1 (Patterning & Z-Autofocus):</b><br>"
            "• Autofocuses and measures optical transform between projector and camera.<br>"
            "• Corrects tag projection (aligning Top-Left corner to camera center).<br>"
            "• Calculates grid size & step distance so 2x2 tags fit within camera FOV.<br>"
            "• Steps through grid, autofocuses, and exposes small ArUco tags in UV.<br>"
            "<i>After Pass 1 completes, remove and develop the wafer.</i><br><br>"
            "<b>Pass 2 (Optical Scan & Calibration):</b><br>"
            "• Jog stage until at least one tag is visible in camera.<br>"
            "• Software estimates wafer orientation and origin transform.<br>"
            "• Scans clusters to measure tag-to-tag relative pixel vectors in camera view.<br>"
            "• Solves global optical coordinates via Least Squares without relying on stage position."
        )
        info_lbl.setWordWrap(True)
        info_layout.addWidget(info_lbl)
        main_layout.addWidget(info_box)

        # 2. Pass 1 Group (Single-Column)
        p1_box = QGroupBox("Pass 1: Pattern Grid & Z Autofocus")
        p1_layout = QVBoxLayout(p1_box)
        p1_layout.setSpacing(6)

        # Calibration Range (L)
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Calibration Range:"))
        self.spin_cal_range = QDoubleSpinBox()
        self.spin_cal_range.setRange(500.0, 100000.0)
        self.spin_cal_range.setValue(10000.0)
        self.spin_cal_range.setSingleStep(1000.0)
        self.spin_cal_range.setSuffix(" µm")
        self.spin_cal_range.valueChanged.connect(self._update_computed_grid_label)
        range_row.addWidget(self.spin_cal_range, 1)
        p1_layout.addLayout(range_row)

        # Camera Pixel Scale
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Camera Pixel Scale:"))
        self.spin_pixel_scale = QDoubleSpinBox()
        self.spin_pixel_scale.setRange(0.01, 100.0)
        self.spin_pixel_scale.setDecimals(3)
        self.spin_pixel_scale.setValue(1.0)
        self.spin_pixel_scale.setSingleStep(0.1)
        self.spin_pixel_scale.setSuffix(" µm/px")
        self.spin_pixel_scale.valueChanged.connect(self._update_computed_grid_label)
        scale_row.addWidget(self.spin_pixel_scale, 1)
        p1_layout.addLayout(scale_row)

        # Dynamic Grid Calculation Display
        self.lbl_grid_info = QLabel("Computed Grid: Calculating...")
        self.lbl_grid_info.setStyleSheet("color: #4da6ff; font-weight: bold; font-size: 11px;")
        p1_layout.addWidget(self.lbl_grid_info)

        # Exposure Time (default 8.0 s)
        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Exposure Time:"))
        self.spin_exposure_time = QDoubleSpinBox()
        self.spin_exposure_time.setRange(0.1, 120.0)
        self.spin_exposure_time.setValue(8.0)
        self.spin_exposure_time.setSingleStep(1.0)
        self.spin_exposure_time.setSuffix(" s")
        exp_row.addWidget(self.spin_exposure_time, 1)
        p1_layout.addLayout(exp_row)

        # Autofocus Checkbox
        self.chk_autofocus = QCheckBox("Enable Autofocus")
        self.chk_autofocus.setChecked(True)
        p1_layout.addWidget(self.chk_autofocus)

        # Pass 1 Output File
        p1_layout.addWidget(QLabel("Pass 1 Output JSON:"))
        self.txt_p1_file = QLineEdit(os.path.abspath("stage_calibration_pass1.json"))
        p1_layout.addWidget(self.txt_p1_file)

        # Pass 1 Start Button
        self.btn_start_p1 = QPushButton("Run Pass 1 (Optical Align, Pattern & Z)")
        self.btn_start_p1.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_start_p1.clicked.connect(self._on_start_pass1_clicked)
        p1_layout.addWidget(self.btn_start_p1)

        main_layout.addWidget(p1_box)

        # 3. Pass 2 Group (Single-Column)
        p2_box = QGroupBox("Pass 2: Optical Scan & Least Squares Map")
        p2_layout = QVBoxLayout(p2_box)
        p2_layout.setSpacing(6)

        p2_layout.addWidget(QLabel("Pass 1 Input JSON:"))
        p2_in_row = QHBoxLayout()
        self.txt_p2_input = QLineEdit(os.path.abspath("stage_calibration_pass1.json"))
        p2_in_row.addWidget(self.txt_p2_input, 1)
        self.btn_browse_p2_input = QPushButton("Browse...")
        self.btn_browse_p2_input.clicked.connect(self._on_browse_p2_input_clicked)
        p2_in_row.addWidget(self.btn_browse_p2_input)
        p2_layout.addLayout(p2_in_row)

        p2_layout.addWidget(QLabel("Final Calibration JSON:"))
        p2_out_row = QHBoxLayout()
        self.txt_p2_output = QLineEdit(os.path.abspath("stage_calibration.json"))
        p2_out_row.addWidget(self.txt_p2_output, 1)
        self.btn_browse_p2_output = QPushButton("Browse...")
        self.btn_browse_p2_output.clicked.connect(self._on_browse_p2_output_clicked)
        p2_out_row.addWidget(self.btn_browse_p2_output)
        p2_layout.addLayout(p2_out_row)

        scale2_row = QHBoxLayout()
        scale2_row.addWidget(QLabel("Camera Pixel Scale:"))
        self.spin_p2_pixel_scale = QDoubleSpinBox()
        self.spin_p2_pixel_scale.setRange(0.01, 100.0)
        self.spin_p2_pixel_scale.setDecimals(3)
        self.spin_p2_pixel_scale.setValue(1.0)
        self.spin_p2_pixel_scale.setSuffix(" µm/px")
        scale2_row.addWidget(self.spin_p2_pixel_scale, 1)
        p2_layout.addLayout(scale2_row)

        self.chk_apply_now = QCheckBox("Apply calibration to stage immediately upon completion")
        self.chk_apply_now.setChecked(True)
        p2_layout.addWidget(self.chk_apply_now)

        self.btn_start_p2 = QPushButton("Run Pass 2 (Scan & Reconstruct Global Map)")
        self.btn_start_p2.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_start_p2.clicked.connect(self._on_start_pass2_clicked)
        p2_layout.addWidget(self.btn_start_p2)

        main_layout.addWidget(p2_box)

        # 4. Status & Controls (No progress bar)
        status_box = QGroupBox("Status & Controls")
        status_layout = QVBoxLayout(status_box)

        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setWordWrap(True)
        status_layout.addWidget(self.lbl_status)

        self.btn_abort = QPushButton("Abort Operation")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._on_abort_clicked)
        status_layout.addWidget(self.btn_abort)

        main_layout.addWidget(status_box)
        main_layout.addStretch(1)

        self.setWidget(container)

        self.bridge.operation_progress.connect(self._on_progress)
        self.bridge.operation_finished.connect(self._on_operation_finished)

        self._update_computed_grid_label()

    def _update_computed_grid_label(self):
        cal_range = self.spin_cal_range.value()
        pixel_scale = self.spin_pixel_scale.value()

        cam_w, cam_h = 1920, 1080
        if self.engine.camera is not None:
            frame = self.engine.camera.get_latest_frame()
            if frame is not None:
                cam_h, cam_w = frame.shape[:2]

        n, d = compute_calibration_grid_params(cal_range, pixel_scale, (cam_w, cam_h))
        fov_um = min(cam_w, cam_h) * pixel_scale
        self.lbl_grid_info.setText(
            f"Computed Grid: {n} x {n} ({n*n} tags) | Step: {d:.1f} µm (FOV: {fov_um:.1f} µm)"
        )

    def _on_browse_p2_input_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pass 1 JSON File", self.txt_p2_input.text(), "JSON Files (*.json)"
        )
        if path:
            self.txt_p2_input.setText(path)

    def _on_browse_p2_output_clicked(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Select Calibration JSON Save Path", self.txt_p2_output.text(), "JSON Files (*.json)"
        )
        if path:
            self.txt_p2_output.setText(path)

    def _on_start_pass1_clicked(self):
        cfg = StageCalibrationPatternAndZConfig(
            calibration_range=self.spin_cal_range.value(),
            pixel_scale=self.spin_pixel_scale.value(),
            exposure_time=self.spin_exposure_time.value(),
            autofocus_enabled=self.chk_autofocus.isChecked(),
            output_json_path=self.txt_p1_file.text().strip(),
        )
        op = StageCalibrationPatternAndZOperation(config=cfg)
        self._active_op = op
        self.lbl_status.setText("Status: Starting Pass 1...")
        self.btn_abort.setEnabled(True)
        self.bridge.start_operation(op)

    def _on_start_pass2_clicked(self):
        cfg = StageCalibrationScanConfig(
            input_json_path=self.txt_p2_input.text().strip(),
            output_json_path=self.txt_p2_output.text().strip(),
            pixel_scale=self.spin_p2_pixel_scale.value(),
            apply_to_stage=self.chk_apply_now.isChecked(),
        )
        op = StageCalibrationScanOperation(config=cfg)
        self._active_op = op
        self.lbl_status.setText("Status: Starting Pass 2...")
        self.btn_abort.setEnabled(True)
        self.bridge.start_operation(op)

    def _on_abort_clicked(self):
        if self._active_op is not None and hasattr(self._active_op, "abort"):
            self._active_op.abort()
        self.bridge.abort_operation()

    def _on_progress(self, pct: float, msg: str):
        if self._active_op is not None:
            self.lbl_status.setText(f"Status: {msg}")

    def _on_operation_finished(self, op_or_name=None, err=None):
        if self._active_op is not None:
            self.btn_abort.setEnabled(False)
            if err:
                self.lbl_status.setText(f"Status: Error - {err}")
            else:
                self.lbl_status.setText("Status: Operation complete successfully!")
            self._active_op = None

    def update_lock_state(self, is_busy: bool):
        self.btn_start_p1.setEnabled(not is_busy)
        self.btn_start_p2.setEnabled(not is_busy)
        self.spin_cal_range.setEnabled(not is_busy)
        self.spin_pixel_scale.setEnabled(not is_busy)
        self.spin_exposure_time.setEnabled(not is_busy)
        self.chk_autofocus.setEnabled(not is_busy)
        self.txt_p1_file.setEnabled(not is_busy)
        self.txt_p2_input.setEnabled(not is_busy)
        self.txt_p2_output.setEnabled(not is_busy)
        self.btn_browse_p2_input.setEnabled(not is_busy)
        self.btn_browse_p2_output.setEnabled(not is_busy)
        self.spin_p2_pixel_scale.setEnabled(not is_busy)
        self.chk_apply_now.setEnabled(not is_busy)
