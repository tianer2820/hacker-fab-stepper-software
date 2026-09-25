import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from operations.ml_data_collection import MLDataCollectionConfig, MLDataCollectionOperation
from ui.bridge import QtEngineBridge


class MLDataCollectionTabWidget(QScrollArea):
    """Tab 4: Automated ML data collection with projector calibration and spiral patterning."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge
        self._active_op: Optional[MLDataCollectionOperation] = None

        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # 1. Info / Instructions
        info_box = QGroupBox("ML Data Collection Procedure")
        info_layout = QVBoxLayout(info_box)
        info_lbl = QLabel(
            "<b>Automated ML Dataset Generation</b><br><br>"
            "1. Performs initial autofocus without UV offset.<br>"
            "2. Projects a 2x2 ArUco calibration grid in Red light to compute relative projector-to-camera alignment.<br>"
            "3. Moves in a 2D spiral pattern from the current position.<br>"
            "4. At each step, autofocuses, exposes the generated random cross marker pattern in UV, then returns to Red focus Z under solid Red illumination.<br>"
            "5. Saves the camera capture, transformed ground-truth pattern, and JSON marker annotations in image space."
        )
        info_lbl.setWordWrap(True)
        info_layout.addWidget(info_lbl)
        main_layout.addWidget(info_box)

        # 2. Marker Configuration
        marker_box = QGroupBox("Marker Parameters")
        marker_grid = QGridLayout(marker_box)

        # Target Scale (%)
        marker_grid.addWidget(QLabel("Target Scale (%):"), 0, 0)
        self.spin_target_scale = QDoubleSpinBox()
        self.spin_target_scale.setRange(0.5, 50.0)
        self.spin_target_scale.setDecimals(1)
        self.spin_target_scale.setSingleStep(0.5)
        self.spin_target_scale.setValue(8.0)
        self.spin_target_scale.setSuffix(" %")
        marker_grid.addWidget(self.spin_target_scale, 0, 1)

        # Scale Jitter (+/- %)
        marker_grid.addWidget(QLabel("Scale Jitter (± %):"), 1, 0)
        self.spin_scale_jitter = QDoubleSpinBox()
        self.spin_scale_jitter.setRange(0.0, 20.0)
        self.spin_scale_jitter.setDecimals(1)
        self.spin_scale_jitter.setSingleStep(0.5)
        self.spin_scale_jitter.setValue(2.0)
        self.spin_scale_jitter.setSuffix(" %")
        marker_grid.addWidget(self.spin_scale_jitter, 1, 1)

        # Markers Per Pattern
        marker_grid.addWidget(QLabel("Markers Per Pattern:"), 2, 0)
        self.spin_marker_count = QSpinBox()
        self.spin_marker_count.setRange(1, 200)
        self.spin_marker_count.setSingleStep(1)
        self.spin_marker_count.setValue(20)
        marker_grid.addWidget(self.spin_marker_count, 2, 1)

        main_layout.addWidget(marker_box)

        # 3. Patterning & Exposure
        pattern_box = QGroupBox("Patterning & Capture")
        pattern_grid = QGridLayout(pattern_box)

        # Total Patterns
        pattern_grid.addWidget(QLabel("Total Patterns:"), 0, 0)
        self.spin_total_patterns = QSpinBox()
        self.spin_total_patterns.setRange(1, 100)
        self.spin_total_patterns.setSingleStep(1)
        self.spin_total_patterns.setValue(10)
        pattern_grid.addWidget(self.spin_total_patterns, 0, 1)

        # Pattern Gap (Spiral Step)
        pattern_grid.addWidget(QLabel("Pattern Gap:"), 1, 0)
        self.spin_pattern_gap = QDoubleSpinBox()
        self.spin_pattern_gap.setRange(10.0, 50000.0)
        self.spin_pattern_gap.setDecimals(1)
        self.spin_pattern_gap.setSingleStep(100.0)
        self.spin_pattern_gap.setValue(1000.0)
        self.spin_pattern_gap.setSuffix(" µm")
        pattern_grid.addWidget(self.spin_pattern_gap, 1, 1)

        # Exposure Duration
        pattern_grid.addWidget(QLabel("Exposure Time:"), 2, 0)
        self.spin_exposure_time = QDoubleSpinBox()
        self.spin_exposure_time.setRange(0.05, 300.0)
        self.spin_exposure_time.setDecimals(2)
        self.spin_exposure_time.setSingleStep(0.5)
        self.spin_exposure_time.setValue(2.0)
        self.spin_exposure_time.setSuffix(" s")
        pattern_grid.addWidget(self.spin_exposure_time, 2, 1)

        # Stabilization Delay
        pattern_grid.addWidget(QLabel("Stabilization Delay:"), 3, 0)
        self.spin_stabilization_delay = QDoubleSpinBox()
        self.spin_stabilization_delay.setRange(0.1, 10.0)
        self.spin_stabilization_delay.setDecimals(1)
        self.spin_stabilization_delay.setSingleStep(0.5)
        self.spin_stabilization_delay.setValue(1.0)
        self.spin_stabilization_delay.setSuffix(" s")
        pattern_grid.addWidget(self.spin_stabilization_delay, 3, 1)

        main_layout.addWidget(pattern_box)

        # 4. Output Folder
        output_box = QGroupBox("Dataset Output Folder")
        output_layout = QHBoxLayout(output_box)

        default_dir = os.path.abspath("stepper_captures/ml_data")
        self.txt_save_dir = QLineEdit(default_dir)
        output_layout.addWidget(self.txt_save_dir)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        output_layout.addWidget(self.btn_browse)

        main_layout.addWidget(output_box)

        # 5. Start button & Status
        action_box = QGroupBox("Execution")
        action_layout = QVBoxLayout(action_box)

        self.btn_start = QPushButton("Start ML Data Collection")
        self.btn_start.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_start.clicked.connect(self._on_start_clicked)
        action_layout.addWidget(self.btn_start)

        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet("color: #666; font-size: 11px;")
        action_layout.addWidget(self.lbl_status)

        main_layout.addWidget(action_box)
        main_layout.addStretch()

        self.setWidget(container)

    def _on_browse_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select ML Data Output Folder",
            self.txt_save_dir.text() or os.getcwd(),
        )
        if folder:
            self.txt_save_dir.setText(folder)

    def _on_start_clicked(self):
        save_dir = self.txt_save_dir.text().strip()
        if not save_dir:
            save_dir = os.path.abspath("stepper_captures/ml_data")
            self.txt_save_dir.setText(save_dir)

        config = MLDataCollectionConfig(
            total_patterns=self.spin_total_patterns.value(),
            pattern_gap=self.spin_pattern_gap.value(),
            target_scale_pct=self.spin_target_scale.value(),
            scale_jitter_pct=self.spin_scale_jitter.value(),
            marker_count=self.spin_marker_count.value(),
            exposure_time=self.spin_exposure_time.value(),
            stabilization_delay=self.spin_stabilization_delay.value(),
            save_directory=save_dir,
            grid_n=2,
            autofocus_config=getattr(self.engine, "autofocus_config", None),
        )

        op = MLDataCollectionOperation(config=config)
        self._active_op = op
        self.lbl_status.setText("Status: Running ML data collection...")
        self.bridge.start_operation(op)

    def on_operation_finished(self, op_or_name=None, err=None):
        from core.operation import Operation

        op = op_or_name if isinstance(op_or_name, Operation) else getattr(self, "_active_op", None)
        if isinstance(op, MLDataCollectionOperation) or op_or_name == "ML Data Collection":
            if err is not None:
                self.lbl_status.setText(f"Status: Failed - {err}")
            else:
                self.lbl_status.setText("Status: Complete")

    def update_lock_state(self, is_busy: bool):
        self.btn_start.setEnabled(not is_busy)
        self.spin_target_scale.setEnabled(not is_busy)
        self.spin_scale_jitter.setEnabled(not is_busy)
        self.spin_marker_count.setEnabled(not is_busy)
        self.spin_total_patterns.setEnabled(not is_busy)
        self.spin_pattern_gap.setEnabled(not is_busy)
        self.spin_exposure_time.setEnabled(not is_busy)
        self.spin_stabilization_delay.setEnabled(not is_busy)
        self.txt_save_dir.setEnabled(not is_busy)
        self.btn_browse.setEnabled(not is_busy)
