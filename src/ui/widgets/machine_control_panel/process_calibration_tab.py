from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.engine import StepperEngine
from ui.bridge import QtEngineBridge


class ProcessCalibrationTabWidget(QScrollArea):
    """Tab 3: Process Calibration / FEM matrix placeholder."""

    def __init__(self, engine: StepperEngine, bridge: QtEngineBridge, parent: QWidget = None):
        super().__init__(parent)
        self.engine = engine
        self.bridge = bridge

        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        proc_layout = QVBoxLayout(container)
        proc_layout.setContentsMargins(6, 6, 6, 6)
        proc_layout.setSpacing(8)

        proc_box = QGroupBox("Process Calibration")
        proc_box_layout = QVBoxLayout(proc_box)
        proc_lbl = QLabel(
            "<b>Exposure & Process Calibration Matrix</b><br><br>"
            "This module will automate exposure dose matrix testing (FEM - Focus Exposure Matrix) "
            "and photoresist process calibration across varying exposure times and Z focal planes.<br><br>"
            "<i>Status: Coming Soon</i>"
        )
        proc_lbl.setWordWrap(True)
        proc_box_layout.addWidget(proc_lbl)

        self.btn_fem_placeholder = QPushButton("Generate FEM Array (Placeholder)")
        self.btn_fem_placeholder.setEnabled(False)
        proc_box_layout.addWidget(self.btn_fem_placeholder)

        proc_layout.addWidget(proc_box)
        proc_layout.addStretch()

        self.setWidget(container)
