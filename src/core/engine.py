from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from camera.camera_module import CameraModule
    from projector import ProjectorController
    from stage_control.stage_controller import StageController

from .chip_project import ChipProject
from .events import Event, EventBus
from .machine_config import MachineConfig
from .operation import ExecutionContext, Operation, OperationManager


class StepperEngine:
    """Core process controller for the lithography stepper.

    Adheres strictly to the 3-pillar architecture:
    1. Chip Project (data): Self-contained ChipProject emitting layer/project events.
    2. Operation (workflow): Single active operation lifecycle managed by OperationManager.
    3. Machine Control (hardware): Directly accessible submodules (stage, projector, camera, dlpc).
    """

    def __init__(
        self,
        stage: StageController,
        projector: ProjectorController,
        camera: CameraModule,
        dlpc: Optional[Any] = None,
        delay_func: Optional[Callable[[float], None]] = None,
        warning_callback: Optional[Callable[[str], None]] = None,
        machine_config: Optional[MachineConfig] = None,
        autofocus_config: Optional[Any] = None,
    ):
        self.stage = stage
        self.projector = projector
        self.camera = camera
        self.dlpc = dlpc
        
        self.delay_func = delay_func or time.sleep
        self.warning_callback = warning_callback or self._default_warning

        # Link event bus
        self.event_bus = EventBus()
        self.stage.event_bus = self.event_bus
        self.projector.event_bus = self.event_bus
        self.camera.event_bus = self.event_bus

        self.projector.dlpc = self.dlpc

        # 1. Machine Configuration
        if machine_config is not None:
            self.machine_config = machine_config
        else:
            self.machine_config = MachineConfig.load_or_default("machine_config.toml")

        # For backwards compatibility if old autofocus_config was passed in tests
        if autofocus_config is not None:
            if hasattr(autofocus_config, "uv_z_offset"):
                self.machine_config.autofocus.uv_z_offset = autofocus_config.uv_z_offset
            elif isinstance(autofocus_config, dict):
                self.machine_config.autofocus.uv_z_offset = float(autofocus_config.get("uv_z_offset", 0.0))

        # Auto-load stage calibration from config if available
        self._init_stage_calibration()

        # 2. Chip Project Data Object
        self.project = ChipProject(events=self.event_bus)
        self.projector.set_project(self.project)

        # 3. Execution Context & Operation Manager
        self.context = ExecutionContext(
            stage=self.stage,
            projector=self.projector,
            camera=self.camera,
            project=self.project,
            event_bus=self.event_bus,
            machine_config=self.machine_config,
            warning_callback=self.create_warning,
            delay_func=self.delay_func,
        )
        self.operations = OperationManager(self.context, self.event_bus)

        # Snapshot folder
        self.snapshot_directory = Path("stepper_captures")
        self.snapshot_directory.mkdir(exist_ok=True)

    def _init_stage_calibration(self):
        cal_path = self.machine_config.stage_calibration.calibration_file_path
        if cal_path and os.path.exists(cal_path) and self.stage is not None:
            try:
                from stage_control.calibration import StageCalibration
                cal = StageCalibration.load_from_json(cal_path)
                self.stage.set_calibration(cal)
                print(f"[StepperEngine] Loaded stage calibration from {cal_path}", flush=True)
            except Exception as e:
                print(f"[StepperEngine] Warning: Could not load stage calibration from {cal_path}: {e}", flush=True)

    def set_machine_config(self, config: MachineConfig):
        """Sets the active machine config, updates context and calibration, and emits change event."""
        self.machine_config = config
        self.context.machine_config = config
        self._init_stage_calibration()
        self.event_bus.emit(Event.MACHINE_CONFIG_CHANGED, self.machine_config)

    def save_machine_config(self, path: str = "machine_config.toml"):
        """Saves current machine config to TOML file."""
        self.machine_config.save(path)

    def load_machine_config(self, path: str = "machine_config.toml"):
        """Loads machine config from TOML file and applies it."""
        config = MachineConfig.load(path)
        self.set_machine_config(config)

    def _default_warning(self, msg: str):
        print(f"Warning: {msg}")

    def create_warning(self, msg: str):
        self.event_bus.emit(Event.WARNING_MESSAGE, msg)
        self.warning_callback(msg)

    # -------------------------------------------------------------------------
    # 1. Chip Project Persistence
    # -------------------------------------------------------------------------
    def load_project(self, path: str):
        self.project = ChipProject.load(path, events=self.event_bus)
        self.context.project = self.project
        self.projector.set_project(self.project)
        self.event_bus.emit(Event.PROJECT_CHANGED, self.project)
        self.event_bus.emit(Event.ACTIVE_LAYER_CHANGED, self.project.active_layer_index)

    def save_project(self, path: str):
        self.project.save(path)

    def new_project(self):
        self.project = ChipProject(events=self.event_bus)
        self.context.project = self.project
        self.projector.set_project(self.project)
        self.event_bus.emit(Event.PROJECT_CHANGED, self.project)
        self.event_bus.emit(Event.ACTIVE_LAYER_CHANGED, self.project.active_layer_index)

    # -------------------------------------------------------------------------
    # 2. Operations
    # -------------------------------------------------------------------------
    def run_operation(self, operation: Operation) -> Any:
        """Executes an operation synchronously using the OperationManager."""
        return self.operations.run(operation)

    @property
    def current_operation(self) -> Optional[Operation]:
        """Returns the currently active operation or None."""
        return self.operations.current_operation

    def abort_operation(self):
        """Aborts the currently running operation."""
        self.operations.abort_current()

