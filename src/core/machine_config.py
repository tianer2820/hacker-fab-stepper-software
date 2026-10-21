from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import toml


@dataclass
class AutofocusStageConfig:
    grid_n: int = 2
    sweep_range: float = 250.0
    threshold: float = 5.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> AutofocusStageConfig:
        if not d:
            return cls()
        return cls(
            grid_n=int(d.get("grid_n", 2)),
            sweep_range=float(d.get("sweep_range", 250.0)),
            threshold=float(d.get("threshold", 5.0)),
        )


@dataclass
class AutofocusMachineConfig:
    uv_z_offset: float = 0.0
    sharpness_z_range: float = 20.0
    sharpness_threshold: float = 0.5
    settle_delay: float = 0.1
    stages: List[AutofocusStageConfig] = field(
        default_factory=lambda: [
            AutofocusStageConfig(grid_n=2, sweep_range=250.0, threshold=5.0),
            AutofocusStageConfig(grid_n=4, sweep_range=10.0, threshold=0.5),
        ]
    )

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> AutofocusMachineConfig:
        if not d:
            return cls()
        stages_data = d.get("stages", [])
        stages = [AutofocusStageConfig.from_dict(s) for s in stages_data] if stages_data else [
            AutofocusStageConfig(grid_n=2, sweep_range=250.0, threshold=5.0),
            AutofocusStageConfig(grid_n=4, sweep_range=10.0, threshold=0.5),
        ]
        return cls(
            uv_z_offset=float(d.get("uv_z_offset", 0.0)),
            sharpness_z_range=float(d.get("sharpness_z_range", 20.0)),
            sharpness_threshold=float(d.get("sharpness_threshold", 0.5)),
            settle_delay=float(d.get("settle_delay", 0.1)),
            stages=stages,
        )


@dataclass
class ExposureMachineConfig:
    default_exposure_time_ms: float = 8000.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> ExposureMachineConfig:
        if not d:
            return cls()
        return cls(
            default_exposure_time_ms=float(d.get("default_exposure_time_ms", 8000.0)),
        )


@dataclass
class StageCalibrationMachineConfig:
    calibration_file_path: str = "stage_calibration.json"
    camera_pixel_size: float = 1.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> StageCalibrationMachineConfig:
        if not d:
            return cls()
        pixel_size = d.get("camera_pixel_size", d.get("pixel_scale", 1.0))
        return cls(
            calibration_file_path=str(d.get("calibration_file_path", "stage_calibration.json")),
            camera_pixel_size=float(pixel_size),
        )


@dataclass
class MachineConfig:
    autofocus: AutofocusMachineConfig = field(default_factory=AutofocusMachineConfig)
    exposure: ExposureMachineConfig = field(default_factory=ExposureMachineConfig)
    stage_calibration: StageCalibrationMachineConfig = field(default_factory=StageCalibrationMachineConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "autofocus": {
                "uv_z_offset": self.autofocus.uv_z_offset,
                "sharpness_z_range": self.autofocus.sharpness_z_range,
                "sharpness_threshold": self.autofocus.sharpness_threshold,
                "settle_delay": self.autofocus.settle_delay,
                "stages": [asdict(s) for s in self.autofocus.stages],
            },
            "exposure": {
                "default_exposure_time_ms": self.exposure.default_exposure_time_ms,
            },
            "stage_calibration": {
                "calibration_file_path": self.stage_calibration.calibration_file_path,
                "camera_pixel_size": self.stage_calibration.camera_pixel_size,
            },
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> MachineConfig:
        if not d:
            return cls()
        return cls(
            autofocus=AutofocusMachineConfig.from_dict(d.get("autofocus")),
            exposure=ExposureMachineConfig.from_dict(d.get("exposure")),
            stage_calibration=StageCalibrationMachineConfig.from_dict(d.get("stage_calibration")),
        )

    def save(self, path: str = "machine_config.toml") -> None:
        """Saves configuration to a TOML file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            toml.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str = "machine_config.toml") -> MachineConfig:
        """Loads configuration from a TOML file."""
        with open(path, "r") as f:
            data = toml.load(f)
        return cls.from_dict(data)

    @classmethod
    def load_or_default(cls, path: str = "machine_config.toml") -> MachineConfig:
        """Loads configuration if file exists, otherwise returns default config."""
        if os.path.exists(path):
            try:
                return cls.load(path)
            except Exception as e:
                print(f"[MachineConfig] Error loading {path}: {e}. Using defaults.", flush=True)
                return cls()
        return cls()

