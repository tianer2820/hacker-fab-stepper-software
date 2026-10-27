# Hacker Fab
# Stage Coordinate Calibration Module
# Provides 2D bilinear interpolation for XY distortion and surface-relative Z mapping

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class StageCalibrationPoint:
    grid_index: Tuple[int, int]  # (row, col) or (j, i)
    marker_id: int
    raw_xy: Tuple[float, float]
    corr_xy: Tuple[float, float]
    z_focus: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_index": list(self.grid_index),
            "marker_id": int(self.marker_id),
            "raw_xy": [float(self.raw_xy[0]), float(self.raw_xy[1])],
            "corr_xy": [float(self.corr_xy[0]), float(self.corr_xy[1])],
            "z_focus": float(self.z_focus),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageCalibrationPoint":
        return cls(
            grid_index=tuple(d["grid_index"]),
            marker_id=int(d["marker_id"]),
            raw_xy=(float(d["raw_xy"][0]), float(d["raw_xy"][1])),
            corr_xy=(float(d["corr_xy"][0]), float(d["corr_xy"][1])),
            z_focus=float(d["z_focus"]),
        )


class StageCalibration:
    """Manages stage coordinate transformation and height map leveling."""

    def __init__(
        self,
        grid_shape: Tuple[int, int],  # (Ny, Nx) -> (rows, cols)
        points: List[StageCalibrationPoint],
        stage_type: str = "",
        timestamp: Optional[str] = None,
        enabled: bool = True,
        step_distance: Optional[float] = None,
        calibration_range: Optional[float] = None,
        dict_id: Optional[int] = None,
        flip_horizontal: bool = False,
    ):
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.points = points
        self.stage_type = stage_type
        self.timestamp = timestamp or datetime.now().isoformat()
        self.enabled = enabled
        self.step_distance = float(step_distance) if step_distance is not None else None
        self.calibration_range = float(calibration_range) if calibration_range is not None else None
        try:
            self.dict_id = int(dict_id) if dict_id is not None else None
        except (TypeError, ValueError):
            self.dict_id = None
        self.flip_horizontal = bool(flip_horizontal)

        self._ny, self._nx = self.grid_shape
        self._raw_x_grid = np.zeros((self._ny, self._nx), dtype=np.float64)
        self._raw_y_grid = np.zeros((self._ny, self._nx), dtype=np.float64)
        self._corr_x_grid = np.zeros((self._ny, self._nx), dtype=np.float64)
        self._corr_y_grid = np.zeros((self._ny, self._nx), dtype=np.float64)
        self._z_focus_grid = np.zeros((self._ny, self._nx), dtype=np.float64)

        self._raw_x_coords: np.ndarray = np.array([])
        self._raw_y_coords: np.ndarray = np.array([])

        self._build_grids()

    def _build_grids(self):
        """Populates grid matrices from calibration points."""
        for pt in self.points:
            r, c = pt.grid_index
            if 0 <= r < self._ny and 0 <= c < self._nx:
                self._raw_x_grid[r, c] = pt.raw_xy[0]
                self._raw_y_grid[r, c] = pt.raw_xy[1]
                self._corr_x_grid[r, c] = pt.corr_xy[0]
                self._corr_y_grid[r, c] = pt.corr_xy[1]
                self._z_focus_grid[r, c] = pt.z_focus

        if self._ny > 0 and self._nx > 0:
            self._raw_x_coords = self._raw_x_grid[0, :]
            self._raw_y_coords = self._raw_y_grid[:, 0]

    def _interpolate_grid_bilinear(
        self,
        x: float,
        y: float,
        grid: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
    ) -> float:
        """Performs 2D bilinear interpolation on a regular mesh."""
        nx = len(x_coords)
        ny = len(y_coords)

        if nx < 2 or ny < 2:
            return float(grid[0, 0]) if grid.size > 0 else 0.0

        # Find column cell index
        if x <= x_coords[0]:
            i = 0
            u = 0.0
        elif x >= x_coords[-1]:
            i = nx - 2
            u = 1.0
        else:
            i = int(np.searchsorted(x_coords, x) - 1)
            i = max(0, min(i, nx - 2))
            u = (x - x_coords[i]) / max(x_coords[i + 1] - x_coords[i], 1e-9)

        # Find row cell index
        if y <= y_coords[0]:
            j = 0
            v = 0.0
        elif y >= y_coords[-1]:
            j = ny - 2
            v = 1.0
        else:
            j = int(np.searchsorted(y_coords, y) - 1)
            j = max(0, min(j, ny - 2))
            v = (y - y_coords[j]) / max(y_coords[j + 1] - y_coords[j], 1e-9)

        q00 = grid[j, i]
        q10 = grid[j, i + 1]
        q01 = grid[j + 1, i]
        q11 = grid[j + 1, i + 1]

        val = (1.0 - u) * (1.0 - v) * q00 + u * (1.0 - v) * q10 + (1.0 - u) * v * q01 + u * v * q11
        return float(val)

    def forward_map_xy(self, x_raw: float, y_raw: float) -> Tuple[float, float]:
        """Maps raw machine coordinates (X, Y) to corrected optical coordinates (X, Y)."""
        if not self.has_xy_calibration or self._nx < 2 or self._ny < 2:
            return (float(x_raw), float(y_raw))

        x_corr = self._interpolate_grid_bilinear(
            x_raw, y_raw, self._corr_x_grid, self._raw_x_coords, self._raw_y_coords
        )
        y_corr = self._interpolate_grid_bilinear(
            x_raw, y_raw, self._corr_y_grid, self._raw_x_coords, self._raw_y_coords
        )
        return (x_corr, y_corr)


    def get_z_offset(self, x_corr: float, y_corr: float) -> float:
        """Returns the interpolated focal surface height Z_raw for given corrected (X, Y)."""
        assert self._z_focus_grid.size != 0
        assert self._nx >= 2 or self._ny >= 2

        # Map corrected XY to raw XY to evaluate the raw grid height map
        x_raw, y_raw = x_corr, y_corr
        # x_raw, y_raw = self.inverse_map_xy(x_corr, y_corr)
        return self._interpolate_grid_bilinear(
            x_raw, y_raw, self._z_focus_grid, self._raw_x_coords, self._raw_y_coords
        )

    def corr_to_raw(
        self,
        corr_coords: list[float],
    ) -> Dict[str, float]:
        """Translates corrected target coordinates to raw machine target coordinates.

        Z is surface-relative: Z_raw = Z_surface(X_corr, Y_corr) + Z_corr.
        """
        corr_x, corr_y, corr_z = corr_coords
        z_surf = self.get_z_offset(corr_x, corr_y)

        raw_dict = {
            'x': corr_x,
            'y': corr_y,
            'z': corr_z + z_surf,
        }
        return raw_dict


    @property
    def has_xy_calibration(self) -> bool:
        """Returns True if XY coordinates contain measured optical correction, False if identity (Z-leveling only)."""
        if not self.points:
            return False
        return any(
            abs(p.corr_xy[0] - p.raw_xy[0]) > 1e-4 or abs(p.corr_xy[1] - p.raw_xy[1]) > 1e-4
            for p in self.points
        )

    def save_to_json(self, file_path: str):
        """Serializes calibration data to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            "version": "2.0",
            "timestamp": self.timestamp,
            "stage_type": self.stage_type,
            "enabled": self.enabled,
            "grid_shape": list(self.grid_shape),
            "step_distance": self.step_distance,
            "calibration_range": self.calibration_range,
            "dict_id": self.dict_id,
            "flip_horizontal": self.flip_horizontal,
            "bounds_raw": {
                "x_min": float(np.min(self._raw_x_grid)) if self._raw_x_grid.size > 0 else 0.0,
                "x_max": float(np.max(self._raw_x_grid)) if self._raw_x_grid.size > 0 else 0.0,
                "y_min": float(np.min(self._raw_y_grid)) if self._raw_y_grid.size > 0 else 0.0,
                "y_max": float(np.max(self._raw_y_grid)) if self._raw_y_grid.size > 0 else 0.0,
            },
            "points": [pt.to_dict() for pt in self.points],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: str) -> "StageCalibration":
        """Loads and instantiates StageCalibration from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = [StageCalibrationPoint.from_dict(p) for p in data.get("points", [])]
        grid_shape = tuple(data.get("grid_shape", (0, 0)))
        stage_type = data.get("stage_type", "")
        timestamp = data.get("timestamp", "")
        enabled = data.get("enabled", True)
        step_distance = data.get("step_distance")
        calibration_range = data.get("calibration_range")
        dict_id = data.get("dict_id")
        flip_horizontal = data.get("flip_horizontal", False)

        return cls(
            grid_shape=grid_shape,
            points=points,
            stage_type=stage_type,
            timestamp=timestamp,
            enabled=enabled,
            step_distance=step_distance,
            calibration_range=calibration_range,
            dict_id=dict_id,
            flip_horizontal=flip_horizontal,
        )
