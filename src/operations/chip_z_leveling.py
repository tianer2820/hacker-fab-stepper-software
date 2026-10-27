# Hacker Fab
# Chip Z Leveling Operation
# Performs 4-point autofocus around the chip corners to establish a temporary leveling plane

from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np

from core.operation import ExecutionContext, Operation, sub_progress
from operations.autofocus import AutofocusOperation


def fit_plane(points: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Fits a plane Delta_Z = a*X + b*Y + c to measured (X, Y, Delta_Z) points using least squares.

    Returns:
        (a, b, c) plane coefficients.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError(f"Need at least 3 points to fit a plane, got {len(pts)}")
    # A has shape (N, 3): [X, Y, 1]
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts), dtype=np.float64)])
    b = pts[:, 2]
    coef, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


class ChipZLevelingOperation(Operation):
    """Temporary chip Z leveling operation using 4-point autofocus plane fitting.

    Saves current XY coordinate, moves to 4 corners of the chip defined by chip_size,
    runs autofocus at each point, returns to the original coordinate, fits a plane to
    the focal height deviations, and sets the plane offset on the StageCalibration.
    """

    def __init__(
        self,
        chip_size_mm: Optional[float] = None,
        chip_size: Optional[float] = None,
        stabilization_delay: float = 0.5,
    ):
        super().__init__("Chip Z Leveling")
        size = chip_size_mm if chip_size_mm is not None else (chip_size if chip_size is not None else 10.0)
        self.chip_size_mm = float(size)
        self.stabilization_delay = float(stabilization_delay)
        self._current_sub_op: Optional[Operation] = None
        self.fitted_plane: Optional[Tuple[float, float, float]] = None
        self.measured_points: List[Tuple[float, float, float]] = []

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def _run_sub_op(
        self,
        op: Operation,
        context: ExecutionContext,
        report_progress: Callable[[float, str], None],
    ) -> Optional[str]:
        self._current_sub_op = op
        err = op.execute(context, report_progress)
        self._current_sub_op = None
        return err

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if context.stage is None:
            return "Stage controller not available in execution context"

        stage = context.stage
        cal = stage.get_calibration()
        if cal is None:
            msg = "No StageCalibration loaded on StageController. Chip Z leveling requires an active stage calibration."
            if context.warning_callback:
                context.warning_callback(msg)
            return msg

        if self.is_aborted:
            return "Chip Z Leveling aborted"

        # Save initial stage position
        orig_x, orig_y, orig_z = stage.get_position()

        # Clear any previously set chip Z plane so measurements are relative to base calibration
        cal.clear_chip_z_plane()

        # Compute 4 measurement corners (in µm) centered around current XY position
        chip_size_um = self.chip_size_mm * 1000.0
        h = chip_size_um / 2.0

        points = [
            (orig_x - h, orig_y - h),
            (orig_x + h, orig_y - h),
            (orig_x + h, orig_y + h),
            (orig_x - h, orig_y + h),
        ]

        report_progress(0.0, f"Starting Chip Z Leveling for {self.chip_size_mm:.1f} mm chip...")
        self.measured_points = []

        try:
            for idx, (px, py) in enumerate(points):
                if self.is_aborted:
                    return "Chip Z Leveling aborted"

                prog_start = (idx / 4.0) * 0.85
                prog_end = ((idx + 1) / 4.0) * 0.85

                report_progress(
                    prog_start,
                    f"Point {idx + 1}/4: Moving to ({px:.1f}, {py:.1f}) µm...",
                )

                # Move in corrected coordinates with z=0.0 (surface height)
                if not stage.move_absolute({"x": px, "y": py, "z": 0.0}):
                    msg = f"Failed to move stage to point {idx + 1} ({px:.1f}, {py:.1f}) µm"
                    if context.warning_callback:
                        context.warning_callback(msg)
                    return msg

                if self.stabilization_delay > 0:
                    context.delay_func(self.stabilization_delay)

                # Autofocus in red light
                af_op = AutofocusOperation(uv_z_offset=0.0)
                af_reporter = sub_progress(
                    report_progress,
                    prog_start + 0.1 * (prog_end - prog_start),
                    prog_end,
                    prefix=f"Point {idx + 1}/4 - AF",
                )
                af_err = self._run_sub_op(af_op, context, af_reporter)
                if self.is_aborted:
                    return "Chip Z Leveling aborted"
                if af_err is not None:
                    return f"Autofocus failed at point {idx + 1} ({px:.1f}, {py:.1f}): {af_err}"

                if af_op.best_red_z is not None:
                    z_focus_corr = float(af_op.best_red_z)
                else:
                    z_focus_corr = float(stage.get_position()[2])

                # Since stage was moved to z=0.0, the best Z in corrected coords is delta_z
                self.measured_points.append((px, py, z_focus_corr))

            # Move back to original point
            report_progress(0.85, "Returning stage to original position...")
            if not stage.move_absolute({"x": orig_x, "y": orig_y, "z": orig_z}):
                msg = "Failed to return stage to original position"
                if context.warning_callback:
                    context.warning_callback(msg)
                return msg

            # Calculate chip Z plane using the 4 measured coordinates
            a, b, c = fit_plane(self.measured_points)
            self.fitted_plane = (a, b, c)

            # Apply temporary plane offset to stage calibration
            cal.set_chip_z_plane((a, b, c))

            report_progress(
                1.0,
                f"Chip Z Leveling complete. Plane: ΔZ = {a:.6e}*X + {b:.6e}*Y + {c:.2f} µm",
            )
            return None

        finally:
            # Ensure stage returns to initial position if aborted or error occurred before returning
            current_pos = stage.get_position()
            if (
                abs(current_pos[0] - orig_x) > 1e-3
                or abs(current_pos[1] - orig_y) > 1e-3
                or abs(current_pos[2] - orig_z) > 1e-3
            ):
                stage.move_absolute({"x": orig_x, "y": orig_y, "z": orig_z})


# Alias matching operation naming convention
chip_z_leveling = ChipZLevelingOperation

