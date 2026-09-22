from typing import Callable, Optional

from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext, Operation
from lib.ar_tag import (
    compute_focus_score,
    detect_ar_tags,
    generate_ar_tag_grid,
)


class OpticsCalibrationOperation(Operation):
    """Calibrates the chromatic focal Z offset between Red and UV illumination.

    Process:
    1. User places a bare silicon chip on the stage and sets rough manual focus.
    2. Projects ArUco calibration tags in Red illumination via generated image source.
    3. Checks that tags are detected by the camera; aborts with a warning if not detected.
    4. Sweeps Z to maximize Red focal sharpness, finding Z_red.
    5. Switches to UV illumination and sweeps Z to find Z_uv.
    6. Computes chromatic offset: ΔZ = Z_uv - Z_red.
    7. Disables projector illumination and reports the offset.
    """

    def __init__(
        self,
        grid_n: int = 2,
        sweep_range: float = 30.0,
        sweep_step: float = 2.0,
    ):
        super().__init__("Optics Calibration")
        self.grid_n = grid_n
        self.sweep_range = sweep_range
        self.sweep_step = sweep_step
        self.red_best_z: Optional[float] = None
        self.uv_best_z: Optional[float] = None
        self.uv_z_offset: Optional[float] = None

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        report_progress(0.05, "Starting Optics Calibration...")
        proj_size = context.projector.projector_size()

        try:
            # 1. Project Red ArUco tags using software generated image source
            report_progress(0.1, "Projecting Red ArUco calibration tags...")
            red_img = generate_ar_tag_grid(self.grid_n, canvas_size=proj_size, color_mode=ColorMode.RED)
            context.projector.set_generated_image(red_img)
            context.projector.set_color_mode(ColorMode.RED)
            context.projector.set_image_source(ProjectorImageSource.GENERATED)
            context.projector.set_on(True)
            context.delay_func(0.5)

            if self.is_aborted:
                return "Optics calibration aborted"

            # 2. Check that AR tags are detected in Red illumination
            frame = context.camera.get_latest_frame()
            detected, corners, ids = detect_ar_tags(frame)
            if detected == 0:
                msg = (
                    "No ArUco tags detected in Red illumination! "
                    "Please verify that a bare silicon chip is on the stage, "
                    "rough focus is set, and the camera/projector are active."
                )
                if context.warning_callback:
                    context.warning_callback(msg)
                return msg

            # 3. Optimize focus in Red light
            report_progress(0.2, f"Detected {detected} tags. Optimizing Red focal sharpness...")
            z_start = context.stage.get_position()[2]
            best_red_score = -1.0
            best_red_z = z_start

            steps = max(1, int(self.sweep_range / self.sweep_step))
            for i in range(-steps, steps + 1):
                if self.is_aborted:
                    return "Optics calibration aborted"
                z = z_start + i * self.sweep_step
                context.stage.move_absolute({"z": z})
                context.delay_func(0.1)
                img = context.camera.get_latest_frame()
                score = compute_focus_score(img, blue_only=False)
                if score > best_red_score:
                    best_red_score = score
                    best_red_z = z

            # Move to best Red Z
            context.stage.move_absolute({"z": best_red_z})
            context.delay_func(0.2)
            self.red_best_z = best_red_z
            report_progress(0.5, f"Red focus locked at Z = {best_red_z:.2f} µm")

            if self.is_aborted:
                return "Optics calibration aborted"

            # 4. Switch to UV illumination with generated UV pattern
            report_progress(0.55, "Switching to UV illumination...")
            uv_img = generate_ar_tag_grid(self.grid_n, canvas_size=proj_size, color_mode=ColorMode.UV)
            context.projector.set_generated_image(uv_img)
            context.projector.set_color_mode(ColorMode.UV)
            context.projector.set_image_source(ProjectorImageSource.GENERATED)
            context.projector.set_on(True)
            context.delay_func(0.5)

            if self.is_aborted:
                return "Optics calibration aborted"

            # 5. Optimize focus in UV light
            report_progress(0.65, "Optimizing UV focal sharpness...")
            best_uv_score = -1.0
            best_uv_z = best_red_z

            for i in range(-steps, steps + 1):
                if self.is_aborted:
                    return "Optics calibration aborted"
                z = best_red_z + i * self.sweep_step
                context.stage.move_absolute({"z": z})
                context.delay_func(0.1)
                img = context.camera.get_latest_frame()
                score = compute_focus_score(img, blue_only=True)
                if score > best_uv_score:
                    best_uv_score = score
                    best_uv_z = z

            # Move to best UV Z
            context.stage.move_absolute({"z": best_uv_z})
            context.delay_func(0.2)
            self.uv_best_z = best_uv_z

            # 6. Compute UV - Red Z offset
            self.uv_z_offset = best_uv_z - best_red_z
            report_progress(
                1.0,
                f"Optics Calibration Complete: Red Z = {best_red_z:.2f} µm, "
                f"UV Z = {best_uv_z:.2f} µm, Offset ΔZ = {self.uv_z_offset:+.2f} µm",
            )
            return None

        finally:
            # Unconditionally turn off projector and restore source
            context.projector.set_on(False)
            context.projector.set_generated_image(None)
            context.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)
