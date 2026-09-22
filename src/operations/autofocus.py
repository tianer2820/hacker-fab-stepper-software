import os
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext, Operation
from lib.ar_tag import (
    compute_focus_score,
    detect_ar_tags,
    generate_ar_tag_grid,
)


def execute_autofocus(
    has_homing: bool,
    get_autofocus_base: Callable[[], float],
    get_current_z: Callable[[], float],
    move_absolute: Callable[[dict[str, float]], bool],
    move_relative: Callable[[dict[str, float]], bool],
    get_camera_image: Callable[[], Optional[np.ndarray]],
    delay: Callable[[float], None] = time.sleep,
    on_warning: Optional[Callable[[str], None]] = None,
    blue_only: bool = False,
    log: bool = False,
    projector: Optional[Any] = None,
    projector_size: Tuple[int, int] = (1920, 1080),
    min_detection_rate: float = 0.85,
    uv_z_offset: float = 0.0,
    is_aborted: Optional[Callable[[], bool]] = None,
) -> bool:
    """Executes iterative ArUco autofocus independently of the GUI framework.

    Iterates through tag grids (2x2 -> 4x4 -> 8x8). For each level, optimizes focal
    sharpness. The iteration stops when fewer than min_detection_rate (default 85%) of tags
    can be detected. Finally, applies the UV-Red Z offset if operating in red mode.

    Returns True if autofocus completed successfully, False otherwise.
    """
    if log:
        try:
            os.mkdir("aftest")
        except FileExistsError:
            pass
        log_file = open("aftest/log.csv", "w")
    else:
        log_file = None

    try:
        print("Starting Autofocus...")
        current_z = get_current_z() if get_current_z is not None else get_autofocus_base()
        best_overall_z = current_z

        # Target illumination color
        target_mode = ColorMode.UV if blue_only else ColorMode.RED

        # Iterative grid progression: 2x2 (4 large tags) -> 4x4 (16 tags) -> 8x8 (64 tags)
        grid_sizes = [2, 4, 8]

        for grid_n in grid_sizes:
            if is_aborted and is_aborted():
                return False

            total_tags = grid_n * grid_n

            # 1. Generate and project ArUco grid using generated image source
            if projector is not None:
                grid_img = generate_ar_tag_grid(
                    grid_n=grid_n,
                    canvas_size=projector_size,
                    color_mode=target_mode,
                )
                projector.set_generated_image(grid_img)
                projector.set_color_mode(target_mode)
                projector.set_image_source(ProjectorImageSource.GENERATED)
                projector.set_on(True)
                delay(0.2)

            # 2. Check tag detection
            frame = get_camera_image()
            det_count, _, _ = detect_ar_tags(frame, flip_horizontal=True)
            detection_rate = det_count / total_tags if total_tags > 0 else 0.0

            # If moving to a finer grid (N > 2) and detection rate is below threshold, stop iteration
            if grid_n > 2 and detection_rate < min_detection_rate:
                print(
                    f"Grid {grid_n}x{grid_n} detection rate ({detection_rate*100:.1f}%) "
                    f"< {min_detection_rate*100:.1f}%. Stopping iteration."
                )
                break

            # 3. Determine sweep range & step size for this grid level
            if grid_n == 2:
                sweep_range = 20.0
                step_size = 4.0
            elif grid_n == 4:
                sweep_range = 10.0
                step_size = 2.0
            else:
                sweep_range = 4.0
                step_size = 1.0

            # 4. Sweep Z around current best Z to maximize sharpness
            best_score = -1.0
            best_grid_z = current_z
            steps = int(sweep_range / step_size)

            for i in range(-steps, steps + 1):
                if is_aborted and is_aborted():
                    return False

                target_z = current_z + (i * step_size)
                if move_absolute({"z": target_z}):
                    delay(0.1)
                else:
                    if on_warning:
                        on_warning("Failed autofocus, z-stage can't go past boundary limits")
                    return False

                img = get_camera_image()
                score = compute_focus_score(img, blue_only=blue_only)
                print(f"focus average: {score}")

                if score > best_score:
                    best_score = score
                    best_grid_z = target_z

            print(f"Fine grain sampling done, best focus is: {best_score}")

            # Move to best Z found in this grid iteration
            move_absolute({"z": best_grid_z})
            current_z = best_grid_z
            best_overall_z = best_grid_z
            delay(0.2)

            # Verify detection rate at best focus
            frame_best = get_camera_image()
            det_count_best, _, _ = detect_ar_tags(frame_best, flip_horizontal=True)
            rate_best = det_count_best / total_tags if total_tags > 0 else 0.0

            # If detection falls below threshold, do not progress to smaller tags
            if rate_best < min_detection_rate:
                print(
                    f"Post-focus {grid_n}x{grid_n} detection rate ({rate_best*100:.1f}%) "
                    f"< {min_detection_rate*100:.1f}%. Halting further refinement."
                )
                break

        # 5. Apply UV-Red Z offset if in red mode and offset is configured
        if not blue_only and abs(uv_z_offset) > 1e-6:
            uv_focal_z = best_overall_z + uv_z_offset
            move_absolute({"z": uv_focal_z})
            print(f"Applied UV-Red Z offset: {uv_z_offset:+.2f} µm (Z: {best_overall_z:.2f} -> {uv_focal_z:.2f} µm)")

        print("Autofocus Complete.")
        return True

    finally:
        if projector is not None:
            projector.set_on(False)
            projector.set_generated_image(None)
            projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)
        if log_file:
            log_file.close()


@dataclass
class AutofocusConfig:
    enabled: bool = True
    uv_z_offset: float = 0.0
    min_detection_rate: float = 0.85

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "AutofocusConfig":
        if not d:
            return cls()
        return cls(
            enabled=bool(d.get("enabled", True)),
            uv_z_offset=float(d.get("uv_z_offset", 0.0)),
            min_detection_rate=float(d.get("min_detection_rate", 0.85)),
        )


class AutofocusOperation(Operation):
    """Performs autofocus calibration using iterative ArUco tag grids."""

    def __init__(
        self,
        blue_only: bool = False,
        log: bool = False,
        config: Optional[Union[AutofocusConfig, dict]] = None,
    ):
        super().__init__("Autofocus")
        self.blue_only = blue_only
        self.log = log
        if isinstance(config, dict):
            self.config = AutofocusConfig.from_dict(config)
        else:
            self.config = config or AutofocusConfig()

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if not self.config.enabled:
            report_progress(1.0, "Autofocus disabled in config")
            return "Autofocus disabled in config"

        report_progress(0.1, "Executing iterative ArUco autofocus...")

        success = execute_autofocus(
            has_homing=context.stage.has_homing(),
            get_autofocus_base=lambda: context.stage.get_position()[2],
            get_current_z=lambda: context.stage.get_position()[2],
            move_absolute=context.stage.move_absolute,
            move_relative=context.stage.move_relative,
            get_camera_image=context.camera.get_latest_frame,
            delay=context.delay_func,
            on_warning=context.warning_callback,
            blue_only=self.blue_only,
            log=self.log,
            projector=context.projector,
            projector_size=context.projector.projector_size(),
            min_detection_rate=self.config.min_detection_rate,
            uv_z_offset=self.config.uv_z_offset,
            is_aborted=lambda: self.is_aborted,
        )

        if not success:
            return "Autofocus failed or aborted"

        report_progress(1.0, "Autofocus complete")
        return None