from dataclasses import dataclass
from typing import Callable, Optional, Union
import cv2
import numpy as np

from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext, Operation, sub_progress
from lib.ar_tag import detect_ar_tags, generate_ar_tag_grid
from operations.maximize_image_sharpness import MaximizeImageSharpnessOperation


@dataclass
class AutofocusConfig:
    enabled: bool = True
    uv_z_offset: float = 0.0
    min_detection_rate: float = 0.5

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "AutofocusConfig":
        if not d:
            return cls()
        return cls(
            enabled=bool(d.get("enabled", True)),
            uv_z_offset=float(d.get("uv_z_offset", 0.0)),
            min_detection_rate=float(d.get("min_detection_rate", 0.5)),
        )


class AutofocusOperation(Operation):
    """Performs autofocus calibration using iterative ArUco tag grids and MaximizeImageSharpnessOperation."""

    def __init__(
        self,
        log: bool = False,
        config: Optional[Union[AutofocusConfig, dict]] = None,
    ):
        super().__init__("Autofocus")
        self.log = log
        if isinstance(config, dict):
            self.config = AutofocusConfig.from_dict(config)
        else:
            self.config = config or AutofocusConfig()
        self._current_sub_op: Optional[Operation] = None
        self.best_red_z: Optional[float] = None

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if not self.config.enabled:
            report_progress(1.0, "Autofocus disabled in config")
            return "Autofocus disabled in config"

        if self.is_aborted:
            return "Autofocus aborted"

        report_progress(0.05, "Starting iterative ArUco autofocus...")

        projector = context.projector
        assert projector is not None, "Projector not found"
        proj_size = projector.projector_size()

        best_overall_z = context.stage.get_position()[2]

        try:
            # Iterative grid progression: 2x2 (4 large tags) -> 4x4 (16 tags) -> 8x8 (64 tags)
            grid_sizes = [2, 4]
            step_configs = [
                (250, 5),
                (10, 0.5),
            ]

            stage_pct_span = 0.85 / len(grid_sizes)
            for idx, grid_n in enumerate(grid_sizes):
                if self.is_aborted:
                    return "Autofocus aborted"

                total_tags = grid_n * grid_n
                stage_start = 0.05 + idx * stage_pct_span
                stage_end = stage_start + stage_pct_span
                report_progress(stage_start, f"Projecting {grid_n}x{grid_n} ArUco grid...")

                # 1. Generate and project ArUco grid using generated image source
                grid_img = generate_ar_tag_grid(
                    grid_n=grid_n,
                    canvas_size=proj_size,
                )
                projector.set_generated_image(grid_img)
                projector.set_color_mode(ColorMode.RED)
                projector.set_image_source(ProjectorImageSource.GENERATED)
                projector.set_on(True)
                context.delay_func(1.0)

                # 2. Determine search range & threshold for this grid level
                sweep_range, threshold = step_configs[idx]

                # 3. Maximize image sharpness via MaximizeImageSharpnessOperation
                sharp_start = stage_start + 0.05 * stage_pct_span
                sub_reporter = sub_progress(
                    report_progress,
                    sharp_start,
                    stage_end,
                    prefix=f"Grid {grid_n}x{grid_n}",
                )
                sharp_op = MaximizeImageSharpnessOperation(
                    z_range=sweep_range,
                    threshold=threshold,
                )
                self._current_sub_op = sharp_op
                if self.is_aborted:
                    sharp_op.abort()

                err = sharp_op.execute(context, report_progress=sub_reporter)
                self._current_sub_op = None

                if self.is_aborted:
                    return "Autofocus aborted"

                if err is not None:
                    return f"Autofocus failed: {err}"

                if sharp_op.result is not None:
                    best_overall_z = sharp_op.result.best_z

                # 5. Verify detection rate at best focus
                context.delay_func(1.0)
                frame_best = context.camera.get_latest_frame()

                # Disabled because the failure rate is too high

                # det_count_best, _, _ = detect_ar_tags(frame_best, flip_horizontal=True)
                # rate_best = det_count_best / total_tags if total_tags > 0 else 0.0

                # # If detection falls below threshold, do not progress to smaller tags
                # if rate_best < self.config.min_detection_rate:
                #     print(
                #         f"Post-focus {grid_n}x{grid_n} detection rate ({rate_best*100:.1f}%) "
                #         f"< {self.config.min_detection_rate*100:.1f}%. Halting further refinement.",
                #         flush=True,
                #     )
                #     break

            self.best_red_z = float(best_overall_z)

            # 4. Apply UV-Red Z offset if offset is configured
            if abs(self.config.uv_z_offset) > 1e-6:
                report_progress(0.92, "Applying UV-Red Z offset...")
                uv_focal_z = best_overall_z + self.config.uv_z_offset
                if not context.stage.move_absolute({"z": uv_focal_z}):
                    msg = f"Failed to apply UV-Red Z offset to {uv_focal_z:.2f} µm"
                    if context.warning_callback:
                        context.warning_callback(msg)
                    return msg
                print(f"Applied UV-Red Z offset: {self.config.uv_z_offset:+.2f} µm (Z: {best_overall_z:.2f} -> {uv_focal_z:.2f} µm)", flush=True)

            report_progress(1.0, f"Autofocus complete at Z = {best_overall_z:.2f} µm")
            return None

        finally:
            if projector is not None:
                projector.set_on(False)
                projector.set_generated_image(None)
                projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)