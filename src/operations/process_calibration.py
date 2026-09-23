from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np

from core.chip_project import PatterningSettings
from core.events import ProjectorImageSource
from core.operation import ExecutionContext, Operation
from lib.gen_calibration_pattern import generate_litho_target
from operations.autofocus import AutofocusOperation
from operations.exposure import ExposureOperation
from operations.movement import JogOperation


def generate_spiral_offsets(count: int, step_distance: float) -> List[Tuple[float, float]]:
    """Generates `count` (dx, dy) coordinate offsets in spiral order starting from center (0, 0).
    
    Order:
      0: (0, 0)
      1: (+d, 0)
      2: (+d, +d)
      3: (0, +d)
      4: (-d, +d)
      ...
    """
    if count <= 0:
        return []
    offsets: List[Tuple[float, float]] = [(0.0, 0.0)]
    if count == 1:
        return offsets

    x, y = 0, 0
    dx, dy = 1, 0
    segment_length = 1
    segment_passed = 0
    turns = 0

    while len(offsets) < count:
        x += dx
        y += dy
        offsets.append((float(x * step_distance), float(y * step_distance)))
        segment_passed += 1

        if segment_passed == segment_length:
            segment_passed = 0
            # Turn 90 degrees CCW: (1,0) -> (0,1) -> (-1,0) -> (0,-1)
            dx, dy = -dy, dx
            turns += 1
            if turns % 2 == 0:
                segment_length += 1

    return offsets[:count]


@dataclass
class ProcessCalibrationConfig:
    min_exposure: float = 1.0  # seconds
    max_exposure: float = 5.0  # seconds
    sweep_steps: int = 5
    motion_distance: float = 1000.0  # µm


class ProcessCalibrationOperation(Operation):
    """Automates process calibration (FEM) across an array of positions in a 2D spiral order.
    
    At each position:
      1. Moves stage to spiral location (starting at center for step 0).
      2. Performs autofocus.
      3. Generates calibration pattern fitting the projector resolution with exposure length as title.
      4. Sets generated pattern to projector and exposes for the configured duration.
    """

    def __init__(
        self,
        min_exposure: float = 1.0,
        max_exposure: float = 5.0,
        sweep_steps: int = 5,
        motion_distance: float = 1000.0,
        config: Optional[ProcessCalibrationConfig] = None,
    ):
        super().__init__("Process Calibration")
        if config is not None:
            self.min_exposure = config.min_exposure
            self.max_exposure = config.max_exposure
            self.sweep_steps = config.sweep_steps
            self.motion_distance = config.motion_distance
        else:
            self.min_exposure = min_exposure
            self.max_exposure = max_exposure
            self.sweep_steps = sweep_steps
            self.motion_distance = motion_distance

        self._current_sub_op: Optional[Operation] = None

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def _run_sub_op(
        self,
        op: Operation,
        context: ExecutionContext,
        progress_cb: Callable[[float, str], None],
    ) -> Optional[str]:
        if self.is_aborted:
            return "Aborted"
        self._current_sub_op = op
        try:
            return op.execute(context, progress_cb)
        finally:
            self._current_sub_op = None

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if self.is_aborted:
            return "Process calibration aborted"

        projector = context.projector
        stage = context.stage
        if projector is None or stage is None:
            return "Projector or stage not found in context"

        total_steps = max(1, int(self.sweep_steps))
        if total_steps == 1:
            exposures = [float(self.min_exposure)]
        else:
            exposures = [float(e) for e in np.linspace(self.min_exposure, self.max_exposure, total_steps)]

        offsets = generate_spiral_offsets(total_steps, self.motion_distance)

        # Record starting position
        start_x, start_y, _ = stage.get_position()
        pw, ph = projector.projector_size()
        square_size = min(pw, ph)

        report_progress(0.0, f"Starting process calibration ({total_steps} steps)...")

        try:
            for step_idx, (exp_s, (off_x, off_y)) in enumerate(zip(exposures, offsets)):
                if self.is_aborted:
                    break

                pct_base = step_idx / total_steps
                pct_step = 1.0 / total_steps
                step_num = step_idx + 1

                # 1. Move stage (step 0 starts at current position, subsequent steps move to spiral offset)
                if step_idx > 0:
                    target_x = start_x + off_x
                    target_y = start_y + off_y
                    report_progress(
                        pct_base,
                        f"Step {step_num}/{total_steps}: Moving to spiral pos ({target_x:.1f}, {target_y:.1f}) µm...",
                    )
                    jog_op = JogOperation({"x": target_x, "y": target_y}, relative=False)
                    err = self._run_sub_op(jog_op, context, lambda p, m: None)
                    if err or self.is_aborted:
                        break

                # 2. Autofocus
                report_progress(
                    pct_base + 0.15 * pct_step,
                    f"Step {step_num}/{total_steps}: Autofocusing...",
                )
                af_config = getattr(context, "autofocus_config", None)
                af_op = AutofocusOperation(blue_only=False, config=af_config)
                err = self._run_sub_op(
                    af_op,
                    context,
                    lambda p, m: report_progress(
                        pct_base + (0.15 + 0.35 * p) * pct_step,
                        f"Step {step_num}/{total_steps} - AF: {m}",
                    ),
                )
                if err or self.is_aborted:
                    if err:
                        return f"Autofocus failed at step {step_num}: {err}"
                    break

                # 3. Generate pattern fitting projector resolution with exposure length as title
                report_progress(
                    pct_base + 0.55 * pct_step,
                    f"Step {step_num}/{total_steps}: Generating pattern for {exp_s:.4g}s...",
                )
                title = f"{exp_s:.4g}s"
                sub_title = f"STEP: {step_num}/{total_steps}\nEXP: {exp_s:.4g}s"
                pattern_sq = generate_litho_target(
                    size=square_size,
                    main_text=title,
                    sub_text=sub_title,
                )

                # Fit pattern into projector canvas
                canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
                if pattern_sq.ndim == 2:
                    pattern_rgb = cv2.cvtColor(pattern_sq, cv2.COLOR_GRAY2RGB)
                else:
                    pattern_rgb = pattern_sq

                y_offset = (ph - square_size) // 2
                x_offset = (pw - square_size) // 2
                canvas[y_offset : y_offset + square_size, x_offset : x_offset + square_size] = pattern_rgb

                # 4. Set generated image and expose
                projector.set_generated_image(canvas)
                projector.set_image_source(ProjectorImageSource.GENERATED)

                duration_ms = exp_s * 1000.0
                report_progress(
                    pct_base + 0.6 * pct_step,
                    f"Step {step_num}/{total_steps}: Exposing {exp_s:g}s ({int(duration_ms)} ms)...",
                )
                settings = PatterningSettings(exposure_time=duration_ms)
                exp_op = ExposureOperation(layer_index=None, settings=settings)
                err = self._run_sub_op(
                    exp_op,
                    context,
                    lambda p, m: report_progress(
                        pct_base + (0.6 + 0.4 * p) * pct_step,
                        f"Step {step_num}/{total_steps} - {m}",
                    ),
                )
                if err or self.is_aborted:
                    break

        finally:
            if projector is not None:
                projector.set_on(False)
                projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

        if self.is_aborted:
            report_progress(1.0, "Process calibration aborted")
            return "Process calibration aborted"
        else:
            report_progress(1.0, "Process calibration complete")
            return None
