import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union

from core.operation import ExecutionContext, Operation
from lib.ar_tag import compute_focus_score


@dataclass
class SharpnessOptimizationResult:
    """Outcome of the sharpness maximization search."""
    best_z: float
    best_score: float
    iterations_completed: int
    converged: bool
    measured_points: Dict[float, float] = field(default_factory=dict)


class MaximizeImageSharpnessOperation(Operation):
    """Operation that maximizes image sharpness along the Z-axis without modifying the projector.

    Uses an iterative 3-point search (similar to binary/ternary search) that evaluates the bottom,
    middle, and top Z positions, selecting the two sharpest points to form the next sub-range.
    """

    def __init__(
        self,
        z_range: Union[float, Tuple[float, float]] = 20.0,
        threshold: float = 0.5,
        max_iterations: int = 10,
        blue_only: bool = False,
        settle_delay: float = 0.5,
    ):
        super().__init__("Maximize Image Sharpness")
        self.z_range = z_range
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.blue_only = blue_only
        self.settle_delay = settle_delay
        self.result: Optional[SharpnessOptimizationResult] = None

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if self.is_aborted:
            return "Sharpness maximization aborted"

        report_progress(0.05, "Starting image sharpness maximization...")

        current_z = context.stage.get_position()[2]

        # Determine initial search interval [z_low, z_high]
        if isinstance(self.z_range, (int, float)):
            half_range = abs(float(self.z_range))
            z_low = current_z - half_range
            z_high = current_z + half_range
        elif isinstance(self.z_range, (tuple, list)) and len(self.z_range) == 2:
            z_low = min(float(self.z_range[0]), float(self.z_range[1]))
            z_high = max(float(self.z_range[0]), float(self.z_range[1]))
        else:
            return f"Invalid z_range: {self.z_range}. Must be float or Tuple[float, float]."

        if z_high <= z_low:
            z_high = z_low + self.threshold

        # Cache for measured points: round(z, 4) -> score
        cache: Dict[float, float] = {}
        best_z = current_z
        best_score = -1.0

        def measure(z_target: float) -> Optional[float]:
            nonlocal best_z, best_score
            key = round(float(z_target), 4)
            if key in cache:
                return cache[key]

            if self.is_aborted:
                return None

            if not context.stage.move_absolute({"z": z_target}):
                msg = f"Failed to move Z stage to {z_target:.3f} µm (boundary limit reached)"
                if context.warning_callback:
                    context.warning_callback(msg)
                return None

            context.delay_func(self.settle_delay)
            frame = context.camera.get_latest_frame()
            score = compute_focus_score(frame, blue_only=self.blue_only)
            cache[key] = score

            if score > best_score:
                best_score = score
                best_z = z_target

            return score

        # Initial sampling of bottom, middle, top
        z_mid = (z_low + z_high) / 2.0

        s_low = measure(z_low)
        if s_low is None:
            return "Sharpness maximization aborted" if self.is_aborted else "Failed to move Z stage to lower limit"

        s_mid = measure(z_mid)
        if s_mid is None:
            return "Sharpness maximization aborted" if self.is_aborted else "Failed to move Z stage to middle position"

        s_high = measure(z_high)
        if s_high is None:
            return "Sharpness maximization aborted" if self.is_aborted else "Failed to move Z stage to upper limit"

        iterations = 0
        converged = False

        while iterations < self.max_iterations:
            interval_width = z_high - z_low
            if interval_width <= self.threshold:
                converged = True
                break

            if self.is_aborted:
                return "Sharpness maximization aborted"

            iterations += 1

            # We have three points: (z_low, s_low), (z_mid, s_mid), (z_high, s_high)
            # Select the two highest-scoring points to form the next search bracket
            candidates = [
                (z_low, s_low),
                (z_mid, s_mid),
                (z_high, s_high),
            ]
            candidates.sort(key=lambda p: p[1], reverse=True)
            top1, top2 = candidates[0], candidates[1]

            # The middle point should never be worse than both ends, if that happens, something is wrong
            if s_mid < s_low and s_mid < s_high:
                return "Failed to maximize image sharpness, middle point worse than both endpoints"


            # Standard unimodal case: top two points are either [z_low, z_mid] or [z_mid, z_high]
            new_z_low = min(top1[0], top2[0])
            new_z_high = max(top1[0], top2[0])

            # Update scores for the new endpoints (already measured and cached)
            s_low = cache[round(new_z_low, 4)]
            s_high = cache[round(new_z_high, 4)]
            z_low, z_high = new_z_low, new_z_high

            # Check convergence after contraction
            if (z_high - z_low) <= self.threshold:
                converged = True
                break

            # Compute new middle point and sample
            z_mid = (z_low + z_high) / 2.0
            s_mid = measure(z_mid)
            if s_mid is None:
                return "Sharpness maximization aborted" if self.is_aborted else "Failed to move Z stage"

            progress = min(1.0, iterations / max(1, self.max_iterations))
            report_progress(
                progress,
                f"Iter {iterations}/{self.max_iterations}: Z range [{z_low:.2f}, {z_high:.2f}], best score {best_score:.2f}",
            )

        # Finally, reposition stage to the global best Z position found
        if not context.stage.move_absolute({"z": best_z}):
            msg = f"Failed to return Z stage to best position {best_z:.3f} µm"
            if context.warning_callback:
                context.warning_callback(msg)
            return msg

        context.delay_func(self.settle_delay)

        self.result = SharpnessOptimizationResult(
            best_z=best_z,
            best_score=best_score,
            iterations_completed=iterations,
            converged=converged,
            measured_points=cache,
        )

        report_progress(1.0, f"Sharpness maximized at Z = {best_z:.2f} µm (score = {best_score:.2f})")
        return None
