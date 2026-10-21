# Hacker Fab
# Stage Calibration Operations:
# Pass 1: Autofocus, Projector-to-Camera Optical Transform, 3-Step Iterative Correction, and Patterning
# Pass 2: Chip Pose Estimation, Multi-Tag Camera Scanning, Relative Least Squares Global Reconstruction

from dataclasses import dataclass
from datetime import datetime
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext, Operation, sub_progress
from lib.ar_tag import detect_ar_tags, get_aruco_dict
from lib.projector_camera_transform import (
    compute_projector_to_camera_homography,
    generate_calibration_tag_grid_with_coords,
    transform_points_projector_to_camera,
)
from operations.autofocus import AutofocusConfig, AutofocusOperation
from operations.exposure import ExposureOperation, ExposureOperationConfig
from stage_control.calibration import StageCalibration, StageCalibrationPoint

DEFAULT_ARUCO_DICT = cv2.aruco.DICT_4X4_1000


def compute_calibration_grid_params(
    cal_range: float,
    pixel_scale: float,
    camera_size: Tuple[int, int] = (1920, 1080),
) -> Tuple[int, float]:
    """Computes square grid_size N and step_distance d.

    Ensures step_distance <= 0.4 * min(FOV_x, FOV_y) so that any 2x2 cluster
    of 4 adjacent tags fits within a single camera frame.

    Returns:
        (grid_size, step_distance) where grid_size is N (for N x N grid).
    """
    cam_w, cam_h = camera_size
    fov = min(cam_w, cam_h) * float(pixel_scale)
    target_pitch = max(100.0, 0.4 * fov)

    intervals = max(1, int(round(cal_range / target_pitch)))
    grid_size = intervals + 1
    step_distance = float(cal_range / intervals)
    return grid_size, step_distance


def generate_spiral_grid_indices(nx: int, ny: int) -> List[Tuple[int, int]]:
    """Generates (row, col) grid indices in center-to-edge spiral order.

    Guarantees all (r, c) in [0, ny) x [0, nx) are visited exactly once,
    starting from the center grid cell ((ny - 1) // 2, (nx - 1) // 2).
    """
    total = nx * ny
    if total <= 0:
        return []

    c0 = (nx - 1) // 2
    r0 = (ny - 1) // 2

    indices: List[Tuple[int, int]] = [(r0, c0)]
    visited = {(r0, c0)}

    x, y = 0, 0
    dx, dy = 1, 0
    segment_length = 1
    segment_passed = 0
    turns = 0

    max_steps = max(nx, ny) * max(nx, ny) * 4 + 100
    steps = 0
    while len(indices) < total and steps < max_steps:
        x += dx
        y += dy
        steps += 1

        c = c0 + x
        r = r0 + y
        if 0 <= r < ny and 0 <= c < nx:
            if (r, c) not in visited:
                visited.add((r, c))
                indices.append((r, c))

        segment_passed += 1
        if segment_passed == segment_length:
            segment_passed = 0
            dx, dy = -dy, dx
            turns += 1
            if turns % 2 == 0:
                segment_length += 1

    return indices



def generate_canonical_ar_tag(
    marker_id: int,
    dict_id: int = DEFAULT_ARUCO_DICT,
    size: int = 400,
    quiet_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a square marker image with white quiet zone and returns canonical corners.

    Corners order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    dictionary = get_aruco_dict(dict_id)
    quiet = max(4, int(size * quiet_ratio))
    total_sz = size + 2 * quiet

    canvas = np.zeros((total_sz, total_sz), dtype=np.uint8)
    canvas[:, :] = 255  # white quiet zone background
    marker_img = cv2.aruco.generateImageMarker(dictionary, int(marker_id), size)
    canvas[quiet : quiet + size, quiet : quiet + size] = marker_img

    corners = np.array(
        [
            [0.0, 0.0],
            [float(total_sz), 0.0],
            [float(total_sz), float(total_sz)],
            [0.0, float(total_sz)],
        ],
        dtype=np.float32,
    )
    return canvas, corners


def warp_marker_to_projector(
    marker_id: int,
    target_proj_corners: np.ndarray,
    canvas_size: Tuple[int, int],
    dict_id: int = DEFAULT_ARUCO_DICT,
) -> np.ndarray:
    """Warps an ArUco marker onto the projector canvas to match target_proj_corners."""
    pw, ph = canvas_size
    canonical_img, canonical_corners = generate_canonical_ar_tag(marker_id, dict_id=dict_id, size=300)

    M = cv2.getPerspectiveTransform(canonical_corners, target_proj_corners.astype(np.float32))
    warped_gray = cv2.warpPerspective(canonical_img, M, (pw, ph), flags=cv2.INTER_LINEAR)

    bgr = np.zeros((ph, pw, 3), dtype=np.uint8)
    bgr[:, :, 0] = warped_gray
    bgr[:, :, 1] = warped_gray
    bgr[:, :, 2] = warped_gray
    return bgr


def solve_global_positions_from_relative_measurements(
    grid_size: int,
    measurements: List[Tuple[int, int, float, float]],
    anchor_id: int,
    anchor_pos: Tuple[float, float],
    nominal_positions: Dict[int, Tuple[float, float]],
) -> Dict[int, Tuple[float, float]]:
    """Reconstructs global optical positions using least squares from pairwise relative vectors.

    Args:
        grid_size: N for an N x N grid. Total markers M = N * N.
        measurements: List of (id_A, id_B, dx, dy) where x_B - x_A = dx and y_B - y_A = dy.
        anchor_id: Marker ID to anchor (e.g. 0).
        anchor_pos: (X, Y) coordinate for the anchor.
        nominal_positions: Dict mapping marker_id -> nominal (X, Y) fallback.

    Returns:
        Dict mapping marker_id -> reconstructed (X, Y).
    """
    total_markers = grid_size * grid_size
    if len(measurements) == 0:
        return dict(nominal_positions)

    K = len(measurements)
    A = np.zeros((K + 1, total_markers), dtype=np.float64)
    bx = np.zeros(K + 1, dtype=np.float64)
    by = np.zeros(K + 1, dtype=np.float64)

    for k, (id_a, id_b, dx, dy) in enumerate(measurements):
        if 0 <= id_a < total_markers and 0 <= id_b < total_markers:
            A[k, id_a] = -1.0
            A[k, id_b] = 1.0
            bx[k] = dx
            by[k] = dy

    # Anchor row
    A[K, anchor_id] = 1.0
    bx[K] = anchor_pos[0]
    by[K] = anchor_pos[1]

    sol_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)

    # Check for connected markers
    observed = set()
    for id_a, id_b, _, _ in measurements:
        observed.add(id_a)
        observed.add(id_b)
    observed.add(anchor_id)

    results: Dict[int, Tuple[float, float]] = {}
    for m_id in range(total_markers):
        if m_id in observed and not (np.isnan(sol_x[m_id]) or np.isnan(sol_y[m_id])):
            results[m_id] = (float(sol_x[m_id]), float(sol_y[m_id]))
        else:
            results[m_id] = nominal_positions.get(m_id, (0.0, 0.0))

    return results


@dataclass
class StageCalibrationPatternAndZConfig:
    calibration_range: float = 10000.0  # µm (total square span L)
    pixel_scale: float = 1.0  # µm per camera pixel
    exposure_time: float = 8.0  # seconds
    stabilization_delay: float = 3  # seconds
    autofocus_enabled: bool = True
    uv_z_offset: float = 0.0
    output_json_path: str = "stage_calibration_pass1.json"
    dict_id: int = DEFAULT_ARUCO_DICT

    # Backwards compatibility fields
    grid_size: Optional[int] = None
    step_distance: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "StageCalibrationPatternAndZConfig":
        if not d:
            return cls()
        return cls(
            calibration_range=float(d.get("calibration_range", 10000.0)),
            pixel_scale=float(d.get("pixel_scale", 1.0)),
            exposure_time=float(d.get("exposure_time", 8.0)),
            stabilization_delay=float(d.get("stabilization_delay", 0.5)),
            autofocus_enabled=bool(d.get("autofocus_enabled", True)),
            uv_z_offset=float(d.get("uv_z_offset", 0.0)),
            output_json_path=str(d.get("output_json_path", "stage_calibration_pass1.json")),
            dict_id=int(d.get("dict_id", DEFAULT_ARUCO_DICT)),
            grid_size=int(d["grid_size"]) if "grid_size" in d else None,
            step_distance=float(d["step_distance"]) if "step_distance" in d else None,
        )


class StageCalibrationPatternAndZOperation(Operation):
    """Pass 1: Autofocus, Optical Homography Measurement, 3-Step Iterative Correction, and Patterning."""

    def __init__(
        self,
        config: Optional[Union[StageCalibrationPatternAndZConfig, dict]] = None,
    ):
        super().__init__("Stage Calibration - Pass 1 (Pattern & Z)")
        if isinstance(config, dict):
            self.config = StageCalibrationPatternAndZConfig.from_dict(config)
        else:
            self.config = config or StageCalibrationPatternAndZConfig()
        self._current_sub_op: Optional[Operation] = None
        self.recorded_points: List[Dict[str, Any]] = []
        self.corrected_proj_corners: Optional[np.ndarray] = None
        self.flip_horizontal: bool = False

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
        if context.stage is None or context.projector is None or context.camera is None:
            return "Stage, Projector, or Camera not available in execution context"

        stage = context.stage
        projector = context.projector
        camera = context.camera
        pw, ph = projector.projector_size()

        # Step 0: Get camera frame dimensions
        init_frame = camera.get_latest_frame()
        cam_h, cam_w = init_frame.shape[:2] if init_frame is not None else (1080, 1920)
        u_c, v_c = cam_w / 2.0, cam_h / 2.0

        # Calculate grid parameters: N and d
        if self.config.grid_size is not None and self.config.step_distance is not None:
            nx = ny = self.config.grid_size
            step_dist = self.config.step_distance
        else:
            nx, step_dist = compute_calibration_grid_params(
                cal_range=self.config.calibration_range,
                pixel_scale=self.config.pixel_scale,
                camera_size=(cam_w, cam_h),
            )
            ny = nx

        total_points = nx * ny
        start_x, start_y, start_z = stage.get_raw_position()

        try:
            # =========================================================================
            # STEP 1: INITIAL AUTOFOCUS
            # =========================================================================
            report_progress(0.01, "Step 1/3: Running initial autofocus...")
            if self.config.autofocus_enabled:
                af_op = AutofocusOperation(
                    config=AutofocusConfig(
                        enabled=True,
                        uv_z_offset=self.config.uv_z_offset,
                    )
                )
                af_reporter = sub_progress(report_progress, 0.01, 0.06, prefix="Initial AF")
                af_err = self._run_sub_op(af_op, context, af_reporter)
                if af_err and self.is_aborted:
                    return "Operation aborted during autofocus"
                if af_op.best_red_z is not None:
                    start_z = float(af_op.best_red_z)

            if self.is_aborted:
                return "Operation aborted"

            # =========================================================================
            # STEP 2: MEASURE OPTICAL TRANSFORM (PROJECTOR TO CAMERA)
            # =========================================================================
            report_progress(0.06, "Step 2/3: Measuring optical transform (Projector -> Camera)...")
            grid_img, tag_coords = generate_calibration_tag_grid_with_coords(
                grid_n=2,
                canvas_size=(pw, ph),
                dict_id=cv2.aruco.DICT_4X4_250,
            )
            projector.set_generated_image(grid_img)
            projector.set_color_mode(ColorMode.RED)
            projector.set_image_source(ProjectorImageSource.GENERATED)
            projector.set_on(True)
            context.delay_func(self.config.stabilization_delay)

            cal_frame = camera.get_latest_frame()
            flip_horizontal = False
            det_count, corners, ids = detect_ar_tags(cal_frame, dict_id=cv2.aruco.DICT_4X4_250, flip_horizontal=False)
            if det_count < 2:
                det_count_flip, corners_flip, ids_flip = detect_ar_tags(cal_frame, dict_id=cv2.aruco.DICT_4X4_250, flip_horizontal=True)
                if det_count_flip > det_count:
                    det_count, corners, ids = det_count_flip, corners_flip, ids_flip
                    flip_horizontal = True
            self.flip_horizontal = flip_horizontal

            H_proj_to_cam, matched_count = compute_projector_to_camera_homography(corners, ids, tag_coords)
            projector.set_on(False)

            if H_proj_to_cam is None or matched_count < 4:
                return (
                    f"Projector-to-camera optical calibration failed: insufficient tag matches ({matched_count}/4 points). "
                    "Cannot calibrate stage without calibrated projection."
                )

            try:
                H_cam_to_proj = np.linalg.inv(H_proj_to_cam)
            except np.linalg.LinAlgError as e:
                return f"Failed to invert projector-to-camera homography matrix: {e}"

            # =========================================================================
            # STEP 3: 3-STEP ITERATIVE CORRECTION (TOP-LEFT CORNER AT CAMERA CENTER)
            # =========================================================================
            report_progress(0.10, "Step 3/3: Iteratively correcting projected tag alignment (3 passes)...")
            tag_sz_cam = max(24.0, min(cam_w, cam_h) * 0.15)
            # Target camera corners: Top-Left at (u_c, v_c), sitting in bottom-right quadrant
            cam_corners = np.array([
                [u_c, v_c],
                [u_c + tag_sz_cam, v_c],
                [u_c + tag_sz_cam, v_c + tag_sz_cam],
                [u_c, v_c + tag_sz_cam],
            ], dtype=np.float64)

            for iter_idx in range(3):
                if self.is_aborted:
                    return "Operation aborted"

                # Map target camera corners to projector coordinates
                proj_pts = transform_points_projector_to_camera(cam_corners, H_cam_to_proj)
                self.corrected_proj_corners = proj_pts

                # Project test marker in Red illumination
                test_img = warp_marker_to_projector(
                    marker_id=0,
                    target_proj_corners=proj_pts,
                    canvas_size=(pw, ph),
                    dict_id=self.config.dict_id,
                )
                projector.set_generated_image(test_img)
                projector.set_color_mode(ColorMode.RED)
                projector.set_on(True)
                context.delay_func(self.config.stabilization_delay)

                iter_frame = camera.get_latest_frame()
                c_cnt, c_corners, c_ids = detect_ar_tags(iter_frame, dict_id=self.config.dict_id, flip_horizontal=self.flip_horizontal)
                if c_cnt > 0 and c_corners:
                    detected_tl = c_corners[0].reshape(-1, 2)[0]  # Top-left corner
                    err = detected_tl - np.array([u_c, v_c])
                    # Adjust camera-space corners opposite to error
                    cam_corners[:, 0] -= err[0]
                    cam_corners[:, 1] -= err[1]

                projector.set_on(False)

            # Final corrected projector corners
            self.corrected_proj_corners = transform_points_projector_to_camera(cam_corners, H_cam_to_proj)

            # =========================================================================
            # STEP 4: PATTERNING & Z SURVEY (SPIRAL ORDER FROM CENTER TO EDGE)
            # =========================================================================
            report_progress(0.15, f"Patterning {nx}x{ny} grid (step: {step_dist:.1f} µm, range: {self.config.calibration_range:.1f} µm)...")
            self.recorded_points = []
            loop_span = 0.85
            base_offset = 0.15

            c0 = (nx - 1) // 2
            r0 = (ny - 1) // 2
            spiral_indices = generate_spiral_grid_indices(nx, ny)

            for point_idx, (r, c) in enumerate(spiral_indices):
                if self.is_aborted:
                    return "Operation aborted"

                prog_frac = base_offset + (point_idx / total_points) * loop_span
                marker_id = r * nx + c

                target_raw_x = start_x + (c - c0) * step_dist
                target_raw_y = start_y + (r - r0) * step_dist

                report_progress(
                    prog_frac,
                    f"Point {point_idx + 1}/{total_points} (Grid [{r}, {c}], Tag ID {marker_id}): Moving to ({target_raw_x:.1f}, {target_raw_y:.1f})...",
                )
                stage.move_absolute_raw({"x": target_raw_x, "y": target_raw_y})
                context.delay_func(self.config.stabilization_delay)

                # Autofocus
                z_focus = start_z
                if self.config.autofocus_enabled:
                    af_op = AutofocusOperation(
                        config=AutofocusConfig(
                            enabled=True,
                            uv_z_offset=self.config.uv_z_offset,
                        )
                    )
                    self._run_sub_op(af_op, context, lambda p, m: None)
                    if af_op.best_red_z is not None:
                        z_focus = float(af_op.best_red_z)
                    else:
                        z_focus = float(stage.get_raw_position()[2])
                else:
                    z_focus = float(stage.get_raw_position()[2])

                # Expose corrected single ArUco tag
                tag_img = warp_marker_to_projector(
                    marker_id=marker_id,
                    target_proj_corners=self.corrected_proj_corners,
                    canvas_size=(pw, ph),
                    dict_id=self.config.dict_id,
                )
                projector.set_generated_image(tag_img)
                projector.set_image_source(ProjectorImageSource.GENERATED)

                exp_config = ExposureOperationConfig(exposure_time=self.config.exposure_time * 1000.0)
                exp_op = ExposureOperation(layer_index=None, config=exp_config)
                pt_span = loop_span / total_points
                exp_reporter = sub_progress(
                    report_progress,
                    prog_frac + 0.5 * pt_span,
                    prog_frac + pt_span,
                    prefix=f"Point {point_idx + 1}/{total_points} (Grid [{r}, {c}], Tag ID {marker_id})",
                )
                exp_err = self._run_sub_op(
                    exp_op,
                    context,
                    exp_reporter,
                )
                if exp_err or self.is_aborted:
                    return exp_err or "Operation aborted during exposure"

                self.recorded_points.append({
                    "grid_index": [r, c],
                    "marker_id": marker_id,
                    "raw_xy": [float(target_raw_x), float(target_raw_y)],
                    "z_focus": float(z_focus),
                })

            # Save Pass 1 JSON
            pass1_data = {
                "version": "2.0",
                "timestamp": datetime.now().isoformat(),
                "grid_shape": [ny, nx],
                "step_distance": step_dist,
                "calibration_range": self.config.calibration_range,
                "pixel_scale": self.config.pixel_scale,
                "dict_id": self.config.dict_id,
                "flip_horizontal": self.flip_horizontal,
                "points": self.recorded_points,
            }
            os.makedirs(os.path.dirname(os.path.abspath(self.config.output_json_path)), exist_ok=True)
            with open(self.config.output_json_path, "w", encoding="utf-8") as f:
                json.dump(pass1_data, f, indent=2)

            report_progress(
                1.0,
                f"Pass 1 complete! Saved {len(self.recorded_points)} points to {self.config.output_json_path}. "
                "Please develop wafer, mount, and jog stage so at least one tag is visible to run Pass 2.",
            )
            return None

        finally:
            projector.set_on(False)
            projector.set_generated_image(None)
            projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)


@dataclass
class StageCalibrationScanConfig:
    input_json_path: str = "stage_calibration_pass1.json"
    output_json_path: str = "stage_calibration.json"
    pixel_scale: float = 1.0  # µm per camera pixel
    stabilization_delay: float = 0.5  # seconds
    apply_to_stage: bool = True
    dict_id: int = DEFAULT_ARUCO_DICT

    # Backwards compatibility fields
    pixel_scale_x: Optional[float] = None
    pixel_scale_y: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "StageCalibrationScanConfig":
        if not d:
            return cls()
        ps = float(d.get("pixel_scale", d.get("pixel_scale_x", 1.0)))
        return cls(
            input_json_path=str(d.get("input_json_path", "stage_calibration_pass1.json")),
            output_json_path=str(d.get("output_json_path", "stage_calibration.json")),
            pixel_scale=ps,
            stabilization_delay=float(d.get("stabilization_delay", 0.5)),
            apply_to_stage=bool(d.get("apply_to_stage", True)),
            dict_id=int(d.get("dict_id", DEFAULT_ARUCO_DICT)),
        )


class StageCalibrationScanOperation(Operation):
    """Pass 2: Chip pose estimation, multi-tag camera scanning, and Least Squares global reconstruction."""

    def __init__(
        self,
        config: Optional[Union[StageCalibrationScanConfig, dict]] = None,
    ):
        super().__init__("Stage Calibration - Pass 2 (Optical Scan)")
        if isinstance(config, dict):
            self.config = StageCalibrationScanConfig.from_dict(config)
        else:
            self.config = config or StageCalibrationScanConfig()
        self.calibration: Optional[StageCalibration] = None
        self.flip_horizontal: bool = False

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if context.stage is None or context.camera is None:
            return "Stage or Camera not available in execution context"

        if not os.path.exists(self.config.input_json_path):
            return f"Pass 1 input file not found: {self.config.input_json_path}"

        with open(self.config.input_json_path, "r", encoding="utf-8") as f:
            pass1_data = json.load(f)

        grid_shape = tuple(pass1_data.get("grid_shape", (0, 0)))
        ny, nx = grid_shape
        step_dist = float(pass1_data.get("step_distance", 5000.0))
        dict_id = int(pass1_data.get("dict_id", self.config.dict_id))
        raw_points = pass1_data.get("points", [])

        if len(raw_points) == 0:
            return "No calibration points found in input file"

        # Lookup dicts for nominal data
        nominal_dict: Dict[int, Dict[str, Any]] = {int(p["marker_id"]): p for p in raw_points}
        nominal_pos: Dict[int, Tuple[float, float]] = {
            int(p["marker_id"]): (float(p["raw_xy"][0]), float(p["raw_xy"][1])) for p in raw_points
        }
        z_dict: Dict[int, float] = {int(p["marker_id"]): float(p["z_focus"]) for p in raw_points}

        stage = context.stage
        camera = context.camera

        # =========================================================================
        # STEP 1: INITIAL CHIP POSE ESTIMATION FROM CURRENT CAMERA VIEW
        # =========================================================================
        report_progress(0.05, "Step 1/3: Detecting visible tag to estimate chip orientation and pose...")
        frame = camera.get_latest_frame()
        if frame is None:
            return "Failed to acquire camera frame"

        cam_h, cam_w = frame.shape[:2]
        u_c, v_c = cam_w / 2.0, cam_h / 2.0

        flip_horizontal = bool(pass1_data.get("flip_horizontal", False))
        det_count, corners, ids = detect_ar_tags(frame, dict_id=dict_id, flip_horizontal=flip_horizontal)
        if det_count == 0 or ids is None:
            det_count_alt, corners_alt, ids_alt = detect_ar_tags(frame, dict_id=dict_id, flip_horizontal=not flip_horizontal)
            if det_count_alt > 0 and ids_alt is not None:
                det_count, corners, ids = det_count_alt, corners_alt, ids_alt
                flip_horizontal = not flip_horizontal
        self.flip_horizontal = flip_horizontal

        if det_count == 0 or ids is None:
            return (
                "No ArUco tag detected in current camera view. "
                "Please jog the stage so at least one calibration tag is visible in the camera, then retry."
            )

        # Collect all detected markers that exist in Pass 1
        matched_tags: List[Tuple[int, np.ndarray]] = []
        for idx, mid in enumerate(ids.flatten()):
            mid_int = int(mid)
            if mid_int in nominal_dict:
                matched_tags.append((mid_int, corners[idx].reshape(-1, 2)))

        if not matched_tags:
            return f"Detected tag IDs ({ids.flatten().tolist()}) not found in Pass 1 calibration dataset."

        # Compute tag orientation angles and wafer origin estimates for all detected tags
        cur_stage_x, cur_stage_y, _ = stage.get_raw_position()
        thetas: List[float] = []
        for mid_int, tag_corners in matched_tags:
            vec_x = tag_corners[1] - tag_corners[0]
            thetas.append(float(np.arctan2(vec_x[1], vec_x[0])))

        # Circular mean of orientation angles to reduce measurement error
        mean_sin = float(np.mean([np.sin(t) for t in thetas]))
        mean_cos = float(np.mean([np.cos(t) for t in thetas]))
        theta = float(np.arctan2(mean_sin, mean_cos))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_chip = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        R_inv = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)

        # Average wafer origin in stage coordinates across all detected tags
        wafer_origin_estimates: List[np.ndarray] = []
        for mid_int, tag_corners in matched_tags:
            tag_cam_center = np.mean(tag_corners, axis=0)
            delta_cam_px = tag_cam_center - np.array([u_c, v_c])
            delta_cam_um = delta_cam_px * self.config.pixel_scale
            tag_stage_pos = np.array([cur_stage_x + delta_cam_um[0], cur_stage_y + delta_cam_um[1]], dtype=np.float64)

            ref_grid_r, ref_grid_c = nominal_dict[mid_int]["grid_index"]
            tag_wafer_offset = np.array([ref_grid_c * step_dist, ref_grid_r * step_dist], dtype=np.float64)
            # In wafer space: tag_stage_pos = wafer_origin + R_chip @ tag_wafer_offset
            origin_est = tag_stage_pos - (R_chip @ tag_wafer_offset)
            wafer_origin_estimates.append(origin_est)

        wafer_stage_origin = np.mean(wafer_origin_estimates, axis=0)

        report_progress(
            0.12,
            f"Pose estimated from {len(matched_tags)} tag(s) (IDs {[t[0] for t in matched_tags]}): "
            f"angle {np.degrees(theta):.2f}°, origin ({wafer_stage_origin[0]:.1f}, {wafer_stage_origin[1]:.1f}) µm",
        )

        # Function mapping nominal wafer offset (dx, dy) to stage coordinate
        def wafer_to_stage_pos(wafer_xy: np.ndarray) -> np.ndarray:
            return wafer_stage_origin + (R_chip @ wafer_xy)

        # =========================================================================
        # STEP 2: SCAN 2x2 CELLS & RECORD RELATIVE INTRA-FRAME MEASUREMENTS
        # =========================================================================
        # Stage position does NOT matter for distance: only camera pixels are measured!
        measurements: List[Tuple[int, int, float, float]] = []
        num_cells = max(1, (ny - 1) * (nx - 1))
        cell_idx = 0

        for r in range(ny - 1):
            for c in range(nx - 1):
                if self.is_aborted:
                    return "Operation aborted"

                cell_idx += 1
                prog = 0.15 + (cell_idx / num_cells) * 0.70

                # Center of the 4 tags in nominal wafer coordinates
                cell_center_wafer = np.array([(c + 0.5) * step_dist, (r + 0.5) * step_dist], dtype=np.float64)
                target_stage_pos = wafer_to_stage_pos(cell_center_wafer)

                # Move stage to center of 2x2 window
                avg_z = z_dict.get(r * nx + c, stage.get_raw_position()[2])
                report_progress(
                    prog,
                    f"Scanning Cell [{r}, {c}] ({cell_idx}/{num_cells}): Moving to cluster center...",
                )
                stage.move_absolute_raw({"x": float(target_stage_pos[0]), "y": float(target_stage_pos[1]), "z": avg_z})
                context.delay_func(self.config.stabilization_delay)

                cell_frame = camera.get_latest_frame()
                if cell_frame is None:
                    continue

                det_cnt, det_corners, det_ids = detect_ar_tags(
                    cell_frame, dict_id=dict_id, flip_horizontal=self.flip_horizontal
                )

                if det_cnt >= 2 and det_ids is not None:
                    centers_in_frame: Dict[int, np.ndarray] = {}
                    for idx_m, mid_val in enumerate(det_ids.flatten()):
                        m_int = int(mid_val)
                        if m_int in nominal_dict:
                            pts = det_corners[idx_m].reshape(-1, 2)
                            centers_in_frame[m_int] = np.mean(pts, axis=0)

                    # Compute all pairwise relative vectors in wafer space
                    marker_ids_found = list(centers_in_frame.keys())
                    for i in range(len(marker_ids_found)):
                        for j in range(i + 1, len(marker_ids_found)):
                            id_a = marker_ids_found[i]
                            id_b = marker_ids_found[j]
                            # Intra-frame displacement in camera pixels
                            disp_cam_px = centers_in_frame[id_b] - centers_in_frame[id_a]
                            disp_cam_um = disp_cam_px * self.config.pixel_scale
                            # Rotate into wafer coordinate frame
                            disp_wafer = R_inv @ disp_cam_um
                            measurements.append((id_a, id_b, float(disp_wafer[0]), float(disp_wafer[1])))

        report_progress(0.88, f"Collected {len(measurements)} relative tag-to-tag distance measurements.")

        # =========================================================================
        # STEP 3: RECONSTRUCT GLOBAL OPTICAL COORDINATES VIA LEAST SQUARES
        # =========================================================================
        report_progress(0.92, "Step 3/3: Solving global coordinates via Least Squares...")
        anchor_id = 0
        anchor_pos = nominal_pos.get(0, (cur_stage_x, cur_stage_y))


        reconstructed_positions = solve_global_positions_from_relative_measurements(
            grid_size=nx,
            measurements=measurements,
            anchor_id=anchor_id,
            anchor_pos=anchor_pos,
            nominal_positions=nominal_pos,
        )

        cal_points: List[StageCalibrationPoint] = []
        for p in raw_points:
            m_id = int(p["marker_id"])
            grid_idx = tuple(p["grid_index"])
            raw_xy = (float(p["raw_xy"][0]), float(p["raw_xy"][1]))
            corr_xy = reconstructed_positions.get(m_id, raw_xy)
            z_focus = float(p["z_focus"])

            cal_points.append(
                StageCalibrationPoint(
                    grid_index=grid_idx,
                    marker_id=m_id,
                    raw_xy=raw_xy,
                    corr_xy=corr_xy,
                    z_focus=z_focus,
                )
            )

        self.calibration = StageCalibration(
            grid_shape=grid_shape,
            points=cal_points,
            stage_type=type(stage).__name__,
        )
        self.calibration.save_to_json(self.config.output_json_path)

        if self.config.apply_to_stage:
            stage.set_calibration(self.calibration)
            print("[Stage Calibration] Applied new calibration directly to stage controller.")

        report_progress(1.0, f"Pass 2 complete! Calibration map saved to {self.config.output_json_path}.")
        return None
