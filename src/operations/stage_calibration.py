# Hacker Fab
# Stage Calibration Operations:
# Stage 1: Multi-Point Autofocus Sweep across Wafer Surface (Z Focal Plane Mapping)
# Stage 2: Optical Alignment, Distortion Correction, and ArUco Tag Grid Patterning (using Stage 1 Z Surface)
# Stage 3: Chip Pose Estimation, Multi-Tag Camera Scanning, Relative Least Squares Global Reconstruction

from datetime import datetime
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext, Operation, sub_progress
from lib.ar_tag import detect_ar_tags
from lib.edge_artag import (
    EdgeArucoDetector,
    EdgeDetectorParameters,
    detect_edge_markers,
)
from lib.projector_camera_transform import (
    compute_projector_to_camera_homography,
    generate_calibration_tag_grid_with_coords,
    transform_points_projector_to_camera,
)
from operations.autofocus import AutofocusOperation
from operations.exposure import ExposureOperation
from stage_control.calibration import StageCalibration, StageCalibrationPoint

def get_aruco_dict(dict_id: Any = None) -> cv2.aruco.Dictionary:
    """Returns a custom ArUco dictionary generated with 5x5 markers and 10000 IDs."""
    dict_type = getattr(cv2.aruco, "Dictionary", None)
    if isinstance(dict_type, type) and isinstance(dict_id, dict_type):
        return dict_id
    if hasattr(cv2.aruco, "Dictionary") and hasattr(cv2.aruco.Dictionary, "create"):
        return cv2.aruco.Dictionary.create(10000, 5, 12345)
    elif hasattr(cv2.aruco, "extendDictionary"):
        return cv2.aruco.extendDictionary(10000, 5, 12345)
    elif hasattr(cv2.aruco, "Dictionary_create"):
        return cv2.aruco.Dictionary_create(10000, 5, 12345)
    return cv2.aruco.Dictionary.create(10000, 5, 12345)


DEFAULT_ARUCO_DICT = get_aruco_dict()


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


# =============================================================================
# STAGE 1: Z FOCUS CALIBRATION OPERATION
# =============================================================================


class StageCalibrationZFocusOperation(Operation):
    """Stage 1: Multi-point autofocus sweep across the wafer surface to map the Z focal plane."""

    def __init__(
        self,
        calibration_range: float = 10000.0,
        camera_pixel_size: Optional[float] = None,
        stabilization_delay: float = 1.0,
        output_json_path: Optional[str] = None,
        apply_to_stage: bool = True,
        grid_size: Optional[int] = None,
        step_distance: Optional[float] = None,
        dict_id: Any = DEFAULT_ARUCO_DICT,
    ):
        super().__init__("Stage Calibration - Stage 1 (Z Focus)")
        self.calibration_range = float(calibration_range)
        self.camera_pixel_size = float(camera_pixel_size) if camera_pixel_size is not None else None
        self.stabilization_delay = float(stabilization_delay)
        self.output_json_path = output_json_path
        self.apply_to_stage = bool(apply_to_stage)
        self.grid_size = int(grid_size) if grid_size is not None else None
        self.step_distance = float(step_distance) if step_distance is not None else None
        if isinstance(dict_id, (int, float)) and not isinstance(dict_id, bool):
            self.dict_id = int(dict_id)
        else:
            self.dict_id = dict_id

        self._current_sub_op: Optional[Operation] = None
        self.calibration: Optional[StageCalibration] = None
        self.recorded_points: List[StageCalibrationPoint] = []

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
        if context.stage is None or context.camera is None:
            return "Stage or Camera not available in execution context"

        stage = context.stage
        camera = context.camera

        pixel_scale = (
            self.camera_pixel_size
            if self.camera_pixel_size is not None
            else (
                context.machine_config.stage_calibration.camera_pixel_size
                if context.machine_config is not None
                else 1.0
            )
        )
        output_path = (
            self.output_json_path
            or (
                context.machine_config.stage_calibration.calibration_file_path
                if context.machine_config is not None
                else "stage_calibration.json"
            )
        )

        # Get camera frame dimensions
        init_frame = camera.get_latest_frame()
        cam_h, cam_w = init_frame.shape[:2] if init_frame is not None else (1080, 1920)

        # Calculate grid parameters: N and d
        if self.grid_size is not None and self.step_distance is not None:
            nx = ny = self.grid_size
            step_dist = self.step_distance
        else:
            nx, step_dist = compute_calibration_grid_params(
                cal_range=self.calibration_range,
                pixel_scale=pixel_scale,
                camera_size=(cam_w, cam_h),
            )
            ny = nx

        total_points = nx * ny
        start_x, start_y, start_z = stage.get_raw_position()
        c0 = (nx - 1) // 2
        r0 = (ny - 1) // 2
        spiral_indices = generate_spiral_grid_indices(nx, ny)

        report_progress(0.0, f"Starting Stage 1: Z focus survey on {nx}x{ny} grid (step {step_dist:.1f} µm)...")
        self.recorded_points = []

        for point_idx, (r, c) in enumerate(spiral_indices):
            if self.is_aborted:
                return "Stage 1 calibration aborted"

            prog_start = point_idx / total_points
            prog_end = (point_idx + 1) / total_points
            marker_id = r * nx + c

            target_raw_x = start_x + (c - c0) * step_dist
            target_raw_y = start_y + (r - r0) * step_dist

            report_progress(
                prog_start,
                f"Point {point_idx + 1}/{total_points} (Grid [{r}, {c}]): Moving to ({target_raw_x:.1f}, {target_raw_y:.1f}) µm...",
            )
            stage.move_absolute_raw({"x": target_raw_x, "y": target_raw_y})
            context.delay_func(self.stabilization_delay)

            # Autofocus at current point (Red focal plane measurement, UV offset=0.0)
            af_op = AutofocusOperation(uv_z_offset=0.0)
            af_reporter = sub_progress(
                report_progress,
                prog_start + 0.1 * (prog_end - prog_start),
                prog_end,
                prefix=f"Point {point_idx + 1}/{total_points} - AF",
            )
            af_err = self._run_sub_op(af_op, context, af_reporter)
            if af_err and self.is_aborted:
                return "Stage 1 calibration aborted during autofocus"

            if af_op.best_red_z is not None:
                z_focus = float(af_op.best_red_z)
            else:
                z_focus = float(stage.get_raw_position()[2])

            pt = StageCalibrationPoint(
                grid_index=(r, c),
                marker_id=marker_id,
                raw_xy=(float(target_raw_x), float(target_raw_y)),
                corr_xy=(float(target_raw_x), float(target_raw_y)),  # Identity mapping until Stage 3
                z_focus=float(z_focus),
            )
            self.recorded_points.append(pt)

        # Sort points by grid_index for deterministic ordering
        self.recorded_points.sort(key=lambda p: (p.grid_index[0], p.grid_index[1]))

        self.calibration = StageCalibration(
            grid_shape=(ny, nx),
            points=self.recorded_points,
            stage_type=type(stage).__name__,
            step_distance=step_dist,
            calibration_range=self.calibration_range,
            dict_id=self.dict_id,
        )
        self.calibration.save_to_json(output_path)

        if self.apply_to_stage:
            stage.set_calibration(self.calibration)
            print("[Stage Calibration] Applied Z-leveling calibration map to stage controller.")

        report_progress(
            1.0,
            f"Stage 1 complete! Saved Z focal plane with {len(self.recorded_points)} points to {output_path}. "
            "Stage Z-leveling is now active.",
        )
        return None


# =============================================================================
# STAGE 2: PATTERNING OPERATION
# =============================================================================


class StageCalibrationPatternOperation(Operation):
    """Stage 2: Measure optical transform, correct alignment, and pattern the ArUco tag grid using calibrated Z heights."""

    def __init__(
        self,
        input_json_path: Optional[str] = None,
        exposure_time: Optional[float] = None,
        stabilization_delay: float = 1.0,
        uv_z_offset: Optional[float] = None,
        dict_id: Any = DEFAULT_ARUCO_DICT,
    ):
        super().__init__("Stage Calibration - Stage 2 (Patterning)")
        self.input_json_path = input_json_path
        self.exposure_time = float(exposure_time) if exposure_time is not None else None
        self.stabilization_delay = float(stabilization_delay)
        self.uv_z_offset = float(uv_z_offset) if uv_z_offset is not None else None
        if isinstance(dict_id, (int, float)) and not isinstance(dict_id, bool):
            self.dict_id = int(dict_id)
        else:
            self.dict_id = dict_id

        self._current_sub_op: Optional[Operation] = None
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

        input_path = (
            self.input_json_path
            or (
                context.machine_config.stage_calibration.calibration_file_path
                if context.machine_config is not None
                else "stage_calibration.json"
            )
        )

        if not os.path.exists(input_path):
            return f"Stage 1 calibration file not found: {input_path}. Please run Stage 1 (Z Focus) first."

        cal = StageCalibration.load_from_json(input_path)
        if len(cal.points) == 0:
            return f"No points found in calibration file: {input_path}"

        stage = context.stage
        projector = context.projector
        camera = context.camera
        pw, ph = projector.projector_size()

        effective_exp_time = (
            self.exposure_time
            if self.exposure_time is not None
            else (
                context.machine_config.exposure.default_exposure_time_ms / 1000.0
                if context.machine_config is not None
                else 8.0
            )
        )
        effective_uv_offset = (
            self.uv_z_offset
            if self.uv_z_offset is not None
            else (
                context.machine_config.autofocus.uv_z_offset
                if context.machine_config is not None
                else 0.0
            )
        )

        init_frame = camera.get_latest_frame()
        cam_h, cam_w = init_frame.shape[:2] if init_frame is not None else (1080, 1920)
        u_c, v_c = cam_w / 2.0, cam_h / 2.0

        ny, nx = cal.grid_shape
        step_dist = cal.step_distance or 1000.0
        dict_id = cal.dict_id or self.dict_id
        pts_by_id: Dict[int, StageCalibrationPoint] = {p.marker_id: p for p in cal.points}

        total_points = nx * ny
        start_x, start_y, start_z = stage.get_raw_position()

        try:
            # =========================================================================
            # STEP 1: MEASURE OPTICAL TRANSFORM (PROJECTOR TO CAMERA)
            # =========================================================================
            report_progress(0.02, "Step 1/2: Measuring optical transform (Projector -> Camera)...")
            grid_img, tag_coords = generate_calibration_tag_grid_with_coords(
                grid_n=2,
                canvas_size=(pw, ph),
                dict_id=cv2.aruco.DICT_4X4_250,
            )
            projector.set_generated_image(grid_img)
            projector.set_color_mode(ColorMode.RED)
            projector.set_image_source(ProjectorImageSource.GENERATED)
            projector.set_on(True)
            context.delay_func(self.stabilization_delay)

            cal_frame = camera.get_latest_frame()
            flip_horizontal = False
            det_count, corners, ids = detect_ar_tags(cal_frame, dict_id=cv2.aruco.DICT_4X4_250, flip_horizontal=False)
            if det_count < 2:
                det_count_flip, corners_flip, ids_flip = detect_ar_tags(
                    cal_frame, dict_id=cv2.aruco.DICT_4X4_250, flip_horizontal=True
                )
                if det_count_flip > det_count:
                    det_count, corners, ids = det_count_flip, corners_flip, ids_flip
                    flip_horizontal = True
            self.flip_horizontal = flip_horizontal

            H_proj_to_cam, matched_count = compute_projector_to_camera_homography(corners, ids, tag_coords)
            projector.set_on(False)

            if H_proj_to_cam is None or matched_count < 4:
                return (
                    f"Projector-to-camera optical calibration failed: insufficient tag matches ({matched_count}/4 points). "
                    "Cannot pattern stage grid without calibrated projection."
                )

            try:
                H_cam_to_proj = np.linalg.inv(H_proj_to_cam)
            except np.linalg.LinAlgError as e:
                return f"Failed to invert projector-to-camera homography matrix: {e}"

            # 3-step iterative correction so marker 0 sits with top-left at camera center
            report_progress(0.08, "Iteratively correcting tag projection alignment (3 passes)...")
            tag_sz_cam = max(24.0, min(cam_w, cam_h) * 0.15)
            cam_corners = np.array([
                [u_c, v_c],
                [u_c + tag_sz_cam, v_c],
                [u_c + tag_sz_cam, v_c + tag_sz_cam],
                [u_c, v_c + tag_sz_cam],
            ], dtype=np.float64)

            for _ in range(3):
                if self.is_aborted:
                    return "Stage 2 calibration aborted"

                proj_pts = transform_points_projector_to_camera(cam_corners, H_cam_to_proj)
                self.corrected_proj_corners = proj_pts

                test_img = warp_marker_to_projector(
                    marker_id=0,
                    target_proj_corners=proj_pts,
                    canvas_size=(pw, ph),
                    dict_id=dict_id,
                )
                projector.set_generated_image(test_img)
                projector.set_color_mode(ColorMode.RED)
                projector.set_on(True)
                context.delay_func(self.stabilization_delay)

                iter_frame = camera.get_latest_frame()
                c_cnt, c_corners, _ = detect_ar_tags(iter_frame, dict_id=dict_id, flip_horizontal=self.flip_horizontal)
                if c_cnt > 0 and c_corners:
                    detected_tl = c_corners[0].reshape(-1, 2)[0]
                    err = detected_tl - np.array([u_c, v_c])
                    cam_corners[:, 0] -= err[0]
                    cam_corners[:, 1] -= err[1]

                projector.set_on(False)

            self.corrected_proj_corners = transform_points_projector_to_camera(cam_corners, H_cam_to_proj)

            # =========================================================================
            # STEP 2: EXPOSE ARUCO GRID USING CALIBRATED Z FOCUS SURFACE
            # =========================================================================
            report_progress(0.15, f"Step 2/2: Patterning {nx}x{ny} grid using Stage 1 calibrated Z heights...")
            loop_span = 0.85
            base_offset = 0.15
            c0 = (nx - 1) // 2
            r0 = (ny - 1) // 2
            spiral_indices = generate_spiral_grid_indices(nx, ny)

            for point_idx, (r, c) in enumerate(spiral_indices):
                if self.is_aborted:
                    return "Stage 2 calibration aborted"

                prog_frac = base_offset + (point_idx / total_points) * loop_span
                marker_id = r * nx + c

                pt = pts_by_id.get(marker_id)
                target_raw_x = pt.raw_xy[0] if pt else (start_x + (c - c0) * step_dist)
                target_raw_y = pt.raw_xy[1] if pt else (start_y + (r - r0) * step_dist)
                calibrated_z = (pt.z_focus if pt else start_z) + effective_uv_offset

                report_progress(
                    prog_frac,
                    f"Point {point_idx + 1}/{total_points} (Grid [{r}, {c}], Tag ID {marker_id}): Moving to ({target_raw_x:.1f}, {target_raw_y:.1f}, {calibrated_z:.1f} µm)...",
                )
                stage.move_absolute_raw({"x": target_raw_x, "y": target_raw_y, "z": calibrated_z})
                context.delay_func(self.stabilization_delay)

                # Expose single ArUco marker directly (no autofocus sweep needed!)
                tag_img = warp_marker_to_projector(
                    marker_id=marker_id,
                    target_proj_corners=self.corrected_proj_corners,
                    canvas_size=(pw, ph),
                    dict_id=dict_id,
                )
                projector.set_generated_image(tag_img)
                projector.set_image_source(ProjectorImageSource.GENERATED)

                exp_op = ExposureOperation(layer_index=None, exposure_time=effective_exp_time * 1000.0)
                pt_span = loop_span / total_points
                exp_reporter = sub_progress(
                    report_progress,
                    prog_frac + 0.1 * pt_span,
                    prog_frac + pt_span,
                    prefix=f"Point {point_idx + 1}/{total_points} (Grid [{r}, {c}], Tag ID {marker_id})",
                )
                exp_err = self._run_sub_op(exp_op, context, exp_reporter)
                if exp_err or self.is_aborted:
                    return exp_err or "Stage 2 calibration aborted during exposure"

            # Update calibration file to record dict_id and flip_horizontal
            cal.flip_horizontal = self.flip_horizontal
            cal.dict_id = dict_id
            cal.save_to_json(input_path)

            report_progress(
                1.0,
                f"Stage 2 complete! {total_points} ArUco markers patterned using calibrated Z heights. "
                "Please develop wafer, mount, and jog stage so at least one tag is visible to run Stage 3.",
            )
            return None

        finally:
            projector.set_on(False)
            projector.set_generated_image(None)
            projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)


# =============================================================================
# STAGE 3: SCANNING & RECONSTRUCTION OPERATION
# =============================================================================


class StageCalibrationScanOperation(Operation):
    """Stage 3: Chip pose estimation, multi-tag camera scanning, and Least Squares global reconstruction."""

    def __init__(
        self,
        input_json_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        camera_pixel_size: Optional[float] = None,
        stabilization_delay: float = 0.5,
        apply_to_stage: bool = True,
        dict_id: Any = DEFAULT_ARUCO_DICT,
        edge_detector_params: Optional[EdgeDetectorParameters] = None,
    ):
        super().__init__("Stage Calibration - Stage 3 (Optical Scan)")
        self.input_json_path = input_json_path
        self.output_json_path = output_json_path
        self.camera_pixel_size = float(camera_pixel_size) if camera_pixel_size is not None else None
        self.stabilization_delay = float(stabilization_delay)
        self.apply_to_stage = bool(apply_to_stage)
        if isinstance(dict_id, (int, float)) and not isinstance(dict_id, bool):
            self.dict_id = int(dict_id)
        else:
            self.dict_id = dict_id
        self.edge_detector_params = edge_detector_params
        self.calibration: Optional[StageCalibration] = None
        self.flip_horizontal: bool = False

    def _detect_markers(
        self,
        frame: Optional[np.ndarray],
        dict_id: Any = None,
        flip_horizontal: bool = False,
    ) -> Tuple[int, List[np.ndarray], Optional[np.ndarray]]:
        """Detects ArUco markers using edge_artag detector."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return 0, [], None

        img = cv2.flip(frame, 1) if flip_horizontal else frame
        dictionary = get_aruco_dict(dict_id)
        corners, ids, _ = detect_edge_markers(
            img,
            dictionary=dictionary,
            parameters=self.edge_detector_params,
        )
        count = len(ids) if ids is not None else 0
        if count == 0:
            # Fall back to standard OpenCV detector as safety net
            count, corners, ids = detect_ar_tags(frame, dict_id=dict_id, flip_horizontal=flip_horizontal)
        return count, corners, ids

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if context.stage is None or context.camera is None:
            return "Stage or Camera not available in execution context"

        input_path = (
            self.input_json_path
            or (
                context.machine_config.stage_calibration.calibration_file_path
                if context.machine_config is not None
                else "stage_calibration.json"
            )
        )
        output_path = self.output_json_path or input_path

        if not os.path.exists(input_path):
            return f"Stage calibration input file not found: {input_path}"

        with open(input_path, "r", encoding="utf-8") as f:
            stage1_data = json.load(f)

        effective_pixel_scale = (
            self.camera_pixel_size
            if self.camera_pixel_size is not None
            else (
                context.machine_config.stage_calibration.camera_pixel_size
                if context.machine_config is not None
                else 1.0
            )
        )

        grid_shape = tuple(stage1_data.get("grid_shape", (0, 0)))
        ny, nx = grid_shape
        step_dist = float(stage1_data.get("step_distance", 5000.0))
        raw_dict_id = stage1_data.get("dict_id", self.dict_id)
        if isinstance(raw_dict_id, (int, float)) and not isinstance(raw_dict_id, bool):
            dict_id = int(raw_dict_id)
        else:
            dict_id = raw_dict_id
        raw_points = stage1_data.get("points", [])

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

        det_count, corners, ids = self._detect_markers(frame, dict_id=dict_id, flip_horizontal=False)
        flip_horizontal = False
        if det_count == 0:
            det_count, corners, ids = self._detect_markers(frame, dict_id=dict_id, flip_horizontal=True)
            flip_horizontal = True
        self.flip_horizontal = flip_horizontal

        if det_count == 0 or ids is None or len(ids) == 0:
            return (
                "No ArUco tags detected in camera field of view. "
                "Please jog stage to bring at least one patterned tag into view."
            )

        cur_stage_x, cur_stage_y, cur_stage_z = stage.get_raw_position()

        # Find all detected tags that exist in nominal points
        matched_tags: List[Tuple[int, np.ndarray]] = []
        for idx_det, mid_val in enumerate(ids.flatten()):
            mid_int = int(mid_val)
            if mid_int in nominal_dict:
                matched_tags.append((mid_int, corners[idx_det].reshape(-1, 2)))

        if not matched_tags:
            return f"Detected tag IDs {ids.flatten().tolist()} do not belong to the patterned calibration grid."

        # Estimate rotation angle theta
        angles = []
        for _, tag_corners in matched_tags:
            edge_top = tag_corners[1] - tag_corners[0]
            angles.append(np.arctan2(edge_top[1], edge_top[0]))
        theta = float(np.median(angles))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_chip = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        R_inv = np.linalg.inv(R_chip)

        # Estimate wafer origin from matched tags
        wafer_origin_estimates = []
        for mid_int, tag_corners in matched_tags:
            tag_cam_center = np.mean(tag_corners, axis=0)
            delta_cam_px = tag_cam_center - np.array([u_c, v_c])
            delta_cam_um = delta_cam_px * effective_pixel_scale
            tag_stage_pos = np.array([cur_stage_x + delta_cam_um[0], cur_stage_y + delta_cam_um[1]], dtype=np.float64)

            ref_grid_r, ref_grid_c = nominal_dict[mid_int]["grid_index"]
            tag_wafer_offset = np.array([ref_grid_c * step_dist, ref_grid_r * step_dist], dtype=np.float64)
            origin_est = tag_stage_pos - (R_chip @ tag_wafer_offset)
            wafer_origin_estimates.append(origin_est)

        wafer_stage_origin = np.mean(wafer_origin_estimates, axis=0)

        report_progress(
            0.12,
            f"Pose estimated from {len(matched_tags)} tag(s) (IDs {[t[0] for t in matched_tags]}): "
            f"angle {np.degrees(theta):.2f}°, origin ({wafer_stage_origin[0]:.1f}, {wafer_stage_origin[1]:.1f}) µm",
        )

        def wafer_to_stage_pos(wafer_xy: np.ndarray) -> np.ndarray:
            return wafer_stage_origin + (R_chip @ wafer_xy)

        # =========================================================================
        # STEP 2: SCAN 2x2 CELLS & RECORD RELATIVE INTRA-FRAME MEASUREMENTS
        # =========================================================================
        measurements: List[Tuple[int, int, float, float]] = []
        num_cells = max(1, (ny - 1) * (nx - 1))
        cell_idx = 0

        for r in range(ny - 1):
            for c in range(nx - 1):
                if self.is_aborted:
                    return "Stage 3 calibration aborted"

                cell_idx += 1
                prog = 0.15 + (cell_idx / num_cells) * 0.70

                cell_center_wafer = np.array([(c + 0.5) * step_dist, (r + 0.5) * step_dist], dtype=np.float64)
                target_stage_pos = wafer_to_stage_pos(cell_center_wafer)

                avg_z = z_dict.get(r * nx + c, stage.get_raw_position()[2])
                report_progress(
                    prog,
                    f"Scanning Cell [{r}, {c}] ({cell_idx}/{num_cells}): Moving to cluster center...",
                )
                stage.move_absolute_raw({"x": float(target_stage_pos[0]), "y": float(target_stage_pos[1]), "z": avg_z})
                context.delay_func(self.stabilization_delay)

                cell_frame = camera.get_latest_frame()
                if cell_frame is None:
                    continue

                det_cnt, det_corners, det_ids = self._detect_markers(
                    cell_frame, dict_id=dict_id, flip_horizontal=self.flip_horizontal
                )

                if det_cnt >= 2 and det_ids is not None:
                    centers_in_frame: Dict[int, np.ndarray] = {}
                    for idx_m, mid_val in enumerate(det_ids.flatten()):
                        m_int = int(mid_val)
                        if m_int in nominal_dict:
                            pts = det_corners[idx_m].reshape(-1, 2)
                            centers_in_frame[m_int] = np.mean(pts, axis=0)

                    marker_ids_found = list(centers_in_frame.keys())
                    for i in range(len(marker_ids_found)):
                        for j in range(i + 1, len(marker_ids_found)):
                            id_a = marker_ids_found[i]
                            id_b = marker_ids_found[j]
                            disp_cam_px = centers_in_frame[id_b] - centers_in_frame[id_a]
                            disp_cam_um = disp_cam_px * effective_pixel_scale
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
            step_distance=step_dist,
            calibration_range=float(stage1_data.get("calibration_range", 10000.0)),
            dict_id=dict_id,
            flip_horizontal=self.flip_horizontal,
        )
        self.calibration.save_to_json(output_path)

        if self.apply_to_stage:
            stage.set_calibration(self.calibration)
            print("[Stage Calibration] Applied new calibration directly to stage controller.")

        report_progress(1.0, f"Stage 3 complete! Calibration map saved to {output_path}.")
        return None
