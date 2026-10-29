from dataclasses import dataclass, field
import json
import os
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from lib.edge_artag import EdgeDetectorParameters
from lib.stage_calibration_reconstruction.detector import detect_markers_in_frame
from lib.stage_calibration_reconstruction.solver import (
    estimate_chip_rotation,
    extract_pairwise_measurements,
    solve_global_positions,
)
from stage_control.calibration import StageCalibration, StageCalibrationPoint


@dataclass
class ReconstructionDiagnostics:
    total_images: int = 0
    images_with_tags: int = 0
    total_tag_detections: int = 0
    unique_tags_found: int = 0
    nominal_tags_count: int = 0
    total_pairwise_constraints: int = 0
    estimated_rotation_deg: float = 0.0
    mean_residual_um: float = 0.0
    max_residual_um: float = 0.0
    per_image_results: Dict[str, Dict] = field(default_factory=dict)


class ReconstructionPipeline:
    """Processes a directory of scanned photos to reconstruct the global optical stage calibration."""

    def __init__(
        self,
        images_dir: str,
        stage1_json_path: str,
        camera_pixel_size: float = 1.0,
        detector_params: Optional[EdgeDetectorParameters] = None,
        flip_horizontal: bool = False,
    ):
        self.images_dir = images_dir
        self.stage1_json_path = stage1_json_path
        self.camera_pixel_size = float(camera_pixel_size)
        self.detector_params = detector_params
        self.flip_horizontal = flip_horizontal

    def run(
        self,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[Optional[StageCalibration], ReconstructionDiagnostics, Optional[str]]:
        """Executes the reconstruction pipeline.

        Returns:
            (StageCalibration or None, ReconstructionDiagnostics, error_message or None)
        """
        diag = ReconstructionDiagnostics()

        if not os.path.exists(self.stage1_json_path):
            return None, diag, f"Stage 1 calibration file not found: {self.stage1_json_path}"
        if not os.path.isdir(self.images_dir):
            return None, diag, f"Images directory not found: {self.images_dir}"

        try:
            with open(self.stage1_json_path, "r", encoding="utf-8") as f:
                s1_data = json.load(f)
        except Exception as e:
            return None, diag, f"Failed to load Stage 1 JSON: {e}"

        grid_shape = tuple(s1_data.get("grid_shape", (0, 0)))
        ny, nx = grid_shape
        step_dist = float(s1_data.get("step_distance", 1000.0))
        raw_points = s1_data.get("points", [])
        if len(raw_points) == 0:
            return None, diag, "No calibration points found in Stage 1 file"

        nominal_dict: Dict[int, Dict] = {int(p["marker_id"]): p for p in raw_points}
        nominal_pos: Dict[int, Tuple[float, float]] = {
            int(p["marker_id"]): (float(p["raw_xy"][0]), float(p["raw_xy"][1])) for p in raw_points
        }
        diag.nominal_tags_count = len(nominal_dict)

        # List image files
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        all_files = sorted([
            f for f in os.listdir(self.images_dir)
            if os.path.splitext(f.lower())[1] in valid_exts
        ])
        diag.total_images = len(all_files)
        if diag.total_images == 0:
            return None, diag, f"No image files found in {self.images_dir}"

        # 1. Detection across all images
        image_detections: List[Tuple[str, List[np.ndarray], np.ndarray]] = []
        all_angles: List[float] = []
        observed_ids: set[int] = set()

        for idx, fname in enumerate(all_files):
            if progress_cb:
                progress_cb(0.1 + 0.6 * (idx / diag.total_images), f"Detecting tags in {fname} ({idx + 1}/{diag.total_images})...")

            img_path = os.path.join(self.images_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            cnt, corners, ids = detect_markers_in_frame(
                img,
                parameters=self.detector_params,
                flip_horizontal=self.flip_horizontal,
            )

            diag.per_image_results[fname] = {
                "detected_count": cnt,
                "ids": ids.flatten().tolist() if ids is not None else [],
            }

            if cnt > 0 and ids is not None:
                diag.images_with_tags += 1
                diag.total_tag_detections += cnt
                for mid in ids.flatten():
                    observed_ids.add(int(mid))

                image_detections.append((fname, corners, ids))
                rot = estimate_chip_rotation(corners, ids, nominal_dict)
                if rot is not None:
                    all_angles.append(rot)

        diag.unique_tags_found = len(observed_ids)
        if not image_detections:
            return None, diag, "No ArUco markers detected in any of the scanned images"

        # 2. Chip orientation estimation
        theta = float(np.median(all_angles)) if all_angles else 0.0
        diag.estimated_rotation_deg = float(np.degrees(theta))

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_chip = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        R_inv = np.linalg.inv(R_chip)

        # 3. Pairwise relative measurements
        measurements: List[Tuple[int, int, float, float]] = []
        for _, corners, ids in image_detections:
            intra_meas = extract_pairwise_measurements(
                corners=corners,
                ids=ids,
                nominal_dict=nominal_dict,
                pixel_scale=self.camera_pixel_size,
                R_inv=R_inv,
            )
            measurements.extend(intra_meas)

        diag.total_pairwise_constraints = len(measurements)
        if not measurements:
            return None, diag, "No multi-tag frames found to establish pairwise relative constraints"

        if progress_cb:
            progress_cb(0.8, "Solving global coordinates via Least Squares...")

        # 4. Global least squares reconstruction
        anchor_id = 0
        if 0 not in nominal_pos and nominal_pos:
            anchor_id = list(nominal_pos.keys())[0]
        anchor_pos = nominal_pos.get(anchor_id, (0.0, 0.0))

        reconstructed_positions = solve_global_positions(
            grid_shape=grid_shape,
            measurements=measurements,
            anchor_id=anchor_id,
            anchor_pos=anchor_pos,
            nominal_positions=nominal_pos,
        )

        # 5. Compute residuals
        residuals = []
        for id_a, id_b, dx, dy in measurements:
            if id_a in reconstructed_positions and id_b in reconstructed_positions:
                pa = reconstructed_positions[id_a]
                pb = reconstructed_positions[id_b]
                err_x = (pb[0] - pa[0]) - dx
                err_y = (pb[1] - pa[1]) - dy
                residuals.append(float(np.hypot(err_x, err_y)))

        if residuals:
            diag.mean_residual_um = float(np.mean(residuals))
            diag.max_residual_um = float(np.max(residuals))

        # 6. Build StageCalibration object
        cal_points: List[StageCalibrationPoint] = []
        for p in raw_points:
            m_id = int(p["marker_id"])
            grid_idx = tuple(p["grid_index"])
            raw_xy = (float(p["raw_xy"][0]), float(p["raw_xy"][1]))
            corr_xy = reconstructed_positions.get(m_id, raw_xy)
            z_focus = float(p.get("z_focus", 0.0))

            cal_points.append(
                StageCalibrationPoint(
                    grid_index=grid_idx,
                    marker_id=m_id,
                    raw_xy=raw_xy,
                    corr_xy=corr_xy,
                    z_focus=z_focus,
                )
            )

        cal = StageCalibration(
            grid_shape=grid_shape,
            points=cal_points,
            stage_type=s1_data.get("stage_type", ""),
            step_distance=step_dist,
            calibration_range=float(s1_data.get("calibration_range", 10000.0)),
            flip_horizontal=self.flip_horizontal,
        )

        if progress_cb:
            progress_cb(1.0, "Reconstruction complete!")

        return cal, diag, None

