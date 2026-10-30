from dataclasses import dataclass, field
import json
import os
import re
from typing import Callable, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from lib.edge_artag import EdgeDetectorParameters
from lib.stage_calibration_reconstruction.detector import detect_markers_in_frame
from lib.stage_calibration_reconstruction.solver import (
    estimate_camera_to_stage_transform,
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
    detected_marker_ids: Set[int] = field(default_factory=set)
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

        # Load scan manifest if available
        manifest_path = os.path.join(self.images_dir, "scan_manifest.json")
        manifest_stage_xy: Dict[str, Tuple[float, float]] = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    m_data = json.load(mf)
                    for ph in m_data.get("photos", []):
                        fn = ph.get("filename")
                        raw_pos = ph.get("actual_raw_pos") or ph.get("target_xy")
                        if fn and raw_pos:
                            manifest_stage_xy[fn] = (float(raw_pos[0]), float(raw_pos[1]))
            except Exception as e:
                print(f"[ReconstructionPipeline] Warning loading scan_manifest.json: {e}")

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
        image_detections: List[Tuple[str, List[np.ndarray], np.ndarray, Optional[Tuple[float, float]], Tuple[int, int]]] = []
        all_angles: List[float] = []
        observed_ids: set[int] = set()

        for idx, fname in enumerate(all_files):
            if progress_cb:
                progress_cb(0.1 + 0.5 * (idx / diag.total_images), f"Detecting tags in {fname} ({idx + 1}/{diag.total_images})...")

            img_path = os.path.join(self.images_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            # Determine stage coordinates for this photo
            photo_stage_xy: Optional[Tuple[float, float]] = None
            if fname in manifest_stage_xy:
                photo_stage_xy = manifest_stage_xy[fname]
            else:
                m_match = re.search(r"_x([+-]?\d+(?:\.\d+)?)_y([+-]?\d+(?:\.\d+)?)", fname)
                if m_match:
                    photo_stage_xy = (float(m_match.group(1)), float(m_match.group(2)))

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

                image_detections.append((fname, corners, ids, photo_stage_xy, (w, h)))
                rot = estimate_chip_rotation(corners, ids, nominal_dict)
                if rot is not None:
                    all_angles.append(rot)

        diag.unique_tags_found = len(observed_ids)
        diag.detected_marker_ids = set(observed_ids)
        if not image_detections:
            return None, diag, "No ArUco markers detected in any of the scanned images"

        # 2. Chip orientation estimation from ArUco marker corners
        theta = float(np.median(all_angles)) if all_angles else 0.0
        diag.estimated_rotation_deg = float(np.degrees(theta))

        # Collect multi-frame observations for camera-to-stage transform: (marker_id, stage_xy, cam_offset_um)
        multi_frame_obs: List[Tuple[int, Tuple[float, float], Tuple[float, float]]] = []
        # Marker detections with camera offsets: marker_id -> list of (stage_xy or None, cam_offset_um)
        marker_observations: Dict[int, List[Tuple[Optional[Tuple[float, float]], Tuple[float, float]]]] = {}

        for _, corners, ids, photo_stage_xy, (w, h) in image_detections:
            for idx_m, mid_val in enumerate(ids.flatten()):
                m_int = int(mid_val)
                pts = corners[idx_m].reshape(-1, 2)
                center_px = np.mean(pts, axis=0)
                off_x_px = center_px[0] - (w / 2.0)
                off_y_px = center_px[1] - (h / 2.0)
                off_x_um = float(off_x_px) * self.camera_pixel_size
                off_y_um = float(off_y_px) * self.camera_pixel_size
                cam_offset_um = (off_x_um, off_y_um)

                marker_observations.setdefault(m_int, []).append((photo_stage_xy, cam_offset_um))
                if photo_stage_xy is not None:
                    multi_frame_obs.append((m_int, photo_stage_xy, cam_offset_um))

        if progress_cb:
            progress_cb(0.65, "Calculating camera-to-stage transformation...")

        # 3. Calculate camera-to-stage transform FIRST
        T_cam_to_stage = estimate_camera_to_stage_transform(
            multi_frame_observations=multi_frame_obs,
            default_rotation=theta,
            flip_horizontal=self.flip_horizontal,
        )

        # 4. Transform detected tags into actual stage space
        observed_stage_positions: Dict[int, Tuple[float, float]] = {}
        for m_id, obs_list in marker_observations.items():
            valid_stage_pts: List[Tuple[float, float]] = []
            for s_xy, c_off in obs_list:
                if s_xy is not None:
                    # p_stage = s_xy + T_cam_to_stage @ c_off
                    sx = s_xy[0] + T_cam_to_stage[0, 0] * c_off[0] + T_cam_to_stage[0, 1] * c_off[1]
                    sy = s_xy[1] + T_cam_to_stage[1, 0] * c_off[0] + T_cam_to_stage[1, 1] * c_off[1]
                    valid_stage_pts.append((sx, sy))

            if valid_stage_pts:
                avg_x = float(np.mean([p[0] for p in valid_stage_pts]))
                avg_y = float(np.mean([p[1] for p in valid_stage_pts]))
                observed_stage_positions[m_id] = (avg_x, avg_y)
            elif m_id in nominal_pos:
                observed_stage_positions[m_id] = nominal_pos[m_id]

        # Extract intra-frame pairwise relative measurements directly in stage space
        stage_measurements: List[Tuple[int, int, float, float]] = []
        for _, corners, ids, photo_stage_xy, (w, h) in image_detections:
            centers_in_frame: Dict[int, Tuple[float, float]] = {}
            for idx_m, mid_val in enumerate(ids.flatten()):
                m_int = int(mid_val)
                if m_int in nominal_dict:
                    pts = corners[idx_m].reshape(-1, 2)
                    c_px = np.mean(pts, axis=0)
                    off_x_um = float(c_px[0] - (w / 2.0)) * self.camera_pixel_size
                    off_y_um = float(c_px[1] - (h / 2.0)) * self.camera_pixel_size
                    centers_in_frame[m_int] = (off_x_um, off_y_um)

            m_ids = sorted(list(centers_in_frame.keys()))
            for i in range(len(m_ids)):
                for j in range(i + 1, len(m_ids)):
                    id_a = m_ids[i]
                    id_b = m_ids[j]
                    dc_x = centers_in_frame[id_b][0] - centers_in_frame[id_a][0]
                    dc_y = centers_in_frame[id_b][1] - centers_in_frame[id_a][1]
                    ds_x = T_cam_to_stage[0, 0] * dc_x + T_cam_to_stage[0, 1] * dc_y
                    ds_y = T_cam_to_stage[1, 0] * dc_x + T_cam_to_stage[1, 1] * dc_y
                    stage_measurements.append((id_a, id_b, float(ds_x), float(ds_y)))

        diag.total_pairwise_constraints = len(stage_measurements)

        if progress_cb:
            progress_cb(0.8, "Solving global coordinates via Least Squares...")

        # 5. Global least squares reconstruction:
        # - Solves detected tags in stage space directly (without generated constraints)
        # - Matches global rotation & translation to nominal positions
        # - Assigns undetected tags from fitted rigid transform
        # - Aligns origin to anchor marker (0, 0)
        anchor_id = 0
        if 0 not in nominal_pos and nominal_pos:
            anchor_id = list(nominal_pos.keys())[0]
        anchor_pos = nominal_pos.get(anchor_id, (0.0, 0.0))

        reconstructed_positions, global_rot_mat = solve_global_positions(
            grid_shape=grid_shape,
            measurements=stage_measurements,
            anchor_id=anchor_id,
            anchor_pos=anchor_pos,
            nominal_positions=nominal_pos,
            observed_positions=observed_stage_positions,
            step_distance=step_dist,
            default_rotation=theta,
        )

        # 6. Compute residuals directly in stage space
        residuals = []
        for id_a, id_b, ds_x, ds_y in stage_measurements:
            if id_a in reconstructed_positions and id_b in reconstructed_positions:
                pa = reconstructed_positions[id_a]
                pb = reconstructed_positions[id_b]
                err_x = (pb[0] - pa[0]) - ds_x
                err_y = (pb[1] - pa[1]) - ds_y
                residuals.append(float(np.hypot(err_x, err_y)))

        if residuals:
            diag.mean_residual_um = float(np.mean(residuals))
            diag.max_residual_um = float(np.max(residuals))

        # 7. Build StageCalibration object
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
            cam_to_stage_matrix=T_cam_to_stage,
            detected_marker_ids=observed_ids,
        )

        if progress_cb:
            progress_cb(1.0, "Reconstruction complete!")

        return cal, diag, None
