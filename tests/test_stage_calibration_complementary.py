import json
import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from lib.stage_calibration_reconstruction.pipeline import (
    ReconstructionDiagnostics,
    ReconstructionPipeline,
)
from operations.stage_calibration import DEFAULT_ARUCO_DICT
from tests.test_stage_calibration import generate_etched_ar_tag


class TestStageCalibrationComplementary(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.primary_dir = os.path.join(self.tmp_dir, "primary_images")
        self.comp_dir = os.path.join(self.tmp_dir, "comp_images")
        os.makedirs(self.primary_dir, exist_ok=True)
        os.makedirs(self.comp_dir, exist_ok=True)

        # Create a Stage 1 calibration JSON with 2x2 grid (markers 0, 1, 2, 3)
        self.stage1_path = os.path.join(self.tmp_dir, "stage1.json")
        stage1_data = {
            "version": "1.0",
            "stage_type": "dummy",
            "grid_shape": [2, 2],
            "step_distance": 1000.0,
            "calibration_range": 2000.0,
            "points": [
                {"grid_index": [0, 0], "marker_id": 0, "raw_xy": [0.0, 0.0], "z_focus": 50.0},
                {"grid_index": [0, 1], "marker_id": 1, "raw_xy": [1000.0, 0.0], "z_focus": 50.0},
                {"grid_index": [1, 0], "marker_id": 2, "raw_xy": [0.0, 1000.0], "z_focus": 50.0},
                {"grid_index": [1, 1], "marker_id": 3, "raw_xy": [1000.0, 1000.0], "z_focus": 50.0},
            ],
        }
        with open(self.stage1_path, "w", encoding="utf-8") as f:
            json.dump(stage1_data, f, indent=2)

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _create_synthetic_image(self, marker_defs, canvas_size=(600, 800)):
        """marker_defs: list of (marker_id, center_x, center_y, tag_size)"""
        h, w = canvas_size
        img = np.full((h, w, 3), 220, dtype=np.uint8)
        for mid, cx, cy, sz in marker_defs:
            tag = generate_etched_ar_tag(mid, sz, dictionary=DEFAULT_ARUCO_DICT)
            th, tw = tag.shape[:2]
            x0 = int(round(cx - tw / 2.0))
            y0 = int(round(cy - th / 2.0))
            if 0 <= y0 and y0 + th <= h and 0 <= x0 and x0 + tw <= w:
                img[y0 : y0 + th, x0 : x0 + tw] = np.stack([tag] * 3, axis=-1)
        return img

    def test_reconstruction_with_and_without_complementary_dir(self):
        # 1. Create primary scan images (markers 0 and 1 in photo 1; markers 2 and 3 in photo 2)
        # Photo 1: Marker 0 at (250, 300), Marker 1 at (550, 300) -> distance dx = 300 px
        p1 = self._create_synthetic_image([(0, 250, 300, 120), (1, 550, 300, 120)])
        cv2.imwrite(os.path.join(self.primary_dir, "scan_0000_r0_c0_x0.0_y0.0.png"), p1)

        # Photo 2: Marker 2 at (250, 300), Marker 3 at (550, 300) -> distance dx = 300 px
        p2 = self._create_synthetic_image([(2, 250, 300, 120), (3, 550, 300, 120)])
        cv2.imwrite(os.path.join(self.primary_dir, "scan_0001_r1_c0_x0.0_y1000.0.png"), p2)

        # Run pipeline WITHOUT complementary directory
        pipeline_single = ReconstructionPipeline(
            images_dir=self.primary_dir,
            stage1_json_path=self.stage1_path,
            camera_pixel_size=1.0,
        )
        cal1, diag1, err1 = pipeline_single.run()
        self.assertIsNone(err1)
        self.assertIsNotNone(cal1)
        self.assertEqual(diag1.primary_pairwise_constraints, 2)  # (0, 1) and (2, 3)
        self.assertEqual(diag1.complementary_pairwise_constraints, 0)
        self.assertEqual(diag1.total_pairwise_constraints, 2)

        # 2. Create complementary scan images rotated by ~90 degrees or with vertical marker pairs (0 and 2)
        # Comp Photo 1: Marker 0 at (400, 150), Marker 2 at (400, 450) -> distance dy = 300 px
        # Comp Photo 2: Marker 1 at (400, 150), Marker 3 at (400, 450) -> distance dy = 300 px
        c1 = self._create_synthetic_image([(0, 400, 150, 120), (2, 400, 450, 120)])
        cv2.imwrite(os.path.join(self.comp_dir, "comp_0000.png"), c1)

        c2 = self._create_synthetic_image([(1, 400, 150, 120), (3, 400, 450, 120)])
        cv2.imwrite(os.path.join(self.comp_dir, "comp_0001.png"), c2)

        # Run pipeline WITH complementary directory
        pipeline_dual = ReconstructionPipeline(
            images_dir=self.primary_dir,
            stage1_json_path=self.stage1_path,
            camera_pixel_size=1.0,
            complementary_images_dir=self.comp_dir,
        )
        cal2, diag2, err2 = pipeline_dual.run()
        self.assertIsNone(err2)
        self.assertIsNotNone(cal2)
        self.assertEqual(diag2.primary_pairwise_constraints, 2)
        self.assertEqual(diag2.complementary_pairwise_constraints, 2)  # (0, 2) and (1, 3)
        self.assertEqual(diag2.total_pairwise_constraints, 4)

        # Global cam_to_stage matrix should be computed strictly from primary scan
        np.testing.assert_allclose(cal1.cam_to_stage_matrix, cal2.cam_to_stage_matrix, atol=1e-5)

    def test_detect_markers_batch(self):
        from lib.stage_calibration_reconstruction.detector import detect_markers_batch

        img1 = self._create_synthetic_image([(0, 250, 300, 120), (1, 550, 300, 120)])
        img2 = self._create_synthetic_image([(2, 250, 300, 120)])
        img3 = self._create_synthetic_image([(3, 400, 300, 120)])

        prog_calls = []

        def prog_cb(completed, total):
            prog_calls.append((completed, total))

        results = detect_markers_batch([img1, img2, img3], progress_cb=prog_cb)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 2)  # img1 has 2 markers
        self.assertEqual(results[1][0], 1)  # img2 has 1 marker
        self.assertEqual(results[2][0], 1)  # img3 has 1 marker
        self.assertTrue(len(prog_calls) >= 3)
        self.assertEqual(prog_calls[-1], (3, 3))

    def test_undetected_tag_opposite_neighbor_averaging(self):
        from lib.stage_calibration_reconstruction.solver import (
            match_global_transform_and_assign_undetected,
        )

        grid_shape = (3, 3)
        nominal_positions = {
            r * 3 + c: (float(c * 1000.0), float(r * 1000.0))
            for r in range(3)
            for c in range(3)
        }

        # Case A: Center tag 4 is missing, opposite neighbors 3 & 5 (+-x) and 1 & 7 (+-y) are detected
        # Tag 3: (0, 1010), Tag 5: (2000, 1030) -> x midpoint = (1000, 1020)
        # Tag 1: (1005, 0), Tag 7: (1015, 2000) -> y midpoint = (1010, 1000)
        # Both pairs present: average of midpoints = (1005.0, 1010.0)
        detected_a = {
            0: (0.0, 0.0),
            1: (1005.0, 0.0),
            2: (2000.0, 0.0),
            3: (0.0, 1010.0),
            # 4 is undetected
            5: (2000.0, 1030.0),
            6: (0.0, 2000.0),
            7: (1015.0, 2000.0),
            8: (2000.0, 2000.0),
        }

        res_a, _, _ = match_global_transform_and_assign_undetected(
            detected_positions=detected_a,
            nominal_positions=nominal_positions,
            grid_shape=grid_shape,
        )
        self.assertAlmostEqual(res_a[4][0], 1005.0, places=3)
        self.assertAlmostEqual(res_a[4][1], 1010.0, places=3)

        # Case B: Only horizontal opposite neighbors (3 and 5) detected for 4 (1 and 7 missing)
        detected_b = {
            0: (0.0, 0.0),
            2: (2000.0, 0.0),
            3: (0.0, 1010.0),
            5: (2000.0, 1030.0),
            6: (0.0, 2000.0),
            8: (2000.0, 2000.0),
        }
        res_b, _, _ = match_global_transform_and_assign_undetected(
            detected_positions=detected_b,
            nominal_positions=nominal_positions,
            grid_shape=grid_shape,
        )
        self.assertAlmostEqual(res_b[4][0], 1000.0, places=3)
        self.assertAlmostEqual(res_b[4][1], 1020.0, places=3)

        # Case C: Only vertical opposite neighbors (1 and 7) detected for 4 (3 and 5 missing)
        detected_c = {
            0: (0.0, 0.0),
            1: (1005.0, 0.0),
            2: (2000.0, 0.0),
            6: (0.0, 2000.0),
            7: (1015.0, 2000.0),
            8: (2000.0, 2000.0),
        }
        res_c, _, _ = match_global_transform_and_assign_undetected(
            detected_positions=detected_c,
            nominal_positions=nominal_positions,
            grid_shape=grid_shape,
        )
        self.assertAlmostEqual(res_c[4][0], 1010.0, places=3)
        self.assertAlmostEqual(res_c[4][1], 1000.0, places=3)

        # Case D: Corner tag 0 missing (has no opposite horizontal or vertical neighbors)
        # Should fallback to global rigid alignment
        detected_d = {
            m: (pos[0] + 50.0, pos[1] + 100.0)
            for m, pos in nominal_positions.items()
            if m != 0
        }
        res_d, _, _ = match_global_transform_and_assign_undetected(
            detected_positions=detected_d,
            nominal_positions=nominal_positions,
            grid_shape=grid_shape,
        )
        self.assertAlmostEqual(res_d[0][0], 50.0, places=1)
        self.assertAlmostEqual(res_d[0][1], 100.0, places=1)

    def test_estimate_camera_to_stage_transform_orthogonal_rigid(self):
        from lib.stage_calibration_reconstruction.solver import estimate_camera_to_stage_transform

        # Simulate skewed stage observations where motor X and Y are non-orthogonal (e.g. shear of 0.1)
        # Stage moves:
        # Move 1: stage moves by (1000, 50) while camera image shifts by (-1000, 0)
        # Move 2: stage moves by (0, 1000) while camera image shifts by (0, -1000)
        # Marker 0 observed at stage (0, 0) with cam_offset (0, 0)
        # Marker 0 observed at stage (1000, 50) with cam_offset (-1000, 0)
        # Marker 1 observed at stage (0, 0) with cam_offset (0, 0)
        # Marker 1 observed at stage (0, 1000) with cam_offset (0, -1000)
        obs = [
            (0, (0.0, 0.0), (0.0, 0.0)),
            (0, (1000.0, 50.0), (-1000.0, 0.0)),
            (1, (0.0, 0.0), (0.0, 0.0)),
            (1, (0.0, 1000.0), (0.0, -1000.0)),
        ]

        T = estimate_camera_to_stage_transform(obs, default_rotation=0.0)

        # T must be strictly orthogonal (rigid rotation + uniform scale):
        # 1. Columns must be orthogonal: dot product == 0
        dot_cols = T[0, 0] * T[0, 1] + T[1, 0] * T[1, 1]
        self.assertAlmostEqual(dot_cols, 0.0, places=5)

        # 2. Columns must have equal length (isotropic scale):
        len_col0 = np.hypot(T[0, 0], T[1, 0])
        len_col1 = np.hypot(T[0, 1], T[1, 1])
        self.assertAlmostEqual(len_col0, len_col1, places=5)
        self.assertAlmostEqual(len_col0, 1.0, delta=0.05)

    def test_additional_stage_rotation_in_calibration(self):
        from stage_control.calibration import StageCalibration, StageCalibrationPoint
        from core.machine_config import MachineConfig

        # Create a 3x3 grid centered around (5000, 5000)
        coords = [4000.0, 5000.0, 6000.0]
        pts = []
        mid = 0
        for r, y in enumerate(coords):
            for c, x in enumerate(coords):
                pts.append(StageCalibrationPoint((r, c), mid, (x, y), (x, y), 0.0))
                mid += 1

        # Base camera mounted at 90 deg relative to stage: (x_cam -> -y_stage, y_cam -> x_stage)
        # R_90 = [[0, -1], [1, 0]]
        base_T = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
        cal = StageCalibration(
            grid_shape=(3, 3),
            points=pts,
            cam_to_stage_matrix=base_T,
            additional_rotation_deg=0.0,
        )

        # Baseline: target optical coordinate c = (5000, -5000)
        # raw_space_target = base_T @ [5000, -5000] = [5000, 5000]
        raw_x0, raw_y0 = cal.inverse_map_xy(5000.0, -5000.0)
        self.assertAlmostEqual(raw_x0, 5000.0, places=3)
        self.assertAlmostEqual(raw_y0, 5000.0, places=3)

        # Apply +5 deg additional rotation
        cal.set_additional_rotation(5.0)
        self.assertEqual(cal.additional_rotation_deg, 5.0)

        # Forward map and inverse map should remain mutually inverse
        target_cam = (5100.0, -4900.0)
        rx, ry = cal.inverse_map_xy(target_cam[0], target_cam[1])
        fwd_x, fwd_y = cal.forward_map_xy(rx, ry)
        self.assertAlmostEqual(fwd_x, target_cam[0], places=3)
        self.assertAlmostEqual(fwd_y, target_cam[1], places=3)

        # Verify MachineConfig serialization round-trip
        cfg = MachineConfig()
        cfg.stage_calibration.additional_stage_rotation_deg = 2.5
        cfg.stage_calibration.additional_stage_skew = 0.005
        d = cfg.to_dict()
        self.assertEqual(d["stage_calibration"]["additional_stage_rotation_deg"], 2.5)
        self.assertEqual(d["stage_calibration"]["additional_stage_skew"], 0.005)
        cfg_loaded = MachineConfig.from_dict(d)
        self.assertEqual(cfg_loaded.stage_calibration.additional_stage_rotation_deg, 2.5)
        self.assertEqual(cfg_loaded.stage_calibration.additional_stage_skew, 0.005)

    def test_additional_stage_skew_in_calibration(self):
        from stage_control.calibration import StageCalibration, StageCalibrationPoint

        # Create a 3x3 grid centered around (5000, 5000)
        coords = [4000.0, 5000.0, 6000.0]
        pts = []
        mid = 0
        for r, y in enumerate(coords):
            for c, x in enumerate(coords):
                pts.append(StageCalibrationPoint((r, c), mid, (x, y), (x, y), 0.0))
                mid += 1

        # Identity cam to stage matrix for straightforward skew verification:
        # dx_stage = dx_cam, dy_stage = dy_cam
        identity_T = np.eye(2, dtype=np.float64)
        cal = StageCalibration(
            grid_shape=(3, 3),
            points=pts,
            cam_to_stage_matrix=identity_T,
            additional_rotation_deg=0.0,
            additional_skew=0.0,
        )

        # Baseline: with 0 skew, (5000, 5000) -> (5000, 5000)
        rx_base, ry_base = cal.inverse_map_xy(5000.0, 5000.0)
        self.assertAlmostEqual(rx_base, 5000.0, places=3)
        self.assertAlmostEqual(ry_base, 5000.0, places=3)

        # Apply skew of +0.01 (dy/dx = 0.01)
        # S = [[1, 0], [0.01, 1]]
        cal.set_additional_skew(0.01)
        self.assertEqual(cal.additional_skew, 0.01)

        # Under skew=0.01, (5000, 5000) -> ry = 5000 + 0.01*5000 = 5050
        rx0, ry0 = cal.inverse_map_xy(5000.0, 5000.0)
        self.assertAlmostEqual(rx0, 5000.0, places=3)
        self.assertAlmostEqual(ry0, 5050.0, places=3)

        # Moving +100 um along X should produce +1 um shift along Y (100 * 0.01 = 1)
        rx1, ry1 = cal.inverse_map_xy(5100.0, 5000.0)
        # Expected: rx = 5100.0, ry = 5000.0 + 0.01 * 5100.0 = 5051.0
        self.assertAlmostEqual(rx1, 5100.0, places=3)
        self.assertAlmostEqual(ry1, 5051.0, places=3)

        # Verify dy/dx ratio: delta_y / delta_x == 0.01
        dy = ry1 - ry0
        dx = rx1 - rx0
        self.assertAlmostEqual(dy / dx, 0.01, places=5)

        # Forward map and inverse map should remain mutually inverse
        fwd_x, fwd_y = cal.forward_map_xy(rx1, ry1)
        self.assertAlmostEqual(fwd_x, 5100.0, places=3)
        self.assertAlmostEqual(fwd_y, 5000.0, places=3)


if __name__ == "__main__":
    unittest.main()
