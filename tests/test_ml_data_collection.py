import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from camera.camera_module import DummyCamera
from core.engine import StepperEngine
from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext
from lib.ar_tag import detect_ar_tags
from lib.cross_marker import generate_cross_pattern, get_base_cross_vertices
from lib.projector_camera_transform import (
    compute_projector_to_camera_homography,
    generate_calibration_tag_grid_with_coords,
    transform_marker_to_camera_space,
    transform_points_projector_to_camera,
    warp_projector_image_to_camera,
)
from operations.ml_data_collection import MLDataCollectionConfig, MLDataCollectionOperation
from projector import DummyProjector
from stage_control.dummy_stage import DummyStage
from ui.bridge import QtEngineBridge
from ui.widgets.machine_control_panel.machine_control_panel_widget import MachineControlPanelWidget
from ui.widgets.machine_control_panel.ml_data_collection_tab import MLDataCollectionTabWidget


app = QApplication.instance() or QApplication([])


class TestCrossMarkerGeneration(unittest.TestCase):
    def test_base_cross_vertices(self):
        b = 10.0
        verts = get_base_cross_vertices(b)
        self.assertEqual(verts.shape, (12, 2))
        # Top-most point is at y = -1.5b
        self.assertAlmostEqual(np.min(verts[:, 1]), -15.0)
        # Bottom-most point is at y = +1.5b
        self.assertAlmostEqual(np.max(verts[:, 1]), 15.0)
        # Left-most point is at x = -1.5b
        self.assertAlmostEqual(np.min(verts[:, 0]), -15.0)
        # Right-most (longer arm) is at x = +2.5b
        self.assertAlmostEqual(np.max(verts[:, 0]), 25.0)

    def test_generate_cross_pattern(self):
        canvas_w, canvas_h = 800, 600
        canvas, markers = generate_cross_pattern(
            canvas_size=(canvas_w, canvas_h),
            target_scale_pct=8.0,
            scale_jitter_pct=2.0,
            marker_count=10,
            random_seed=42,
        )
        self.assertEqual(canvas.shape, (canvas_h, canvas_w, 3))
        self.assertEqual(len(markers), 10)

        longer_edge = max(canvas_w, canvas_h)
        for m in markers:
            self.assertGreaterEqual(m.scale_pct, 6.0 - 1e-5)
            self.assertLessEqual(m.scale_pct, 10.0 + 1e-5)
            # Center inside canvas
            self.assertGreater(m.center_proj[0], 0)
            self.assertLess(m.center_proj[0], canvas_w)
            self.assertGreater(m.center_proj[1], 0)
            self.assertLess(m.center_proj[1], canvas_h)
            # All vertices inside canvas
            for vx, vy in m.vertices_proj:
                self.assertGreaterEqual(vx, 0)
                self.assertLess(vx, canvas_w)
                self.assertGreaterEqual(vy, 0)
                self.assertLess(vy, canvas_h)

        # Verify no markers overlap (enclosing circles check)
        for i in range(len(markers)):
            for j in range(i + 1, len(markers)):
                m1 = markers[i]
                m2 = markers[j]
                b1 = m1.scale_px / 4.0
                b2 = m2.scale_px / 4.0
                r1 = 2.6 * b1
                r2 = 2.6 * b2
                dist = np.hypot(m1.center_proj[0] - m2.center_proj[0], m1.center_proj[1] - m2.center_proj[1])
                self.assertGreaterEqual(dist, r1 + r2)


class TestProjectorCameraTransform(unittest.TestCase):
    def test_calibration_grid_and_homography(self):
        proj_w, proj_h = 640, 480
        grid_img, tag_coords = generate_calibration_tag_grid_with_coords(
            grid_n=2,
            canvas_size=(proj_w, proj_h),
        )
        self.assertEqual(grid_img.shape, (proj_h, proj_w, 3))
        self.assertEqual(len(tag_coords), 4)

        # Detect ArUco tags directly on generated pattern
        count, corners, ids = detect_ar_tags(grid_img)
        self.assertEqual(count, 4)

        # Estimate identity homography
        H, matched = compute_projector_to_camera_homography(corners, ids, tag_coords)
        self.assertIsNotNone(H)
        self.assertEqual(matched, 16)
        # Check that H is close to identity (since camera frame == projector frame in this test)
        norm_H = H / H[2, 2]
        np.testing.assert_allclose(norm_H, np.eye(3), atol=0.1)

    def test_transform_points_and_marker(self):
        # Create a homography representing translation + scaling
        # x_cam = 0.5 * x_proj + 50, y_cam = 0.5 * y_proj + 30
        H = np.array(
            [
                [0.5, 0.0, 50.0],
                [0.0, 0.5, 30.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        pts = np.array([[100.0, 200.0], [300.0, 400.0]])
        transformed = transform_points_projector_to_camera(pts, H)
        expected = np.array([[100.0, 130.0], [200.0, 230.0]])
        np.testing.assert_allclose(transformed, expected, atol=1e-5)

        # Warp image test
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:200, 100:200] = 255
        warped = warp_projector_image_to_camera(img, H, (320, 240))
        self.assertEqual(warped.shape, (240, 320, 3))


class TestMLDataCollectionOperation(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.stage = DummyStage()
        self.stage.current_pos = [1000.0, 2000.0, 100.0]
        self.projector = DummyProjector()
        self.camera = DummyCamera()

        # Generate a synthetic camera frame with 2x2 ArUco tags for calibration
        proj_w, proj_h = self.projector.projector_size()
        cal_img, _ = generate_calibration_tag_grid_with_coords(grid_n=2, canvas_size=(proj_w, proj_h))
        self.camera._active = True
        self.camera._latest_frame = cal_img.copy()

        self.context = ExecutionContext(
            stage=self.stage,
            projector=self.projector,
            camera=self.camera,
            delay_func=lambda s: None,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_ml_data_collection_run(self):
        config = MLDataCollectionConfig(
            total_patterns=2,
            pattern_gap=500.0,
            target_scale_pct=8.0,
            scale_jitter_pct=1.0,
            marker_count=5,
            exposure_time=0.1,
            stabilization_delay=0.0,
            save_directory=self.test_dir,
            grid_n=2,
        )

        op = MLDataCollectionOperation(config=config)

        # Mock sub-operations to avoid slow camera loops
        with patch.object(op, "_run_sub_op", wraps=op._run_sub_op) as mock_run:
            def side_effect_sub_op(sub_op, ctx, cb):
                if sub_op.name == "Autofocus":
                    sub_op.best_red_z = 100.0
                    return None
                elif sub_op.name in ("Exposure", "Layer Exposure"):
                    return None
                elif sub_op.name in ("Jog", "Move"):
                    return sub_op.execute(ctx, cb)
                return sub_op.execute(ctx, cb)

            mock_run.side_effect = side_effect_sub_op

            progress_reports = []
            err = op.execute(self.context, lambda p, m: progress_reports.append((p, m)))
            self.assertIsNone(err)

        # Verify output files
        pattern1_img = os.path.join(self.test_dir, "pattern_0001.png")
        pattern1_gt = os.path.join(self.test_dir, "pattern_0001_projected.png")
        pattern1_json = os.path.join(self.test_dir, "pattern_0001.json")
        pattern2_img = os.path.join(self.test_dir, "pattern_0002.png")
        pattern2_gt = os.path.join(self.test_dir, "pattern_0002_projected.png")
        pattern2_json = os.path.join(self.test_dir, "pattern_0002.json")
        master_json = os.path.join(self.test_dir, "dataset_metadata.json")

        self.assertTrue(os.path.exists(pattern1_img))
        self.assertTrue(os.path.exists(pattern1_gt))
        self.assertTrue(os.path.exists(pattern1_json))
        self.assertTrue(os.path.exists(pattern2_img))
        self.assertTrue(os.path.exists(pattern2_gt))
        self.assertTrue(os.path.exists(pattern2_json))
        self.assertTrue(os.path.exists(master_json))

        # Check JSON contents
        with open(pattern1_json, "r") as f:
            rec1 = json.load(f)
        self.assertEqual(rec1["step_index"], 1)
        self.assertEqual(rec1["image_filename"], "pattern_0001.png")
        self.assertEqual(rec1["projected_gt_filename"], "pattern_0001_projected.png")
        self.assertIn("markers", rec1)
        self.assertEqual(len(rec1["markers"]), 5)
        for m in rec1["markers"]:
            self.assertIn("center_img", m)
            self.assertIn("rotation_img_deg", m)
            self.assertIn("scale_img_px", m)

        with open(master_json, "r") as f:
            master = json.load(f)
        self.assertEqual(master["total_patterns_collected"], 2)
        self.assertIn("homography_matrix", master)


class TestMLDataCollectionTabWidget(unittest.TestCase):
    def setUp(self):
        self.stage = DummyStage()
        self.projector = DummyProjector()
        self.camera = DummyCamera()
        self.engine = StepperEngine(stage=self.stage, projector=self.projector, camera=self.camera)
        self.bridge = QtEngineBridge(engine=self.engine)

    def test_tab_widget_controls_and_locking(self):
        panel = MachineControlPanelWidget(self.engine, self.bridge)
        ml_tab = panel.ml_tab

        # Test defaults
        self.assertAlmostEqual(ml_tab.spin_target_scale.value(), 8.0)
        self.assertAlmostEqual(ml_tab.spin_scale_jitter.value(), 2.0)
        self.assertEqual(ml_tab.spin_marker_count.value(), 20)
        self.assertEqual(ml_tab.spin_total_patterns.value(), 10)
        self.assertAlmostEqual(ml_tab.spin_pattern_gap.value(), 1000.0)

        # Test lock state
        ml_tab.update_lock_state(is_busy=True)
        self.assertFalse(ml_tab.btn_start.isEnabled())
        self.assertFalse(ml_tab.spin_target_scale.isEnabled())
        self.assertFalse(ml_tab.spin_total_patterns.isEnabled())

        ml_tab.update_lock_state(is_busy=False)
        self.assertTrue(ml_tab.btn_start.isEnabled())
        self.assertTrue(ml_tab.spin_target_scale.isEnabled())
        self.assertTrue(ml_tab.spin_total_patterns.isEnabled())

        # Test start triggers operation
        with patch.object(self.bridge, "start_operation") as mock_start:
            ml_tab.btn_start.click()
            mock_start.assert_called_once()
            op = mock_start.call_args[0][0]
            self.assertIsInstance(op, MLDataCollectionOperation)
            self.assertEqual(op.config.total_patterns, 10)

        # Test finish updates status
        ml_tab.on_operation_finished(op, err=None)
        self.assertEqual(ml_tab.lbl_status.text(), "Status: Complete")

        ml_tab.on_operation_finished(op, err="Device timeout")
        self.assertEqual(ml_tab.lbl_status.text(), "Status: Failed - Device timeout")


if __name__ == "__main__":
    unittest.main()
