import os
import sys
import json
import cv2
import numpy as np
import tempfile

import unittest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stage_control.calibration import StageCalibration, StageCalibrationPoint
from stage_control.dummy_stage import DummyStage


class TestStageCalibration(unittest.TestCase):
    def setUp(self):
        # Create a simple 2x2 calibration grid
        # Raw coords: (0, 0), (1000, 0), (0, 1000), (1000, 1000)
        # Optical coords: simulated 10 µm shift in X and 20 µm shift in Y, and tilted Z
        self.points = [
            StageCalibrationPoint((0, 0), 0, (0.0, 0.0), (10.0, 20.0), 100.0),
            StageCalibrationPoint((0, 1), 1, (1000.0, 0.0), (1010.0, 20.0), 110.0),
            StageCalibrationPoint((1, 0), 2, (0.0, 1000.0), (10.0, 1020.0), 120.0),
            StageCalibrationPoint((1, 1), 3, (1000.0, 1000.0), (1010.0, 1020.0), 130.0),
        ]
        self.cal = StageCalibration(
            grid_shape=(2, 2),
            points=self.points,
            stage_type="dummy",
        )

    def test_json_serialization(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            self.cal.save_to_json(temp_path)
            loaded = StageCalibration.load_from_json(temp_path)

            self.assertEqual(loaded.grid_shape, (2, 2))
            self.assertEqual(loaded.stage_type, "dummy")
            self.assertEqual(len(loaded.points), 4)
            self.assertEqual(loaded.points[0].marker_id, 0)
            self.assertAlmostEqual(loaded.points[0].z_focus, 100.0)
            self.assertAlmostEqual(loaded.points[3].raw_xy[0], 1000.0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_forward_and_inverse_mapping(self):
        # Raw midpoint (500, 500)
        cx, cy = self.cal.forward_map_xy(500.0, 500.0)
        self.assertAlmostEqual(cx, 510.0, places=3)
        self.assertAlmostEqual(cy, 520.0, places=3)

        # Invert back to raw
        rx, ry = self.cal.inverse_map_xy(cx, cy)
        self.assertAlmostEqual(rx, 500.0, places=2)
        self.assertAlmostEqual(ry, 500.0, places=2)

    def test_surface_relative_z(self):
        # At raw midpoint (500, 500), Z_focus should be average of (100, 110, 120, 130) = 115.0
        z_surf = self.cal.get_z_surface(510.0, 520.0)
        self.assertAlmostEqual(z_surf, 115.0, places=2)

        # to_raw with Z=0 (focal plane) should return raw Z = 115.0
        target_corr = {"x": 510.0, "y": 520.0, "z": 0.0}
        raw_dict = self.cal.to_raw(target_corr)
        self.assertAlmostEqual(raw_dict["x"], 500.0, places=2)
        self.assertAlmostEqual(raw_dict["y"], 500.0, places=2)
        self.assertAlmostEqual(raw_dict["z"], 115.0, places=2)

        # to_corr at raw (500, 500, 115) should return corr Z = 0.0
        corr_pos = self.cal.to_corr((500.0, 500.0, 115.0))
        self.assertAlmostEqual(corr_pos[0], 510.0, places=2)
        self.assertAlmostEqual(corr_pos[1], 520.0, places=2)
        self.assertAlmostEqual(corr_pos[2], 0.0, places=2)


class TestStageControllerCalibration(unittest.TestCase):
    def test_dummy_stage_with_calibration(self):
        stage = DummyStage(delay=0.0)
        self.assertIsNone(stage.get_calibration())

        # Uncalibrated move
        stage.move_absolute({"x": 100.0, "y": 200.0, "z": 50.0})
        self.assertEqual(stage.get_position(), (100.0, 200.0, 50.0))
        self.assertEqual(stage.get_raw_position(), (100.0, 200.0, 50.0))

        # Attach calibration
        points = [
            StageCalibrationPoint((0, 0), 0, (0.0, 0.0), (10.0, 20.0), 100.0),
            StageCalibrationPoint((0, 1), 1, (1000.0, 0.0), (1010.0, 20.0), 100.0),
            StageCalibrationPoint((1, 0), 2, (0.0, 1000.0), (10.0, 1020.0), 100.0),
            StageCalibrationPoint((1, 1), 3, (1000.0, 1000.0), (1010.0, 1020.0), 100.0),
        ]
        cal = StageCalibration((2, 2), points, stage_type="dummy")
        stage.set_calibration(cal)

        # Move absolute in corrected space: target (510.0, 520.0, 0.0)
        # Should command raw stage to (500.0, 500.0, 100.0)
        stage.move_absolute({"x": 510.0, "y": 520.0, "z": 0.0})
        
        raw_pos = stage.get_raw_position()
        self.assertAlmostEqual(raw_pos[0], 500.0, places=2)
        self.assertAlmostEqual(raw_pos[1], 500.0, places=2)
        self.assertAlmostEqual(raw_pos[2], 100.0, places=2)

        # get_position() reports corrected space
        corr_pos = stage.get_position()
        self.assertAlmostEqual(corr_pos[0], 510.0, places=2)
        self.assertAlmostEqual(corr_pos[1], 520.0, places=2)
        self.assertAlmostEqual(corr_pos[2], 0.0, places=2)

        # Relative move: +10 in x in corrected space
        stage.move_relative({"x": 10.0})
        corr_pos2 = stage.get_position()
        self.assertAlmostEqual(corr_pos2[0], 520.0, places=2)

        # Bypass raw move
        stage.move_absolute_raw({"x": 0.0, "y": 0.0, "z": 0.0})
        self.assertEqual(stage.get_raw_position(), (0.0, 0.0, 0.0))


class MockProjector:
    def __init__(self):
        self.is_on = False
        self.color_mode = None
        self.image_source = None
        self.generated_image = None

    def projector_size(self):
        return (1920, 1080)

    def set_on(self, on: bool):
        self.is_on = on

    def set_color_mode(self, mode):
        self.color_mode = mode

    def set_image_source(self, src):
        self.image_source = src

    def set_generated_image(self, img):
        self.generated_image = img

    def wait_for_display(self, timeout: float = 2.0) -> bool:
        return True


class MockCamera:
    def __init__(self, frame=None):
        self.frame = frame

    def get_latest_frame(self):
        return self.frame


class TestStageCalibrationOperations(unittest.TestCase):
    def test_compute_calibration_grid_params(self):
        from operations.stage_calibration import compute_calibration_grid_params
        # If camera FOV is 1080 µm, step pitch ~ 432 µm
        n, d = compute_calibration_grid_params(cal_range=10000.0, pixel_scale=1.0, camera_size=(1920, 1080))
        self.assertGreaterEqual(n, 2)
        self.assertLessEqual(d, 0.45 * 1080.0)

    def test_solve_global_positions_least_squares(self):
        from operations.stage_calibration import solve_global_positions_from_relative_measurements
        # In a 2x2 grid (markers 0, 1, 2, 3)
        # 0: (0, 0), 1: (100, 0), 2: (0, 100), 3: (100, 100)
        measurements = [
            (0, 1, 100.0, 0.0),   # 1 relative to 0
            (0, 2, 0.0, 100.0),   # 2 relative to 0
            (1, 3, 0.0, 100.0),   # 3 relative to 1
            (2, 3, 100.0, 0.0),   # 3 relative to 2
        ]
        nominal = {0: (0.0, 0.0), 1: (100.0, 0.0), 2: (0.0, 100.0), 3: (100.0, 100.0)}
        recon = solve_global_positions_from_relative_measurements(
            grid_size=2,
            measurements=measurements,
            anchor_id=0,
            anchor_pos=(0.0, 0.0),
            nominal_positions=nominal,
        )
        self.assertAlmostEqual(recon[0][0], 0.0, places=2)
        self.assertAlmostEqual(recon[0][1], 0.0, places=2)
        self.assertAlmostEqual(recon[1][0], 100.0, places=2)
        self.assertAlmostEqual(recon[1][1], 0.0, places=2)
        self.assertAlmostEqual(recon[2][0], 0.0, places=2)
        self.assertAlmostEqual(recon[2][1], 100.0, places=2)
        self.assertAlmostEqual(recon[3][0], 100.0, places=2)
        self.assertAlmostEqual(recon[3][1], 100.0, places=2)

    def test_three_stage_calibration_end_to_end(self):
        from core.operation import ExecutionContext
        from operations.stage_calibration import (
            StageCalibrationZFocusOperation,
            StageCalibrationPatternOperation,
            StageCalibrationScanOperation,
            generate_canonical_ar_tag,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            s1_path = os.path.join(tmpdir, "s1_focus.json")
            s3_path = os.path.join(tmpdir, "final_cal.json")

            stage = DummyStage(delay=0.0)
            projector = MockProjector()
            # Synthesize camera frame showing tag ID 0 and tag ID 1
            tag0_img, _ = generate_canonical_ar_tag(marker_id=0, size=80)
            tag1_img, _ = generate_canonical_ar_tag(marker_id=1, size=80)
            h, w = tag0_img.shape
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[150 : 150 + h, 100 : 100 + w, :] = np.stack([tag0_img] * 3, axis=-1)
            frame[150 : 150 + h, 350 : 350 + w, :] = np.stack([tag1_img] * 3, axis=-1)
            camera = MockCamera(frame=frame)

            ctx = ExecutionContext(
                stage=stage,
                camera=camera,
                projector=projector,
                delay_func=lambda s: None,
            )

            # 1. Run Stage 1 (Z Focus)
            op1 = StageCalibrationZFocusOperation(
                calibration_range=2000.0,
                camera_pixel_size=1.0,
                grid_size=2,
                step_distance=1000.0,
                stabilization_delay=0.0,
                output_json_path=s1_path,
                apply_to_stage=True,
            )
            err1 = op1.execute(ctx, lambda p, m: None)
            self.assertIsNone(err1)
            self.assertTrue(os.path.exists(s1_path))
            self.assertEqual(len(op1.recorded_points), 4)
            # Stage should now have Z-leveling only
            self.assertIsNotNone(stage.get_calibration())
            self.assertFalse(stage.get_calibration().has_xy_calibration)

            # 2. Run Stage 2 (Patterning)
            op2 = StageCalibrationPatternOperation(
                input_json_path=s1_path,
                exposure_time=0.01,
                stabilization_delay=0.0,
            )
            err2 = op2.execute(ctx, lambda p, m: None)
            self.assertIsNone(err2)

            # 3. Run Stage 3 (Optical Scan & Reconstruction)
            op3 = StageCalibrationScanOperation(
                input_json_path=s1_path,
                output_json_path=s3_path,
                camera_pixel_size=1.0,
                stabilization_delay=0.0,
                apply_to_stage=True,
            )
            err3 = op3.execute(ctx, lambda p, m: None)
            self.assertIsNone(err3)
            self.assertTrue(os.path.exists(s3_path))
            self.assertIsNotNone(op3.calibration)
            self.assertIsNotNone(stage.get_calibration())
            self.assertEqual(stage.get_calibration().grid_shape, (2, 2))
            self.assertTrue(stage.get_calibration().has_xy_calibration)

    def test_generate_spiral_grid_indices(self):
        from operations.stage_calibration import generate_spiral_grid_indices

        # 3x3 grid: center is (1, 1), should visit all 9 cells
        spiral_3x3 = generate_spiral_grid_indices(3, 3)
        self.assertEqual(len(spiral_3x3), 9)
        self.assertEqual(spiral_3x3[0], (1, 1))
        self.assertEqual(len(set(spiral_3x3)), 9)
        for r, c in spiral_3x3:
            self.assertTrue(0 <= r < 3 and 0 <= c < 3)

        # 2x2 grid: center is (0, 0), should visit all 4 cells
        spiral_2x2 = generate_spiral_grid_indices(2, 2)
        self.assertEqual(len(spiral_2x2), 4)
        self.assertEqual(spiral_2x2[0], (0, 0))
        self.assertEqual(len(set(spiral_2x2)), 4)

        # 5x5 grid: center is (2, 2), should visit all 25 cells
        spiral_5x5 = generate_spiral_grid_indices(5, 5)
        self.assertEqual(len(spiral_5x5), 25)
        self.assertEqual(spiral_5x5[0], (2, 2))
        self.assertEqual(len(set(spiral_5x5)), 25)

    def test_stage2_fails_explicitly_if_projection_uncalibrated(self):
        from core.operation import ExecutionContext
        from operations.stage_calibration import (
            StageCalibrationPatternOperation,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            s1_path = os.path.join(tmpdir, "stage1.json")
            StageCalibration(
                grid_shape=(2, 2),
                points=[
                    StageCalibrationPoint((0, 0), 0, (0.0, 0.0), (0.0, 0.0), 100.0),
                    StageCalibrationPoint((0, 1), 1, (1000.0, 0.0), (1000.0, 0.0), 100.0),
                    StageCalibrationPoint((1, 0), 2, (0.0, 1000.0), (0.0, 1000.0), 100.0),
                    StageCalibrationPoint((1, 1), 3, (1000.0, 1000.0), (1000.0, 1000.0), 100.0),
                ],
                stage_type="dummy",
            ).save_to_json(s1_path)

            stage = DummyStage(delay=0.0)
            projector = MockProjector()
            # Blank frame: 0 tags detected
            camera = MockCamera(frame=np.zeros((480, 640, 3), dtype=np.uint8))
            ctx = ExecutionContext(stage=stage, camera=camera, projector=projector, delay_func=lambda s: None)

            op2 = StageCalibrationPatternOperation(
                input_json_path=s1_path,
                exposure_time=0.01,
                stabilization_delay=0.0,
            )
            err = op2.execute(ctx, lambda p, m: None)
            self.assertIsNotNone(err)
            self.assertIn("Projector-to-camera optical calibration failed", err)

    def test_stage2_fails_explicitly_if_homography_inversion_fails(self):
        from unittest.mock import patch
        from core.operation import ExecutionContext
        from operations.stage_calibration import (
            StageCalibrationPatternOperation,
            generate_canonical_ar_tag,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            s1_path = os.path.join(tmpdir, "stage1.json")
            StageCalibration(
                grid_shape=(2, 2),
                points=[
                    StageCalibrationPoint((0, 0), 0, (0.0, 0.0), (0.0, 0.0), 100.0),
                    StageCalibrationPoint((0, 1), 1, (1000.0, 0.0), (1000.0, 0.0), 100.0),
                    StageCalibrationPoint((1, 0), 2, (0.0, 1000.0), (0.0, 1000.0), 100.0),
                    StageCalibrationPoint((1, 1), 3, (1000.0, 1000.0), (1000.0, 1000.0), 100.0),
                ],
                stage_type="dummy",
            ).save_to_json(s1_path)

            stage = DummyStage(delay=0.0)
            projector = MockProjector()
            tag_img, _ = generate_canonical_ar_tag(marker_id=0, size=100)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            h, w = tag_img.shape
            frame[100 : 100 + h, 100 : 100 + w, 0] = tag_img
            frame[100 : 100 + h, 100 : 100 + w, 1] = tag_img
            frame[100 : 100 + h, 100 : 100 + w, 2] = tag_img
            camera = MockCamera(frame=frame)
            ctx = ExecutionContext(stage=stage, camera=camera, projector=projector, delay_func=lambda s: None)

            op2 = StageCalibrationPatternOperation(
                input_json_path=s1_path,
                exposure_time=0.01,
                stabilization_delay=0.0,
            )
            # Patch findHomography to return a singular matrix (all zeros)
            with patch("cv2.findHomography", return_value=(np.zeros((3, 3), dtype=np.float64), np.ones((4, 1)))):
                err = op2.execute(ctx, lambda p, m: None)
                self.assertIsNotNone(err)
                self.assertIn("Failed to invert projector-to-camera homography", err)

    def test_stage2_uses_exposure_operation(self):
        from unittest.mock import patch
        from core.operation import ExecutionContext
        from operations.exposure import ExposureOperation
        from operations.stage_calibration import (
            StageCalibrationPatternOperation,
            generate_canonical_ar_tag,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            s1_path = os.path.join(tmpdir, "stage1.json")
            StageCalibration(
                grid_shape=(2, 2),
                points=[
                    StageCalibrationPoint((0, 0), 0, (0.0, 0.0), (0.0, 0.0), 100.0),
                    StageCalibrationPoint((0, 1), 1, (1000.0, 0.0), (1000.0, 0.0), 100.0),
                    StageCalibrationPoint((1, 0), 2, (0.0, 1000.0), (0.0, 1000.0), 100.0),
                    StageCalibrationPoint((1, 1), 3, (1000.0, 1000.0), (1000.0, 1000.0), 100.0),
                ],
                stage_type="dummy",
            ).save_to_json(s1_path)

            stage = DummyStage(delay=0.0)
            projector = MockProjector()
            tag_img, _ = generate_canonical_ar_tag(marker_id=0, size=100)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            h, w = tag_img.shape
            frame[100 : 100 + h, 100 : 100 + w, 0] = tag_img
            frame[100 : 100 + h, 100 : 100 + w, 1] = tag_img
            frame[100 : 100 + h, 100 : 100 + w, 2] = tag_img
            camera = MockCamera(frame=frame)
            ctx = ExecutionContext(stage=stage, camera=camera, projector=projector, delay_func=lambda s: None)

            op = StageCalibrationPatternOperation(
                input_json_path=s1_path,
                exposure_time=0.01,
                stabilization_delay=0.0,
            )

            with patch("operations.stage_calibration.ExposureOperation", wraps=ExposureOperation) as mock_exp_cls:
                err = op.execute(ctx, lambda p, m: None)
                self.assertIsNone(err)
                self.assertEqual(mock_exp_cls.call_count, 4)
                # Verify arguments passed to ExposureOperation
                call_args = mock_exp_cls.call_args_list[0]
                self.assertIsNone(call_args.kwargs.get("layer_index"))
                self.assertAlmostEqual(call_args.kwargs.get("exposure_time"), 10.0)

    def test_pass2_multi_tag_pose_averaging(self):
        from core.operation import ExecutionContext
        from operations.stage_calibration import (
            StageCalibrationScanOperation,
            generate_canonical_ar_tag,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            p1_path = os.path.join(tmpdir, "pass1.json")
            p2_path = os.path.join(tmpdir, "cal.json")

            # Create Pass 1 data with 4 tags in 2x2 grid
            pass1_data = {
                "version": "2.0",
                "timestamp": "2026-10-14T12:00:00",
                "grid_shape": [2, 2],
                "step_distance": 1000.0,
                "calibration_range": 2000.0,
                "pixel_scale": 1.0,
                "dict_id": int(cv2.aruco.DICT_4X4_1000),
                "flip_horizontal": False,
                "points": [
                    {"grid_index": [0, 0], "marker_id": 0, "raw_xy": [0.0, 0.0], "z_focus": 100.0},
                    {"grid_index": [0, 1], "marker_id": 1, "raw_xy": [1000.0, 0.0], "z_focus": 100.0},
                    {"grid_index": [1, 0], "marker_id": 2, "raw_xy": [0.0, 1000.0], "z_focus": 100.0},
                    {"grid_index": [1, 1], "marker_id": 3, "raw_xy": [1000.0, 1000.0], "z_focus": 100.0},
                ],
            }
            with open(p1_path, "w", encoding="utf-8") as f:
                json.dump(pass1_data, f)

            # Synthesize frame containing BOTH tag 0 and tag 1
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            tag0_img, _ = generate_canonical_ar_tag(marker_id=0, size=80)
            tag1_img, _ = generate_canonical_ar_tag(marker_id=1, size=80)
            h, w = tag0_img.shape
            # Tag 0 on the left, Tag 1 on the right
            frame[150 : 150 + h, 100 : 100 + w, :] = np.stack([tag0_img]*3, axis=-1)
            frame[150 : 150 + h, 350 : 350 + w, :] = np.stack([tag1_img]*3, axis=-1)

            stage = DummyStage(delay=0.0)
            camera = MockCamera(frame=frame)
            projector = MockProjector()
            ctx = ExecutionContext(stage=stage, camera=camera, projector=projector, delay_func=lambda s: None)

            op2 = StageCalibrationScanOperation(
                input_json_path=p1_path,
                output_json_path=p2_path,
                camera_pixel_size=1.0,
                stabilization_delay=0.0,
                apply_to_stage=True,
            )
            err = op2.execute(ctx, lambda p, m: None)
            self.assertIsNone(err)
            self.assertTrue(os.path.exists(p2_path))



class TestStageCalibrationUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])

    def test_ui_widgets_and_load_flow(self):
        from core.engine import StepperEngine
        from ui.bridge import QtEngineBridge
        from ui.widgets.machine_control_panel.machine_control_panel_widget import MachineControlPanelWidget
        from ui.widgets.machine_control_panel.stage_calibration_tab import StageCalibrationTabWidget

        from projector import DummyProjector
        from camera import DummyCamera

        stage = DummyStage(delay=0.0)
        projector = DummyProjector()
        camera = DummyCamera()
        engine = StepperEngine(stage=stage, projector=projector, camera=camera)
        bridge = QtEngineBridge(engine)

        panel = MachineControlPanelWidget(engine, bridge)
        self.assertEqual(panel.tabs.count(), 6)
        self.assertEqual(panel.tabs.tabText(4), "Stage Calibration")
        self.assertEqual(panel.tabs.tabText(5), "Machine Config")
        self.assertIsInstance(panel.stage_cal_tab, StageCalibrationTabWidget)

        # Check config tab calibration controls
        config_tab = panel.config_tab
        self.assertTrue(hasattr(config_tab, "btn_apply_stage_cal"))
        self.assertTrue(hasattr(config_tab, "btn_clear_stage_cal"))
        self.assertTrue(hasattr(config_tab, "lbl_cal_status"))
        self.assertIn("Raw", config_tab.lbl_cal_status.text())

        # Test setting Z-leveling calibration updates UI
        cal_z = StageCalibration(
            (2, 2),
            [
                StageCalibrationPoint((0, 0), 0, (0.0, 0.0), (0.0, 0.0), 100.0),
                StageCalibrationPoint((0, 1), 1, (100.0, 0.0), (100.0, 0.0), 100.0),
            ],
            stage_type="dummy",
        )
        stage.set_calibration(cal_z)
        config_tab._refresh_calibration_status()
        self.assertIn("Active: Z-Leveling Only", config_tab.lbl_cal_status.text())
        self.assertTrue(config_tab.btn_clear_stage_cal.isEnabled())

        # Test setting Full XY + Z calibration updates UI
        cal_full = StageCalibration(
            (2, 2),
            [
                StageCalibrationPoint((0, 0), 0, (0.0, 0.0), (5.0, 5.0), 100.0),
                StageCalibrationPoint((0, 1), 1, (100.0, 0.0), (105.0, 5.0), 100.0),
            ],
            stage_type="dummy",
        )
        stage.set_calibration(cal_full)
        config_tab._refresh_calibration_status()
        self.assertIn("Active: Full XY + Z", config_tab.lbl_cal_status.text())
        self.assertTrue(config_tab.btn_clear_stage_cal.isEnabled())

        # Test clear
        config_tab._on_clear_stage_cal_clicked()
        self.assertIn("Raw", config_tab.lbl_cal_status.text())
        self.assertFalse(config_tab.btn_clear_stage_cal.isEnabled())
        self.assertIsNone(stage.get_calibration())

        # Check Stage Calibration Tab fields & defaults
        stage_cal_tab = panel.stage_cal_tab
        self.assertTrue(hasattr(stage_cal_tab, "btn_start_s1"))
        self.assertTrue(hasattr(stage_cal_tab, "btn_start_s2"))
        self.assertTrue(hasattr(stage_cal_tab, "btn_start_s3"))
        self.assertEqual(stage_cal_tab.spin_exposure_time.value(), 8.0)
        self.assertEqual(stage_cal_tab.spin_cal_range.value(), 10000.0)
        self.assertFalse(hasattr(stage_cal_tab, "spin_pixel_scale"))
        self.assertEqual(panel.config_tab.spin_camera_pixel_size.value(), 1.0)
        self.assertIn("Computed Grid", stage_cal_tab.lbl_grid_info.text())
        self.assertFalse(hasattr(stage_cal_tab, "progress_bar"))




if __name__ == "__main__":
    unittest.main()


