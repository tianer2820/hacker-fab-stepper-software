import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.operation import ExecutionContext
from operations.chip_z_leveling import ChipZLevelingOperation, fit_plane
from stage_control.calibration import StageCalibration, StageCalibrationPoint
from stage_control.dummy_stage import DummyStage


class TestChipZLeveling(unittest.TestCase):
    def setUp(self):
        # 2x2 base calibration grid
        self.points = [
            StageCalibrationPoint((0, 0), 0, (0.0, 0.0), (0.0, 0.0), 100.0),
            StageCalibrationPoint((0, 1), 1, (10000.0, 0.0), (10000.0, 0.0), 110.0),
            StageCalibrationPoint((1, 0), 2, (0.0, 10000.0), (0.0, 10000.0), 120.0),
            StageCalibrationPoint((1, 1), 3, (10000.0, 10000.0), (10000.0, 10000.0), 130.0),
        ]
        self.cal = StageCalibration(
            grid_shape=(2, 2),
            points=self.points,
            stage_type="dummy",
        )

    def test_fit_plane(self):
        # Test exact plane: Delta_Z = 0.002 * X - 0.001 * Y + 15.0
        coords = [
            (1000.0, 2000.0, 0.002 * 1000.0 - 0.001 * 2000.0 + 15.0),
            (5000.0, 2000.0, 0.002 * 5000.0 - 0.001 * 2000.0 + 15.0),
            (5000.0, 8000.0, 0.002 * 5000.0 - 0.001 * 8000.0 + 15.0),
            (1000.0, 8000.0, 0.002 * 1000.0 - 0.001 * 8000.0 + 15.0),
        ]
        a, b, c = fit_plane(coords)
        self.assertAlmostEqual(a, 0.002, places=6)
        self.assertAlmostEqual(b, -0.001, places=6)
        self.assertAlmostEqual(c, 15.0, places=4)

    def test_calibration_plane_offset(self):
        # Base z offset at (5000, 5000) should be 115.0
        base_z = self.cal.get_base_z_offset(5000.0, 5000.0)
        self.assertAlmostEqual(base_z, 115.0, places=2)
        self.assertAlmostEqual(self.cal.get_z_offset(5000.0, 5000.0), 115.0, places=2)

        # Set chip Z plane: Delta_Z = 0.001 * X + 0.002 * Y + 5.0
        plane = (0.001, 0.002, 5.0)
        self.cal.set_chip_z_plane(plane)
        self.assertEqual(self.cal.get_chip_z_plane(), plane)

        # Expected offset at (5000, 5000): 0.001*5000 + 0.002*5000 + 5.0 = 5 + 10 + 5 = 20.0
        # Total z offset: 115.0 + 20.0 = 135.0
        self.assertAlmostEqual(self.cal.get_base_z_offset(5000.0, 5000.0), 115.0, places=2)
        self.assertAlmostEqual(self.cal.get_z_offset(5000.0, 5000.0), 135.0, places=2)

        # Verify corr_to_raw includes this offset
        raw_coords = self.cal.corr_to_raw([5000.0, 5000.0, 10.0])
        self.assertAlmostEqual(raw_coords["z"], 145.0, places=2)

        # Clear chip plane
        self.cal.clear_chip_z_plane()
        self.assertIsNone(self.cal.get_chip_z_plane())
        self.assertAlmostEqual(self.cal.get_z_offset(5000.0, 5000.0), 115.0, places=2)

    def test_chip_z_plane_not_saved_to_disk(self):
        self.cal.set_chip_z_plane((0.001, 0.002, 5.0))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = tf.name

        try:
            self.cal.save_to_json(temp_path)
            loaded = StageCalibration.load_from_json(temp_path)
            self.assertIsNone(loaded.get_chip_z_plane())
            self.assertAlmostEqual(loaded.get_z_offset(5000.0, 5000.0), 115.0, places=2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_chip_z_leveling_requires_calibration(self):
        stage = DummyStage()
        stage.set_calibration(None)

        context = ExecutionContext(
            stage=stage,
            camera=MagicMock(),
            projector=MagicMock(),
            machine_config=None,
            delay_func=lambda d: None,
        )

        op = ChipZLevelingOperation(chip_size_mm=10.0)
        err = op.execute(context, lambda p, m: None)
        self.assertIsNotNone(err)
        self.assertIn("No StageCalibration loaded", err)

    def test_chip_z_leveling_operation_execution(self):
        stage = DummyStage()
        stage.set_calibration(self.cal)
        stage.move_absolute({"x": 5000.0, "y": 5000.0, "z": 0.0})

        mock_projector = MagicMock()
        mock_projector.projector_size.return_value = (100, 100)
        mock_camera = MagicMock()
        mock_camera.get_latest_frame.return_value = None

        context = ExecutionContext(
            stage=stage,
            camera=mock_camera,
            projector=mock_projector,
            machine_config=None,
            delay_func=lambda d: None,
        )

        op = ChipZLevelingOperation(chip_size_mm=4.0, stabilization_delay=0.0)

        # Mock sub-operation execution so autofocus returns fixed best_red_z values per point
        # Point 1: (3000, 3000) -> 5.0
        # Point 2: (7000, 3000) -> 9.0
        # Point 3: (7000, 7000) -> 13.0
        # Point 4: (3000, 7000) -> 9.0
        # This corresponds to: Delta_Z = 0.001 * X + 0.001 * Y - 1.0
        # (3000: 3 + 3 - 1 = 5; 7000, 3000: 7 + 3 - 1 = 9; etc.)
        call_count = 0
        expected_z = [5.0, 9.0, 13.0, 9.0]

        def mock_run_sub_op(sub_op, ctx, reporter):
            nonlocal call_count
            sub_op.best_red_z = expected_z[call_count]
            call_count += 1
            return None

        op._run_sub_op = mock_run_sub_op

        progress_reports = []
        err = op.execute(context, lambda p, m: progress_reports.append((p, m)))

        self.assertIsNone(err)
        self.assertEqual(call_count, 4)

        # Verify returned to original position
        self.assertEqual(list(stage.get_position()), [5000.0, 5000.0, 0.0])

        # Verify plane fitted and set
        plane = self.cal.get_chip_z_plane()
        self.assertIsNotNone(plane)
        self.assertAlmostEqual(plane[0], 0.001, places=5)
        self.assertAlmostEqual(plane[1], 0.001, places=5)
        self.assertAlmostEqual(plane[2], -1.0, places=3)


if __name__ == "__main__":
    unittest.main()
