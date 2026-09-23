import os
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.chip_project import ChipLayer, ChipProject
from core.events import Event, EventBus
from core.engine import StepperEngine
from projector import DummyProjector
from camera.camera_module import DummyCamera
from stage_control.dummy_stage import DummyStage
from ui.bridge import QtEngineBridge
from ui.widgets.workflow_panel import WorkflowPanelWidget
from PySide6.QtWidgets import QApplication


class TestPatternScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a test pattern image: 200x100 (width 200, height 100)
        cls.test_img_path = str(Path(__file__).resolve().parent / "test_pattern_temp.png")
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[20:80, 40:160] = [255, 255, 255]
        cv2.imwrite(cls.test_img_path, img)

        if QApplication.instance() is None:
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_img_path):
            os.remove(cls.test_img_path)

    def test_default_scale_disabled(self):
        project = ChipProject()
        layer = project.active_layer
        self.assertEqual(layer.scale_w, -1)
        self.assertEqual(layer.scale_h, -1)
        self.assertIsNone(layer._pattern_cache)

        layer.set_pattern_path(self.test_img_path)
        # Verify lazy evaluation: _pattern_cache should still be None
        self.assertIsNone(layer._pattern_cache)

        tiles, coords = layer.generate_tiles()
        self.assertIsNotNone(layer._pattern_cache)
        # Original size was 200x100
        self.assertEqual(layer._pattern_cache.shape[1], 200)
        self.assertEqual(layer._pattern_cache.shape[0], 100)

    def test_explicit_scaling_and_laziness(self):
        project = ChipProject()
        layer = project.active_layer
        layer.set_pattern_path(self.test_img_path)

        # Set scaling
        layer.scale_w = 400
        layer.scale_h = 300

        # Lazy: pattern should not be loaded or scaled yet
        self.assertIsNone(layer._pattern_cache)
        self.assertEqual(len(layer._tile_cache), 0)

        # Generate tiles triggers lazy scaling
        tiles, coords = layer.generate_tiles()
        self.assertIsNotNone(layer._pattern_cache)
        self.assertEqual(layer._pattern_cache.shape[1], 400)
        self.assertEqual(layer._pattern_cache.shape[0], 300)
        self.assertEqual(tiles[0].shape[1], 400)
        self.assertEqual(tiles[0].shape[0], 300)

    def test_rescaling_with_force_generate_tiles(self):
        project = ChipProject()
        layer = project.active_layer
        layer.set_pattern_path(self.test_img_path)
        layer.generate_tiles()
        self.assertEqual(layer._pattern_cache.shape[1], 200)
        self.assertEqual(layer._pattern_cache.shape[0], 100)

        # Update scale
        layer.scale_w = 500
        layer.scale_h = 250
        # Re-generating with force=True reloads and applies new scale
        layer.generate_tiles(force=True)
        self.assertEqual(layer._pattern_cache.shape[1], 500)
        self.assertEqual(layer._pattern_cache.shape[0], 250)

    def test_aspect_ratio_scaling_when_one_dimension_set(self):
        project = ChipProject()
        layer = project.active_layer
        layer.set_pattern_path(self.test_img_path)

        # Only scale_w set
        layer.scale_w = 400
        layer.scale_h = -1
        layer.generate_tiles(force=True)
        # Original is 200w x 100h -> 400w x 200h
        self.assertEqual(layer._pattern_cache.shape[1], 400)
        self.assertEqual(layer._pattern_cache.shape[0], 200)

        # Only scale_h set
        layer.scale_w = -1
        layer.scale_h = 50
        layer.generate_tiles(force=True)
        # Original is 200w x 100h -> 100w x 50h
        self.assertEqual(layer._pattern_cache.shape[1], 100)
        self.assertEqual(layer._pattern_cache.shape[0], 50)

    def test_serialization(self):
        layer = ChipLayer(name="Test Layer", pattern_path="/dummy/path.png", scale_w=1920, scale_h=1080)
        data = layer.to_disk()
        self.assertEqual(data["scale_w"], 1920)
        self.assertEqual(data["scale_h"], 1080)

        restored = ChipLayer.from_disk(data)
        self.assertEqual(restored.name, "Test Layer")
        self.assertEqual(restored.scale_w, 1920)
        self.assertEqual(restored.scale_h, 1080)
        self.assertEqual(restored.pattern_path, "/dummy/path.png")

    def test_gui_pattern_scaling_and_projector_match(self):
        stage = DummyStage()
        projector = DummyProjector(size=(1280, 720))
        camera = DummyCamera()
        engine = StepperEngine(stage=stage, projector=projector, camera=camera)
        bridge = QtEngineBridge(engine)

        widget = WorkflowPanelWidget(engine, bridge)
        layer_subpanel = widget.layer_subpanel

        # Check default UI values
        self.assertEqual(layer_subpanel.spin_scale_w.value(), -1)
        self.assertEqual(layer_subpanel.spin_scale_h.value(), -1)

        # Set pattern path
        engine.project.active_layer.set_pattern_path(self.test_img_path)
        engine.event_bus.emit(Event.PROJECT_CHANGED, engine.project)

        # Click Match Projector Resolution button
        layer_subpanel.btn_match_projector.click()
        self.assertEqual(layer_subpanel.spin_scale_w.value(), 1280)
        self.assertEqual(layer_subpanel.spin_scale_h.value(), 720)
        self.assertEqual(engine.project.active_layer.scale_w, 1280)
        self.assertEqual(engine.project.active_layer.scale_h, 720)

        # Pattern should still be lazily unscaled
        self.assertIsNone(engine.project.active_layer._pattern_cache)

        # Click Generate Tiles
        layer_subpanel.btn_regenerate_tiles.click()
        self.assertIsNotNone(engine.project.active_layer._pattern_cache)
        self.assertEqual(engine.project.active_layer._pattern_cache.shape[1], 1280)
        self.assertEqual(engine.project.active_layer._pattern_cache.shape[0], 720)


if __name__ == "__main__":
    unittest.main()
