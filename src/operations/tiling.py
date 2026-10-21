from typing import Callable, Optional

from core.chip_project import PatterningSettings
from core.events import ColorMode, ProjectorImageSource
from core.operation import ExecutionContext, Operation, sub_progress
from operations.autofocus import AutofocusOperation
from operations.exposure import ExposureOperation
from operations.movement import JogOperation


class TiledExposureOperation(Operation):
    """Runs a multi-tile step-and-repeat exposure sequence composing lower-level operations directly."""

    def __init__(
        self,
        layer_index: int,
        settings: Optional[PatterningSettings] = None,
        uv_z_offset: Optional[float] = None,
    ):
        super().__init__("Tiled Exposure")
        self.layer_index = layer_index
        self.settings = settings
        self.uv_z_offset = uv_z_offset
        self._current_sub_op: Optional[Operation] = None

    def abort(self):
        super().abort()
        if self._current_sub_op is not None:
            self._current_sub_op.abort()

    def execute(self, context: ExecutionContext, report_progress: Callable[[float, str], None]) -> Optional[str]:
        if context.project is None or self.layer_index >= len(context.project.layers):
            return "Invalid project or layer index"

        context.project.select_layer(self.layer_index)
        layer = context.project.layers[self.layer_index]

        # 1. Get the tiling path for the active layer
        pos = context.stage.get_position()
        start_pos = (pos[0], pos[1])
        tiles, tile_coords = layer.generate_tiles()
        total_tiles = len(tiles)

        if total_tiles == 0:
            report_progress(1.0, "No tiles to expose")
            return None

        report_progress(0.0, f"Beginning tiled exposure ({total_tiles} tiles)...")

        last_error: Optional[str] = None
        try:
            for tile_idx, (offset_x, offset_y) in enumerate(tile_coords):
                if self.is_aborted:
                    break

                pct_base = tile_idx / total_tiles
                pct_step = 1.0 / total_tiles

                # Step 1: Set the projector to black
                context.projector.set_on(False)

                # Step 2: Move to the position
                tx = start_pos[0] + offset_x
                ty = start_pos[1] + offset_y
                jog_reporter = sub_progress(
                    report_progress,
                    pct_base,
                    pct_base + 0.15 * pct_step,
                    prefix=f"Tile {tile_idx + 1}/{total_tiles}",
                )
                report_progress(
                    pct_base,
                    f"Tile {tile_idx + 1}/{total_tiles} - Moving stage to ({tx:.1f}, {ty:.1f})...",
                )
                jog_op = JogOperation({"x": tx, "y": ty}, relative=False)
                err = self._run_sub_op(jog_op, context, jog_reporter)
                if err or self.is_aborted:
                    last_error = err
                    break

                # Step 3: Set the active tile
                context.project.select_tile(tile_idx)

                # Step 4: Source to active layer
                context.projector.set_image_source(ProjectorImageSource.ACTIVE_LAYER)

                # Step 5: Set projector to red and run auto focus
                context.projector.set_color_mode(ColorMode.RED)
                af_reporter = sub_progress(
                    report_progress,
                    pct_base + 0.15 * pct_step,
                    pct_base + 0.60 * pct_step,
                    prefix=f"Tile {tile_idx + 1}/{total_tiles}",
                )
                af_op = AutofocusOperation(uv_z_offset=self.uv_z_offset)
                err = self._run_sub_op(af_op, context, af_reporter)
                if err or self.is_aborted:
                    last_error = err
                    break

                # Step 6: Then, do an exposure
                exp_reporter = sub_progress(
                    report_progress,
                    pct_base + 0.60 * pct_step,
                    pct_base + 1.0 * pct_step,
                    prefix=f"Tile {tile_idx + 1}/{total_tiles}",
                )
                exp_time = self.settings.exposure_time if self.settings is not None else None
                exp_op = ExposureOperation(
                    layer_index=self.layer_index,
                    exposure_time=exp_time,
                    tile_index=tile_idx,
                )
                err = self._run_sub_op(exp_op, context, exp_reporter)
                if err or self.is_aborted:
                    last_error = err
                    break
        finally:
            context.projector.set_on(False)

        if self.is_aborted:
            report_progress(1.0, "Tiled exposure aborted")
            return "Tiled exposure aborted"
        elif last_error is not None:
            msg = f"Tiled exposure failed: {last_error}"
            report_progress(1.0, msg)
            return msg
        else:
            report_progress(1.0, "Tiled exposure complete")
            return None

    def _run_sub_op(
        self,
        op: Operation,
        context: ExecutionContext,
        progress_cb: Callable[[float, str], None],
    ) -> Optional[str]:
        if self.is_aborted:
            return "Aborted"
        self._current_sub_op = op
        try:
            return op.execute(context, progress_cb)
        finally:
            self._current_sub_op = None
