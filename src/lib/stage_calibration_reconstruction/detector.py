from typing import List, Optional, Tuple
import cv2
import numpy as np

from lib.edge_artag import (
    EdgeArucoDetector,
    EdgeDetectorParameters,
    detect_edge_markers,
)
from operations.stage_calibration import DEFAULT_ARUCO_DICT


def detect_markers_in_frame(
    frame: Optional[np.ndarray],
    parameters: Optional[EdgeDetectorParameters] = None,
    dictionary: Optional[cv2.aruco.Dictionary] = None,
    flip_horizontal: bool = False,
) -> Tuple[int, List[np.ndarray], Optional[np.ndarray]]:
    """Detects ArUco markers using edge_artag without fallback.

    Args:
        frame: BGR or grayscale numpy image.
        parameters: Optional EdgeDetectorParameters for fine-tuning.
        dictionary: Optional cv2.aruco.Dictionary (defaults to DEFAULT_ARUCO_DICT).
        flip_horizontal: Whether to flip the frame horizontally before detection.

    Returns:
        (count, corners_list, ids_array)
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return 0, [], None

    img = cv2.flip(frame, 1) if flip_horizontal else frame
    dict_obj = dictionary if dictionary is not None else DEFAULT_ARUCO_DICT

    corners, ids, _ = detect_edge_markers(
        img,
        dictionary=dict_obj,
        parameters=parameters,
    )
    count = len(ids) if ids is not None else 0
    return count, corners, ids


def draw_detected_markers(
    image: np.ndarray,
    corners: List[np.ndarray],
    ids: Optional[np.ndarray],
    flip_horizontal: bool = False,
) -> np.ndarray:
    """Draws detected ArUco marker boundaries, corner origins, and IDs on an image copy."""
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    annotated = cv2.flip(image, 1) if flip_horizontal else image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    if ids is None or len(corners) == 0:
        return annotated

    for idx, c in enumerate(corners):
        pts = c.reshape(-1, 2).astype(np.int32)
        # Bounding polygon
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # Top-left corner highlight (origin of the tag)
        cv2.circle(annotated, tuple(pts[0]), radius=5, color=(0, 0, 255), thickness=-1)
        # ID label
        mid = int(ids.flatten()[idx])
        center = np.mean(pts, axis=0).astype(np.int32)
        cv2.putText(
            annotated,
            f"ID:{mid}",
            (center[0] - 20, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated

