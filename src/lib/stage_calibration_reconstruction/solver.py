from typing import Dict, List, Optional, Tuple
import numpy as np


def estimate_chip_rotation(
    corners: List[np.ndarray],
    ids: np.ndarray,
    nominal_dict: Dict[int, Dict],
) -> Optional[float]:
    """Estimates the wafer/chip rotation angle theta (radians) from detected tag corners."""
    angles = []
    for idx_det, mid_val in enumerate(ids.flatten()):
        mid_int = int(mid_val)
        if mid_int in nominal_dict:
            tag_corners = corners[idx_det].reshape(-1, 2)
            edge_top = tag_corners[1] - tag_corners[0]
            angles.append(np.arctan2(edge_top[1], edge_top[0]))

    if not angles:
        return None
    return float(np.median(angles))


def extract_pairwise_measurements(
    corners: List[np.ndarray],
    ids: np.ndarray,
    nominal_dict: Dict[int, Dict],
    pixel_scale: float,
    R_inv: np.ndarray,
) -> List[Tuple[int, int, float, float]]:
    """Extracts all relative pairwise distance measurements (id_a, id_b, dx_wafer, dy_wafer)

    from detected tags in a single camera frame.
    """
    centers_in_frame: Dict[int, np.ndarray] = {}
    for idx_m, mid_val in enumerate(ids.flatten()):
        m_int = int(mid_val)
        if m_int in nominal_dict:
            pts = corners[idx_m].reshape(-1, 2)
            centers_in_frame[m_int] = np.mean(pts, axis=0)

    measurements: List[Tuple[int, int, float, float]] = []
    marker_ids_found = list(centers_in_frame.keys())
    for i in range(len(marker_ids_found)):
        for j in range(i + 1, len(marker_ids_found)):
            id_a = marker_ids_found[i]
            id_b = marker_ids_found[j]
            disp_cam_px = centers_in_frame[id_b] - centers_in_frame[id_a]
            disp_cam_um = disp_cam_px * pixel_scale
            disp_wafer = R_inv @ disp_cam_um
            measurements.append((id_a, id_b, float(disp_wafer[0]), float(disp_wafer[1])))

    return measurements


def solve_global_positions(
    grid_shape: Tuple[int, int],
    measurements: List[Tuple[int, int, float, float]],
    anchor_id: int,
    anchor_pos: Tuple[float, float],
    nominal_positions: Dict[int, Tuple[float, float]],
) -> Dict[int, Tuple[float, float]]:
    """Reconstructs global optical positions using least squares from pairwise relative vectors.

    Args:
        grid_shape: (ny, nx) for the grid. Total markers M = ny * nx.
        measurements: List of (id_A, id_B, dx, dy) where x_B - x_A = dx and y_B - y_A = dy.
        anchor_id: Marker ID to anchor (e.g. 0).
        anchor_pos: (X, Y) coordinate for the anchor.
        nominal_positions: Dict mapping marker_id -> nominal (X, Y) fallback.

    Returns:
        Dict mapping marker_id -> reconstructed (X, Y).
    """
    ny, nx = grid_shape
    total_markers = ny * nx
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
    if 0 <= anchor_id < total_markers:
        A[K, anchor_id] = 1.0
        bx[K] = anchor_pos[0]
        by[K] = anchor_pos[1]
    else:
        # Fallback anchor to first available nominal id
        first_id = list(nominal_positions.keys())[0] if nominal_positions else 0
        A[K, first_id] = 1.0
        f_pos = nominal_positions.get(first_id, (0.0, 0.0))
        bx[K] = f_pos[0]
        by[K] = f_pos[1]

    sol_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)

    # Check observed markers
    observed = set()
    for id_a, id_b, _, _ in measurements:
        observed.add(id_a)
        observed.add(id_b)
    observed.add(anchor_id)

    results: Dict[int, Tuple[float, float]] = {}
    for m_id in nominal_positions:
        if m_id in observed and 0 <= m_id < total_markers and not (np.isnan(sol_x[m_id]) or np.isnan(sol_y[m_id])):
            results[m_id] = (float(sol_x[m_id]), float(sol_y[m_id]))
        else:
            results[m_id] = nominal_positions.get(m_id, (0.0, 0.0))

    return results

