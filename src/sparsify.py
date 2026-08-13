"""
Stage 8: Control point sparsification.

Converts dense primitive paths into sparse control polylines.
"""
from __future__ import annotations
import math
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .types import (
    NormalizedConfig, GridMap, DensePath, ControlPoint, ControlPath,
    SparseControlPaths, SparsifyReport, StageStatus, Point, PointF,
    angle_diff, euclidean_distance_f
)
from .geometry import (
    check_segment_collision, compute_control_length, compute_turns,
    point_to_segment_distance_f
)


def _compute_max_error(
    dense_points: List[Point],
    a: PointF,
    b: PointF,
) -> float:
    """Compute maximum geometric error of replacing dense segment with line AB.

    Args:
        dense_points: Subset of dense points between A and B (inclusive)
        a: Point A (start of segment)
        b: Point B (end of segment)

    Returns:
        Maximum distance from any dense point to line AB
    """
    max_err = 0.0
    for p in dense_points:
        d = point_to_segment_distance_f((float(p[0]), float(p[1])), a, b)
        if d > max_err:
            max_err = d
    return max_err


def _safe_to_remove(
    dense_points: List[Point],
    control_points: List[ControlPoint],
    idx: int,
    hard_mask: np.ndarray,
    map_size: int,
    beta_theta: float,
    beta10: float,
    beta13: float,
) -> bool:
    """Check if control point at idx can be safely removed.

    Attempts to replace AB-BC with AC directly.

    Args:
        dense_points: All dense path points
        control_points: Current control points
        idx: Index of point to remove
        hard_mask: Hard obstacle mask
        map_size: Map size
        beta_theta: Max turning angle
        beta10: Min turning interval
        beta13: Max geometric error

    Returns:
        True if safe to remove
    """
    if idx <= 0 or idx >= len(control_points) - 1:
        return False  # Don't remove endpoints

    a = (control_points[idx - 1].x, control_points[idx - 1].y)
    b = (control_points[idx].x, control_points[idx].y)
    c = (control_points[idx + 1].x, control_points[idx + 1].y)

    # Check non-zero
    if (abs(a[0] - c[0]) < 1e-9 and abs(a[1] - c[1]) < 1e-9):
        return False

    # Check collision of new segment AC
    ax, ay = int(round(a[0])), int(round(a[1]))
    cx, cy = int(round(c[0])), int(round(c[1]))
    if check_segment_collision((ax, ay), (cx, cy), hard_mask, map_size):
        return False

    # Get dense points between A and C
    start_idx = control_points[idx - 1].dense_source_index
    end_idx = control_points[idx + 1].dense_source_index
    segment_points = dense_points[start_idx:end_idx + 1]

    # Check geometric error
    err = _compute_max_error(segment_points, a, c)
    if err > beta_13:
        return False

    # Check turning angles at surrounding points
    # After removing B, need to check angle at A-1->A->C and A->C->C+1
    if idx - 2 >= 0:
        prev = (control_points[idx - 2].x, control_points[idx - 2].y)
        v1 = (a[0] - prev[0], a[1] - prev[1])
        v2 = (c[0] - a[0], c[1] - a[1])
        t1 = np.arctan2(v1[1], v1[0])
        t2 = np.arctan2(v2[1], v2[0])
        if angle_diff(t1, t2) > beta_theta + 1e-6:
            return False

    if idx + 2 < len(control_points):
        next_pt = (control_points[idx + 2].x, control_points[idx + 2].y)
        v1 = (c[0] - a[0], c[1] - a[1])
        v2 = (next_pt[0] - c[0], next_pt[1] - c[1])
        t1 = np.arctan2(v1[1], v1[0])
        t2 = np.arctan2(v2[1], v2[0])
        if angle_diff(t1, t2) > beta_theta + 1e-6:
            return False

    # Check turning interval (beta10)
    # After removal, the distance from A to C along the path must be >= beta10
    dist_ac = euclidean_distance_f(a, c)
    if dist_ac < beta10 - 1e-6:
        return False

    return True


def _sparsify_path(
    dense_path: DensePath,
    hard_mask: np.ndarray,
    map_size: int,
    beta_theta: float,
    beta10: float,
    beta11: float,
    beta12: float,
    beta13: float,
    od_id: int,
    candidate_id: int,
) -> Tuple[ControlPath, dict]:
    """Convert a dense path to a sparse control path.

    Args:
        dense_path: Dense path
        hard_mask: Hard obstacle mask
        map_size: Map size
        beta_theta: Max turning angle
        beta10: Min turning interval
        beta11: Initial min spacing preference
        beta12: Max control point spacing
        beta13: Max geometric error
        od_id: OD ID
        candidate_id: Candidate ID

    Returns:
        (ControlPath, report dict)
    """
    dense_points = dense_path.points
    n_dense = len(dense_points)

    # Step 1: Initial control points (start, all primitive endpoints, end)
    control_points: List[ControlPoint] = []
    control_points.append(ControlPoint(
        point_id=0,
        sequence_index=0,
        x=float(dense_points[0][0]),
        y=float(dense_points[0][1]),
        dense_source_index=0,
        source_type="start",
        retention_reason="path_start",
        is_movable=False,
    ))

    # Add primitive endpoints
    for seg_start, seg_end, prim_id in dense_path.motion_segments:
        if seg_end < n_dense:
            idx = seg_end
            control_points.append(ControlPoint(
                point_id=len(control_points),
                sequence_index=len(control_points),
                x=float(dense_points[idx][0]),
                y=float(dense_points[idx][1]),
                dense_source_index=idx,
                source_type="primitive_end",
                retention_reason="primitive_endpoint",
                is_movable=True,
            ))

    # Add end point
    end_idx = n_dense - 1
    control_points.append(ControlPoint(
        point_id=len(control_points),
        sequence_index=len(control_points),
        x=float(dense_points[end_idx][0]),
        y=float(dense_points[end_idx][1]),
        dense_source_index=end_idx,
        source_type="end",
        retention_reason="path_end",
        is_movable=False,
    ))

    # Step 2: Safe deletion
    deleted_count = 0
    # Multiple passes: keep removing until no more points can be removed
    changed = True
    while changed:
        changed = False
        i = 1
        while i < len(control_points) - 1:
            if _safe_to_remove(dense_points, control_points, i, hard_mask, map_size,
                               beta_theta, beta10, beta13):
                control_points.pop(i)
                deleted_count += 1
                changed = True
                # Don't increment i - the next point shifted into this position
            else:
                i += 1

    # Re-number sequence indices
    for i, cp in enumerate(control_points):
        cp.sequence_index = i
        cp.point_id = i

    print(f"[Stage 8]    OD {od_id}: deleted {deleted_count} points, "
          f"{len(control_points)} remaining")

    # Step 3: Insert overlap support points (skipped for now - will be added in full impl)

    # Step 4: Insert max spacing points (beta12)
    spacing_inserts = 0
    i = 0
    while i < len(control_points) - 1:
        a = (control_points[i].x, control_points[i].y)
        b = (control_points[i + 1].x, control_points[i + 1].y)
        dist = euclidean_distance_f(a, b)
        if dist > beta12:
            n_segments = int(math.ceil(dist / beta12))
            for j in range(1, n_segments):
                t = j / n_segments
                x = a[0] + t * (b[0] - a[0])
                y = a[1] + t * (b[1] - a[1])
                # Find nearest dense point
                dense_idx = _find_nearest_dense_index(dense_points, (x, y))
                cp = ControlPoint(
                    point_id=len(control_points),
                    sequence_index=i + j,
                    x=x,
                    y=y,
                    dense_source_index=dense_idx,
                    source_type="spacing_insert",
                    retention_reason=f"max_spacing_{beta12}",
                    is_movable=True,
                )
                control_points.insert(i + j, cp)
                spacing_inserts += 1
            i += n_segments
        else:
            i += 1

    # Re-number after spacing inserts
    for i, cp in enumerate(control_points):
        cp.sequence_index = i
        cp.point_id = i

    # Step 5: Handle short spacing (beta11) - remove collinear close points
    short_exceptions = 0
    i = 1
    while i < len(control_points) - 1:
        a = (control_points[i - 1].x, control_points[i - 1].y)
        b = (control_points[i].x, control_points[i].y)
        dist = euclidean_distance_f(a, b)
        if dist < beta11 and control_points[i].source_type == "spacing_insert":
            # Try to remove collinear point
            prev = (control_points[i - 1].x, control_points[i - 1].y)
            next_pt = (control_points[i + 1].x, control_points[i + 1].y)
            # Check if collinear (angle close to 0 or pi)
            v1 = (b[0] - prev[0], b[1] - prev[1])
            v2 = (next_pt[0] - b[0], next_pt[1] - b[1])
            t1 = np.arctan2(v1[1], v1[0])
            t2 = np.arctan2(v2[1], v2[0])
            if angle_diff(t1, t2) < 1e-6:
                # Collinear, safe to remove
                control_points.pop(i)
                continue
        i += 1

    # Re-number
    for i, cp in enumerate(control_points):
        cp.sequence_index = i
        cp.point_id = i

    segment_ids = list(range(len(control_points) - 1))
    control_path = ControlPath(
        od_id=od_id,
        points=control_points,
        selected_dense_candidate_id=candidate_id,
        segment_ids=segment_ids,
    )

    report = {
        "od_id": od_id,
        "initial_points": n_dense,
        "deleted": deleted_count,
        "spacing_inserts": spacing_inserts,
        "short_exceptions": short_exceptions,
        "final_points": len(control_points),
    }

    return control_path, report


def _find_nearest_dense_index(dense_points: List[Point], target: PointF) -> int:
    """Find the nearest dense point index to a target coordinate."""
    best_idx = 0
    best_dist = float('inf')
    for i, dp in enumerate(dense_points):
        d = (dp[0] - target[0]) ** 2 + (dp[1] - target[1]) ** 2
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def run_stage8(
    config: NormalizedConfig,
    grid_map: GridMap,
    selected_dense_paths: List[DensePath],
) -> Tuple[SparseControlPaths, SparsifyReport, float]:
    """Run Stage 8: Sparsify selected dense paths into control paths.

    Args:
        config: Normalized configuration
        grid_map: Grid map
        selected_dense_paths: Selected dense paths from Stage 7

    Returns:
        Tuple of (SparseControlPaths, SparsifyReport, elapsed_time)
    """
    t0 = time.time()
    print(f"[Stage 8] Sparsifying {len(selected_dense_paths)} dense paths ...")

    hard_mask = grid_map.hard_obstacle_mask
    map_size = grid_map.size

    control_paths = []
    per_path_reports = []

    for od_idx, dense_path in enumerate(selected_dense_paths):
        print(f"[Stage 8]  OD {od_idx}: {len(dense_path.points)} dense points")
        cp, report = _sparsify_path(
            dense_path, hard_mask, map_size,
            config.beta_theta_rad, config.beta10, config.beta11,
            config.beta12, config.beta13,
            od_idx, 0,
        )
        control_paths.append(cp)
        per_path_reports.append(report)

    result = SparseControlPaths(paths=control_paths)
    report = SparsifyReport(per_path=per_path_reports)

    elapsed = time.time() - t0
    print(f"[Stage 8] Completed in {elapsed:.2f}s")
    return result, report, elapsed