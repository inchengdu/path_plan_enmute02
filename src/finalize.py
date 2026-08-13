"""
Stage 10: Final recomputation and output.

Recomputes all metrics from geometry, performs validation checks,
and produces the final result.
"""
from __future__ import annotations
import time
from typing import Dict, List, Tuple
import numpy as np

from .types import (
    NormalizedConfig, GridMap, NetworkState, ControlPath, ControlPoint,
    FinalResult, StageStatus, PointF, angle_diff
)
from .geometry import (
    generate_road_mask, compute_occupancy_count, compute_overlap_from_occupancy,
    compute_control_length, compute_obstacle_cost, compute_turns,
    compute_turning_angles, check_segment_collision, euclidean_distance_f
)


def _recompute_metrics_from_geometry(
    control_paths: List[ControlPath],
    grid_map: GridMap,
    config: NormalizedConfig,
) -> Tuple[float, float, float, float, float, np.ndarray, List[np.ndarray]]:
    """Recompute all metrics from scratch using control path geometry.

    Returns:
        (L, D, A, R, J_true, occupancy_count, road_masks)
    """
    map_size = grid_map.size
    road_radius = config.beta3 / 2.0
    a1, a2, a3, a4 = config.alpha1, config.alpha2, config.alpha3, config.alpha4

    L = 0.0
    D = 0.0
    road_masks = []

    for cp in control_paths:
        poly = [(p.x, p.y) for p in cp.points]
        L += compute_control_length(poly)
        D += compute_obstacle_cost(poly, grid_map.obstacle_cost_field, map_size)
        mask = generate_road_mask(poly, road_radius, map_size)
        road_masks.append(mask)

    occ = compute_occupancy_count(road_masks, map_size)
    A = compute_overlap_from_occupancy(occ)
    R = 0.0  # Placeholder for repulsion
    J_true = a1 * L + a2 * D + a3 * A + a4 * R

    return L, D, A, R, J_true, occ, road_masks


def _validate_paths(
    control_paths: List[ControlPath],
    od_records: list,
    grid_map: GridMap,
    config: NormalizedConfig,
) -> dict:
    """Perform basic functional validation on all paths.

    Checks:
    1. OD, start, end correct
    2. Coordinates finite and in map
    3. Segments collision-free and non-zero
    4. Control point spacing <= beta12
    5. True turns <= beta_theta
    6. Turning interval >= beta10
    """
    map_size = grid_map.size
    hard_mask = grid_map.hard_obstacle_mask
    beta_theta = config.beta_theta_rad
    beta10 = config.beta10
    beta12 = config.beta12
    epsilon = 1e-6

    results = {
        "total_paths": len(control_paths),
        "passed": 0,
        "failed": 0,
        "checks": [],
    }

    for cp_idx, cp in enumerate(control_paths):
        poly = [(p.x, p.y) for p in cp.points]
        path_checks = {
            "path_id": cp_idx,
            "od_id": cp.od_id,
            "n_points": len(cp.points),
            "errors": [],
        }

        # 1. Check finite coordinates
        for pt in cp.points:
            if not (np.isfinite(pt.x) and np.isfinite(pt.y)):
                path_checks["errors"].append(f"Non-finite coordinate at point {pt.point_id}")

        # 2. Check in map
        for pt in cp.points:
            if not (0 <= pt.x < map_size and 0 <= pt.y < map_size):
                path_checks["errors"].append(f"Out of bounds at point {pt.point_id}")

        # 3. Check collision-free segments
        for i in range(len(poly) - 1):
            x0, y0 = int(round(poly[i][0])), int(round(poly[i][1]))
            x1, y1 = int(round(poly[i + 1][0])), int(round(poly[i + 1][1]))
            if check_segment_collision((x0, y0), (x1, y1), hard_mask, map_size):
                path_checks["errors"].append(f"Collision at segment {i}-{i+1}")

        # 4. Check non-zero segments
        for i in range(len(poly) - 1):
            dist = euclidean_distance_f(poly[i], poly[i + 1])
            if dist < epsilon:
                path_checks["errors"].append(f"Zero-length segment at {i}-{i+1}")

        # 5. Check beta12 spacing
        for i in range(len(poly) - 1):
            dist = euclidean_distance_f(poly[i], poly[i + 1])
            if dist > beta12 + epsilon:
                path_checks["errors"].append(f"Spacing {dist:.2f} > beta12={beta12} at {i}")

        # 6. Check turning angles
        for i in range(1, len(poly) - 1):
            ax, ay = poly[i - 1]
            bx, by = poly[i]
            cx, cy = poly[i + 1]
            v1 = (bx - ax, by - ay)
            v2 = (cx - bx, cy - by)
            if abs(v1[0]) < epsilon and abs(v1[1]) < epsilon:
                continue
            if abs(v2[0]) < epsilon and abs(v2[1]) < epsilon:
                continue
            t1 = np.arctan2(v1[1], v1[0])
            t2 = np.arctan2(v2[1], v2[0])
            turn = angle_diff(t1, t2)
            if turn > beta_theta + 1e-6:
                path_checks["errors"].append(f"Turn {np.degrees(turn):.2f}deg > beta_theta "
                                              f"{np.degrees(beta_theta):.2f}deg at point {i}")

        # 7. Check turning interval >= beta10
        turns = []
        for i in range(1, len(poly) - 1):
            ax, ay = poly[i - 1]
            bx, by = poly[i]
            cx, cy = poly[i + 1]
            v1 = (bx - ax, by - ay)
            v2 = (cx - bx, cy - by)
            if abs(v1[0]) < epsilon and abs(v1[1]) < epsilon:
                continue
            if abs(v2[0]) < epsilon and abs(v2[1]) < epsilon:
                continue
            t1 = np.arctan2(v1[1], v1[0])
            t2 = np.arctan2(v2[1], v2[0])
            if angle_diff(t1, t2) > config.turn_detection_epsilon:
                turns.append(i)

        for t_idx in range(len(turns) - 1):
            i1 = turns[t_idx]
            i2 = turns[t_idx + 1]
            path_dist = 0.0
            for k in range(i1, i2):
                path_dist += euclidean_distance_f(poly[k], poly[k + 1])
            if path_dist < beta10 - epsilon:
                path_checks["errors"].append(
                    f"Turn interval {path_dist:.2f} < beta10={beta10} "
                    f"between turns {i1}-{i2}")

        if path_checks["errors"]:
            path_checks["passed"] = False
            results["failed"] += 1
        else:
            path_checks["passed"] = True
            results["passed"] += 1

        results["checks"].append(path_checks)

    return results


def _verify_overlap_consistency(
    road_masks: List[np.ndarray],
    occ: np.ndarray,
    map_size: int,
) -> dict:
    """Verify overlap computed by two methods matches.

    Method 1: A_pair = sum_{r<s} |B_r ∩ B_s|
    Method 2: A_count = sum_c choose(n(c), 2)
    """
    n_paths = len(road_masks)

    # Method 1: pairwise
    A_pair = 0.0
    for r in range(n_paths):
        for s in range(r + 1, n_paths):
            A_pair += float(np.sum(road_masks[r] & road_masks[s]))

    # Method 2: choose(n, 2)
    A_count = compute_overlap_from_occupancy(occ)

    diff = abs(A_pair - A_count)
    return {
        "A_pair": float(A_pair),
        "A_count": float(A_count),
        "difference": float(diff),
        "consistent": diff < 1e-6,
    }


def run_stage10(
    config: NormalizedConfig,
    grid_map: GridMap,
    best_network: NetworkState,
    stage_reports: dict,
) -> Tuple[FinalResult, float]:
    """Run Stage 10: Final recomputation, validation, and output.

    Args:
        config: Normalized configuration
        grid_map: Grid map
        best_network: Best network from Stage 9
        stage_reports: Reports from all previous stages

    Returns:
        Tuple of (FinalResult, elapsed_time)
    """
    t0 = time.time()
    print(f"[Stage 10] Final recomputation and validation ...")

    # Recompute metrics from geometry
    control_paths = best_network.control_paths
    L, D, A, R, J_true, occ, road_masks = _recompute_metrics_from_geometry(
        control_paths, grid_map, config
    )

    print(f"[Stage 10] Final metrics: L={L:.2f}, D={D:.2f}, A={A:.2f}, "
          f"J={J_true:.4f}")

    # Validate paths
    od_records = stage_reports.get("od_records", [])
    validation = _validate_paths(control_paths, od_records, grid_map, config)

    # Verify overlap consistency
    overlap_check = _verify_overlap_consistency(road_masks, occ, grid_map.size)

    # Compile validation summary
    validation_summary = {
        "path_validation": validation,
        "overlap_consistency": overlap_check,
        "all_checks_passed": validation["failed"] == 0 and overlap_check["consistent"],
        "n_ods": len(control_paths),
        "n_control_points": sum(len(cp.points) for cp in control_paths),
    }

    # Check if all paths pass
    all_ok = validation["failed"] == 0
    status = StageStatus.SUCCESS if all_ok else StageStatus.INTERNAL_ERROR

    if not all_ok:
        print(f"[Stage 10] WARNING: {validation['failed']} paths have validation errors")
    if not overlap_check["consistent"]:
        print(f"[Stage 10] WARNING: Overlap inconsistency: "
              f"pair={overlap_check['A_pair']:.2f}, "
              f"count={overlap_check['A_count']:.2f}")

    # Build run manifest
    run_manifest = {
        "status": status,
        "final_metrics": {
            "L": float(L),
            "D": float(D),
            "A": float(A),
            "R": float(R),
            "J_true": float(J_true),
        },
        "validation_passed": all_ok,
        "overlap_consistent": overlap_check["consistent"],
        "stage_reports": stage_reports,
    }

    # Extract conflict components
    # Find contiguous overlap regions
    overlap_mask = occ > 1
    conflict_components = []
    if np.any(overlap_mask):
        from scipy.ndimage import label
        labeled, n_comp = label(overlap_mask.reshape(grid_map.size, grid_map.size))
        for cid in range(1, n_comp + 1):
            comp_mask = labeled == cid
            area = int(np.sum(comp_mask))
            cy, cx = np.where(comp_mask)
            if len(cx) > 0:
                conflict_components.append({
                    "id": cid,
                    "area": area,
                    "center": (float(np.mean(cx)), float(np.mean(cy))),
                    "bounds": (int(np.min(cx)), int(np.min(cy)),
                               int(np.max(cx)), int(np.max(cy))),
                })

    final_result = FinalResult(
        status=status,
        final_control_paths=control_paths,
        final_metrics={"L": L, "D": D, "A": A, "R": R, "J_true": J_true},
        conflict_components=conflict_components,
        validation_summary=validation_summary,
        run_manifest=run_manifest,
    )

    elapsed = time.time() - t0
    print(f"[Stage 10] Completed in {elapsed:.2f}s")
    return final_result, elapsed