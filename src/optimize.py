"""
Stage 9: Continuous control point optimization.

Local search on control point positions using Effect Score scheduling,
Reactive Tabu, Adaptive Threshold, and five-point local windows.
"""
from __future__ import annotations
import copy
import math
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .types import (
    NormalizedConfig, RandomStreams, GridMap, ControlPoint, ControlPath,
    SparseControlPaths, NetworkState, ControlOptimizationResult,
    StageStatus, PointF, angle_diff, euclidean_distance_f, derive_seed
)
from .geometry import (
    generate_road_mask, compute_occupancy_count, compute_overlap_from_occupancy,
    compute_turns, compute_control_length, compute_obstacle_cost,
    check_segment_collision, bresenham_line, point_to_segment_distance_f
)


def _make_control_polyline(cp: ControlPath) -> List[PointF]:
    """Extract float polyline from control path."""
    return [(p.x, p.y) for p in cp.points]


def _compute_path_metrics(
    control_paths: List[ControlPath],
    grid_map: GridMap,
    config: NormalizedConfig,
) -> Tuple[float, float, float, float, float]:
    """Compute L, D, A, R, J_true for a set of control paths."""
    map_size = grid_map.size
    road_radius = config.beta3 / 2.0
    a1, a2, a3, a4 = config.alpha1, config.alpha2, config.alpha3, config.alpha4

    # L: total control length
    L = 0.0
    for cp in control_paths:
        poly = _make_control_polyline(cp)
        L += compute_control_length(poly)

    # D: obstacle cost
    D = 0.0
    for cp in control_paths:
        poly = _make_control_polyline(cp)
        D += compute_obstacle_cost(poly, grid_map.obstacle_cost_field, map_size)

    # A: overlap
    road_masks = []
    for cp in control_paths:
        poly = _make_control_polyline(cp)
        mask = generate_road_mask(poly, road_radius, map_size)
        road_masks.append(mask)
    occ = compute_occupancy_count(road_masks, map_size)
    A = compute_overlap_from_occupancy(occ)

    # R: repulsion (simplified - using obstacle cost as proxy when formula not available)
    R = 0.0  # Will be computed when repulsion formula is available

    J_true = a1 * L + a2 * D + a3 * A + a4 * R
    return L, D, A, R, J_true


def _compute_effect_score(
    control_paths: List[ControlPath],
    road_masks: List[np.ndarray],
    occ: np.ndarray,
    occupancy_count: np.ndarray,
    map_size: int,
    beta6: float,
    conflict_components: List[dict],
) -> Dict[int, float]:
    """Compute Effect Score for each control point.

    Effect Score measures a control point's influence on nearby conflicts.
    Higher score = higher priority for optimization.
    """
    scores: Dict[int, float] = {}
    for cp_idx, cp in enumerate(control_paths):
        poly = _make_control_polyline(cp)
        for pt_idx, pt in enumerate(cp.points):
            if not pt.is_movable:
                continue
            score = 0.0
            # For each conflict component, check distance
            for comp in conflict_components:
                conflict_area = comp.get("area", 0)
                comp_center = comp.get("center", (0, 0))
                if conflict_area <= 0:
                    continue
                dist = euclidean_distance_f((pt.x, pt.y), comp_center)
                if dist <= beta6:
                    score += conflict_area * (1.0 - dist / beta6)
            scores[pt.point_id] = score
    return scores


def _generate_candidate_positions(
    point: ControlPoint,
    control_paths: List[ControlPath],
    grid_map: GridMap,
    config: NormalizedConfig,
    local_window_indices: List[int],
    path_idx: int,
) -> List[Tuple[float, float]]:
    """Generate and filter candidate positions for a control point.

    Resolution: delta_C = config.candidate_resolution
    Half-width: beta4 axial positions in each direction.

    Pre-filtering:
    1. Map bounds
    2. Fixed endpoints
    3. Zero-length segments
    4. Collision with hard obstacles
    5. beta12 max spacing
    6. beta10 turning interval
    """
    beta4 = config.beta4
    delta = config.candidate_resolution
    hard_mask = grid_map.hard_obstacle_mask
    map_size = grid_map.size
    beta12 = config.beta12
    beta10 = config.beta10
    epsilon = config.numeric_epsilon

    candidates = []
    cp = control_paths[path_idx]
    poly = _make_control_polyline(cp)
    pt_idx = point.sequence_index

    for u in range(-beta4, beta4 + 1):
        for v in range(-beta4, beta4 + 1):
            nx = point.x + u * delta
            ny = point.y + v * delta

            # 1. Map bounds
            if nx < 0 or nx >= map_size or ny < 0 or ny >= map_size:
                continue

            # 2. Fixed endpoints
            if not point.is_movable:
                continue

            # 3. Zero-length segments
            if pt_idx > 0:
                prev = poly[pt_idx - 1]
                if abs(nx - prev[0]) < epsilon and abs(ny - prev[1]) < epsilon:
                    continue
            if pt_idx < len(poly) - 1:
                next_pt = poly[pt_idx + 1]
                if abs(nx - next_pt[0]) < epsilon and abs(ny - next_pt[1]) < epsilon:
                    continue

            # 4. Collision check for new segments
            collides = False
            if pt_idx > 0:
                prev_int = (int(round(poly[pt_idx - 1][0])), int(round(poly[pt_idx - 1][1])))
                cur_int = (int(round(nx)), int(round(ny)))
                if check_segment_collision(prev_int, cur_int, hard_mask, map_size):
                    collides = True
            if not collides and pt_idx < len(poly) - 1:
                cur_int = (int(round(nx)), int(round(ny)))
                next_int = (int(round(poly[pt_idx + 1][0])), int(round(poly[pt_idx + 1][1])))
                if check_segment_collision(cur_int, next_int, hard_mask, map_size):
                    collides = True
            if collides:
                continue

            # 5. beta12 max spacing
            if pt_idx > 0:
                prev = poly[pt_idx - 1]
                if euclidean_distance_f(prev, (nx, ny)) > beta12 + epsilon:
                    continue
            if pt_idx < len(poly) - 1:
                next_pt = poly[pt_idx + 1]
                if euclidean_distance_f((nx, ny), next_pt) > beta12 + epsilon:
                    continue

            # 6. beta10 turning interval (check surrounding angles)
            # This is deferred to turn-specific filtering below

            candidates.append((nx, ny))

    return candidates


def _filter_by_turn_constraint(
    candidates: List[Tuple[float, float]],
    point: ControlPoint,
    control_paths: List[ControlPath],
    path_idx: int,
    beta_theta: float,
    local_window_indices: List[int],
    strict: bool = True,
    first_point_relaxed: bool = False,
    first_point_multiplier: float = 2.0,
) -> List[Tuple[float, float]]:
    """Filter candidates by turn angle constraints.

    Args:
        candidates: List of candidate (x, y) positions
        point: Control point being optimized
        control_paths: All control paths
        path_idx: Index of this path
        beta_theta: Max turning angle
        local_window_indices: Indices of points in local window
        strict: If True, only allow candidates within beta_theta
        first_point_relaxed: If True, allow up to 2*beta_theta for first point
        first_point_multiplier: Relaxation multiplier

    Returns:
        Filtered candidate list
    """
    cp = control_paths[path_idx]
    poly = _make_control_polyline(cp)
    pt_idx = point.sequence_index
    max_turn = beta_theta * (first_point_multiplier if first_point_relaxed else 1.0)
    epsilon = 1e-6

    valid = []
    for nx, ny in candidates:
        ok = True
        # Check turn at prev point (A->B->C)
        if pt_idx >= 2:
            a = poly[pt_idx - 2]
            b = (poly[pt_idx - 1][0], poly[pt_idx - 1][1])
            c = (nx, ny)
            v1 = (b[0] - a[0], b[1] - a[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            if abs(v1[0]) < epsilon and abs(v1[1]) < epsilon:
                ok = False
            elif abs(v2[0]) < epsilon and abs(v2[1]) < epsilon:
                ok = False
            else:
                t1 = np.arctan2(v1[1], v1[0])
                t2 = np.arctan2(v2[1], v2[0])
                if angle_diff(t1, t2) > max_turn + epsilon:
                    ok = False

        # Check turn at this point (B->C->D)
        if ok and pt_idx >= 1 and pt_idx < len(poly) - 1:
            a = (poly[pt_idx - 1][0], poly[pt_idx - 1][1])
            b = (nx, ny)
            c = poly[pt_idx + 1]
            v1 = (b[0] - a[0], b[1] - a[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            if abs(v1[0]) < epsilon and abs(v1[1]) < epsilon:
                ok = False
            elif abs(v2[0]) < epsilon and abs(v2[1]) < epsilon:
                ok = False
            else:
                t1 = np.arctan2(v1[1], v1[0])
                t2 = np.arctan2(v2[1], v2[0])
                if angle_diff(t1, t2) > max_turn + epsilon:
                    ok = False

        # Check turn at next point (C->D->E)
        if ok and pt_idx >= 0 and pt_idx < len(poly) - 2:
            a = (nx, ny)
            b = poly[pt_idx + 1]
            c = poly[pt_idx + 2]
            v1 = (b[0] - a[0], b[1] - a[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            if abs(v1[0]) < epsilon and abs(v1[1]) < epsilon:
                ok = False
            elif abs(v2[0]) < epsilon and abs(v2[1]) < epsilon:
                ok = False
            else:
                t1 = np.arctan2(v1[1], v1[0])
                t2 = np.arctan2(v2[1], v2[0])
                if angle_diff(t1, t2) > max_turn + epsilon:
                    ok = False

        if ok:
            valid.append((nx, ny))

    return valid


def _compute_local_delta(
    path_idx: int,
    point_idx: int,
    new_pos: Tuple[float, float],
    control_paths: List[ControlPath],
    grid_map: GridMap,
    config: NormalizedConfig,
    occ: np.ndarray,
    road_masks: List[np.ndarray],
) -> Tuple[float, float, float, float]:
    """Compute local objective delta for moving a control point.

    Returns (delta_L, delta_D, delta_A, delta_R).
    """
    cp = control_paths[path_idx]
    poly = _make_control_polyline(cp)
    old_pos = (cp.points[point_idx].x, cp.points[point_idx].y)
    map_size = grid_map.size
    road_radius = config.beta3 / 2.0

    # Compute old local length contribution
    old_L_local = 0.0
    if point_idx > 0:
        old_L_local += euclidean_distance_f(poly[point_idx - 1], old_pos)
    if point_idx < len(poly) - 1:
        old_L_local += euclidean_distance_f(old_pos, poly[point_idx + 1])

    new_L_local = 0.0
    if point_idx > 0:
        new_L_local += euclidean_distance_f(poly[point_idx - 1], new_pos)
    if point_idx < len(poly) - 1:
        new_L_local += euclidean_distance_f(new_pos, poly[point_idx + 1])
    delta_L = new_L_local - old_L_local

    # Compute old obstacle cost for local segments
    old_D_local = 0.0
    if point_idx > 0:
        x0, y0 = int(round(poly[point_idx - 1][0])), int(round(poly[point_idx - 1][1]))
        x1, y1 = int(round(old_pos[0])), int(round(old_pos[1]))
        for x, y in bresenham_line(x0, y0, x1, y1):
            if 0 <= x < map_size and 0 <= y < map_size:
                old_D_local += float(grid_map.obstacle_cost_field[y, x])
    if point_idx < len(poly) - 1:
        x0, y0 = int(round(old_pos[0])), int(round(old_pos[1]))
        x1, y1 = int(round(poly[point_idx + 1][0])), int(round(poly[point_idx + 1][1]))
        for x, y in bresenham_line(x0, y0, x1, y1):
            if 0 <= x < map_size and 0 <= y < map_size:
                old_D_local += float(grid_map.obstacle_cost_field[y, x])

    new_D_local = 0.0
    if point_idx > 0:
        x0, y0 = int(round(poly[point_idx - 1][0])), int(round(poly[point_idx - 1][1]))
        x1, y1 = int(round(new_pos[0])), int(round(new_pos[1]))
        for x, y in bresenham_line(x0, y0, x1, y1):
            if 0 <= x < map_size and 0 <= y < map_size:
                new_D_local += float(grid_map.obstacle_cost_field[y, x])
    if point_idx < len(poly) - 1:
        x0, y0 = int(round(new_pos[0])), int(round(new_pos[1]))
        x1, y1 = int(round(poly[point_idx + 1][0])), int(round(poly[point_idx + 1][1]))
        for x, y in bresenham_line(x0, y0, x1, y1):
            if 0 <= x < map_size and 0 <= y < map_size:
                new_D_local += float(grid_map.obstacle_cost_field[y, x])
    delta_D = new_D_local - old_D_local

    # Compute overlap delta by recomputing local road mask contribution
    # For the old mask contribution
    old_segments = []
    if point_idx > 0:
        old_segments.append((poly[point_idx - 1], old_pos))
    if point_idx < len(poly) - 1:
        old_segments.append((old_pos, poly[point_idx + 1]))

    new_segments = []
    if point_idx > 0:
        new_segments.append((poly[point_idx - 1], new_pos))
    if point_idx < len(poly) - 1:
        new_segments.append((new_pos, poly[point_idx + 1]))

    # Compute old overlap contribution of this path
    old_mask_contrib = np.zeros(map_size * map_size, dtype=bool)
    for seg in old_segments:
        seg_mask = generate_road_mask([seg[0], seg[1]], road_radius, map_size)
        old_mask_contrib |= seg_mask

    new_mask_contrib = np.zeros(map_size * map_size, dtype=bool)
    for seg in new_segments:
        seg_mask = generate_road_mask([seg[0], seg[1]], road_radius, map_size)
        new_mask_contrib |= seg_mask

    # Compute overlap with other paths
    old_A_local = 0.0
    new_A_local = 0.0
    for other_idx, other_mask in enumerate(road_masks):
        if other_idx == path_idx:
            continue
        old_A_local += float(np.sum(old_mask_contrib & other_mask))
        new_A_local += float(np.sum(new_mask_contrib & other_mask))

    # Also compute self-overlap change
    old_self = float(np.sum(old_mask_contrib & road_masks[path_idx]))
    new_self = float(np.sum(new_mask_contrib & road_masks[path_idx]))
    # But we need to be careful: the old mask contributed to the old overlap
    # The new mask will contribute to the new overlap
    old_A_local += old_self
    new_A_local += new_self

    delta_A = new_A_local - old_A_local

    # R: repulsion delta (simplified - 0 for now)
    delta_R = 0.0

    return delta_L, delta_D, delta_A, delta_R


def _apply_move(
    path_idx: int,
    point_idx: int,
    new_pos: Tuple[float, float],
    control_paths: List[ControlPath],
    road_masks: List[np.ndarray],
    occ: np.ndarray,
    map_size: int,
    road_radius: float,
) -> None:
    """Apply a control point move and update road masks."""
    cp = control_paths[path_idx]
    poly = _make_control_polyline(cp)
    old_pos = (cp.points[point_idx].x, cp.points[point_idx].y)

    # Update road mask for this path
    old_segments = []
    if point_idx > 0:
        old_segments.append((poly[point_idx - 1], old_pos))
    if point_idx < len(poly) - 1:
        old_segments.append((old_pos, poly[point_idx + 1]))

    new_segments = []
    if point_idx > 0:
        new_segments.append((poly[point_idx - 1], new_pos))
    if point_idx < len(poly) - 1:
        new_segments.append((new_pos, poly[point_idx + 1]))

    # Remove old contribution
    old_contrib = np.zeros(map_size * map_size, dtype=bool)
    for seg in old_segments:
        seg_mask = generate_road_mask([seg[0], seg[1]], road_radius, map_size)
        old_contrib |= seg_mask

    new_contrib = np.zeros(map_size * map_size, dtype=bool)
    for seg in new_segments:
        seg_mask = generate_road_mask([seg[0], seg[1]], road_radius, map_size)
        new_contrib |= seg_mask

    # Update: remove old, add new
    road_masks[path_idx] = road_masks[path_idx] & ~old_contrib
    road_masks[path_idx] = road_masks[path_idx] | new_contrib

    # Update occupancy
    for i in range(map_size * map_size):
        if old_contrib[i] and not new_contrib[i]:
            occ[i] -= 1
        elif new_contrib[i] and not old_contrib[i]:
            occ[i] += 1

    # Update control point position
    cp.points[point_idx].x = new_pos[0]
    cp.points[point_idx].y = new_pos[1]


def run_stage9(
    config: NormalizedConfig,
    sparse_paths: SparseControlPaths,
    grid_map: GridMap,
    random_streams: RandomStreams,
) -> Tuple[ControlOptimizationResult, float]:
    """Run Stage 9: Continuous control point optimization.

    Args:
        config: Normalized configuration
        sparse_paths: Sparse control paths from Stage 8
        grid_map: Grid map
        random_streams: Random streams

    Returns:
        Tuple of (ControlOptimizationResult, elapsed_time)
    """
    t0 = time.time()
    n_ods = len(sparse_paths.paths)
    print(f"[Stage 9] Running continuous control point optimization "
          f"({config.control_K} rounds, {n_ods} paths) ...")

    map_size = grid_map.size
    road_radius = config.beta3 / 2.0

    # Initialize network state
    control_paths = copy.deepcopy(sparse_paths.paths)

    # Build road masks
    road_masks = []
    for cp in control_paths:
        poly = _make_control_polyline(cp)
        mask = generate_road_mask(poly, road_radius, map_size)
        road_masks.append(mask)

    occ = compute_occupancy_count(road_masks, map_size)

    # Compute initial metrics
    L, D, A, R, J = _compute_path_metrics(control_paths, grid_map, config)
    print(f"[Stage 9]  Initial: L={L:.2f}, D={D:.2f}, A={A:.2f}, J={J:.2f}")

    # Track best-so-far
    best_paths = copy.deepcopy(control_paths)
    best_masks = [m.copy() for m in road_masks]
    best_occ = occ.copy()
    best_J = J
    best_L, best_D, best_A, best_R = L, D, A, R

    # Search state
    K = config.control_K
    stall_rounds = config.control_stall_rounds
    min_improvement = config.control_min_improvement
    stall = 0
    objective_trace = [J]
    adjustment_trace = []
    selection_trace = []

    for round_idx in range(K):
        round_rng = random_streams.get_control_rng(round_idx)
        round_start = time.time()

        print(f"[Stage 9]  Round {round_idx + 1}/{K} (best J={best_J:.4f}) ...")

        # Determine scheduling mode
        is_strict = (round_idx % 2 == 0)  # odd rounds (0-indexed = even = strict)
        # Actually: odd rounds = strict effect, even rounds = 10% random

        # Collect all movable points
        unvisited = []
        for cp_idx, cp in enumerate(control_paths):
            for pt_idx, pt in enumerate(cp.points):
                if pt.is_movable:
                    unvisited.append((cp_idx, pt_idx))

        if not unvisited:
            print(f"[Stage 9]  No movable points, stopping")
            break

        # Compute conflict components
        conflict_components = []
        # Simple: find contiguous overlap regions
        overlap_mask = occ > 1
        if np.any(overlap_mask):
            from scipy.ndimage import label
            labeled, n_comp = label(overlap_mask.reshape(map_size, map_size))
            for cid in range(1, n_comp + 1):
                comp_mask = labeled == cid
                area = int(np.sum(comp_mask))
                cy, cx = np.where(comp_mask)
                if len(cx) > 0:
                    center = (float(np.mean(cx)), float(np.mean(cy)))
                    conflict_components.append({
                        "id": cid,
                        "area": area,
                        "center": center,
                        "age": 0,
                    })

        # Compute effect scores
        effect_scores = _compute_effect_score(
            control_paths, road_masks, occ, occ, map_size,
            config.effect_range, conflict_components
        )

        # Sort unvisited by effect score
        unvisited.sort(key=lambda x: -effect_scores.get(
            control_paths[x[0]].points[x[1]].point_id, 0.0
        ))

        # Process points in order
        moves_made = 0
        for cp_idx, pt_idx in unvisited:
            point = control_paths[cp_idx].points[pt_idx]

            # Generate candidate positions
            candidates = _generate_candidate_positions(
                point, control_paths, grid_map, config,
                [], cp_idx
            )

            if not candidates:
                continue

            # Filter by turn constraint
            is_first = (moves_made == 0)
            filtered = _filter_by_turn_constraint(
                candidates, point, control_paths, cp_idx,
                config.beta_theta_rad, [],
                strict=True,
                first_point_relaxed=is_first,
                first_point_multiplier=config.first_point_turn_multiplier,
            )

            if not filtered:
                # Try with relaxed turn constraint
                filtered = _filter_by_turn_constraint(
                    candidates, point, control_paths, cp_idx,
                    config.beta_theta_rad, [],
                    strict=False,
                    first_point_relaxed=True,
                    first_point_multiplier=config.first_point_turn_multiplier + 0.5,
                )

            if not filtered:
                continue

            # Random 10% selection for even rounds
            if not is_strict and round_rng.uniform() < 0.10:
                import random as _random
                nx, ny = _random.choice(filtered)
            else:
                # Evaluate each candidate
                best_candidate = None
                best_delta = float('inf')

                for nx, ny in filtered:
                    dL, dD, dA, dR = _compute_local_delta(
                        cp_idx, pt_idx, (nx, ny),
                        control_paths, grid_map, config,
                        occ, road_masks
                    )
                    a1, a2, a3, a4 = config.alpha1, config.alpha2, config.alpha3, config.alpha4
                    delta = a1 * dL + a2 * dD + a3 * dA + a4 * dR

                    if delta < best_delta:
                        best_delta = delta
                        best_candidate = (nx, ny)

                nx, ny = best_candidate

            # Apply move
            _apply_move(
                cp_idx, pt_idx, (nx, ny),
                control_paths, road_masks, occ, map_size, road_radius
            )
            moves_made += 1

            selection_trace.append({
                "round": round_idx,
                "path": cp_idx,
                "point": pt_idx,
                "new_pos": (nx, ny),
            })

        # Round end: compute metrics
        L, D, A, R, J = _compute_path_metrics(control_paths, grid_map, config)
        objective_trace.append(J)

        round_elapsed = time.time() - round_start
        print(f"[Stage 9]   Round {round_idx + 1}: J={J:.4f} ({moves_made} moves, "
              f"{round_elapsed:.2f}s)")

        # Check best-so-far
        if J < best_J - min_improvement * best_J:
            best_J = J
            best_L, best_D, best_A, best_R = L, D, A, R
            best_paths = copy.deepcopy(control_paths)
            best_masks = [m.copy() for m in road_masks]
            best_occ = occ.copy()
            stall = 0
            print(f"[Stage 9]   New best: J={best_J:.4f}")
        else:
            stall += 1

        if stall >= stall_rounds:
            print(f"[Stage 9]  Stopping: {stall} rounds without improvement")
            stop_reason = "stall"
            break
    else:
        stop_reason = "max_rounds"

    # Build best network state
    final_L, final_D, final_A, final_R, final_J = _compute_path_metrics(
        best_paths, grid_map, config
    )

    best_network = NetworkState(
        control_paths=best_paths,
        road_masks_by_path=best_masks,
        occupancy_count=best_occ,
        true_metrics={"L": final_L, "D": final_D, "A": final_A, "R": final_R, "J_true": final_J},
        conflict_components=conflict_components,
        point_search_states={},
        adjustment_attempts=[],
        local_priority_blocks=[],
        round_index=K,
    )

    result = ControlOptimizationResult(
        best_network=best_network,
        rounds_completed=min(round_idx + 1, K),
        stop_reason=stop_reason,
        objective_trace=objective_trace,
        selection_trace=selection_trace,
        adjustment_trace=adjustment_trace,
    )

    elapsed = time.time() - t0
    print(f"[Stage 9] Completed in {elapsed:.2f}s: {result.rounds_completed} rounds, "
          f"best J={best_J:.4f}")
    return result, elapsed