"""
Stage 9: Continuous control point optimization.

Local search on control point positions using Effect Score scheduling,
Reactive Tabu, Adaptive Threshold, and five-point local windows.
"""
from __future__ import annotations
import collections
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


def _is_true_turn(
    i: int,
    poly: List[PointF],
    pt_idx: int,
    new_pos: Optional[PointF],
    eps: float,
) -> bool:
    """True if point i is a true turn, with point pt_idx replaced by new_pos.

    Mirrors the turn detection used by Stage 10 validation (angle > eps).
    Passing pt_idx < 0 leaves the polyline unchanged.
    """
    n = len(poly)
    if i <= 0 or i >= n - 1:
        return False
    ax, ay = poly[i - 1]
    bx, by = poly[i]
    cx, cy = poly[i + 1]
    if new_pos is not None:
        if pt_idx == i - 1:
            ax, ay = new_pos
        elif pt_idx == i:
            bx, by = new_pos
        elif pt_idx == i + 1:
            cx, cy = new_pos
    v1 = (bx - ax, by - ay)
    v2 = (cx - bx, cy - by)
    if abs(v1[0]) < eps and abs(v1[1]) < eps:
        return False
    if abs(v2[0]) < eps and abs(v2[1]) < eps:
        return False
    t1 = math.atan2(v1[1], v1[0])
    t2 = math.atan2(v2[1], v2[0])
    return angle_diff(t1, t2) > eps


def _respects_beta10_interval(
    pt_idx: int,
    new_pos: PointF,
    poly: List[PointF],
    turn_flags: List[bool],
    seglen: List[float],
    pref: List[float],
    eps: float,
    beta10: float,
) -> bool:
    """Check a candidate move keeps consecutive true turns >= beta10 apart.

    Only the local five-point window's turn status can change, so turn flags are
    recomputed there; arcs between consecutive turns are recomputed for every
    pair, adjusting for the two segments that contain the moved point.
    """
    n = len(poly)
    if n < 3:
        return True
    lo = max(1, pt_idx - 2)
    hi = min(n - 2, pt_idx + 2)
    flags = list(turn_flags)
    for i in range(lo, hi + 1):
        flags[i] = _is_true_turn(i, poly, pt_idx, new_pos, eps)

    turns = [i for i in range(n) if flags[i]]
    for a, b in zip(turns, turns[1:]):
        arc = pref[b] - pref[a]
        if pt_idx - 1 >= 0 and a <= pt_idx - 1 < b:
            arc += euclidean_distance_f(poly[pt_idx - 1], new_pos) - seglen[pt_idx - 1]
        if pt_idx + 1 < n and a <= pt_idx < b:
            arc += euclidean_distance_f(new_pos, poly[pt_idx + 1]) - seglen[pt_idx]
        if arc < beta10 - eps:
            return False
    return True


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

            # 6. beta10 turning interval: enforced in _filter_by_turn_constraint
            #    (base-illegal, never relaxed), not here.

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
    beta10: Optional[float] = None,
    turn_epsilon: float = 1e-6,
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
        beta10: Min true-turn interval; enforced as a base-illegal constraint
            (never relaxed, per spec section 8 filter #6)
        turn_epsilon: True-turn detection angle epsilon

    Returns:
        Filtered candidate list
    """
    cp = control_paths[path_idx]
    poly = _make_control_polyline(cp)
    pt_idx = point.sequence_index
    max_turn = beta_theta * (first_point_multiplier if first_point_relaxed else 1.0)
    epsilon = 1e-6

    # Precompute current polyline turn flags and arc-length prefix sums for the
    # beta10 true-turn interval check.
    n = len(poly)
    seglen = [euclidean_distance_f(poly[k], poly[k + 1]) for k in range(n - 1)]
    pref = [0.0] * n
    for k in range(1, n):
        pref[k] = pref[k - 1] + seglen[k - 1]
    turn_flags = [_is_true_turn(i, poly, -1, None, turn_epsilon) for i in range(n)]

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

        # beta10 true-turn interval is base-illegal and never relaxed with
        # the turn-angle relaxation (spec section 8 filter #6).
        if ok and beta10 is not None:
            if not _respects_beta10_interval(
                pt_idx, (nx, ny), poly, turn_flags, seglen, pref,
                turn_epsilon, beta10,
            ):
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

    # Compute overlap with other paths. Overlap A only counts cells shared by
    # DIFFERENT paths (choose(n(c), 2) is zero for n(c)==1), so this path's own
    # road area must NOT be added as "self-overlap".
    old_A_local = 0.0
    new_A_local = 0.0
    for other_idx, other_mask in enumerate(road_masks):
        if other_idx == path_idx:
            continue
        old_A_local += float(np.sum(old_mask_contrib & other_mask))
        new_A_local += float(np.sum(new_mask_contrib & other_mask))

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

    # Update occupancy count vectorized (this is the hot path in Stage 9)
    occ += new_contrib.astype(np.int32) - old_contrib.astype(np.int32)

    # Update control point position
    cp.points[point_idx].x = new_pos[0]
    cp.points[point_idx].y = new_pos[1]


ScoredCandidate = Tuple[Tuple[float, float], float]


def _position_key(
    pos: Tuple[float, float],
    resolution: float,
) -> Tuple[int, int]:
    """Quantize a control point position to the candidate-resolution grid."""
    return (int(round(pos[0] / resolution)), int(round(pos[1] / resolution)))


def _init_point_search_state(point_id: int, config: NormalizedConfig) -> dict:
    """Persistent per-point Reactive Tabu / Adaptive Threshold state."""
    return {
        "point_id": point_id,
        "tabu_positions": {},  # quantized position key -> remaining tenure
        "recent_positions": collections.deque(
            maxlen=config.control_tabu_cycle_detection_window),
        "current_tenure": config.control_tabu_tenure_initial,
        "visits_since_cycle": 0,
        "threshold_ratio": config.adaptive_threshold_initial,
        "stagnant_visits": 0,
        "moves_made": 0,
    }


def _is_position_tabu(
    pos: Tuple[float, float],
    state: dict,
    config: NormalizedConfig,
) -> bool:
    """True if the quantized position key is currently tabooed for this point."""
    key = _position_key(pos, config.candidate_resolution)
    return state["tabu_positions"].get(key, 0) > 0


def _record_move(
    state: dict,
    old_pos: Tuple[float, float],
    new_pos: Tuple[float, float],
    config: NormalizedConfig,
) -> None:
    """Update Reactive Tabu state after an accepted move.

    Taboos the reverse move (old position) and adapts tenure when the point
    cycles back onto a recently visited position (spec section 9).
    """
    new_key = _position_key(new_pos, config.candidate_resolution)

    # Cycle detection over the recent-position window.
    if new_key in state["recent_positions"]:
        state["current_tenure"] = min(
            config.control_tabu_tenure_max,
            state["current_tenure"] + config.control_tabu_tenure_increase_on_cycle,
        )
        state["visits_since_cycle"] = 0
    else:
        state["visits_since_cycle"] += 1
        if (state["visits_since_cycle"]
                >= config.control_tabu_visits_without_cycle_before_decrease
                and state["current_tenure"] > config.control_tabu_tenure_min):
            state["current_tenure"] -= config.control_tabu_tenure_decrease_without_cycle
            state["visits_since_cycle"] = 0

    state["recent_positions"].append(new_key)
    state["moves_made"] += 1

    # Taboo the reverse move with the current tenure.
    old_key = _position_key(old_pos, config.candidate_resolution)
    if old_key != new_key:
        state["tabu_positions"][old_key] = state["current_tenure"]

    # Decay all tabu tenures by one timestep.
    for k in list(state["tabu_positions"]):
        state["tabu_positions"][k] -= 1
        if state["tabu_positions"][k] <= 0:
            del state["tabu_positions"][k]


def _grow_threshold(state: dict, config: NormalizedConfig) -> None:
    """Lower the acceptance bar after a visit that produced no improvement."""
    state["stagnant_visits"] += 1
    state["threshold_ratio"] = min(
        config.adaptive_threshold_max,
        state["threshold_ratio"] + config.adaptive_threshold_growth,
    )


def _accept_improvement(state: dict, config: NormalizedConfig) -> None:
    """An improving move was accepted: decay the adaptive threshold."""
    state["threshold_ratio"] = max(
        0.0, state["threshold_ratio"] * config.adaptive_threshold_decay,
    )
    state["stagnant_visits"] = 0


def _choose_move(
    scored: List[ScoredCandidate],
    state: dict,
    config: NormalizedConfig,
    reference_scale: float,
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Reactive Tabu + Adaptive Threshold candidate selection (spec section 9).

    Within the preserved constraint rank:
      1. improving (delta < 0) Tabu-free candidates win;
      2. lacking those, a Tabu candidate is aspirated only when it strictly
         improves the current objective (proxy for strict legal global-best
         improvement, per `control_tabu_aspiration`);
      3. otherwise only candidates worsening within the per-point adaptive
         threshold are allowed;
      4. nothing within threshold -> the point stays in place and the
         threshold grows for the next visit.

    Returns (chosen position or None when staying, decision reason).
    """
    free: List[ScoredCandidate] = []
    tabu_c: List[ScoredCandidate] = []
    best_free_delta = float("inf")
    for s in scored:
        if _is_position_tabu(s[0], state, config):
            tabu_c.append(s)
        else:
            free.append(s)
            best_free_delta = min(best_free_delta, s[1])

    threshold_abs = state["threshold_ratio"] * reference_scale

    improving = [s for s in free if s[1] < 0]
    if improving:
        chosen = min(improving, key=lambda s: s[1])
        _accept_improvement(state, config)
        return chosen[0], "improved"

    # Aspiration: strictly-improving Tabu candidates that beat every free one.
    aspirants = [
        s for s in tabu_c
        if s[1] < 0 and (not free or s[1] < best_free_delta)
    ]
    if aspirants:
        chosen = min(aspirants, key=lambda s: s[1])
        _accept_improvement(state, config)
        return chosen[0], "tabu_aspiration"

    allowed = [s for s in free if s[1] <= threshold_abs]
    if allowed:
        chosen = min(allowed, key=lambda s: s[1])
        _grow_threshold(state, config)
        return chosen[0], "moved_within_threshold"

    _grow_threshold(state, config)
    return None, "threshold_block"


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
    # Per-point Reactive Tabu / Adaptive Threshold state, persistent across rounds.
    point_states: Dict[int, dict] = {}
    # Adaptive Threshold reference: the objective of the current round-start
    # network (equal to the previous round-end network, or the initial network
    # before the first round).
    round_start_objective = J
    # Initialize to avoid UnboundLocalError when the search breaks before the
    # first full round (e.g. no movable points).
    conflict_components: List[dict] = []
    stop_reason = "max_rounds"

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
            stop_reason = "no_movable_points"
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

        # Adaptive Threshold reference scale: round-start objective per movable
        # point (config `reference_scale = round_start_objective_per_movable_point`).
        reference_scale = round_start_objective / max(1, len(unvisited))

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
                beta10=config.beta10,
                turn_epsilon=config.turn_detection_epsilon,
            )

            relaxed_path = False
            if not filtered:
                # Try with relaxed turn constraint (emergency turn-repair path).
                relaxed_path = True
                filtered = _filter_by_turn_constraint(
                    candidates, point, control_paths, cp_idx,
                    config.beta_theta_rad, [],
                    strict=False,
                    first_point_relaxed=True,
                    first_point_multiplier=config.first_point_turn_multiplier + 0.5,
                    beta10=config.beta10,
                    turn_epsilon=config.turn_detection_epsilon,
                )

            if not filtered:
                selection_trace.append({
                    "round": round_idx,
                    "path": cp_idx,
                    "point": pt_idx,
                    "point_id": point.point_id,
                    "new_pos": None,
                    "moved": False,
                    "reason": "no_candidate",
                })
                continue

            point_state = point_states.setdefault(
                point.point_id, _init_point_search_state(point.point_id, config))

            # Score approved candidates with the local objective increment.
            scored = []
            a1, a2, a3, a4 = config.alpha1, config.alpha2, config.alpha3, config.alpha4
            for nx, ny in filtered:
                dL, dD, dA, dR = _compute_local_delta(
                    cp_idx, pt_idx, (nx, ny),
                    control_paths, grid_map, config,
                    occ, road_masks,
                )
                scored.append(((nx, ny), a1 * dL + a2 * dD + a3 * dA + a4 * dR))

            chosen = None
            reason = "no_candidate"
            if relaxed_path:
                # Emergency turn-repair: keep the historical best-delta behavior;
                # Reactive Tabu / Adaptive Threshold do not apply to repair moves.
                chosen = min(scored, key=lambda s: s[1])[0]
                reason = "relaxed_fallback"
            elif not is_strict and round_rng.uniform() < 0.10:
                # Even rounds: reproducible 10% random draw among Tabu-free
                # candidates (spec section 6.2).
                free = [s for s in scored
                        if not _is_position_tabu(s[0], point_state, config)]
                if not free:
                    free = [s for s in scored if s[1] < 0]
                if free:
                    idx = int(round_rng.integers(0, len(free)))
                    chosen = free[idx][0]
                    reason = "random"
            else:
                chosen, reason = _choose_move(scored, point_state, config, reference_scale)

            # A "move" that lands on the current position is a no-op.
            if chosen is not None and euclidean_distance_f(
                chosen, (point.x, point.y),
            ) <= config.numeric_epsilon:
                chosen = None
                reason = "unchanged"

            if chosen is None:
                selection_trace.append({
                    "round": round_idx,
                    "path": cp_idx,
                    "point": pt_idx,
                    "point_id": point.point_id,
                    "new_pos": None,
                    "moved": False,
                    "reason": reason,
                })
                continue

            old_pos = (point.x, point.y)
            _apply_move(
                cp_idx, pt_idx, chosen,
                control_paths, road_masks, occ, map_size, road_radius,
            )
            if not relaxed_path:
                _record_move(point_state, old_pos, chosen, config)

            moves_made += 1

            selection_trace.append({
                "round": round_idx,
                "path": cp_idx,
                "point": pt_idx,
                "point_id": point.point_id,
                "new_pos": chosen,
                "moved": True,
                "reason": reason,
            })

        # Round end: compute metrics
        L, D, A, R, J = _compute_path_metrics(control_paths, grid_map, config)
        objective_trace.append(J)
        round_start_objective = J

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
        point_search_states=dict(point_states),
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