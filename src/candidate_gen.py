"""
Stage 5: Dense candidate path generation.
"""
from __future__ import annotations
import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Set, Tuple
import heapq
import numpy as np

from .types import (
    NormalizedConfig, RandomStreams, GridMap, PrimitiveLibrary,
    MotionPrimitive, DensePath, ODRecord, CandidateSet, CandidateAttemptLog,
    StageStatus, Point, angle_diff, euclidean_distance, derive_seed
)
from .geometry import (
    check_collision, bresenham_line, closed_winding_area,
    compute_path_physical_length
)
from .od_sampling import _astar_baseline_reconstruct


def _astar_candidate(
    start: Point,
    goal: Point,
    hard_mask: np.ndarray,
    map_size: int,
    primitives: List[MotionPrimitive],
    by_heading: Dict[int, List[MotionPrimitive]],
    heading_angles: List[float],
    max_expanded: int,
    beta_theta: float,
    beta10: float,
    heuristic_weight: float,
    state_bias_rng: np.random.Generator,
    bias_amplitude: float,
) -> Optional[Tuple[List[int], List[Point], float]]:
    """Run randomized candidate A* search.

    Same as baseline but with heuristic weight scaling and per-state random bias.

    Args:
        start: Start point
        goal: Goal point
        hard_mask: Hard obstacle mask
        map_size: Map size
        primitives: All motion primitives
        by_heading: Primitives indexed by heading
        heading_angles: List of heading angles
        max_expanded: Maximum states to expand
        beta_theta: Maximum turning angle
        beta10: Minimum turning interval
        heuristic_weight: Heuristic weight (1.0-1.15)
        state_bias_rng: RNG for state bias
        bias_amplitude: Bias amplitude

    Returns:
        (primitive_sequence, path_points, total_length) or None
    """
    n_headings = len(heading_angles)

    heading_neighbors: Dict[int, List[int]] = {}
    for h_idx in range(n_headings):
        neighbors = []
        for nh_idx in range(n_headings):
            diff = angle_diff(heading_angles[h_idx], heading_angles[nh_idx])
            if diff <= beta_theta + 1e-9:
                neighbors.append(nh_idx)
        heading_neighbors[h_idx] = neighbors

    goal_x, goal_y = goal

    open_heap = []
    open_set: Dict[tuple, Tuple[float, float, int, Optional[tuple], int]] = {}
    closed_set: Set[tuple] = set()
    state_counter = 0

    # Pre-compute state biases (stable per state)
    state_bias_cache: Dict[tuple, float] = {}

    def get_bias(x: int, y: int, h_idx: int) -> float:
        key = (x, y, h_idx)
        if key not in state_bias_cache:
            state_bias_cache[key] = state_bias_rng.uniform(-bias_amplitude, bias_amplitude)
        return state_bias_cache[key]

    # Start state
    for h_idx in range(n_headings):
        h_cost = euclidean_distance(start, goal)
        bias = get_bias(start[0], start[1], h_idx)
        state = (start[0], start[1], h_idx)
        f = h_cost * heuristic_weight + bias
        heapq.heappush(open_heap, (f, h_cost, 0.0, state_counter, state, None, -1))
        open_set[state] = (0.0, h_cost, state_counter, None, -1)
        state_counter += 1

    expanded = 0
    while open_heap and expanded < max_expanded:
        f, h_cost, g, _sid, state, parent, prim_id = heapq.heappop(open_heap)
        x, y, h_idx = state

        if state in closed_set:
            continue

        if state in open_set:
            saved_g, _, _, _, _ = open_set[state]
            if abs(g - saved_g) > 1e-6:
                # This is an older entry
                continue

        closed_set.add(state)
        expanded += 1

        if (x, y) == (goal_x, goal_y):
            prim_seq = []
            cur_state = state
            while cur_state is not None:
                if cur_state in open_set:
                    _, _, _, p_state, p_prim = open_set[cur_state]
                    if p_state is not None:
                        prim_seq.append(p_prim)
                        cur_state = p_state
                    else:
                        break
                else:
                    break
            prim_seq = prim_seq[::-1]
            return (prim_seq, [], g)

        for nh_idx in heading_neighbors.get(h_idx, []):
            for prim in by_heading.get(nh_idx, []):
                nx = x + prim.displacement[0]
                ny = y + prim.displacement[1]

                if not (0 <= nx < map_size and 0 <= ny < map_size):
                    continue

                prim_world = [(ox + x, oy + y) for ox, oy in prim.supercover_offsets]
                if check_collision(prim_world, hard_mask, map_size):
                    continue

                new_state = (nx, ny, nh_idx)
                if new_state in closed_set:
                    continue

                new_g = g + prim.primitive_length
                h_cost = euclidean_distance((nx, ny), goal)
                bias = get_bias(nx, ny, nh_idx)
                new_f = new_g + h_cost * heuristic_weight + bias

                if new_state in open_set:
                    old_g, _, _, _, _ = open_set[new_state]
                    if new_g >= old_g - 1e-9:
                        continue

                open_set[new_state] = (new_g, h_cost, state_counter, state, prim.primitive_id)
                heapq.heappush(open_heap, (new_f, h_cost, new_g, state_counter, new_state, state, prim.primitive_id))
                state_counter += 1

    return None


def _check_candidate_validity(
    candidate: DensePath,
    start: Point,
    goal: Point,
    map_size: int,
    hard_mask: np.ndarray,
    primitives: List[MotionPrimitive],
    beta_theta: float,
    beta10: float,
    max_length: float,
    existing_hashes: Set[str],
    existing_paths: List[DensePath],
    min_area_diff: float,
) -> Tuple[bool, str]:
    """Check candidate validity through a sequence of criteria.

    Args:
        candidate: Candidate path to check
        start: Start point
        goal: Goal point
        map_size: Map size
        hard_mask: Hard obstacle mask
        primitives: Motion primitive library (for actual heading angles)
        beta_theta: Maximum turning angle
        beta10: Minimum turning interval
        max_length: Maximum allowed length
        existing_hashes: Set of existing geometry hashes
        existing_paths: List of existing paths
        min_area_diff: Minimum area difference

    Returns:
        (valid, reject_stage)
    """
    # 1. Check start and end
    if candidate.points[0] != start:
        return False, "start_mismatch"
    if candidate.points[-1] != goal:
        return False, "goal_mismatch"

    # 2. Check bounds and collision
    for x, y in candidate.points:
        if not (0 <= x < map_size and 0 <= y < map_size):
            return False, "out_of_bounds"
        if hard_mask[y, x]:
            return False, "collision"

    # 3. Check turns: actual heading angle between consecutive motion primitives
    #    (the turn constraint applies to primitive transitions, not to the
    #    Bresenham-dense grid cells, whose per-cell direction is quantized to
    #    45-degree steps and would always exceed beta_theta).
    for i in range(1, len(candidate.motion_segments)):
        prev_pid = candidate.motion_segments[i - 1][2]
        cur_pid = candidate.motion_segments[i][2]
        prev_heading = primitives[prev_pid].actual_heading_angle
        cur_heading = primitives[cur_pid].actual_heading_angle
        if angle_diff(prev_heading, cur_heading) > beta_theta + 1e-6:
            return False, "turn_exceeded"

    # 4. Check length
    if candidate.total_physical_length > max_length:
        return False, "length_exceeded"

    # 5. Check duplicate (hash)
    if candidate.geometry_hash in existing_hashes:
        return False, "duplicate_hash"

    # 6. Check area difference
    for existing in existing_paths:
        area_diff = closed_winding_area(candidate.points, existing.points)
        if area_diff < min_area_diff:
            return False, "area_diff_too_small"

    return True, "accepted"


# === Parallel worker context: read-only shared data, initialized once per process ===
_WORKER_CTX: Dict[str, object] = {}


def _init_stage5_worker(config, hard_mask, map_size, primitives, by_heading,
                        heading_angles, random_streams):
    _WORKER_CTX.update(
        config=config, hard_mask=hard_mask, map_size=map_size,
        primitives=primitives, by_heading=by_heading,
        heading_angles=heading_angles, random_streams=random_streams,
    )


def _gen_one_od_candidates(od_idx: int, od_record: ODRecord):
    """Generate candidate paths for a single OD (executed in a worker process).

    Identical logic to the original serial loop; all mutable state is local to
    this function, and randomness is derived from (od_idx, attempt), so results
    are bit-for-bit identical regardless of scheduling order.
    """
    c = _WORKER_CTX
    config = c["config"]
    hard_mask = c["hard_mask"]
    map_size = c["map_size"]
    primitives = c["primitives"]
    by_heading = c["by_heading"]
    heading_angles = c["heading_angles"]
    random_streams = c["random_streams"]

    max_candidates = config.eta7
    min_area_diff = config.eta8
    max_length_ratio = config.max_length_ratio_to_baseline
    consecutive_fail_limit = config.consecutive_failure_limit
    max_attempts = config.max_candidate_attempts_per_od

    candidates = [od_record.baseline_path]
    hashes = {od_record.baseline_path.geometry_hash}
    baseline_length = od_record.baseline_length
    max_length = baseline_length * max_length_ratio
    consecutive_fails = 0
    attempt = 0

    od_attempt_logs: List[CandidateAttemptLog] = []

    while len(candidates) < max_candidates and attempt < max_attempts:
        # Get random stream for this attempt
        attempt_rng = random_streams.get_candidate_rng(od_idx, attempt)
        heuristic_weight = attempt_rng.uniform(config.heuristic_weight_min,
                                                config.heuristic_weight_max)
        bias_amplitude = config.state_bias_amplitude

        # Create state bias RNG with stable seed
        bias_seed = derive_seed(config.config_hash, "candidate_bias", od_idx, attempt)
        bias_rng = np.random.Generator(np.random.PCG64(bias_seed))

        result = _astar_candidate(
            od_record.start, od_record.goal, hard_mask, map_size,
            primitives, by_heading, heading_angles,
            config.candidate_max_expanded,
            config.beta_theta_rad, config.beta10,
            heuristic_weight, bias_rng, bias_amplitude,
        )

        search_status = "no_path" if result is None else "path_found"
        reject_stage = "accepted"
        candidate_length = 0.0
        min_diff = 0.0

        if result is not None:
            prim_seq, _, total_length = result
            candidate = _astar_baseline_reconstruct(
                od_record.start, od_record.goal, primitives, prim_seq
            )
            candidate_length = candidate.total_physical_length

            valid, reject_stage = _check_candidate_validity(
                candidate, od_record.start, od_record.goal,
                map_size, hard_mask, primitives,
                config.beta_theta_rad, config.beta10,
                max_length, hashes, candidates, min_area_diff,
            )

            if valid:
                candidate.geometry_hash = candidate._compute_hash()
                # Re-check hash after proper computation
                if candidate.geometry_hash not in hashes:
                    # Re-check area diff with proper hash
                    ok = True
                    for existing in candidates:
                        area_diff = closed_winding_area(candidate.points, existing.points)
                        if area_diff < min_area_diff:
                            ok = False
                            reject_stage = "area_diff_too_small"
                            break
                    if ok:
                        candidates.append(candidate)
                        hashes.add(candidate.geometry_hash)
                        consecutive_fails = 0
                    else:
                        consecutive_fails += 1
                else:
                    consecutive_fails += 1
                    reject_stage = "duplicate_hash"
            else:
                consecutive_fails += 1
        else:
            consecutive_fails += 1

        log_entry = CandidateAttemptLog(
            od_id=od_idx,
            attempt=attempt,
            heuristic_weight=heuristic_weight,
            seed=bias_seed,
            search_status=search_status,
            reject_stage=reject_stage,
            length=candidate_length,
            min_area_diff=min_diff,
        )
        od_attempt_logs.append(log_entry)
        attempt += 1

        if consecutive_fails >= consecutive_fail_limit:
            break

    stop_reason = "max_candidates" if len(candidates) >= max_candidates else "consecutive_failures"
    cs = CandidateSet(
        od_id=od_idx,
        baseline_path=od_record.baseline_path,
        baseline_length=baseline_length,
        candidates=candidates,
        hashes=hashes,
        stop_reason=stop_reason,
        frozen=False,
    )
    return od_idx, cs, od_attempt_logs


def run_stage5(
    config: NormalizedConfig,
    grid_map: GridMap,
    primitive_library: PrimitiveLibrary,
    od_records: List[ODRecord],
    random_streams: RandomStreams,
) -> Tuple[List[CandidateSet], List[CandidateAttemptLog], float]:
    """Run Stage 5: Generate dense candidate paths for each OD.

    Candidate generation is parallelized across OD pairs. Each OD is fully
    independent (no shared mutable state), and per-attempt randomness is derived
    from (od_idx, attempt), so parallel execution is deterministic and identical
    to the serial baseline.

    Args:
        config: Normalized configuration
        grid_map: Grid map
        primitive_library: Primitive library
        od_records: OD records from Stage 4
        random_streams: Random streams

    Returns:
        Tuple of (CandidateSet list, attempt logs, elapsed_time)
    """
    t0 = time.time()
    n_ods = len(od_records)
    print(f"[Stage 5] Generating dense candidate paths for {n_ods} OD pairs ...")

    hard_mask = grid_map.hard_obstacle_mask
    map_size = grid_map.size
    primitives = primitive_library.primitives
    by_heading = primitive_library.by_heading
    heading_angles = primitive_library.heading_angles

    initargs = (config, hard_mask, map_size, primitives, by_heading,
                heading_angles, random_streams)
    n_workers = min(config.eta10, n_ods, os.cpu_count() or 1)

    if n_workers <= 1:
        # Serial fallback: initialize context in-process and run worker directly
        _init_stage5_worker(*initargs)
        results = [_gen_one_od_candidates(i, od)
                   for i, od in enumerate(od_records)]
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_stage5_worker,
            initargs=initargs,
        ) as ex:
            # ex.map preserves input order == stable_task_index commit semantics
            results = list(ex.map(_gen_one_od_candidates,
                                  range(n_ods), od_records))

    candidate_sets: List[CandidateSet] = [r[1] for r in results]
    all_attempt_logs: List[CandidateAttemptLog] = []
    for _, _, logs in results:
        all_attempt_logs.extend(logs)

    for od_idx, cs, _ in results:
        print(f"[Stage 5]  OD {od_idx}: {len(cs.candidates)} candidates, "
              f"stop_reason={cs.stop_reason}")

    elapsed = time.time() - t0
    print(f"[Stage 5] Completed in {elapsed:.2f}s")
    return candidate_sets, all_attempt_logs, elapsed