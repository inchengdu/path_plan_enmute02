"""
Stage 4: OD sampling and baseline A* path finding.
"""
from __future__ import annotations
import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Set
import heapq
import numpy as np

from .types import (
    NormalizedConfig, RandomStreams, GridMap, PrimitiveLibrary,
    MotionPrimitive, DensePath, ODRecord, ODStageReport,
    StageStatus, Point, derive_seed, angle_diff, euclidean_distance
)
from .geometry import check_collision, bresenham_line, compute_path_physical_length


class AStarState:
    """State for A* search: (x, y, heading_idx)."""
    __slots__ = ('x', 'y', 'heading_idx')

    def __init__(self, x: int, y: int, heading_idx: int):
        self.x = x
        self.y = y
        self.heading_idx = heading_idx

    def __hash__(self):
        return hash((self.x, self.y, self.heading_idx))

    def __eq__(self, other):
        return (self.x, self.y, self.heading_idx) == (other.x, other.y, other.heading_idx)


def _generate_od_proposal(
    rng: np.random.Generator,
    map_size: int,
    hard_mask: np.ndarray,
    clearance: float,
    origin_x_max: float,
    dest_x_min: float,
    batch_size: int,
    existing_points: Set[Point],
) -> List[Tuple[Point, Point]]:
    """Generate a batch of OD proposals.

    Args:
        rng: Random number generator
        map_size: Map size
        hard_mask: Hard obstacle mask
        clearance: Minimum distance from obstacles and boundary
        origin_x_max: Maximum x fraction for origin
        dest_x_min: Minimum x fraction for destination
        batch_size: Number of proposals
        existing_points: Already used OD points

    Returns:
        List of ((origin_x, origin_y), (dest_x, dest_y)) proposals
    """
    proposals = []
    max_x = int(map_size * origin_x_max)
    min_x = int(map_size * dest_x_min)

    for _ in range(batch_size * 3):  # extra attempts
        if len(proposals) >= batch_size:
            break

        origin = (rng.integers(max(1, int(clearance)), max(1, max_x)),
                  rng.integers(max(1, int(clearance)), map_size - int(clearance) - 1))
        dest = (rng.integers(min_x, map_size - int(clearance) - 1),
                rng.integers(max(1, int(clearance)), map_size - int(clearance) - 1))

        # Check validity
        if not (0 <= origin[0] < map_size and 0 <= origin[1] < map_size):
            continue
        if not (0 <= dest[0] < map_size and 0 <= dest[1] < map_size):
            continue
        if hard_mask[origin[1], origin[0]] or hard_mask[dest[1], dest[0]]:
            continue
        if origin[0] >= dest[0]:
            continue
        if origin == dest:
            continue
        if origin in existing_points or dest in existing_points:
            continue

        proposals.append((origin, dest))

    return proposals


def _astar_baseline(
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
) -> Optional[Tuple[List[int], List[Point], float]]:
    """Run baseline A* search for shortest path.

    State: (x, y, heading_idx). No random bias, no soft cost.

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

    Returns:
        (primitive_sequence, path_points, total_length) or None
    """
    # Map heading indices to their angles
    n_headings = len(heading_angles)

    # For each heading, find compatible next headings
    heading_neighbors: Dict[int, List[int]] = {}
    for h_idx in range(n_headings):
        neighbors = []
        for nh_idx in range(n_headings):
            diff = angle_diff(heading_angles[h_idx], heading_angles[nh_idx])
            if diff <= beta_theta + 1e-9:
                neighbors.append(nh_idx)
        heading_neighbors[h_idx] = neighbors

    # Heuristic: Euclidean distance to goal
    goal_x, goal_y = goal

    # Open set: (f, g, h, x, y, heading_idx, parent_state, primitive_id)
    # Use tie-breaking: h, g, stable_id
    open_heap = []
    open_set: Dict[tuple, Tuple[float, float, int, Optional[tuple], int]] = {}
    closed_set: Set[tuple] = set()
    state_counter = 0

    # Start state: try all headings from start
    for h_idx in range(n_headings):
        h_cost = euclidean_distance(start, goal)
        state = (start[0], start[1], h_idx)
        key = (h_cost, 0.0, state_counter)
        heapq.heappush(open_heap, (h_cost, 0.0, state_counter, state, None, -1))
        open_set[state] = (0.0, h_cost, state_counter, None, -1)
        state_counter += 1

    expanded = 0
    while open_heap and expanded < max_expanded:
        f, g, _sid, state, parent, prim_id = heapq.heappop(open_heap)
        x, y, h_idx = state

        if state in closed_set:
            continue

        # Check if in open set with better g
        if state in open_set:
            saved_g, _, _, _, _ = open_set[state]
            if g > saved_g + 1e-9:
                continue

        closed_set.add(state)
        expanded += 1

        # Check if we reached the goal
        if (x, y) == (goal_x, goal_y):
            # Reconstruct primitive sequence by walking parent pointers
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
            path_points = []
            # Reconstruct points from primitives
            cx, cy = start
            path_points.append((cx, cy))
            for pid in prim_seq:
                prim = primitives[pid]
                dx, dy = prim.displacement
                cx += dx
                cy += dy
                # Add intermediate points
                for ox, oy in prim.dense_offsets[1:]:
                    path_points.append((cx - (dx - ox) if dx != 0 else cx,
                                        cy - (dy - oy) if dy != 0 else cy))
                path_points.append((cx, cy))
            # Deduplicate consecutive identical points
            unique = [path_points[0]]
            for p in path_points[1:]:
                if p != unique[-1]:
                    unique.append(p)
            return (prim_seq, unique, g)

        # Expand successors
        for nh_idx in heading_neighbors.get(h_idx, []):
            for prim in by_heading.get(nh_idx, []):
                nx = x + prim.displacement[0]
                ny = y + prim.displacement[1]

                # Check bounds
                if not (0 <= nx < map_size and 0 <= ny < map_size):
                    continue

                # Check collision
                prim_world = [(ox + x, oy + y) for ox, oy in prim.supercover_offsets]
                if check_collision(prim_world, hard_mask, map_size):
                    continue

                new_state = (nx, ny, nh_idx)
                if new_state in closed_set:
                    continue

                new_g = g + prim.primitive_length
                h_cost = euclidean_distance((nx, ny), goal)
                new_f = new_g + h_cost

                if new_state in open_set:
                    old_g, _, _, _, _ = open_set[new_state]
                    if new_g >= old_g - 1e-9:
                        continue

                open_set[new_state] = (new_g, h_cost, state_counter, state, prim.primitive_id)
                heapq.heappush(open_heap, (new_f, new_g, state_counter, new_state, state, prim.primitive_id))
                state_counter += 1

    return None  # No path found


def _astar_baseline_reconstruct(
    start: Point,
    goal: Point,
    primitives: List[MotionPrimitive],
    prim_seq: List[int],
) -> DensePath:
    """Reconstruct DensePath from primitive sequence."""
    points = [start]
    cumulative = [0.0]
    total_length = 0.0
    cx, cy = start

    motion_segments = []
    prim_seq_list = []

    for idx, pid in enumerate(prim_seq):
        prim = primitives[pid]
        dx, dy = prim.displacement
        nx, ny = cx + dx, cy + dy
        prim_seq_list.append(pid)

        # Add intermediate dense points (skip first, it's already added)
        for i, (ox, oy) in enumerate(prim.dense_offsets):
            wx = cx + ox
            wy = cy + oy
            if i == 0:
                continue
            points.append((wx, wy))

        # The last point should be (nx, ny)
        if points[-1] != (nx, ny):
            points.append((nx, ny))

        total_length += prim.primitive_length
        cumulative.append(total_length)
        motion_segments.append((len(points) - len(prim.dense_offsets) + 1,
                                len(points) - 1, pid))
        cx, cy = nx, ny

    # Deduplicate
    unique = [points[0]]
    for p in points[1:]:
        if p != unique[-1]:
            unique.append(p)
    points = unique

    # Detect true turns
    true_turns = []
    for i in range(1, len(points) - 1):
        ax, ay = points[i - 1]
        bx, by = points[i]
        cx_, cy_ = points[i + 1]
        v1 = (bx - ax, by - ay)
        v2 = (cx_ - bx, cy_ - by)
        theta1 = np.arctan2(v1[1], v1[0])
        theta2 = np.arctan2(v2[1], v2[0])
        if angle_diff(theta1, theta2) > 1e-6:
            true_turns.append(i)

    # Compute geometry hash
    data = b",".join(f"{x},{y}".encode() for x, y in points)
    geo_hash = hashlib.blake2b(data, digest_size=8).hexdigest()

    return DensePath(
        points=points,
        primitive_sequence=prim_seq_list,
        motion_segments=motion_segments,
        true_turn_indices=true_turns,
        cumulative_lengths=cumulative,
        total_physical_length=total_length,
        geometry_hash=geo_hash,
    )


# === Parallel worker context: read-only shared data, initialized once per process ===
_WORKER_CTX: Dict[str, object] = {}


def _init_stage4_worker(hard_mask, map_size, primitives, by_heading,
                        heading_angles, max_expanded, beta_theta, beta10):
    _WORKER_CTX.update(
        hard_mask=hard_mask, map_size=map_size, primitives=primitives,
        by_heading=by_heading, heading_angles=heading_angles,
        max_expanded=max_expanded, beta_theta=beta_theta, beta10=beta10,
    )


def _baseline_one(origin: Point, dest: Point):
    """Run baseline A* and reconstruct for a single proposal (in a worker process).

    A* is deterministic (no RNG), so running it in parallel is bit-for-bit
    identical to the serial baseline.
    """
    c = _WORKER_CTX
    result = _astar_baseline(
        origin, dest, c["hard_mask"], c["map_size"],
        c["primitives"], c["by_heading"], c["heading_angles"],
        c["max_expanded"], c["beta_theta"], c["beta10"],
    )
    if result is None:
        return None
    prim_seq, _, _ = result
    dense_path = _astar_baseline_reconstruct(origin, dest, c["primitives"], prim_seq)
    return origin, dest, dense_path


def run_stage4(
    config: NormalizedConfig,
    grid_map: GridMap,
    primitive_library: PrimitiveLibrary,
    random_streams: RandomStreams,
) -> Tuple[List[ODRecord], ODStageReport, float]:
    """Run Stage 4: Sample OD pairs and compute baseline shortest paths.

    Args:
        config: Normalized configuration
        grid_map: Grid map
        primitive_library: Primitive library
        random_streams: Random streams

    Returns:
        Tuple of (ODRecord list, ODStageReport, elapsed_time)
    """
    import numpy as np

    t0 = time.time()
    print(f"[Stage 4] Sampling {config.eta6} OD pairs and computing baseline paths ...")

    rng = random_streams.get_od_rng()
    hard_mask = grid_map.hard_obstacle_mask
    map_size = grid_map.size
    primitives = primitive_library.primitives
    by_heading = primitive_library.by_heading
    heading_angles = primitive_library.heading_angles

    od_records: List[ODRecord] = []
    existing_points: Set[Point] = set()
    batch_size = config.od_proposal_batch_size
    total_proposals = 0
    total_no_path = 0
    total_budget_exhausted = 0

    n_workers = min(config.eta10, batch_size, os.cpu_count() or 1)
    initargs = (hard_mask, map_size, primitives, by_heading, heading_angles,
                config.baseline_max_expanded, config.beta_theta_rad, config.beta10)

    def _accept(origin, dest, dense_path):
        """Accept-and-validate one proposal (runs in the parent process).

        Kept serial to preserve proposal_index acceptance order and the shared
        ``existing_points`` dedup state, matching the serial semantics exactly.
        """
        nonlocal total_no_path, total_budget_exhausted
        if dense_path.points[0] != origin or dense_path.points[-1] != dest:
            return
        if dense_path.total_physical_length > 1e6:
            total_budget_exhausted += 1
            return
        od_id = len(od_records)
        od_records.append(ODRecord(
            od_id=od_id,
            start=origin,
            goal=dest,
            baseline_path=dense_path,
            baseline_length=dense_path.total_physical_length,
        ))
        existing_points.add(origin)
        existing_points.add(dest)
        print(f"[Stage 4]    OD {od_id}: origin={origin}, dest={dest}, "
              f"length={dense_path.total_physical_length:.1f}")

    if n_workers <= 1:
        _init_stage4_worker(*initargs)
    else:
        ex = ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_stage4_worker,
                                 initargs=initargs)
        ex.__enter__()

    try:
        for batch in range(config.max_od_batches):
            if len(od_records) >= config.eta6:
                break

            print(f"[Stage 4]  Batch {batch + 1}: {len(od_records)}/{config.eta6} OD pairs accepted")

            # Proposal generation stays serial: it consumes the shared RNG and
            # depends on existing_points dedup.
            proposals = _generate_od_proposal(
                rng, map_size, hard_mask, config.od_clearance,
                config.origin_x_max_fraction, config.destination_x_min_fraction,
                batch_size, existing_points,
            )
            total_proposals += len(proposals)
            if not proposals:
                continue

            if n_workers <= 1:
                batch_results = [_baseline_one(o, d) for o, d in proposals]
            else:
                # Parallel A*; ex.map preserves proposal order == proposal_index
                # acceptance order.
                batch_results = list(ex.map(
                    _baseline_one,
                    [p[0] for p in proposals],
                    [p[1] for p in proposals],
                ))

            for res in batch_results:
                if len(od_records) >= config.eta6:
                    break
                if res is None:
                    total_no_path += 1
                    continue
                _accept(*res)
    finally:
        if n_workers > 1:
            ex.__exit__(None, None, None)

    # Determine stop reason
    if len(od_records) >= config.eta6:
        stop_reason = StageStatus.SUCCESS
    else:
        stop_reason = StageStatus.BUDGET_EXHAUSTED
        print(f"[Stage 4]  WARNING: Only {len(od_records)}/{config.eta6} OD pairs accepted")

    report = ODStageReport(
        proposals=total_proposals,
        accepted=len(od_records),
        no_path=total_no_path,
        budget_exhausted=total_budget_exhausted,
        stop_reason=stop_reason,
    )

    elapsed = time.time() - t0
    print(f"[Stage 4] Completed in {elapsed:.2f}s: {len(od_records)} OD pairs")
    return od_records, report, elapsed