"""
Stage 7: Multi-OD combination optimization using Reactive Tabu Search.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from .types import (
    NormalizedConfig, RandomStreams, CandidateSet, FrozenCandidateSets,
    CandidatePrecompute, CombinationResult, DensePath, StageStatus
)


def _compute_objective(
    selected: List[int],
    n_ods: int,
    overlap_Q: np.ndarray,
    physical_lengths: List[List[float]],
    flat_candidate_map: List[Tuple[int, int]],
    alpha1: float = 1.0,
    alpha3: float = 1.0,
) -> float:
    """Compute combination objective: overlap + optional length term.

    Args:
        selected: Selected candidate index per OD
        n_ods: Number of OD pairs
        overlap_Q: Overlap matrix (flat indices)
        physical_lengths: Physical lengths per OD per candidate
        flat_candidate_map: (od_idx, cand_idx) -> flat_idx
        alpha1: Length weight
        alpha3: Overlap weight

    Returns:
        Objective value
    """
    total_overlap = 0.0
    for r in range(n_ods):
        for s in range(r + 1, n_ods):
            flat_r = _to_flat(flat_candidate_map, r, selected[r])
            flat_s = _to_flat(flat_candidate_map, s, selected[s])
            total_overlap += overlap_Q[flat_r, flat_s]

    total_length = sum(physical_lengths[r][selected[r]] for r in range(n_ods))
    return alpha3 * total_overlap + alpha1 * total_length


def _to_flat(flat_map: List[Tuple[int, int]], od_idx: int, cand_idx: int) -> int:
    """Convert (od_idx, cand_idx) to flat index."""
    for i, (od, cand) in enumerate(flat_map):
        if od == od_idx and cand == cand_idx:
            return i
    raise ValueError(f"Invalid (od={od_idx}, cand={cand_idx})")


def _compute_overlap_delta(
    od_idx: int,
    old_cand: int,
    new_cand: int,
    current_selected: List[int],
    n_ods: int,
    overlap_Q: np.ndarray,
    flat_candidate_map: List[Tuple[int, int]],
) -> float:
    """Compute overlap delta when OD r changes from old_cand to new_cand.

    Delta = sum_{s != r} (Q_{r new, s cur} - Q_{r old, s cur})
    """
    delta = 0.0
    flat_new = _to_flat(flat_candidate_map, od_idx, new_cand)
    flat_old = _to_flat(flat_candidate_map, od_idx, old_cand)
    for s in range(n_ods):
        if s == od_idx:
            continue
        flat_s = _to_flat(flat_candidate_map, s, current_selected[s])
        delta += overlap_Q[flat_new, flat_s] - overlap_Q[flat_old, flat_s]
    return delta


def _compute_objective_tie_break(
    selected: List[int],
    candidate_indices: List[int],
    n_ods: int,
    overlap_Q: np.ndarray,
    physical_lengths: List[List[float]],
    flat_candidate_map: List[Tuple[int, int]],
) -> Tuple[float, float, float, Tuple[int, ...]]:
    """Compute objective with tie-breakers: overlap, length, candidate index vector."""
    total_overlap = 0.0
    for r in range(n_ods):
        for s in range(r + 1, n_ods):
            flat_r = _to_flat(flat_candidate_map, r, selected[r])
            flat_s = _to_flat(flat_candidate_map, s, selected[s])
            total_overlap += overlap_Q[flat_r, flat_s]
    total_length = sum(physical_lengths[r][selected[r]] for r in range(n_ods))
    return (total_overlap, total_length, 0.0, tuple(selected))


def run_stage7(
    config: NormalizedConfig,
    frozen_sets: FrozenCandidateSets,
    precompute: CandidatePrecompute,
    random_streams: RandomStreams,
) -> Tuple[CombinationResult, List[DensePath], float]:
    """Run Stage 7: Multi-OD combination optimization using Reactive Tabu Search.

    Args:
        config: Normalized configuration
        frozen_sets: Frozen candidate sets
        precompute: Precomputed overlap data
        random_streams: Random streams

    Returns:
        Tuple of (CombinationResult, selected dense paths, elapsed_time)
    """
    t0 = time.time()
    print(f"[Stage 7] Running multi-OD combination optimization ...")

    candidate_sets = frozen_sets.sets
    n_ods = len(candidate_sets)
    overlap_Q = precompute.overlap_Q
    physical_lengths = precompute.physical_lengths

    # Build flat candidate map
    flat_candidate_map = []
    for od_idx, cs in enumerate(candidate_sets):
        for cand_idx in range(len(cs.candidates)):
            flat_candidate_map.append((od_idx, cand_idx))

    # Initial solution: all baseline (candidate index 0)
    current = [0] * n_ods
    current_overlap = _compute_objective(
        current, n_ods, overlap_Q, physical_lengths, flat_candidate_map,
        config.alpha1, config.alpha3
    )

    best = current.copy()
    best_overlap = current_overlap
    best_tie = _compute_objective_tie_break(
        best, list(range(n_ods)), n_ods, overlap_Q, physical_lengths, flat_candidate_map
    )

    print(f"[Stage 7]  Initial objective (all baseline): {best_overlap:.2f}")

    # Reactive Tabu parameters
    tenure = config.tabu_tenure_initial
    tenure_min = config.tabu_tenure_min
    tenure_max = config.tabu_tenure_max
    tabu_tenure: Dict[int, int] = {}  # (od_idx, new_cand) -> remaining tenure
    stall = 0
    max_iter = config.combination_max_iterations
    max_stall = config.combination_stall
    cycle_window = 50
    history: List[Tuple[int, ...]] = []
    objective_trace = [best_overlap]
    rng = random_streams.get_combination_rng()

    for iteration in range(max_iter):
        # Enumerate all single-OD replacement moves
        best_move = None
        best_move_obj = float('inf')
        best_move_tie = None

        for r in range(n_ods):
            for k in range(len(candidate_sets[r].candidates)):
                if k == current[r]:
                    continue

                # Check Tabu
                tabu_key = (r, k)
                is_tabu = tabu_key in tabu_tenure and tabu_tenure[tabu_key] > 0

                # Compute delta
                delta = _compute_overlap_delta(
                    r, current[r], k, current, n_ods,
                    overlap_Q, flat_candidate_map
                )

                new_obj = current_overlap + delta * config.alpha3 + \
                    (physical_lengths[r][k] - physical_lengths[r][current[r]]) * config.alpha1

                # Aspiration: allow Tabu move if it improves global best
                if is_tabu and new_obj < best_overlap - 1e-9:
                    is_tabu = False  # Aspiration

                if is_tabu:
                    continue

                # Compute tie-breaker
                new_selected = current.copy()
                new_selected[r] = k
                tie = _compute_objective_tie_break(
                    new_selected, list(range(n_ods)), n_ods,
                    overlap_Q, physical_lengths, flat_candidate_map
                )

                if best_move is None or new_obj < best_move_obj - 1e-9 or \
                   (abs(new_obj - best_move_obj) < 1e-9 and tie < best_move_tie):
                    best_move = (r, k)
                    best_move_obj = new_obj
                    best_move_tie = tie

        if best_move is None:
            print(f"[Stage 7]  No non-tabu moves at iteration {iteration}")
            break

        # Apply move
        r, k = best_move
        old_cand = current[r]
        current[r] = k
        current_overlap = best_move_obj

        # Update Tabu tenure
        tabu_key = (r, old_cand)  # Tabu the reverse move
        tabu_tenure[tabu_key] = tenure + 1

        # Decay tabu
        to_delete = []
        for key in tabu_tenure:
            tabu_tenure[key] -= 1
            if tabu_tenure[key] <= 0:
                to_delete.append(key)
        for key in to_delete:
            del tabu_tenure[key]

        # Update best
        if current_overlap < best_overlap - 1e-9:
            best = current.copy()
            best_overlap = current_overlap
            best_tie = best_move_tie
            stall = 0
        else:
            stall += 1

        # Cycle detection
        state_tuple = tuple(current)
        history.append(state_tuple)
        if len(history) > cycle_window:
            history.pop(0)
            # Check for cycles
            if len(set(history[-cycle_window:])) < cycle_window:
                tenure = min(tenure + 2, tenure_max)
            else:
                tenure = max(tenure - 1, tenure_min)

        objective_trace.append(current_overlap)

        if (iteration + 1) % 500 == 0:
            print(f"[Stage 7]  Iteration {iteration + 1}: best={best_overlap:.2f}, "
                  f"current={current_overlap:.2f}, tenure={tenure}")

        if stall >= max_stall:
            print(f"[Stage 7]  Stopping: {stall} iterations without improvement")
            break

    # Determine stop reason
    if stall >= max_stall:
        stop_reason = "stall"
    elif iteration >= max_iter - 1:
        stop_reason = "max_iterations"
    else:
        stop_reason = "no_neighborhood"

    print(f"[Stage 7]  Best objective: {best_overlap:.2f} (initial: {objective_trace[0]:.2f})")

    # Get selected dense paths
    selected_paths = []
    for od_idx, cand_idx in enumerate(best):
        selected_paths.append(candidate_sets[od_idx].candidates[cand_idx])

    result = CombinationResult(
        selected_candidate_indices=best,
        initial_objective=objective_trace[0],
        best_objective=best_overlap,
        iterations=iteration + 1,
        stop_reason=stop_reason,
        objective_trace=objective_trace,
    )

    elapsed = time.time() - t0
    print(f"[Stage 7] Completed in {elapsed:.2f}s: {iteration + 1} iterations")
    return result, selected_paths, elapsed