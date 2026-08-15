"""
Stage 7: Multi-OD combination optimization using Reactive Tabu Search.

Optimization objective: the *overlap count* of the dense network. Cells with
more than one selected path covering them (occupancy >= 2) are grouped into
4-connected regions; each region contributes the number of distinct paths
covering it (a region shared by 2 paths counts 2, by 3 paths counts 3). The
search minimizes alpha3 * (overlap count) + alpha1 * (total length).
"""
from __future__ import annotations
import time
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import label as connected_components

from .types import (
    NormalizedConfig, RandomStreams, FrozenCandidateSets,
    CandidatePrecompute, CombinationResult, DensePath,
)


def _overlap_count_from_occ(
    occ: np.ndarray,
    selected_nonzero: List[np.ndarray],
    map_size: int,
) -> float:
    """Overlap count of a selection given its per-cell occupancy.

    Cells with occ >= 2 form 4-connected overlap regions. The overlap count is
    the sum over regions of the number of distinct paths covering that region.
    Computed as the sum over paths of the number of overlap regions the path
    touches (one `connected_components` label per region).
    """
    labels = connected_components((occ >= 2).reshape(map_size, map_size))[0]
    labels = labels.ravel()
    total = 0
    for idx in selected_nonzero:
        uniq = np.unique(labels[idx])
        if uniq.size == 0:
            continue
        if uniq[0] == 0:
            uniq = uniq[1:]  # background: cells of this path not in any overlap region
        total += uniq.size
    return float(total)


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
        precompute: Precomputed road masks, cell indices and lengths
        random_streams: Random streams

    Returns:
        Tuple of (CombinationResult, selected dense paths, elapsed_time)
    """
    t0 = time.time()
    print(f"[Stage 7] Running multi-OD combination optimization "
          f"(objective: overlap count) ...")

    candidate_sets = frozen_sets.sets
    n_ods = len(candidate_sets)
    map_size = config.eta1
    alpha1 = config.alpha1
    alpha3 = config.alpha3

    road_masks = precompute.road_masks
    nonzero_indices = precompute.nonzero_indices
    physical_lengths = precompute.physical_lengths

    # Initial solution: all baseline (candidate index 0)
    current = [0] * n_ods
    occ = np.zeros(map_size * map_size, dtype=np.int32)
    for od in range(n_ods):
        occ += road_masks[od][current[od]]
    selected_nonzero = [nonzero_indices[od][current[od]] for od in range(n_ods)]
    current_overlap = _overlap_count_from_occ(occ, selected_nonzero, map_size)
    current_length = sum(physical_lengths[od][current[od]] for od in range(n_ods))
    current_obj = alpha3 * current_overlap + alpha1 * current_length

    best = current.copy()
    best_overlap = current_overlap
    best_obj = current_obj

    print(f"[Stage 7]  Initial objective (all baseline): {best_obj:.2f} "
          f"(overlap count = {best_overlap:.0f})")

    # Reactive Tabu parameters
    tenure = config.tabu_tenure_initial
    tenure_min = config.tabu_tenure_min
    tenure_max = config.tabu_tenure_max
    tabu_tenure: dict = {}  # (od_idx, new_cand) -> remaining tenure
    stall = 0
    max_iter = config.combination_max_iterations
    max_stall = config.combination_stall
    cycle_window = 50
    history: List[Tuple[int, ...]] = []
    objective_trace = [best_obj]
    rng = random_streams.get_combination_rng()

    for iteration in range(max_iter):
        # Enumerate all single-OD replacement moves
        best_move = None
        best_move_obj = float('inf')
        best_move_tie = None

        for r in range(n_ods):
            old_cand = current[r]
            idx_old = nonzero_indices[r][old_cand]
            length_old = physical_lengths[r][old_cand]
            for k in range(len(candidate_sets[r].candidates)):
                if k == old_cand:
                    continue

                # Check Tabu
                tabu_key = (r, k)
                is_tabu = tabu_key in tabu_tenure and tabu_tenure[tabu_key] > 0

                # Tentative occupancy: replace old candidate by k (cells shared
                # by both end up unchanged).
                idx_new = nonzero_indices[r][k]
                occ_tent = occ.copy()
                occ_tent[idx_old] -= 1
                occ_tent[idx_new] += 1
                sel_nonzero = list(selected_nonzero)
                sel_nonzero[r] = idx_new
                new_overlap = _overlap_count_from_occ(occ_tent, sel_nonzero, map_size)
                new_length = current_length + (physical_lengths[r][k] - length_old)
                new_obj = alpha3 * new_overlap + alpha1 * new_length

                # Aspiration: allow Tabu move if it improves global best
                if is_tabu and new_obj < best_obj - 1e-9:
                    is_tabu = False  # Aspiration

                if is_tabu:
                    continue

                # Tie-break: overlap count, then length, then candidate vector
                tentative = current.copy()
                tentative[r] = k
                tie = (new_overlap, new_length, 0.0, tuple(tentative))

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
        key_old = current[r]
        idx_old = nonzero_indices[r][key_old]
        idx_new = nonzero_indices[r][k]
        occ[idx_old] -= 1
        occ[idx_new] += 1
        selected_nonzero[r] = idx_new
        current[r] = k
        current_overlap = best_move_tie[0]
        current_length = best_move_tie[1]
        current_obj = best_move_obj

        # Update Tabu tenure (tabu the reverse move)
        tabu_tenure[(r, key_old)] = tenure + 1

        # Decay tabu
        to_delete = []
        for key in tabu_tenure:
            tabu_tenure[key] -= 1
            if tabu_tenure[key] <= 0:
                to_delete.append(key)
        for key in to_delete:
            del tabu_tenure[key]

        # Update best
        if current_obj < best_obj - 1e-9:
            best = current.copy()
            best_obj = current_obj
            best_overlap = current_overlap
            stall = 0
        else:
            stall += 1

        # Cycle detection
        state_tuple = tuple(current)
        history.append(state_tuple)
        if len(history) > cycle_window:
            history.pop(0)
            if len(set(history[-cycle_window:])) < cycle_window:
                tenure = min(tenure + 2, tenure_max)
            else:
                tenure = max(tenure - 1, tenure_min)

        objective_trace.append(current_obj)

        if (iteration + 1) % 500 == 0:
            print(f"[Stage 7]  Iteration {iteration + 1}: best={best_obj:.2f} "
                  f"(overlap count = {best_overlap:.0f}), "
                  f"current={current_obj:.2f}, tenure={tenure}")

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

    print(f"[Stage 7]  Best objective: {best_obj:.2f} "
          f"(overlap count = {best_overlap:.0f}, initial: {objective_trace[0]:.2f})")

    # Get selected dense paths
    selected_paths = []
    for od_idx, cand_idx in enumerate(best):
        selected_paths.append(candidate_sets[od_idx].candidates[cand_idx])

    result = CombinationResult(
        selected_candidate_indices=best,
        initial_objective=objective_trace[0],
        best_objective=best_obj,
        iterations=iteration + 1,
        stop_reason=stop_reason,
        objective_trace=objective_trace,
    )

    elapsed = time.time() - t0
    print(f"[Stage 7] Completed in {elapsed:.2f}s: {iteration + 1} iterations")
    return result, selected_paths, elapsed