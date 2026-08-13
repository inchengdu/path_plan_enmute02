"""
Stage 6: Candidate freeze and overlap precomputation.
"""
from __future__ import annotations
import time
from typing import List, Tuple
import numpy as np

from .types import (
    NormalizedConfig, CandidateSet, FrozenCandidateSets,
    CandidatePrecompute, StageStatus
)
from .geometry import generate_road_mask


def run_stage6(
    config: NormalizedConfig,
    candidate_sets: List[CandidateSet],
) -> Tuple[FrozenCandidateSets, CandidatePrecompute, float]:
    """Run Stage 6: Freeze candidates and precompute overlap.

    Args:
        config: Normalized configuration
        candidate_sets: Candidate sets from Stage 5

    Returns:
        Tuple of (FrozenCandidateSets, CandidatePrecompute, elapsed_time)
    """
    t0 = time.time()
    print(f"[Stage 6] Freezing candidates and precomputing overlaps ...")

    map_size = config.eta1
    road_radius = config.beta3 / 2.0

    # Freeze candidates
    frozen_sets = FrozenCandidateSets(sets=candidate_sets, frozen=True)
    for cs in frozen_sets.sets:
        cs.frozen = True

    n_ods = len(candidate_sets)
    od_candidate_counts = [len(cs.candidates) for cs in candidate_sets]
    total_candidates = sum(od_candidate_counts)

    print(f"[Stage 6]  {n_ods} OD pairs, {total_candidates} total candidates")

    # Generate road masks for each candidate
    road_masks: List[List[np.ndarray]] = []
    physical_lengths: List[List[float]] = []

    for od_idx, cs in enumerate(candidate_sets):
        od_masks = []
        od_lengths = []
        for cand_idx, path in enumerate(cs.candidates):
            # Convert dense points to float polylines
            polyline = [(float(x), float(y)) for x, y in path.points]
            mask = generate_road_mask(polyline, road_radius, map_size)
            od_masks.append(mask)
            od_lengths.append(path.total_physical_length)
        road_masks.append(od_masks)
        physical_lengths.append(od_lengths)
        if (od_idx + 1) % max(1, n_ods // 5) == 0:
            print(f"[Stage 6]  Generated road masks for OD {od_idx + 1}/{n_ods}")

    # Precompute overlap matrix Q_{rk, sl} for r != s
    # Build flat index
    flat_masks = []
    flat_od_indices = []
    for od_idx, od_masks in enumerate(road_masks):
        for cand_idx, mask in enumerate(od_masks):
            flat_masks.append(mask)
            flat_od_indices.append(od_idx)

    # Compute overlap matrix (only for different ODs)
    n_flat = len(flat_masks)
    overlap_Q = np.zeros((n_flat, n_flat), dtype=np.float64)

    print(f"[Stage 6]  Computing overlap matrix ({n_flat}x{n_flat}) ...")
    report_interval = max(1, n_flat // 10)
    for i in range(n_flat):
        for j in range(i + 1, n_flat):
            if flat_od_indices[i] == flat_od_indices[j]:
                continue  # Same OD, not needed
            overlap = float(np.sum(flat_masks[i] & flat_masks[j]))
            overlap_Q[i, j] = overlap
            overlap_Q[j, i] = overlap
        if (i + 1) % report_interval == 0:
            print(f"[Stage 6]   Overlap progress: {i + 1}/{n_flat}")

    # Build flat index mapping
    flat_candidate_map = []  # (od_idx, cand_idx) -> flat_idx
    for od_idx in range(n_ods):
        for cand_idx in range(od_candidate_counts[od_idx]):
            flat_candidate_map.append((od_idx, cand_idx))

    precompute = CandidatePrecompute(
        road_masks=road_masks,
        overlap_Q=overlap_Q,
        physical_lengths=physical_lengths,
    )

    elapsed = time.time() - t0
    print(f"[Stage 6] Completed in {elapsed:.2f}s")
    return frozen_sets, precompute, elapsed