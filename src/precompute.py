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
    total_candidates = sum(len(cs.candidates) for cs in candidate_sets)

    print(f"[Stage 6]  {n_ods} OD pairs, {total_candidates} total candidates")

    # Generate road masks for each candidate
    road_masks: List[List[np.ndarray]] = []
    nonzero_indices: List[List[np.ndarray]] = []
    physical_lengths: List[List[float]] = []

    for od_idx, cs in enumerate(candidate_sets):
        od_masks = []
        od_nonzero = []
        od_lengths = []
        for cand_idx, path in enumerate(cs.candidates):
            # Convert dense points to float polylines
            polyline = [(float(x), float(y)) for x, y in path.points]
            mask = generate_road_mask(polyline, road_radius, map_size)
            od_masks.append(mask)
            od_nonzero.append(np.flatnonzero(mask))
            od_lengths.append(path.total_physical_length)
        road_masks.append(od_masks)
        nonzero_indices.append(od_nonzero)
        physical_lengths.append(od_lengths)
        if (od_idx + 1) % max(1, n_ods // 5) == 0:
            print(f"[Stage 6]  Generated road masks for OD {od_idx + 1}/{n_ods}")

    precompute = CandidatePrecompute(
        road_masks=road_masks,
        nonzero_indices=nonzero_indices,
        physical_lengths=physical_lengths,
    )

    elapsed = time.time() - t0
    print(f"[Stage 6] Completed in {elapsed:.2f}s")
    return frozen_sets, precompute, elapsed