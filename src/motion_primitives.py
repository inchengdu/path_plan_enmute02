"""
Stage 3: Motion primitive library construction.
"""
from __future__ import annotations
import math
import time
from typing import Dict, List, Tuple
import numpy as np

from .types import MotionPrimitive, PrimitiveLibrary, angle_diff, wrap_angle
from .geometry import bresenham_line, supercover_line


def _compute_nominal_headings(beta_theta: float) -> List[float]:
    """Compute uniformly spaced nominal headings.

    Initial heading count:
        N_H = 4 * ceil(pi / (2 * beta_theta))

    Headings are uniformly distributed around the circle.
    """
    n_headings = 4 * max(1, int(math.ceil(math.pi / (2.0 * beta_theta))))
    return [2.0 * math.pi * i / n_headings for i in range(n_headings)]


def _compute_quantized_endpoint(
    length: float, heading: float
) -> Tuple[int, int]:
    """Compute quantized integer endpoint from a primitive.

    Uses signed outward ceil:
        Q_up(z) = sign(z) * ceil(|z|)
    """
    dx = length * math.cos(heading)
    dy = length * math.sin(heading)
    qx = int(math.copysign(math.ceil(abs(dx)), dx)) if abs(dx) > 1e-12 else 0
    qy = int(math.copysign(math.ceil(abs(dy)), dy)) if abs(dy) > 1e-12 else 0
    return (qx, qy)


def run_stage3(config) -> Tuple[PrimitiveLibrary, float]:
    """Run Stage 3: Construct motion primitive library.

    Args:
        config: NormalizedConfig (needs beta_theta_rad, beta10, beta14, etc.)

    Returns:
        Tuple of (PrimitiveLibrary, elapsed_time)
    """
    t0 = time.time()
    print(f"[Stage 3] Constructing motion primitive library ...")
    print(f"[Stage 3] beta_theta={config.beta_theta_rad:.6f} rad, "
          f"beta10={config.beta10}, beta14={config.beta14}")

    beta_theta = config.beta_theta_rad
    beta10 = config.beta10
    beta14 = config.beta14
    epsilon = config.numeric_epsilon

    # Build nominal headings
    all_primitives: List[MotionPrimitive] = []
    seen_displacements: Dict[Tuple[int, int], bool] = {}

    heading_refinement_max = config.heading_refinement_max if hasattr(config, 'heading_refinement_max') else 4
    for refinement in range(heading_refinement_max):
        headings = _compute_nominal_headings(beta_theta)
        # Multiply headings by 2^refinement to increase resolution if needed
        n_headings = len(headings)
        if refinement > 0:
            n_headings = len(headings) * (2 ** refinement)
            headings = [2.0 * math.pi * i / n_headings for i in range(n_headings)]

        all_primitives.clear()
        seen_displacements.clear()

        # Enumerate lengths from beta10 to beta14
        for length in range(int(beta10), int(beta14) + 1):
            for heading in headings:
                dx, dy = _compute_quantized_endpoint(float(length), heading)
                disp = (dx, dy)

                # Skip duplicates
                if disp in seen_displacements:
                    continue
                if dx == 0 and dy == 0:
                    continue
                seen_displacements[disp] = True

                # Compute actual geometry
                actual_heading = math.atan2(float(dy), float(dx))
                prim_length = math.sqrt(dx * dx + dy * dy)

                # Check length bounds
                if prim_length < beta10 - epsilon or prim_length > beta14 + epsilon:
                    continue

                # Generate dense offsets (Bresenham)
                dense_offsets = bresenham_line(0, 0, dx, dy)

                # Generate supercover offsets (collision cover)
                supercover_offsets = supercover_line(0, 0, dx, dy)

                primitive = MotionPrimitive(
                    primitive_id=len(all_primitives),
                    displacement=disp,
                    actual_heading_angle=actual_heading,
                    primitive_length=prim_length,
                    dense_offsets=dense_offsets,
                    supercover_offsets=supercover_offsets,
                    nominal_heading_angle=heading,
                )
                all_primitives.append(primitive)

        if not all_primitives:
            continue

        # Check heading coverage
        actual_headings = sorted(set(
            round(p.actual_heading_angle, 10) for p in all_primitives
        ))
        max_gap = 0.0
        for i in range(len(actual_headings)):
            gap = angle_diff(actual_headings[i], actual_headings[(i + 1) % len(actual_headings)])
            if gap > max_gap:
                max_gap = gap

        print(f"[Stage 3]  Refinement {refinement}: {len(all_primitives)} primitives, "
              f"max heading gap={max_gap:.6f} rad")

        if max_gap <= beta_theta + epsilon:
            break
    else:
        print(f"[Stage 3]  WARNING: Max heading gap {max_gap:.6f} > beta_theta {beta_theta:.6f}")

    # Sort by heading, then length, then displacement
    all_primitives.sort(key=lambda p: (
        round(p.actual_heading_angle, 10),
        p.primitive_length,
        p.displacement
    ))

    # Reassign IDs
    for i, p in enumerate(all_primitives):
        p.primitive_id = i

    # Build by_heading index
    unique_headings = sorted(set(
        round(p.actual_heading_angle, 10) for p in all_primitives
    ))
    heading_to_idx = {h: i for i, h in enumerate(unique_headings)}
    by_heading: Dict[int, List[MotionPrimitive]] = {i: [] for i in range(len(unique_headings))}
    for p in all_primitives:
        h_key = round(p.actual_heading_angle, 10)
        idx = heading_to_idx[h_key]
        by_heading[idx].append(p)

    library = PrimitiveLibrary(
        primitives=all_primitives,
        heading_angles=unique_headings,
        by_heading=by_heading,
        construction_parameters={
            "beta_theta_rad": beta_theta,
            "beta10": beta10,
            "beta14": beta14,
            "n_primitives": len(all_primitives),
            "n_headings": len(unique_headings),
        },
    )

    elapsed = time.time() - t0
    print(f"[Stage 3] Completed in {elapsed:.2f}s: {len(all_primitives)} primitives, "
          f"{len(unique_headings)} headings")
    return library, elapsed