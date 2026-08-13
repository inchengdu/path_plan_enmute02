"""
Stage 2: Map generation and cost fields.
"""
from __future__ import annotations
import time
import math
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from .types import (
    NormalizedConfig, RandomStreams, GridMap, ObstacleRecord,
    MapStageReport, StageStatus, Point
)


def _low_freq_radial_shape(
    rng: np.random.Generator,
    harmonic_min: int,
    harmonic_max: int,
    coeff_abs_max: float,
    angular_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a low-frequency radial shape.

    Returns:
        (angles, radii) arrays defining the shape boundary.
    """
    angles = np.linspace(0, 2 * np.pi, angular_samples, endpoint=False)
    n_harmonics = harmonic_max - harmonic_min + 1
    coeffs = rng.uniform(-coeff_abs_max, coeff_abs_max, n_harmonics)
    phases = rng.uniform(0, 2 * np.pi, n_harmonics)

    radii = np.ones(angular_samples)
    for i, (c, p) in enumerate(zip(coeffs, phases)):
        h = harmonic_min + i
        radii += c * np.sin(h * angles + p)

    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    return angles, radii


def _radial_shape_to_grid(
    angles: np.ndarray,
    radii: np.ndarray,
    scale: float,
    center_x: int,
    center_y: int,
    grid_size: int,
) -> np.ndarray:
    """Rasterize a radial shape onto a grid.

    Args:
        angles: Angle samples
        radii: Radii at each angle
        scale: Scale factor
        center_x, center_y: Grid center position
        grid_size: Grid size

    Returns:
        Boolean mask of the shape
    """
    # Compute boundary points
    xs = center_x + (radii * scale * np.cos(angles)).astype(int)
    ys = center_y + (radii * scale * np.sin(angles)).astype(int)

    # Clip to grid
    xs = np.clip(xs, 0, grid_size - 1)
    ys = np.clip(ys, 0, grid_size - 1)

    # Use polygon fill to rasterize
    from skimage.draw import polygon
    rr, cc = polygon(ys, xs, (grid_size, grid_size))
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    mask[rr, cc] = True
    return mask


def _check_compactness(mask: np.ndarray) -> float:
    """Compute compactness = 4*pi*A/P^2."""
    from skimage.measure import perimeter
    area = np.sum(mask)
    if area == 0:
        return 0.0
    perim = perimeter(mask)
    if perim == 0:
        return 0.0
    return 4 * math.pi * area / (perim * perim)


def _check_aspect_ratio(mask: np.ndarray) -> float:
    """Compute aspect ratio (long edge / short edge) of bounding box."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 1.0
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    h = ymax - ymin + 1
    w = xmax - xmin + 1
    return max(h, w) / max(min(h, w), 1)


def _check_boundary_roughness(mask: np.ndarray) -> float:
    """Compute boundary roughness = actual perimeter / convex hull perimeter."""
    from skimage.measure import perimeter
    from scipy.spatial import ConvexHull
    actual_perim = perimeter(mask)
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return 1.0
    try:
        hull = ConvexHull(np.column_stack([xs, ys]))
        hull_perim = hull.area  # for 2D, area = perimeter
        if hull_perim < 1e-6:
            return 1.0
        return actual_perim / hull_perim
    except Exception:
        return 1.0


def _generate_obstacle_shape(
    rng: np.random.Generator,
    target_area: int,
    grid_size: int,
    config: NormalizedConfig,
) -> Optional[np.ndarray]:
    """Generate a single obstacle shape with target area.

    Args:
        rng: Random number generator
        target_area: Desired area in grid cells
        grid_size: Map size
        config: Normalized config

    Returns:
        Boolean mask, or None if generation failed
    """
    for trial in range(80):
        angles, radii = _low_freq_radial_shape(
            rng, config.radial_harmonic_min, config.radial_harmonic_max,
            config.radial_coefficient_abs_max, config.radial_angular_samples
        )

        # Binary search for scale to match target area
        best_mask = None
        best_area_diff = float('inf')

        lo, hi = 1.0, float(grid_size)
        for _ in range(config.shape_scale_binary_search_iterations):
            scale = (lo + hi) / 2.0
            mask = _radial_shape_to_grid(angles, radii, scale,
                                          grid_size // 2, grid_size // 2, grid_size)
            area = np.sum(mask)
            diff = abs(area - target_area)
            if diff < best_area_diff:
                best_area_diff = diff
                best_mask = mask

            if area > target_area:
                hi = scale
            else:
                lo = scale

        if best_mask is None:
            continue

        # Check relative area tolerance
        best_area = np.sum(best_mask)
        if abs(best_area - target_area) / max(target_area, 1) > config.shape_area_relative_tolerance:
            continue

        # Check 4-connectivity
        from scipy.ndimage import label
        labeled, n_features = label(best_mask)
        if n_features != 1:
            continue

        # Check no holes
        from scipy.ndimage import binary_fill_holes
        filled = binary_fill_holes(best_mask)
        if np.any(filled != best_mask):
            continue

        # Check compactness
        comp = _check_compactness(best_mask)
        if comp < config.compactness_min:
            continue

        # Check aspect ratio
        ar = _check_aspect_ratio(best_mask)
        if ar > config.aspect_ratio_max:
            continue

        # Check boundary roughness
        rough = _check_boundary_roughness(best_mask)
        if rough > config.boundary_roughness_max:
            continue

        return best_mask

    return None


def _place_obstacle(
    mask: np.ndarray,
    existing_mask: np.ndarray,
    grid_size: int,
    min_separation: float,
    margin: int,
    rng: np.random.Generator,
    max_trials: int,
) -> Optional[Tuple[int, int]]:
    """Try to place an obstacle mask on the grid.

    Args:
        mask: Obstacle shape mask
        existing_mask: Current raw obstacle mask
        grid_size: Map size
        min_separation: Minimum separation distance
        margin: Boundary margin
        rng: Random number generator
        max_trials: Maximum placement trials

    Returns:
        (center_x, center_y) or None
    """
    shape_ys, shape_xs = np.where(mask)
    shape_h = shape_ys.max() - shape_ys.min() + 1
    shape_w = shape_xs.max() - shape_xs.min() + 1
    half_h = shape_h // 2
    half_w = shape_w // 2

    for _ in range(max_trials):
        cx = rng.integers(margin + half_w, grid_size - margin - half_w)
        cy = rng.integers(margin + half_h, grid_size - margin - half_h)

        # Compute placement
        y_min = cy - half_h
        y_max = y_min + shape_h
        x_min = cx - half_w
        x_max = x_min + shape_w

        # Check bounds
        if x_min < margin or x_max >= grid_size - margin or y_min < margin or y_max >= grid_size - margin:
            continue

        # Try to place
        placed = np.zeros((grid_size, grid_size), dtype=bool)
        placed[y_min:y_max, x_min:x_max] = mask[shape_ys.min():shape_ys.max()+1,
                                                 shape_xs.min():shape_xs.max()+1]

        # Check overlap with existing
        if np.any(placed & existing_mask):
            continue

        # Check separation distance
        if min_separation > 0 and np.any(existing_mask):
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(~existing_mask)
            overlap_region = placed & (dist < min_separation)
            if np.any(overlap_region):
                continue

        return (cx, cy)

    return None


def _compute_obstacle_cost_field(
    raw_mask: np.ndarray,
    beta1: int,
    map_size: int,
) -> np.ndarray:
    """Compute obstacle cost field with gradient falloff.

    Uses Chebyshev distance from raw obstacles with raised cosine profile.
    Beta1 layers: first layer value = 0.9, then decays to 0 after beta1 layers.

    Args:
        raw_mask: Raw obstacle boolean mask
        beta1: Number of gradient layers
        map_size: Map size

    Returns:
        Cost field array
    """
    # Compute Chebyshev distance transform from raw obstacles
    # Invert mask: free space = True, obstacle = False
    free = ~raw_mask
    # For Chebyshev (chessboard) distance, we can use a custom distance transform
    # or compute Manhattan/Chebyshev via repeated dilations
    dist = np.full((map_size, map_size), beta1 + 1, dtype=np.float32)
    dist[raw_mask] = 0.0

    # Multi-pass Chebyshev distance transform
    for _ in range(beta1 + 2):
        prev = dist.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                rolled = np.roll(dist, shift=(dy, dx), axis=(0, 1))
                dist = np.minimum(dist, rolled + 1)
        if np.allclose(prev, dist):
            break

    # Apply raised cosine profile
    cost_field = np.zeros((map_size, map_size), dtype=np.float32)

    # Obstacle cells = 1.0
    cost_field[raw_mask] = 1.0

    # Gradient layers: first layer = 0.9, then decay
    for d in range(1, beta1 + 1):
        layer_mask = (dist == d) & ~raw_mask
        # Raised cosine: 0.5 * (1 + cos(pi * (d-1) / beta1)) * 0.9
        # Normalized so first layer = 0.9
        if d == 1:
            cost_field[layer_mask] = 0.9
        else:
            t = (d - 1) / beta1
            cost_field[layer_mask] = 0.5 * (1.0 + math.cos(math.pi * t)) * 0.9

    return cost_field


def run_stage2(config: NormalizedConfig, random_streams: RandomStreams,
               external_map: Optional[np.ndarray] = None) -> Tuple[GridMap, MapStageReport, float]:
    """Run Stage 2: Generate map, hard obstacles, and cost fields.

    Args:
        config: Normalized configuration
        random_streams: Random streams
        external_map: Optional external raw obstacle mask

    Returns:
        Tuple of (GridMap, MapStageReport, elapsed_time)
    """
    t0 = time.time()
    print(f"[Stage 2] Generating {config.eta1}x{config.eta1} map ...")

    rng = random_streams.get_map_rng()
    map_size = config.eta1
    total_area = int(config.eta2 * map_size * map_size)
    mid_area = (config.eta3 + config.eta4) / 2.0
    n_obstacles = max(1, int(round(total_area / mid_area)))

    print(f"[Stage 2] Target total obstacle area: {total_area} ({config.eta2*100:.1f}%)")
    print(f"[Stage 2] Target obstacle count: ~{n_obstacles}")

    raw_mask = np.zeros((map_size, map_size), dtype=bool)
    obstacle_records = []
    attempts = 0

    for restart in range(config.max_map_restarts):
        if restart > 0:
            print(f"[Stage 2] Restarting map generation (attempt {restart + 1})...")
            raw_mask.fill(False)
            obstacle_records.clear()

        # Allocate areas to obstacles
        remaining_area = total_area
        obstacle_areas = []
        for i in range(n_obstacles):
            if i == n_obstacles - 1:
                area = remaining_area
            else:
                feasible_max = min(config.eta4, remaining_area - (n_obstacles - i - 1) * config.eta3)
                feasible_min = config.eta3
                if feasible_max < feasible_min:
                    break
                area = rng.integers(feasible_min, feasible_max + 1)
            area = max(config.eta3, min(config.eta4, area))
            area = min(area, remaining_area)
            obstacle_areas.append(area)
            remaining_area -= area

        if remaining_area > 0:
            continue

        success = True
        for oid, target_area in enumerate(obstacle_areas):
            # Generate shape
            shape_mask = _generate_obstacle_shape(rng, target_area, map_size, config)
            if shape_mask is None:
                attempts += 1
                success = False
                break

            # Place obstacle
            pos = _place_obstacle(
                shape_mask, raw_mask, map_size,
                config.obstacle_min_separation, 1, rng,
                config.max_placement_trials_per_obstacle
            )
            if pos is None:
                attempts += 1
                success = False
                break

            cx, cy = pos
            shape_ys, shape_xs = np.where(shape_mask)
            shape_h = shape_ys.max() - shape_ys.min() + 1
            shape_w = shape_xs.max() - shape_xs.min() + 1
            half_h = shape_h // 2
            half_w = shape_w // 2

            y_min = cy - half_h
            y_max = y_min + shape_h
            x_min = cx - half_w
            x_max = x_min + shape_w

            placed = np.zeros((map_size, map_size), dtype=bool)
            placed[y_min:y_max, x_min:x_max] = shape_mask[shape_ys.min():shape_ys.max()+1,
                                                          shape_xs.min():shape_xs.max()+1]
            raw_mask |= placed

            area = int(np.sum(placed))
            bbox = (x_min, y_min, x_max, y_max)
            comp = _check_compactness(placed)
            ar = _check_aspect_ratio(placed)
            obstacle_records.append(ObstacleRecord(
                obstacle_id=oid,
                area=area,
                center=(float(cx), float(cy)),
                bounding_box=bbox,
                compactness=comp,
                aspect_ratio=ar,
            ))

            if (oid + 1) % max(1, n_obstacles // 5) == 0:
                print(f"[Stage 2]  Placed obstacle {oid + 1}/{n_obstacles} (area={area})")

        if success:
            break
    else:
        print(f"[Stage 2] WARNING: Map generation succeeded but may have fewer obstacles than planned")

    print(f"[Stage 2] Raw obstacles: {np.sum(raw_mask)} cells, {len(obstacle_records)} obstacles")

    # Construct hard obstacle mask (robot radius dilation)
    struct = np.ones((int(2 * config.robot_radius) + 1, int(2 * config.robot_radius) + 1))
    hard_mask = binary_dilation(raw_mask, structure=struct)

    # Compute distance field (Euclidean distance to hard obstacles)
    hard_dist = distance_transform_edt(~hard_mask)

    # Compute obstacle cost field
    cost_field = _compute_obstacle_cost_field(raw_mask, config.beta1, map_size)

    # Build obstacle_id_map
    from scipy.ndimage import label
    id_map = np.full((map_size, map_size), -1, dtype=np.int32)
    labeled, n_features = label(raw_mask)
    for fid in range(1, n_features + 1):
        id_map[labeled == fid] = fid - 1

    grid_map = GridMap(
        size=map_size,
        raw_obstacle_mask=raw_mask,
        obstacle_id_map=id_map,
        hard_obstacle_mask=hard_mask,
        hard_distance_field=hard_dist,
        obstacle_cost_field=cost_field,
        obstacle_records=obstacle_records,
    )

    report = MapStageReport(
        total_obstacle_area=int(np.sum(raw_mask)),
        obstacle_count=len(obstacle_records),
        retries=attempts,
        stop_reason=StageStatus.SUCCESS,
    )

    elapsed = time.time() - t0
    print(f"[Stage 2] Completed in {elapsed:.2f}s")
    return grid_map, report, elapsed