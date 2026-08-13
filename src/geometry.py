"""
Geometry and collision utilities.
"""
from __future__ import annotations
from typing import List, Tuple, Set, Optional
import numpy as np
from .types import Point, PointF, DensePath, MotionPrimitive, GridMap, angle_diff, wrap_angle, euclidean_distance_f


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Bresenham line algorithm - deterministic order."""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def supercover_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Supercover line - all cells touched by the closed segment including edges/corners."""
    points = []
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [(x0, y0)]
    points_set: Set[Tuple[int, int]] = set()
    for i in range(steps + 1):
        t = i / steps
        x = x0 + dx * t
        y = y0 + dy * t
        rx = int(round(x))
        ry = int(round(y))
        # Add all cells that the line segment passes through
        # For supercover, we add the cell containing the point
        cx = int(np.floor(x)) if x >= 0 else int(np.ceil(x)) - 1
        cy = int(np.floor(y)) if y >= 0 else int(np.ceil(y)) - 1
        # Check adjacent cells
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                nx = cx + ox
                ny = cy + oy
                if _point_in_segment_cell(nx, ny, x0, y0, x1, y1):
                    points_set.add((nx, ny))
        points_set.add((rx, ry))
    return list(points_set)


def _point_in_segment_cell(cx: int, cy: int, x0: int, y0: int, x1: int, y1: int) -> bool:
    """Check if the cell (cx, cy) is intersected by the segment from (x0,y0) to (x1,y1)."""
    # Check if the segment intersects the axis-aligned square [cx, cx+1] x [cy, cy+1]
    # Use the separating axis theorem approach
    left, right = cx, cx + 1.0
    bottom, top = cy, cy + 1.0

    # Clip the segment to the cell boundaries
    dx = x1 - x0
    dy = y1 - y0

    if dx == 0 and dy == 0:
        return left <= x0 <= right and bottom <= y0 <= top

    t_min = 0.0
    t_max = 1.0

    if dx != 0:
        t1 = (left - x0) / dx if dx != 0 else float('inf')
        t2 = (right - x0) / dx if dx != 0 else float('inf')
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)

    if dy != 0:
        t1 = (bottom - y0) / dy if dy != 0 else float('inf')
        t2 = (top - y0) / dy if dy != 0 else float('inf')
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)

    return t_min <= t_max and t_max >= 0 and t_min <= 1


def check_collision(segment_points: List[Tuple[int, int]], hard_mask: np.ndarray, map_size: int) -> bool:
    """Check if any point in the list collides with hard obstacles."""
    for x, y in segment_points:
        if x < 0 or x >= map_size or y < 0 or y >= map_size:
            return True
        if hard_mask[y, x]:
            return True
    return False


def check_segment_collision(p1: Point, p2: Point, hard_mask: np.ndarray, map_size: int) -> bool:
    """Check if the line segment from p1 to p2 collides with hard obstacles."""
    cover = supercover_line(p1[0], p1[1], p2[0], p2[1])
    return check_collision(cover, hard_mask, map_size)


def generate_road_mask(points: List[PointF], road_radius: float, map_size: int) -> np.ndarray:
    """Generate binary road mask for a polyline with given road radius."""
    mask = np.zeros(map_size * map_size, dtype=bool)
    y_coords, x_coords = np.meshgrid(np.arange(map_size), np.arange(map_size), indexing='ij')
    x_flat = x_coords.ravel()
    y_flat = y_coords.ravel()
    # For each segment, compute distance and mark
    for i in range(len(points) - 1):
        ax, ay = points[i]
        bx, by = points[i + 1]
        abx, aby = bx - ax, by - ay
        len2 = abx * abx + aby * aby
        if len2 < 1e-12:
            continue
        # Project each grid point onto the segment
        px = x_flat - ax
        py = y_flat - ay
        t = (px * abx + py * aby) / len2
        t = np.clip(t, 0.0, 1.0)
        projx = ax + t * abx
        projy = ay + t * aby
        dist2 = (x_flat - projx) ** 2 + (y_flat - projy) ** 2
        mask |= dist2 <= road_radius * road_radius
    return mask


def compute_occupancy_count(road_masks: List[np.ndarray], map_size: int) -> np.ndarray:
    """Compute per-cell occupancy count from multiple road masks."""
    count = np.zeros(map_size * map_size, dtype=np.int32)
    for mask in road_masks:
        count += mask.astype(np.int32)
    return count


def compute_overlap_from_occupancy(occupancy_count: np.ndarray) -> float:
    """Compute total overlap A = sum_c choose(n(c), 2)."""
    n = occupancy_count
    return float(np.sum(n * (n - 1) // 2))


def compute_turns(control_points: List[PointF], epsilon: float = 1e-6) -> List[int]:
    """Identify true turn indices in a control polyline."""
    if len(control_points) < 3:
        return []
    turns = []
    for i in range(1, len(control_points) - 1):
        ax, ay = control_points[i - 1]
        bx, by = control_points[i]
        cx, cy = control_points[i + 1]
        v1 = (bx - ax, by - ay)
        v2 = (cx - bx, cy - by)
        theta1 = np.arctan2(v1[1], v1[0])
        theta2 = np.arctan2(v2[1], v2[0])
        diff = angle_diff(theta1, theta2)
        if diff > epsilon:
            turns.append(i)
    return turns


def compute_turning_angles(control_points: List[PointF]) -> List[float]:
    """Compute turning angles at each interior point."""
    if len(control_points) < 3:
        return []
    angles = []
    for i in range(1, len(control_points) - 1):
        ax, ay = control_points[i - 1]
        bx, by = control_points[i]
        cx, cy = control_points[i + 1]
        v1 = (bx - ax, by - ay)
        v2 = (cx - bx, cy - by)
        theta1 = np.arctan2(v1[1], v1[0])
        theta2 = np.arctan2(v2[1], v2[0])
        angles.append(angle_diff(theta1, theta2))
    return angles


def compute_path_physical_length(points: List[Point]) -> float:
    """Compute total physical length of a dense path."""
    total = 0.0
    for i in range(1, len(points)):
        total += np.sqrt((points[i][0] - points[i - 1][0]) ** 2 +
                         (points[i][1] - points[i - 1][1]) ** 2)
    return total


def compute_control_length(points: List[PointF]) -> float:
    """Compute total length of a control polyline."""
    total = 0.0
    for i in range(1, len(points)):
        total += np.sqrt((points[i][0] - points[i - 1][0]) ** 2 +
                         (points[i][1] - points[i - 1][1]) ** 2)
    return total


def closed_winding_area(poly1: List[Point], poly2: List[Point]) -> float:
    """Compute the area of the region bounded by two closed polylines.
    This is used to measure geometric difference between two candidate paths.
    Uses the absolute winding number area method."""
    # Build a set of points from both paths forming a closed walk
    # polylines are open (start != end), so we close them
    all_points = poly1 + poly2[::-1]  # close the loop
    # Use the shoelace formula on the merged polygon
    # When the two paths overlap, this gives the area of the region between them
    area = 0.0
    n = len(all_points)
    for i in range(n):
        x1, y1 = all_points[i]
        x2, y2 = all_points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def compute_obstacle_cost(path_points: List[PointF], cost_field: np.ndarray, map_size: int) -> float:
    """Compute obstacle cost D by sampling cost_field along the path's grid traversal."""
    total = 0.0
    for i in range(len(path_points) - 1):
        x0, y0 = int(round(path_points[i][0])), int(round(path_points[i][1]))
        x1, y1 = int(round(path_points[i + 1][0])), int(round(path_points[i + 1][1]))
        line = bresenham_line(x0, y0, x1, y1)
        for x, y in line:
            if 0 <= x < map_size and 0 <= y < map_size:
                total += float(cost_field[y, x])
    return total


def compute_repulsion_cost(
    path_points: List[PointF],
    other_paths: List[List[PointF]],
    repulsion_radius: float,
    map_size: int,
    cost_field: np.ndarray,
) -> float:
    """Compute repulsion cost R for a path against other paths.
    Uses raised cosine profile: 0.5*(1+cos(pi*d/(beta2+1))) within radius, 0 outside.
    Excludes cells already within the path's own road mask."""
    total = 0.0
    # Generate ordered grid traversal of this path
    visited_cells = []
    for i in range(len(path_points) - 1):
        x0, y0 = int(round(path_points[i][0])), int(round(path_points[i][1]))
        x1, y1 = int(round(path_points[i + 1][0])), int(round(path_points[i + 1][1]))
        line = bresenham_line(x0, y0, x1, y1)
        visited_cells.extend(line)
    # For each visited cell, accumulate repulsion from other paths
    # Build a KD-tree or just do direct distance computation for simplicity
    # Since map can be large, we compute repulsion for each visited cell
    # by finding the minimum distance to each other path's polyline
    cell_set = list(set(visited_cells))
    for cell_x, cell_y in cell_set:
        if not (0 <= cell_x < map_size and 0 <= cell_y < map_size):
            continue
        total_repulsion = 0.0
        for other in other_paths:
            min_dist = float('inf')
            for j in range(len(other) - 1):
                d = point_to_segment_distance_f(
                    (float(cell_x) + 0.5, float(cell_y) + 0.5),
                    other[j], other[j + 1]
                )
                if d < min_dist:
                    min_dist = d
            if min_dist <= repulsion_radius:
                # Raised cosine: 0.5 * (1 + cos(pi * d / (repulsion_radius + 1)))
                total_repulsion += 0.5 * (1.0 + np.cos(np.pi * min_dist / (repulsion_radius + 1.0)))
        total += total_repulsion
    return total


def point_to_segment_distance_f(p: PointF, a: PointF, b: PointF) -> float:
    """Minimum distance from point p to segment ab."""
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    len2 = abx * abx + aby * aby
    if len2 < 1e-12:
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = (apx * abx + apy * aby) / len2
    t = max(0.0, min(1.0, t))
    projx = ax + t * abx
    projy = ay + t * aby
    return np.sqrt((px - projx) ** 2 + (py - projy) ** 2)