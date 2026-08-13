"""
Visualization module for generating PNG figures.
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .types import (
    GridMap, ODRecord, DensePath, ControlPath, NetworkState,
    CombinationResult, NormalizedConfig
)


def save_network_figure(
    grid_map: GridMap,
    od_records: List[ODRecord],
    map_size: int,
    run_id: str,
    title: str,
    save_path: str,
    paths: Optional[List[DensePath]] = None,
    control_paths: Optional[List[ControlPath]] = None,
    show_overlap: bool = False,
    show_control_points: bool = False,
    highlight_turns: bool = False,
    hist_metrics: Optional[dict] = None,
    dpi: int = 150,
    fig_size: Tuple[float, float] = (10.0, 10.0),
) -> bool:
    """Save a network figure to PNG.

    Args:
        grid_map: Grid map
        od_records: OD records
        map_size: Map size
        run_id: Run identifier
        title: Figure title
        save_path: Output path
        paths: Optional dense paths to draw
        control_paths: Optional control paths to draw
        show_overlap: Whether to highlight overlap regions
        show_control_points: Whether to show control points
        highlight_turns: Whether to highlight turn points
        hist_metrics: Optional metrics to display
        dpi: Image DPI
        fig_size: Figure size in inches

    Returns:
        True if saved successfully
    """
    try:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)

        # Background: free space (white)
        bg = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255

        # Raw obstacles (dark gray)
        bg[grid_map.raw_obstacle_mask] = [80, 80, 80]

        # Hard inflation area (light gray) - only where not already raw obstacle
        infl_mask = grid_map.hard_obstacle_mask & ~grid_map.raw_obstacle_mask
        bg[infl_mask] = [180, 180, 180]

        ax.imshow(bg, origin='lower', extent=[0, map_size, 0, map_size])

        # Overlap highlight (semi-transparent red)
        if show_overlap:
            # Compute occupancy from paths
            if paths:
                from .geometry import generate_road_mask, compute_occupancy_count
                road_radius = 2.5  # beta3/2
                road_masks = []
                for p in paths:
                    poly = [(float(x), float(y)) for x, y in p.points]
                    mask = generate_road_mask(poly, road_radius, map_size)
                    road_masks.append(mask)
                occ = compute_occupancy_count(road_masks, map_size)
                overlap_mask = (occ > 1).reshape(map_size, map_size)
                if np.any(overlap_mask):
                    overlay = np.zeros((map_size, map_size, 4), dtype=np.float32)
                    overlay[overlap_mask] = [1.0, 0.0, 0.0, 0.3]
                    ax.imshow(overlay, origin='lower', extent=[0, map_size, 0, map_size])

        # OD colors
        n_ods = len(od_records)
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_ods, 1)))

        # Draw paths
        if paths:
            for od_idx, path in enumerate(paths):
                if od_idx < len(od_records):
                    xs = [p[0] for p in path.points]
                    ys = [p[1] for p in path.points]
                    ax.plot(xs, ys, color=colors[od_idx % 20], linewidth=1.5,
                            alpha=0.8, label=f"OD{od_idx:02d}")

        # Draw control paths
        if control_paths:
            for cp in control_paths:
                od_idx = cp.od_id
                if od_idx < len(od_records):
                    xs = [p.x for p in cp.points]
                    ys = [p.y for p in cp.points]
                    ax.plot(xs, ys, color=colors[od_idx % 20], linewidth=1.5,
                            alpha=0.8)

                    if show_control_points:
                        xs_inner = [p.x for p in cp.points if p.is_movable]
                        ys_inner = [p.y for p in cp.points if p.is_movable]
                        ax.scatter(xs_inner, ys_inner, color=colors[od_idx % 20],
                                   s=3.0, alpha=0.7, zorder=5)

                        # Fixed endpoints
                        xs_fixed = [p.x for p in cp.points if not p.is_movable]
                        ys_fixed = [p.y for p in cp.points if not p.is_movable]
                        ax.scatter(xs_fixed, ys_fixed, color=colors[od_idx % 20],
                                   s=7.0, marker='o', alpha=0.9, zorder=6)

                    if highlight_turns:
                        poly = [(p.x, p.y) for p in cp.points]
                        for i in range(1, len(poly) - 1):
                            xa, ya = poly[i - 1]
                            bx, by = poly[i]
                            cx, cy = poly[i + 1]
                            v1 = (bx - xa, by - ya)
                            v2 = (cx - bx, cy - by)
                            if abs(v1[0]) > 1e-6 or abs(v1[1]) > 1e-6:
                                if abs(v2[0]) > 1e-6 or abs(v2[1]) > 1e-6:
                                    t1 = np.arctan2(v1[1], v1[0])
                                    t2 = np.arctan2(v2[1], v2[0])
                                    from .types import angle_diff
                                    if angle_diff(t1, t2) > 1e-6:
                                        ax.scatter([bx], [by], color=colors[od_idx % 20],
                                                   s=10.0, marker='o', facecolors='none',
                                                   linewidths=1.5, zorder=7)

        # Draw OD markers
        for od_idx, od in enumerate(od_records):
            c = colors[od_idx % 20]
            # Origin (circle)
            ax.scatter([od.start[0]], [od.start[1]], color=c, s=7.0,
                       marker='o', zorder=10)
            ax.text(od.start[0] + 5, od.start[1] + 5,
                    f"O{od_idx:02d}", fontsize=8, color=c, zorder=10)

            # Destination (x marker)
            ax.scatter([od.goal[0]], [od.goal[1]], color=c, s=7.0,
                       marker='x', zorder=10)
            ax.text(od.goal[0] + 5, od.goal[1] + 5,
                    f"D{od_idx:02d}", fontsize=8, color=c, zorder=10)

        # Title and labels
        title_text = f"{run_id} - {title} ({len(od_records)} OD pairs)"
        if hist_metrics:
            metrics_str = ", ".join(f"{k}={v:.2f}" for k, v in hist_metrics.items())
            title_text += f"\n{metrics_str}"
        ax.set_title(title_text, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, map_size)
        ax.set_ylim(0, map_size)
        ax.set_aspect('equal')

        # Legend (only if few enough OD pairs)
        if len(od_records) <= 10:
            ax.legend(fontsize=6, loc='upper right')

        plt.tight_layout()

        # Save to temp file first, then replace. Pass format explicitly so
        # matplotlib doesn't infer it from the .tmp extension.
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        temp_path = save_path + ".tmp"
        plt.savefig(temp_path, dpi=dpi, bbox_inches='tight', format='png')
        os.replace(temp_path, save_path)
        plt.close(fig)

        return True

    except Exception as e:
        print(f"[Visualization] Error saving {save_path}: {e}")
        try:
            plt.close('all')
        except Exception:
            pass
        return False


def save_all_figures(
    grid_map: GridMap,
    od_records: List[ODRecord],
    run_id: str,
    output_dir: str,
    config: NormalizedConfig,
    selected_dense_paths: Optional[List[DensePath]] = None,
    best_network: Optional[NetworkState] = None,
    combination_result: Optional[CombinationResult] = None,
) -> Dict[str, str]:
    """Save all 4 PNG figures.

    Returns:
        Dictionary mapping figure name to path (or empty string on failure)
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    map_size = config.eta1
    figure_paths = {}

    # PNG 1: Map and OD points
    print("[Visualization] Saving PNG 1: Map and OD points ...")
    path1 = os.path.join(fig_dir, "01_map_and_od.png")
    ok = save_network_figure(
        grid_map, od_records, map_size, run_id,
        "Map and OD Points", path1,
    )
    if ok:
        figure_paths["01_map_and_od"] = path1
    else:
        figure_paths["01_map_and_od"] = ""

    # PNG 2: Baseline path network
    print("[Visualization] Saving PNG 2: Baseline path network ...")
    path2 = os.path.join(fig_dir, "02_baseline_path_network.png")
    baseline_paths = [od.baseline_path for od in od_records]
    ok = save_network_figure(
        grid_map, od_records, map_size, run_id,
        "Baseline Path Network", path2,
        paths=baseline_paths,
    )
    if ok:
        figure_paths["02_baseline"] = path2
    else:
        figure_paths["02_baseline"] = ""

    # PNG 3: Combination optimized network
    if selected_dense_paths is not None:
        print("[Visualization] Saving PNG 3: Combination optimized network ...")
        path3 = os.path.join(fig_dir, "03_combination_optimized_network.png")
        metrics = None
        if combination_result:
            metrics = {"initial_A": combination_result.initial_objective,
                       "best_A": combination_result.best_objective}
        ok = save_network_figure(
            grid_map, od_records, map_size, run_id,
            "Combination Optimized Network", path3,
            paths=selected_dense_paths, show_overlap=True,
            hist_metrics=metrics,
        )
        if ok:
            figure_paths["03_combination"] = path3
        else:
            figure_paths["03_combination"] = ""

    # PNG 4: Control point optimized network
    if best_network is not None:
        print("[Visualization] Saving PNG 4: Control point optimized network ...")
        path4 = os.path.join(fig_dir, "04_control_point_optimized_network.png")
        metrics = best_network.true_metrics
        ok = save_network_figure(
            grid_map, od_records, map_size, run_id,
            "Control Point Optimized Network", path4,
            control_paths=best_network.control_paths,
            show_overlap=True, show_control_points=True,
            highlight_turns=True, hist_metrics=metrics,
        )
        if ok:
            figure_paths["04_control_optimized"] = path4
        else:
            figure_paths["04_control_optimized"] = ""

    return figure_paths