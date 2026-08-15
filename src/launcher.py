"""
Launcher: main entry point for the path planning algorithm pipeline.

Orchestrates all 10 stages in sequence, handles data passing,
output management, and error handling.
"""
from __future__ import annotations
import copy
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

from .types import (
    NormalizedConfig, RandomStreams, GridMap, PrimitiveLibrary,
    ODRecord, DensePath, CandidateSet, CandidatePrecompute,
    FrozenCandidateSets, CombinationResult, SparseControlPaths,
    ControlOptimizationResult, NetworkState, FinalResult, StageStatus,
    CandidateAttemptLog, ODStageReport, MapStageReport, SparsifyReport,
)
from .config import run_stage1
from .map_generation import run_stage2
from .motion_primitives import run_stage3
from .od_sampling import run_stage4
from .candidate_gen import run_stage5
from .precompute import run_stage6
from .combination import run_stage7
from .sparsify import run_stage8
from .optimize import run_stage9
from .finalize import run_stage10
from .visualization import save_all_figures
from .io_utils import (
    save_json, save_jsonl, save_npz_compressed, NumpyEncoder,
    setup_output_directory, save_run_manifest
)


class PipelineRunResult:
    """Result of a full pipeline run."""
    def __init__(
        self,
        run_id: str = "",
        status: str = StageStatus.SUCCESS,
        completed_stages: Optional[List[int]] = None,
        final_result: Optional[FinalResult] = None,
        output_directory: str = "",
        figure_paths: Optional[Dict[str, str]] = None,
        failed_stage: int = -1,
        failure_reason: str = "",
    ):
        self.run_id = run_id
        self.status = status
        self.completed_stages = completed_stages or []
        self.final_result = final_result
        self.output_directory = output_directory
        self.figure_paths = figure_paths or {}
        self.failed_stage = failed_stage
        self.failure_reason = failure_reason


def run_pipeline(
    config_path: str,
    output_root: str = "./outputs",
    run_name: str = "",
    external_map_path: str = "",
    overwrite_policy: str = "reject",
) -> PipelineRunResult:
    """Run the full path planning pipeline.

    Args:
        config_path: Path to TOML configuration file
        output_root: Root output directory
        run_name: Run name (auto-generated if empty)
        external_map_path: Path to external map file
        overwrite_policy: Directory overwrite policy

    Returns:
        PipelineRunResult
    """
    overall_t0 = time.time()
    print("=" * 70)
    print("Path Planning Algorithm Pipeline")
    print("=" * 70)

    # Set up output directory
    try:
        output_dir = setup_output_directory(output_root, run_name, overwrite_policy)
    except FileExistsError as e:
        print(f"[Pipeline] {e}")
        return PipelineRunResult(
            status="OUTPUT_EXISTS",
            failure_reason=str(e),
        )

    # Copy input config
    import shutil
    shutil.copy2(config_path, os.path.join(output_dir, "config", "input_config.toml"))

    # Pipeline state
    completed_stages: List[int] = []
    stage_reports: Dict[str, object] = {}
    pipeline_data: Dict[str, object] = {}
    figure_paths: Dict[str, str] = {}
    run_manifest: dict = {
        "run_id": "",
        "status": "RUNNING",
        "completed_stages": [],
        "stage_times": {},
    }

    # === Stage 1: Config ===
    try:
        print("\n" + "=" * 70)
        print("STAGE 1: Configuration Normalization")
        print("=" * 70)
        config, random_streams, manifest, elapsed = run_stage1(config_path)
        run_manifest = manifest
        run_manifest["stage1_time"] = elapsed
        run_manifest["status"] = "RUNNING"
        run_manifest["completed_stages"] = [1]
        pipeline_data["config"] = config
        pipeline_data["random_streams"] = random_streams
        save_json(config.__dict__, os.path.join(output_dir, "config", "normalized_config.json"))
        completed_stages.append(1)
    except Exception as e:
        print(f"[Pipeline] Stage 1 failed: {e}")
        traceback.print_exc()
        return PipelineRunResult(
            run_id=run_manifest.get("run_id", ""),
            status="STAGE_FAILED",
            completed_stages=completed_stages,
            output_directory=output_dir,
            failed_stage=1,
            failure_reason=str(e),
        )

    # === Stage 2: Map ===
    try:
        print("\n" + "=" * 70)
        print("STAGE 2: Map Generation")
        print("=" * 70)
        grid_map, map_report, elapsed = run_stage2(config, random_streams)
        pipeline_data["grid_map"] = grid_map
        stage_reports["map"] = map_report
        run_manifest["stage2_time"] = elapsed
        run_manifest["completed_stages"] = [1, 2]
        import numpy as np
        save_json({
            "size": grid_map.size,
            "n_obstacles": len(grid_map.obstacle_records),
            "total_area": int(np.sum(grid_map.raw_obstacle_mask)),
            "hard_area": int(np.sum(grid_map.hard_obstacle_mask)),
        }, os.path.join(output_dir, "data", "map_metadata.json"))
        save_npz_compressed({
            "raw_obstacle_mask": grid_map.raw_obstacle_mask,
            "hard_obstacle_mask": grid_map.hard_obstacle_mask,
            "hard_distance_field": grid_map.hard_distance_field,
            "obstacle_cost_field": grid_map.obstacle_cost_field,
        }, os.path.join(output_dir, "data", "map_arrays.npz"))
        completed_stages.append(2)
    except Exception as e:
        print(f"[Pipeline] Stage 2 failed: {e}")
        traceback.print_exc()
        return PipelineRunResult(
            run_id=run_manifest.get("run_id", ""),
            status="STAGE_FAILED",
            completed_stages=completed_stages,
            output_directory=output_dir,
            failed_stage=2,
            failure_reason=str(e),
        )

    # === Stage 3: Motion Primitives ===
    try:
        print("\n" + "=" * 70)
        print("STAGE 3: Motion Primitive Library")
        print("=" * 70)
        prim_lib, elapsed = run_stage3(config)
        pipeline_data["primitive_library"] = prim_lib
        run_manifest["stage3_time"] = elapsed
        run_manifest["completed_stages"] = [1, 2, 3]
        save_json({
            "n_primitives": len(prim_lib.primitives),
            "n_headings": len(prim_lib.heading_angles),
            "heading_angles": [float(h) for h in prim_lib.heading_angles[:10]],
        }, os.path.join(output_dir, "data", "primitive_summary.json"))
        completed_stages.append(3)
    except Exception as e:
        print(f"[Pipeline] Stage 3 failed: {e}")
        traceback.print_exc()
        return PipelineRunResult(
            run_id=run_manifest.get("run_id", ""),
            status="STAGE_FAILED",
            completed_stages=completed_stages,
            output_directory=output_dir,
            failed_stage=3,
            failure_reason=str(e),
        )

    # === Stage 4: OD Sampling ===
    try:
        print("\n" + "=" * 70)
        print("STAGE 4: OD Sampling and Baseline Paths")
        print("=" * 70)
        od_records, od_report, elapsed = run_stage4(
            config, grid_map, prim_lib, random_streams
        )
        pipeline_data["od_records"] = od_records
        stage_reports["od"] = od_report
        run_manifest["stage4_time"] = elapsed
        run_manifest["completed_stages"] = [1, 2, 3, 4]
        save_json({
            "n_ods": len(od_records),
            "records": [{
                "od_id": r.od_id,
                "start": list(r.start),
                "goal": list(r.goal),
                "baseline_length": float(r.baseline_length),
            } for r in od_records],
        }, os.path.join(output_dir, "data", "od_records.json"))
        completed_stages.append(4)
    except Exception as e:
        print(f"[Pipeline] Stage 4 failed: {e}")
        traceback.print_exc()
        return PipelineRunResult(
            run_id=run_manifest.get("run_id", ""),
            status="STAGE_FAILED",
            completed_stages=completed_stages,
            output_directory=output_dir,
            failed_stage=4,
            failure_reason=str(e),
        )

    # PNG 1 and PNG 2 after Stage 4
    try:
        print("[Pipeline] Generating PNG 1 and PNG 2 ...")
        figs = save_all_figures(
            grid_map, od_records, run_manifest.get("run_id", ""),
            output_dir, config,
        )
        figure_paths.update(figs)
    except Exception as e:
        print(f"[Pipeline] Figure generation warning: {e}")

    # === Stage 5: Candidate Generation ===
    try:
        print("\n" + "=" * 70)
        print("STAGE 5: Dense Candidate Path Generation")
        print("=" * 70)
        candidate_sets, attempt_logs, elapsed = run_stage5(
            config, grid_map, prim_lib, od_records, random_streams
        )
        pipeline_data["candidate_sets"] = candidate_sets
        stage_reports["candidate_attempts"] = attempt_logs
        run_manifest["stage5_time"] = elapsed
        run_manifest["completed_stages"] = [1, 2, 3, 4, 5]
        save_json({
            "n_ods": len(candidate_sets),
            "candidate_counts": [len(cs.candidates) for cs in candidate_sets],
            "stop_reasons": [cs.stop_reason for cs in candidate_sets],
        }, os.path.join(output_dir, "data", "candidate_sets.json"))
        save_jsonl([a.__dict__ for a in attempt_logs],
                    os.path.join(output_dir, "traces", "candidate_attempts.jsonl"))
        completed_stages.append(5)
    except Exception as e:
        print(f"[Pipeline] Stage 5 failed: {e}")
        traceback.print_exc()
        return PipelineRunResult(
            run_id=run_manifest.get("run_id", ""),
            status="STAGE_FAILED",
            completed_stages=completed_stages,
            output_directory=output_dir,
            failed_stage=5,
            failure_reason=str(e),
        )

    # === Stage 6: Precompute ===
    try:
        print("\n" + "=" * 70)
        print("STAGE 6: Candidate Freeze and Overlap Precomputation")
        print("=" * 70)
        frozen_sets, precompute, elapsed = run_stage6(config, candidate_sets)
        pipeline_data["frozen_sets"] = frozen_sets
        pipeline_data["precompute"] = precompute
        run_manifest["stage6_time"] = elapsed
        run_manifest["completed_stages"] = [1, 2, 3, 4, 5, 6]
        save_json({
            "per_od_candidate_counts": [len(od_masks) for od_masks in precompute.road_masks],
            "coverage_cells_min": float(min(
                int(np.sum(m)) for od_masks in precompute.road_masks for m in od_masks)),
            "coverage_cells_max": float(max(
                int(np.sum(m)) for od_masks in precompute.road_masks for m in od_masks)),
        }, os.path.join(output_dir, "data", "overlap_summary.json"))
        completed_stages.append(6)
    except Exception as e:
        print(f"[Pipeline] Stage 6 failed: {e}")
        traceback.print_exc()
        return PipelineRunResult(
            run_id=run_manifest.get("run_id", ""),
            status="STAGE_FAILED",
            completed_stages=completed_stages,
            output_directory=output_dir,
            failed_stage=6,
            failure_reason=str(e),
        )

    # === Stage 7: Combination ===
    try:
        print("\n" + "=" * 70)
        print("STAGE 7: Multi-OD Combination Optimization")
        print("=" * 70)
        comb_result, selected_paths, elapsed = run_stage7(
            config, frozen_sets, precompute, random_streams
        )
        pipeline_data["combination_result"] = comb_result
        pipeline_data["selected_dense_paths"] = selected_paths
        run_manifest["stage7_time"] = elapsed
        run_manifest["completed_stages"] = [1, 2, 3, 4, 5, 6, 7]
        save_json({
            "selected_indices": comb_result.selected_candidate_indices,
            "initial_objective": float(comb_result.initial_objective),
            "best_objective": float(comb_result.best_objective),
            "iterations": comb_result.iterations,
            "stop_reason": comb_result.stop_reason,
        }, os.path.join(output_dir, "data", "combination_result.json"))
        save_jsonl([{"i": i, "obj": float(v)}
                    for i, v in enumerate(comb_result.objective_trace)],
                    os.path.join(output_dir, "traces", "combination_trace.jsonl"))
        completed_stages.append(7)
    except Exception as e:
        print(f"[Pipeline] Stage 7 failed: {e}")
        traceback.print_exc()
        return PipelineRunResult(
            run_id=run_manifest.get("run_id", ""),
            status="STAGE_FAILED",
            completed_stages=completed_stages,
            output_directory=output_dir,
            failed_stage=7,
            failure_reason=str(e),
        )

    # PNG 3 after Stage 7
    try:
        print("[Pipeline] Generating PNG 3 ...")
        figs = save_all_figures(
            grid_map, od_records, run_manifest.get("run_id", ""),
            output_dir, config,
            selected_dense_paths=selected_paths,
            combination_result=comb_result,
        )
        figure_paths.update(figs)
    except Exception as e:
        print(f"[Pipeline] Figure generation warning: {e}")

    # === Stage 8: Sparsification ===
    try:
        print("\n" + "=" * 70)
        print("STAGE 8: Control Point Sparsification")
        print("=" * 70)
        sparse_paths, sparsify_report, elapsed = run_stage8(
            config, grid_map, selected_paths
        )
        pipeline_data["sparse_control_paths"] = sparse_paths
        stage_reports["sparsify"] = sparsify_report
        run_manifest["stage8_time"] = elapsed
        run_manifest["completed_stages"] = [1, 2, 3, 4, 5, 6, 7, 8]
        save_json({
            "n_paths": len(sparse_paths.paths),
            "per_path": [{
                "od_id": r["od_id"],
                "initial_points": r["initial_points"],
                "deleted": r["deleted"],
                "spacing_inserts": r["spacing_inserts"],
                "final_points": r["final_points"],
            } for r in sparsify_report.per_path],
        }, os.path.join(output_dir, "data", "sparse_control_paths.json"))
        completed_stages.append(8)
    except Exception as e:
        print(f"[Pipeline] Stage 8 failed: {e}")
        traceback.print_exc()
        return PipelineRunResult(
            run_id=run_manifest.get("run_id", ""),
            status="STAGE_FAILED",
            completed_stages=completed_stages,
            output_directory=output_dir,
            failed_stage=8,
            failure_reason=str(e),
        )

    # === Stage 9: Continuous Optimization ===
    try:
        print("\n" + "=" * 70)
        print("STAGE 9: Continuous Control Point Optimization")
        print("=" * 70)
        opt_result, elapsed = run_stage9(
            config, sparse_paths, grid_map, random_streams
        )
        pipeline_data["optimization_result"] = opt_result
        run_manifest["stage9_time"] = elapsed
        run_manifest["completed_stages"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        save_jsonl(opt_result.selection_trace,
                    os.path.join(output_dir, "traces", "control_optimization_trace.jsonl"))
        completed_stages.append(9)
    except Exception as e:
        print(f"[Pipeline] Stage 9 failed: {e}")
        traceback.print_exc()
        return PipelineRunResult(
            run_id=run_manifest.get("run_id", ""),
            status="STAGE_FAILED",
            completed_stages=completed_stages,
            output_directory=output_dir,
            failed_stage=9,
            failure_reason=str(e),
        )

    # PNG 4 after Stage 9
    try:
        print("[Pipeline] Generating PNG 4 ...")
        figs = save_all_figures(
            grid_map, od_records, run_manifest.get("run_id", ""),
            output_dir, config,
            best_network=opt_result.best_network,
        )
        figure_paths.update(figs)
    except Exception as e:
        print(f"[Pipeline] Figure generation warning: {e}")

    # === Stage 10: Finalization ===
    try:
        print("\n" + "=" * 70)
        print("STAGE 10: Final Recalculation and Output")
        print("=" * 70)
        stage_reports["od_records"] = od_records
        final_result, elapsed = run_stage10(
            config, grid_map, opt_result.best_network, stage_reports
        )
        pipeline_data["final_result"] = final_result
        run_manifest["stage10_time"] = elapsed
        run_manifest["completed_stages"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        completed_stages.append(10)
    except Exception as e:
        print(f"[Pipeline] Stage 10 failed: {e}")
        traceback.print_exc()
        return PipelineRunResult(
            run_id=run_manifest.get("run_id", ""),
            status="STAGE_FAILED",
            completed_stages=completed_stages,
            output_directory=output_dir,
            failed_stage=10,
            failure_reason=str(e),
        )

    # Save final outputs
    save_json({
        "status": final_result.status,
        "metrics": final_result.final_metrics,
        "validation": final_result.validation_summary,
        "n_paths": len(final_result.final_control_paths),
        "n_conflict_components": len(final_result.conflict_components),
    }, os.path.join(output_dir, "data", "final_metrics.json"))

    # Save final control paths
    path_data = []
    for cp in final_result.final_control_paths:
        path_data.append({
            "od_id": cp.od_id,
            "n_points": len(cp.points),
            "points": [(p.x, p.y) for p in cp.points],
        })
    save_json(path_data, os.path.join(output_dir, "data", "final_control_paths.json"))

    # Save validation summary
    save_json(final_result.validation_summary,
              os.path.join(output_dir, "validation_summary.json"))

    # Save final run manifest
    total_elapsed = time.time() - overall_t0
    run_manifest["status"] = final_result.status
    run_manifest["total_time"] = total_elapsed
    run_manifest["figure_paths"] = figure_paths
    run_manifest["final_metrics"] = final_result.final_metrics
    save_run_manifest(run_manifest, output_dir)

    # Determine overall status
    all_figures_saved = all(bool(v) for v in figure_paths.values())
    if final_result.status == StageStatus.SUCCESS and all_figures_saved:
        status = "SUCCESS"
    elif final_result.status == StageStatus.SUCCESS:
        status = "COMPUTE_SUCCESS_VISUALIZATION_PARTIAL"
        print(f"[Pipeline] WARNING: Some figures could not be saved")
    else:
        status = final_result.status

    print("\n" + "=" * 70)
    print(f"Pipeline Complete: {status}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Output directory: {output_dir}")
    print(f"Final J_true: {final_result.final_metrics.get('J_true', 'N/A'):.4f}")
    print("=" * 70)

    return PipelineRunResult(
        run_id=run_manifest.get("run_id", ""),
        status=status,
        completed_stages=completed_stages,
        final_result=final_result,
        output_directory=output_dir,
        figure_paths=figure_paths,
        failed_stage=-1,
        failure_reason="",
    )