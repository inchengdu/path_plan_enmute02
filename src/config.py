"""
Stage 1: Configuration normalization and random stream initialization.
"""
from __future__ import annotations
import hashlib
import math
import time
from typing import Any, Dict, Tuple
import numpy as np
import tomllib

from .types import (
    NormalizedConfig, RandomStreams, StageStatus, derive_seed
)


def load_toml_config(path: str) -> dict:
    """Load TOML configuration file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def merge_configs(research: dict, engineering: dict) -> dict:
    """Deep merge two configs, ensuring no duplicate leaf keys."""
    merged = {}
    _deep_merge(merged, research, engineering, "")
    return merged


def _deep_merge(target: dict, src1: dict, src2: dict, path: str):
    """Recursively merge src1 then src2 into target, error on duplicate leaves."""
    all_keys = set(src1.keys()) | set(src2.keys())
    for k in all_keys:
        full_path = f"{path}.{k}" if path else k
        v1 = src1.get(k)
        v2 = src2.get(k)
        if v1 is not None and v2 is not None:
            if isinstance(v1, dict) and isinstance(v2, dict):
                if k not in target:
                    target[k] = {}
                _deep_merge(target[k], v1, v2, full_path)
            else:
                # Both are leaf values - error on duplicate
                raise ValueError(f"Duplicate leaf key: {full_path}")
        elif v1 is not None:
            target[k] = v1
        elif v2 is not None:
            target[k] = v2


def normalize_config(config: dict) -> NormalizedConfig:
    """Normalize raw configuration into standardized form.

    Args:
        config: Merged configuration dictionary (research + engineering)

    Returns:
        NormalizedConfig with all derived parameters
    """
    # Extract research parameters
    obj = config.get("objective", {})
    spatial = config.get("spatial_costs", {})
    turning = config.get("turning_and_spacing", {})
    map_params = config.get("map", {})
    od_params = config.get("od", {})
    cand_params = config.get("candidates", {})
    runtime_params = config.get("runtime", {})
    cp_search = config.get("control_point_search", {})
    eng_config = config.get("engineering_config", config)
    numeric = config.get("numeric", {})
    map_gen = config.get("map_generation", {})
    motion = config.get("motion_primitives", {})
    astar_params = config.get("astar", {})
    comb_search = config.get("combination_search", {})
    control_tabu = config.get("control_tabu", {})
    adaptive = config.get("adaptive_threshold", {})
    conflicts = config.get("conflicts_and_breakout", {})
    optim = config.get("optimization", {})

    # Convert beta_theta from degrees to radians (unique conversion)
    beta_theta_deg = float(turning.get("beta_theta_deg", 10.0))
    beta_theta_rad = math.radians(beta_theta_deg)

    # Derive beta10, beta11
    beta10 = int(math.ceil(1.0 / beta_theta_rad))
    beta11 = beta10

    beta12 = float(turning.get("beta12", 30.0))
    beta13 = float(turning.get("beta13", 2.0))
    beta14 = float(turning.get("beta14", 15.0))

    # Validate relationships
    if not (1 <= beta10 < beta14 < beta12):
        raise ValueError(f"Invalid beta parameter relationship: "
                         f"beta10={beta10}, beta14={beta14}, beta12={beta12}")

    # Check for legacy explicit beta10/beta11
    if "beta10" in turning and turning["beta10"] != beta10:
        raise ValueError(f"Derived beta10={beta10} doesn't match explicit beta10={turning['beta10']}")
    if "beta11" in turning and turning["beta11"] != beta11:
        raise ValueError(f"Derived beta11={beta11} doesn't match explicit beta11={turning['beta11']}")

    # Extract parameters
    eta1 = int(map_params.get("eta1", 1000))
    eta2 = float(map_params.get("eta2", 0.10))
    eta3 = int(map_params.get("eta3", 500))
    eta4 = int(map_params.get("eta4", 3000))
    eta5 = float(map_params.get("eta5", 1.5))
    eta6 = int(od_params.get("eta6", 10))
    eta7 = int(cand_params.get("eta7", 10))
    eta10 = int(runtime_params.get("eta10", 6))

    beta1 = int(spatial.get("beta1", 3))
    beta2 = float(spatial.get("beta2", 3.0))
    beta3 = float(spatial.get("beta3", 5.0))
    beta4 = int(cp_search.get("beta4", 3))
    beta6 = float(cp_search.get("beta6", 25.0))

    alpha1 = float(obj.get("alpha1", 1.0))
    alpha2 = float(obj.get("alpha2", 1.0))
    alpha3 = float(obj.get("alpha3", 1.0))
    alpha4 = float(obj.get("alpha4", 1.0))

    robot_radius = float(config.get("map", {}).get("robot_radius", 1.0))
    K = int(optim.get("K", 30))

    # Build config hash
    config_hash = _compute_config_hash(config)

    return NormalizedConfig(
        schema_version=str(eng_config.get("schema_version", "1.0")),
        research_parameters=dict(config.get("research_config", {})),
        engineering_parameters=dict(config),
        beta_theta_deg=beta_theta_deg,
        beta_theta_rad=beta_theta_rad,
        beta10=beta10,
        beta11=beta11,
        beta12=beta12,
        beta13=beta13,
        beta14=beta14,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        beta4=beta4,
        beta6=beta6,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        alpha4=alpha4,
        eta1=eta1,
        eta2=eta2,
        eta3=eta3,
        eta4=eta4,
        eta5=eta5,
        eta6=eta6,
        eta7=eta7,
        eta10=eta10,
        robot_radius=robot_radius,
        K=K,
        schedule_rule={
            "strict_effect_round_parity": "odd",
            "randomized_round_parity": "even",
            "random_selection_probability": 0.10,
        },
        config_hash=config_hash,
        # Engineering parameters
        numeric_epsilon=float(numeric.get("epsilon", 1e-9)),
        turn_detection_epsilon=float(numeric.get("turn_detection_angle_epsilon_rad", 1e-6)),
        candidate_resolution=float(cp_search.get("candidate_resolution", 1.0)),
        total_area_relative_tolerance=float(map_gen.get("total_area_relative_tolerance", 0.01)),
        max_candidate_attempts_per_od=int(cand_params.get("max_attempts_per_od", 50)),
        consecutive_failure_limit=int(cand_params.get("consecutive_failure_limit", 5)),
        max_length_ratio_to_baseline=float(cand_params.get("max_length_ratio_to_baseline", 1.20)),
        heuristic_weight_min=float(cand_params.get("heuristic_weight_min", 1.00)),
        heuristic_weight_max=float(cand_params.get("heuristic_weight_max", 1.15)),
        state_bias_amplitude=float(cand_params.get("state_bias_amplitude", 3.0)),
        baseline_max_expanded=int(astar_params.get("baseline_max_expanded_states", 1_500_000)),
        candidate_max_expanded=int(astar_params.get("candidate_max_expanded_states", 750_000)),
        combination_max_iterations=int(comb_search.get("max_iterations", 5000)),
        combination_stall=int(comb_search.get("stall_iterations", 500)),
        tabu_tenure_initial=int(comb_search.get("tabu_tenure_initial", 7)),
        tabu_tenure_min=int(comb_search.get("tabu_tenure_min", 5)),
        tabu_tenure_max=int(comb_search.get("tabu_tenure_max", 20)),
        control_K=int(optim.get("K", 30)),
        control_stall_rounds=int(optim.get("stall_rounds", 5)),
        control_min_improvement=float(optim.get("minimum_relative_improvement", 1e-4)),
        control_tabu_tenure_initial=int(control_tabu.get("tenure_initial", 5)),
        control_tabu_tenure_min=int(control_tabu.get("tenure_min", 3)),
        control_tabu_tenure_max=int(control_tabu.get("tenure_max", 12)),
        control_tabu_cycle_detection_window=int(control_tabu.get("cycle_detection_window", 8)),
        control_tabu_tenure_increase_on_cycle=int(control_tabu.get("tenure_increase_on_cycle", 1)),
        control_tabu_visits_without_cycle_before_decrease=int(control_tabu.get("visits_without_cycle_before_decrease", 10)),
        control_tabu_tenure_decrease_without_cycle=int(control_tabu.get("tenure_decrease_without_cycle", 1)),
        control_tabu_aspiration=str(control_tabu.get("aspiration", "strict_legal_global_best_improvement")),
        control_tabu_failed_adjustment_creates_tabu=bool(control_tabu.get("failed_adjustment_creates_tabu", False)),
        adaptive_threshold_initial=float(adaptive.get("initial_ratio", 0.0)),
        adaptive_threshold_growth=float(adaptive.get("growth_ratio_per_stagnant_visit", 0.005)),
        adaptive_threshold_max=float(adaptive.get("max_ratio", 0.05)),
        adaptive_threshold_decay=float(adaptive.get("decay_factor_on_improvement", 0.5)),
        gamma=float(conflicts.get("gamma", 1.0)),
        age_saturation=int(conflicts.get("age_saturation_rounds", 5)),
        max_local_priority_retries=int(cp_search.get("max_local_priority_retries", 2)),
        obstacle_min_separation=float(map_gen.get("obstacle_min_separation", 3.0)),
        od_clearance=float(od_params.get("od_clearance", 10.0)),
        origin_x_max_fraction=float(od_params.get("origin_x_max_fraction", 0.30)),
        destination_x_min_fraction=float(od_params.get("destination_x_min_fraction", 0.70)),
        max_od_batches=int(od_params.get("max_od_batches", 30)),
        od_proposal_batch_size=eta10,
        eta8=float(cand_params.get("eta8", 0.01)),
        heading_refinement_max=int(motion.get("heading_refinement_max", 4)),
        # Map generation
        radial_harmonic_min=int(map_gen.get("radial_harmonic_min", 2)),
        radial_harmonic_max=int(map_gen.get("radial_harmonic_max", 5)),
        radial_coefficient_abs_max=float(map_gen.get("radial_coefficient_abs_max", 0.06)),
        radial_angular_samples=int(map_gen.get("radial_angular_samples", 256)),
        shape_scale_binary_search_iterations=int(map_gen.get("shape_scale_binary_search_iterations", 32)),
        shape_area_relative_tolerance=float(map_gen.get("shape_area_relative_tolerance", 0.01)),
        compactness_min=float(map_gen.get("compactness_min", 0.55)),
        aspect_ratio_max=float(map_gen.get("aspect_ratio_max", 1.5)),
        boundary_roughness_max=float(map_gen.get("boundary_roughness_max", 1.25)),
        max_shape_trials_per_obstacle=int(map_gen.get("max_shape_trials_per_obstacle", 80)),
        max_placement_trials_per_obstacle=int(map_gen.get("max_placement_trials_per_obstacle", 300)),
        max_shape_regenerations_per_obstacle=int(map_gen.get("max_shape_regenerations_per_obstacle", 20)),
        max_map_restarts=int(map_gen.get("max_map_restarts", 10)),
    )


def _compute_config_hash(config: dict) -> str:
    """Compute BLAKE2b hash of the merged config."""
    raw = str(sorted(config.items())).encode()
    h = hashlib.blake2b(raw, digest_size=8)
    return h.hexdigest()


def initialize_random_streams(root_seed: int, config: NormalizedConfig) -> RandomStreams:
    """Initialize independent random streams for each stage.

    Args:
        root_seed: Root random seed
        config: Normalized configuration

    Returns:
        RandomStreams with independent keys for each module
    """
    map_key = derive_seed(root_seed, "map")
    od_key = derive_seed(root_seed, "od")
    combination_key = derive_seed(root_seed, "combination")

    # Candidate keys for each OD and attempt
    candidate_keys = {}
    for od_idx in range(config.eta6 * 2):  # generous upper bound
        for attempt in range(config.max_candidate_attempts_per_od):
            key = f"od_{od_idx}_attempt_{attempt}"
            candidate_keys[key] = derive_seed(root_seed, "candidate", od_idx, attempt)

    # Control optimization keys per round
    control_keys = []
    for round_idx in range(config.control_K + 5):
        control_keys.append(derive_seed(root_seed, "control", round_idx))

    return RandomStreams(
        root_seed=root_seed,
        map_key=map_key,
        od_key=od_key,
        candidate_keys=candidate_keys,
        combination_key=combination_key,
        control_keys=control_keys,
    )


def run_stage1(config_path: str) -> Tuple[NormalizedConfig, RandomStreams, dict, float]:
    """Run Stage 1: Load and normalize configuration, initialize random streams.

    Args:
        config_path: Path to TOML configuration file

    Returns:
        Tuple of (NormalizedConfig, RandomStreams, run_manifest, elapsed_time)
    """
    t0 = time.time()
    print(f"[Stage 1] Loading configuration from {config_path} ...")

    # Load the engineering config (which references research config)
    raw_config = load_toml_config(config_path)

    # Check if there's a research hyperparameters file reference
    research_path = None
    if "parameter_files" in raw_config:
        research_rel = raw_config["parameter_files"].get("research_hyperparameters_file", "")
        if research_rel:
            import os
            base_dir = os.path.dirname(config_path)
            research_path = os.path.join(base_dir, research_rel)

    # Merge research and engineering configs
    if research_path:
        print(f"[Stage 1] Loading research hyperparameters from {research_path}")
        research_config = load_toml_config(research_path)
        merged = merge_configs(research_config, raw_config)
    else:
        merged = raw_config

    # Normalize
    print(f"[Stage 1] Normalizing configuration ...")
    normalized = normalize_config(merged)
    print(f"[Stage 1] beta_theta_deg={normalized.beta_theta_deg} -> beta_theta_rad={normalized.beta_theta_rad:.10f}")
    print(f"[Stage 1] Derived beta10={normalized.beta10}, beta11={normalized.beta11}")
    print(f"[Stage 1] Config hash: {normalized.config_hash}")

    # Initialize random streams
    root_seed = raw_config.get("random", {}).get("root_seed", 20260812)
    print(f"[Stage 1] Root seed: {root_seed}")
    random_streams = initialize_random_streams(root_seed, normalized)

    # Build run manifest
    run_manifest = {
        "run_id": f"run_{normalized.config_hash[:8]}",
        "config_hash": normalized.config_hash,
        "root_seed": root_seed,
        "schema_version": normalized.schema_version,
        "beta_theta_deg": normalized.beta_theta_deg,
        "beta_theta_rad": normalized.beta_theta_rad,
        "beta10": normalized.beta10,
        "beta11": normalized.beta11,
        "stage1_status": StageStatus.SUCCESS,
    }

    elapsed = time.time() - t0
    print(f"[Stage 1] Completed in {elapsed:.2f}s")
    return normalized, random_streams, run_manifest, elapsed