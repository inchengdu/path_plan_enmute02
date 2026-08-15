"""
Core data types for the path planning algorithm.
"""
from __future__ import annotations
import dataclasses
import hashlib
from typing import List, Tuple, Optional, Dict, Set, Any
import numpy as np


# === Geometry Primitives ===

Point = Tuple[int, int]  # (x, y) integer grid coordinates
PointF = Tuple[float, float]  # (x, y) floating point coordinates
Polyline = List[PointF]  # ordered list of control points


# === Status Codes ===

class StageStatus:
    SUCCESS = "SUCCESS"
    INVALID_CONFIG = "INVALID_CONFIG"
    NO_PATH = "NO_PATH"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# === Stage 1: Config & Random Streams ===

@dataclasses.dataclass
class NormalizedConfig:
    schema_version: str
    research_parameters: dict
    engineering_parameters: dict
    beta_theta_deg: float
    beta_theta_rad: float
    beta10: int
    beta11: int
    beta12: float
    beta13: float
    beta14: float
    beta1: int
    beta2: float
    beta3: float
    beta4: int
    beta6: float
    alpha1: float
    alpha2: float
    alpha3: float
    alpha4: float
    eta1: int
    eta2: float
    eta3: int
    eta4: int
    eta5: float
    eta6: int
    eta7: int
    eta10: int
    robot_radius: float
    K: int  # number of optimization rounds
    schedule_rule: dict
    config_hash: str
    # Additional derived engineering parameters
    obstacle_min_separation: float = 3.0
    beta1_chebyshev: bool = True
    candidate_resolution: float = 1.0
    numeric_epsilon: float = 1e-9
    turn_detection_epsilon: float = 1e-6
    max_candidate_attempts_per_od: int = 50
    consecutive_failure_limit: int = 5
    max_length_ratio_to_baseline: float = 1.20
    heuristic_weight_min: float = 1.00
    heuristic_weight_max: float = 1.15
    state_bias_amplitude: float = 3.0
    baseline_max_expanded: int = 1_500_000
    candidate_max_expanded: int = 750_000
    # Combination search
    combination_max_iterations: int = 5000
    combination_stall: int = 500
    tabu_tenure_initial: int = 7
    tabu_tenure_min: int = 5
    tabu_tenure_max: int = 20
    # Control optimization
    control_K: int = 30
    control_stall_rounds: int = 5
    control_min_improvement: float = 1e-4
    local_window_radius: int = 2
    first_point_turn_multiplier: float = 2.0
    # Adaptive threshold
    adaptive_threshold_initial: float = 0.0
    adaptive_threshold_growth: float = 0.005
    adaptive_threshold_max: float = 0.05
    adaptive_threshold_decay: float = 0.5
    # Control-point Reactive Tabu (Stage 9)
    control_tabu_tenure_initial: int = 5
    control_tabu_tenure_min: int = 3
    control_tabu_tenure_max: int = 12
    control_tabu_cycle_detection_window: int = 8
    control_tabu_tenure_increase_on_cycle: int = 1
    control_tabu_visits_without_cycle_before_decrease: int = 10
    control_tabu_tenure_decrease_without_cycle: int = 1
    control_tabu_aspiration: str = "strict_legal_global_best_improvement"
    control_tabu_failed_adjustment_creates_tabu: bool = False
    # Breakout
    gamma: float = 1.0
    age_saturation: int = 5
    # Effect score
    effect_range: float = 25.0  # beta6
    # Local priority block
    max_local_priority_retries: int = 2
    # OD sampling
    od_clearance: float = 10.0
    origin_x_max_fraction: float = 0.30
    destination_x_min_fraction: float = 0.70
    max_od_batches: int = 30
    od_proposal_batch_size: int = 6  # eta10
    # Map generation
    radial_harmonic_min: int = 2
    radial_harmonic_max: int = 5
    radial_coefficient_abs_max: float = 0.06
    radial_angular_samples: int = 256
    shape_scale_binary_search_iterations: int = 32
    shape_area_relative_tolerance: float = 0.01
    compactness_min: float = 0.55
    aspect_ratio_max: float = 1.5
    boundary_roughness_max: float = 1.25
    max_shape_trials_per_obstacle: int = 80
    max_placement_trials_per_obstacle: int = 300
    max_shape_regenerations_per_obstacle: int = 20
    max_map_restarts: int = 10
    total_area_relative_tolerance: float = 0.01
    eta8: float = 0.01  # minimum area difference for candidate diversity
    heading_refinement_max: int = 4

    def get_metric_weights(self) -> Tuple[float, float, float, float]:
        return (self.alpha1, self.alpha2, self.alpha3, self.alpha4)


@dataclasses.dataclass
class RandomStreams:
    root_seed: int
    map_key: int
    od_key: int
    candidate_keys: Dict[str, int]  # key: "od_{idx}_attempt_{attempt}"
    combination_key: int
    control_keys: List[int]  # per round

    def get_map_rng(self) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.map_key))

    def get_od_rng(self) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.od_key))

    def get_candidate_rng(self, od_idx: int, attempt: int) -> np.random.Generator:
        key = f"od_{od_idx}_attempt_{attempt}"
        return np.random.Generator(np.random.PCG64(self.candidate_keys.get(key, self.root_seed + od_idx * 1000 + attempt)))

    def get_combination_rng(self) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.combination_key))

    def get_control_rng(self, round_idx: int) -> np.random.Generator:
        if round_idx < len(self.control_keys):
            return np.random.Generator(np.random.PCG64(self.control_keys[round_idx]))
        return np.random.Generator(np.random.PCG64(self.root_seed + 100000 + round_idx))


# === Stage 2: GridMap ===

@dataclasses.dataclass
class ObstacleRecord:
    obstacle_id: int
    area: int
    center: PointF
    bounding_box: Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax
    compactness: float
    aspect_ratio: float


@dataclasses.dataclass
class GridMap:
    size: int
    raw_obstacle_mask: np.ndarray  # bool 2D [y][x]
    obstacle_id_map: np.ndarray  # int 2D [y][x], -1 for free
    hard_obstacle_mask: np.ndarray  # bool 2D [y][x]
    hard_distance_field: np.ndarray  # float 2D [y][x]
    obstacle_cost_field: np.ndarray  # float 2D [y][x]
    obstacle_records: List[ObstacleRecord]


@dataclasses.dataclass
class MapStageReport:
    total_obstacle_area: int
    obstacle_count: int
    retries: int
    stop_reason: str


# === Stage 3: Motion Primitive ===

@dataclasses.dataclass
class MotionPrimitive:
    primitive_id: int
    displacement: Tuple[int, int]  # (dx, dy)
    actual_heading_angle: float
    primitive_length: float
    dense_offsets: List[Tuple[int, int]]  # ordered grid points
    supercover_offsets: List[Tuple[int, int]]  # collision cover
    nominal_heading_angle: float


@dataclasses.dataclass
class PrimitiveLibrary:
    primitives: List[MotionPrimitive]
    heading_angles: List[float]
    by_heading: Dict[int, List[MotionPrimitive]]  # heading_index -> list
    construction_parameters: dict


# === Stage 4: OD Records ===

@dataclasses.dataclass
class DensePath:
    points: List[Point]  # ordered dense grid points
    primitive_sequence: List[int]  # primitive IDs
    motion_segments: List[Tuple[int, int, int]]  # (start_idx, end_idx, primitive_id)
    true_turn_indices: List[int]  # indices of points where turns happen
    cumulative_lengths: List[float]
    total_physical_length: float
    geometry_hash: str

    def __post_init__(self):
        if not self.geometry_hash:
            self.geometry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = b",".join(f"{x},{y}".encode() for x, y in self.points)
        return hashlib.blake2b(data, digest_size=8).hexdigest()


@dataclasses.dataclass
class ODRecord:
    od_id: int
    start: Point
    goal: Point
    baseline_path: DensePath
    baseline_length: float


@dataclasses.dataclass
class ODStageReport:
    proposals: int
    accepted: int
    no_path: int
    budget_exhausted: int
    stop_reason: str


# === Stage 5: Candidate Sets ===

@dataclasses.dataclass
class CandidateSet:
    od_id: int
    baseline_path: DensePath
    baseline_length: float
    candidates: List[DensePath]
    hashes: Set[str]
    stop_reason: str
    frozen: bool = False


@dataclasses.dataclass
class CandidateAttemptLog:
    od_id: int
    attempt: int
    heuristic_weight: float
    seed: int
    search_status: str
    reject_stage: str
    length: float
    min_area_diff: float


# === Stage 6: Precompute ===

@dataclasses.dataclass
class FrozenCandidateSets:
    sets: List[CandidateSet]
    frozen: bool = True


@dataclasses.dataclass
class CandidatePrecompute:
    road_masks: List[List[np.ndarray]]  # [od_idx][candidate_idx] -> flat bool road mask (map_size^2,)
    nonzero_indices: List[List[np.ndarray]]  # [od_idx][candidate_idx] -> flat indices of mask cells
    physical_lengths: List[List[float]]  # [od_idx][candidate_idx]
    optional_soft_costs: Optional[List[List[float]]] = None


# === Stage 7: Combination ===

@dataclasses.dataclass
class CombinationResult:
    selected_candidate_indices: List[int]  # per OD
    initial_objective: float
    best_objective: float
    iterations: int
    stop_reason: str
    objective_trace: List[float]


# === Stage 8: Sparsification ===

@dataclasses.dataclass
class ControlPoint:
    point_id: int
    sequence_index: int
    x: float
    y: float
    dense_source_index: int  # index in original dense path
    source_type: str  # "start", "end", "primitive_end", "deletion_survivor", "overlap_support", "spacing_insert", "short_spacing_exception"
    retention_reason: str
    is_movable: bool


@dataclasses.dataclass
class ControlPath:
    od_id: int
    points: List[ControlPoint]
    selected_dense_candidate_id: int
    segment_ids: List[int]


@dataclasses.dataclass
class SparseControlPaths:
    paths: List[ControlPath]


@dataclasses.dataclass
class SparsifyReport:
    per_path: List[dict]


# === Stage 9: Continuous Optimization ===

@dataclasses.dataclass
class NetworkState:
    control_paths: List[ControlPath]
    road_masks_by_path: List[np.ndarray]  # per path bitset
    occupancy_count: np.ndarray  # per grid cell, how many paths cover it
    true_metrics: Dict[str, float]  # L, D, A, R, J_true
    conflict_components: List[dict]
    point_search_states: Dict[int, dict]  # point_id -> state
    adjustment_attempts: List[dict]
    local_priority_blocks: List[list]
    round_index: int


@dataclasses.dataclass
class ControlOptimizationResult:
    best_network: NetworkState
    rounds_completed: int
    stop_reason: str
    objective_trace: List[float]
    selection_trace: List[dict]
    adjustment_trace: List[dict]


# === Stage 10: Final ===

@dataclasses.dataclass
class FinalResult:
    status: str
    final_control_paths: List[ControlPath]
    final_metrics: Dict[str, float]
    conflict_components: List[dict]
    validation_summary: dict
    run_manifest: dict


# === Utility functions ===

def derive_seed(root_seed: int, module: str, *args) -> int:
    """Derive a deterministic sub-seed using BLAKE2b."""
    data = f"{root_seed}:{module}".encode()
    for arg in args:
        data += f":{arg}".encode()
    h = hashlib.blake2b(data, digest_size=8)
    return int.from_bytes(h.digest(), byteorder='big', signed=False) & 0x7FFFFFFF


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a: float, b: float) -> float:
    """Absolute wrapped angle difference."""
    return abs(wrap_angle(a - b))


def chebyshev_distance(p1: Point, p2: Point) -> int:
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def euclidean_distance(p1: Point, p2: Point) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def euclidean_distance_f(p1: PointF, p2: PointF) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_segment_distance(p: PointF, a: PointF, b: PointF) -> float:
    """Minimum distance from point p to segment ab."""
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    t = (apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-30)
    t = max(0.0, min(1.0, t))
    projx = ax + t * abx
    projy = ay + t * aby
    return np.sqrt((px - projx) ** 2 + (py - projy) ** 2)