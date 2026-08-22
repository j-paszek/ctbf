"""Generate the fresh CTBF v5 depth-by-placement shortlist robustness bank."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import gc
import math
import multiprocessing
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from algorithm_evaluation.hypothesis_height_ambiguity_trend import (
    ambiguity_case_summary,
)
from algorithm_evaluation.hypothesis_trend_distance_preflight import (
    _validate_production_distance,
    candidate_graph_summary,
    distance_matrix_summary,
)
from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.paper_pipeline_runner import (
    measured_stage,
    validate_reconstruction_input,
)
from algorithm_evaluation.process_isolation import (
    TRUTH_BLOCK_SIMULATION_WORKER_UNIT,
    FreshSpawnPerTaskExecutor,
    fresh_process_contract,
)
from algorithm_evaluation.simulator_growth_probe import (
    _file_sha256,
    _validate_standard_base_config,
)
from algorithm_evaluation.simulator_reconstruction_intuition_probe import (
    _canonical_cells_at_generation,
    _event_totals,
    _profile,
    truth_sampling_diagnostics,
)
from algorithm_evaluation.simulator_sampling_fraction_truth_probe import (
    hybrid_sample_size,
)
from algorithm_evaluation.v5_shortlist_robustness_common import (
    BANK_ID_BY_SIMULATOR_REGIME,
    BANK_CONFIG_NAME,
    BANK_MANIFEST_NAME,
    BANK_SCHEMA_VERSION,
    BASELINE_SIMULATOR_REGIME,
    BIOPSY_LOWER_BOUND,
    CASE_METADATA_SCHEMA_VERSION,
    CNA_EVENT_PROBABILITY_BY_SIMULATOR_REGIME,
    DEFAULT_BANK_ID,
    DEFAULT_BASE_SEED,
    DEFAULT_BLOCK_COUNT,
    DISTANCE_EXECUTION_SCHEMA_VERSION,
    DISTANCE_EXECUTION_SEMANTICS,
    HEIGHTS,
    HEIGHTS_BY_SIMULATOR_REGIME,
    PLACEMENT_POLICIES,
    PREFLIGHT_CONTRACT_MODE_BY_SIMULATOR_REGIME,
    PRODUCTION_CONTRACT_MODE_BY_SIMULATOR_REGIME,
    PRODUCTION_SCIENTIFIC_ROLE_BY_SIMULATOR_REGIME,
    SAMPLING_RULE,
    SEED_NAMESPACE,
    SELECTED_V2_ARM_IDS,
    SHORTLIST_ARM_IDS,
    SIMULATOR_OVERRIDES_BY_REGIME,
    SIMULATOR_REGIMES,
    SMOKE_CONTRACT_MODE_BY_SIMULATOR_REGIME,
    V2_COMPLETE_ARM_IDS,
    TARGET_FRACTION,
    case_id,
    condition_paths,
    derived_seed,
    ensure_new_output_root,
    placement_schedule,
    serialize_distance,
    serialize_truth,
    truth_path,
    validate_positive_integer,
    write_json,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    canonical_json_digest,
)
from ctbs import (
    Cnp2CnpFileDistanceProvider,
    CtbsRuntimeConfig,
    DistanceMatrix,
    load_ctbs_runtime_config,
    unique_cells_by_cell_id,
)
from distance_semantics import stable_distance_label_key
from simulator import (
    CancerCellEvolutionSimulator,
    Genotype,
    SimulationResourceLimitExceeded,
)
from simulator_config import load_simulator_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "simulator_examples" / "default.json"
DEFAULT_SIMULATION_TIMEOUT_SECONDS = 300
DEFAULT_DISTANCE_TIMEOUT_SECONDS = 1200
DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS = 540
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = 1_048_576
DEFAULT_DISTANCE_WORKERS = 1
MAX_DISTANCE_WORKERS = 32

DistanceCompute = Callable[[Sequence[Genotype]], DistanceMatrix]
SimulatorFactory = Callable[[Mapping[str, Any], int], CancerCellEvolutionSimulator]
SIMULATION_WORKER_GRACE_SECONDS = 60


def _typed_error(error: BaseException, stage: str) -> dict[str, Any]:
    record = {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error)[:4096],
    }
    if isinstance(error, SimulationResourceLimitExceeded):
        record["resource_limit"] = error.as_dict()
        record["simulation_outcome"] = getattr(error, "simulation_outcome", None)
    return record


def _resource_audit_error(
    resources: Mapping[str, Any],
    stage: str,
) -> RuntimeError | None:
    memory = resources.get("memory")
    if not isinstance(memory, Mapping) or memory.get("peak_rss_bytes") is None:
        return RuntimeError(
            f"{stage} has no auditable process-tree peak-RSS measurement."
        )
    return None


def _simulate_preserving_guard_outcome(
    factory: SimulatorFactory,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[CancerCellEvolutionSimulator, dict[str, Any] | None]:
    simulator = factory(config, seed)
    try:
        simulator.run_simulation()
    except SimulationResourceLimitExceeded as error:
        failure = _typed_error(error, "simulation_resource_guard")
        stored_outcome = failure.get("simulation_outcome")
        if (
            not isinstance(stored_outcome, Mapping)
            or stored_outcome.get("status") != "resource_limit_exceeded"
            or stored_outcome.get("failure_generation") is None
        ):
            graph_outcome = simulator.tree.graph.get("simulation_outcome")
            usable_graph_outcome = (
                isinstance(graph_outcome, Mapping)
                and graph_outcome.get("status") == "resource_limit_exceeded"
                and graph_outcome.get("failure_generation") is not None
            )
            failure["simulation_outcome"] = (
                dict(graph_outcome)
                if usable_graph_outcome
                else {
                    "status": "resource_limit_exceeded",
                    "configured_final_generation": int(config["NUMBER_OF_GENERATIONS"]),
                    "last_retained_generation": error.generation,
                    "extinction_generation": None,
                    "failure_generation": error.generation,
                    "resource_limit": error.as_dict(),
                }
            )
        return simulator, failure
    return simulator, None


def _default_simulator_factory(
    config: Mapping[str, Any],
    seed: int,
) -> CancerCellEvolutionSimulator:
    return CancerCellEvolutionSimulator(config, seed=seed)


def _execute_isolated_simulation_stage(
    config: Mapping[str, Any],
    seed: int,
    timeout_seconds: int,
    rss_limit_bytes: int,
):
    return measured_stage(
        lambda: _simulate_preserving_guard_outcome(
            _default_simulator_factory,
            config,
            seed,
        ),
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )


def _sample_generation(
    simulator: CancerCellEvolutionSimulator,
    generation: int,
    *,
    base_seed: int,
    block_index: int,
) -> tuple[list[Genotype], dict[str, Any]]:
    available = _canonical_cells_at_generation(simulator, int(generation))
    if not available:
        raise ValueError(
            f"Required generation {generation} is empty in block {block_index + 1}."
        )
    realized = hybrid_sample_size(
        len(available),
        TARGET_FRACTION,
        BIOPSY_LOWER_BOUND,
    )
    seed = derived_seed("sampling", base_seed, block_index, int(generation))
    if realized == len(available):
        chosen = list(available)
        mode = "all_available"
        seed_used = False
    else:
        permutation = np.random.Generator(np.random.PCG64(seed)).permutation(
            len(available)
        )
        chosen = [available[int(index)] for index in permutation[:realized]]
        mode = "seeded_without_replacement"
        seed_used = True
    chosen.sort(
        key=lambda cell: (
            _profile(cell),
            stable_distance_label_key(cell.cell_id),
        )
    )
    return chosen, {
        "generation": int(generation),
        "available_distinct_state_count": len(available),
        "target_fraction": TARGET_FRACTION,
        "fractional_count": int(math.ceil(TARGET_FRACTION * len(available))),
        "biopsy_lower_bound": BIOPSY_LOWER_BOUND,
        "realized_occurrence_count": len(chosen),
        "selection_mode": mode,
        "sampling_seed": int(seed),
        "sampling_seed_used": seed_used,
        "seed_key_scope": "truth_block_and_generation_shared_across_conditions",
    }


def _reconstruction_input(
    case_id_value: str,
    *,
    height: int,
    policy: str,
    generations: Sequence[int],
    selected_levels: Sequence[Sequence[Genotype]],
) -> dict[str, Any]:
    payload = {
        "schema_version": "ctbf-v5-reconstruction-input-v1",
        "case_id": case_id_value,
        "condition_id": f"H{height}_{policy}_fraction_50_lower_6",
        "sampling_rule": SAMPLING_RULE,
        "placement_policy": policy,
        "levels": [
            {
                "biopsy_level": level_index,
                "generation": int(generation),
                "states": [
                    {
                        "state_label": cell.cell_id,
                        "cnp": list(_profile(cell)),
                    }
                    for cell in cells
                ],
            }
            for level_index, (generation, cells) in enumerate(
                zip(generations, selected_levels)
            )
        ],
    }
    validate_reconstruction_input(payload)
    return payload


def _distance_cells(
    selected_levels: Sequence[Sequence[Genotype]],
) -> list[Genotype]:
    occurrences = [cell for level in selected_levels for cell in level]
    return sorted(
        unique_cells_by_cell_id(occurrences),
        key=lambda cell: stable_distance_label_key(cell.cell_id),
    )


@dataclass(frozen=True)
class _PreparedCase:
    case_id_value: str
    input_payload: dict[str, Any]
    selected_levels: tuple[tuple[Genotype, ...], ...]
    distance_cells: tuple[Genotype, ...]
    metadata_base: dict[str, Any]


@dataclass(frozen=True)
class _DistanceTask:
    ordinal: int
    case_id_value: str
    distance_cells: tuple[Genotype, ...]
    runtime_config: CtbsRuntimeConfig | None
    distance_compute: DistanceCompute | None
    distance_timeout_seconds: int
    rss_limit_bytes: int


@dataclass(frozen=True)
class _DistanceOutcome:
    ordinal: int
    case_id_value: str
    distance: DistanceMatrix | None
    distance_runtime: dict[str, Any] | None
    failure: dict[str, Any] | None


def _prepare_case(
    *,
    truth_at_height,
    block_index: int,
    height: int,
    policy: str,
    generations: Sequence[int],
    placement_seed: int | None,
    sampled_by_generation: Mapping[int, tuple[list[Genotype], dict[str, Any]]],
    base_seed: int,
) -> _PreparedCase:
    case_id_value = case_id(block_index, height, policy)
    selected_levels = tuple(
        tuple(sampled_by_generation[int(value)][0]) for value in generations
    )
    sampling = [sampled_by_generation[int(value)][1] for value in generations]
    distance_cells = tuple(_distance_cells(selected_levels))
    if not distance_cells:
        raise ValueError(f"{case_id_value} has no unique observed states.")
    input_payload = _reconstruction_input(
        case_id_value,
        height=height,
        policy=policy,
        generations=generations,
        selected_levels=selected_levels,
    )
    occurrence_count = sum(len(level) for level in selected_levels)
    truth_diagnostics = truth_sampling_diagnostics(
        truth_at_height,
        selected_levels,
    )
    truth_diagnostics["normalized_minimum_invented_edge_fraction"] = (
        float(
            truth_diagnostics[
                "minimum_invented_edges_for_observed_only_arborescence"
            ]
        )
        / max(1, int(truth_diagnostics["selected_occurrence_count"]) - 1)
    )
    metadata_base = {
        "schema_version": CASE_METADATA_SCHEMA_VERSION,
        "case_id": case_id_value,
        "block_index": block_index,
        "height": int(height),
        "placement_policy": policy,
        "generations": [int(value) for value in generations],
        "placement_schedule_seed": placement_seed,
        "placement_schedule_rng": (
            "numpy_pcg64_without_replacement" if placement_seed is not None else None
        ),
        "sampling_rule": SAMPLING_RULE,
        "sampling": sampling,
        "reconstruction_seed": int(
            derived_seed("reconstruction", base_seed, block_index, int(height))
        ),
        "reconstruction_seed_scope": "truth_block_and_height_shared_across_policies",
        "canonical_input_order": "profile_then_stable_state_label",
        "selected_occurrence_count": occurrence_count,
        "selected_unique_state_count": len(distance_cells),
        "repeated_state_across_biopsy_occurrence_count": (
            occurrence_count - len(distance_cells)
        ),
        "truth_sampling_diagnostics": truth_diagnostics,
    }
    return _PreparedCase(
        case_id_value=case_id_value,
        input_payload=input_payload,
        selected_levels=selected_levels,
        distance_cells=distance_cells,
        metadata_base=metadata_base,
    )


def _execute_distance_task(task: _DistanceTask) -> _DistanceOutcome:
    try:
        compute = task.distance_compute
        if compute is None:
            if task.runtime_config is None:
                raise RuntimeError("Production distance task has no runtime config.")
            compute = Cnp2CnpFileDistanceProvider(task.runtime_config).compute
        distance, distance_runtime, distance_error = measured_stage(
            lambda: compute(task.distance_cells),
            timeout_seconds=task.distance_timeout_seconds,
            rss_limit_bytes=task.rss_limit_bytes,
        )
        if distance_error is None:
            distance_error = _resource_audit_error(
                distance_runtime,
                "distance computation",
            )
        if distance_error is not None:
            raise RuntimeError(
                f"Distance computation failed for {task.case_id_value}: "
                f"{distance_error}"
            ) from distance_error
        if distance is None:
            raise RuntimeError(
                f"Distance computation returned no value for {task.case_id_value}."
            )
        _validate_production_distance(distance)
    except Exception as error:
        return _DistanceOutcome(
            ordinal=task.ordinal,
            case_id_value=task.case_id_value,
            distance=None,
            distance_runtime=None,
            failure=_typed_error(error, "case_generation"),
        )
    return _DistanceOutcome(
        ordinal=task.ordinal,
        case_id_value=task.case_id_value,
        distance=distance,
        distance_runtime=distance_runtime,
        failure=None,
    )


def _run_distance_tasks(
    tasks: Sequence[_DistanceTask],
    *,
    block_index: int,
    requested_worker_count: int,
) -> tuple[list[_DistanceOutcome], dict[str, Any]]:
    start = time.perf_counter_ns()
    try:
        machine_cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        machine_cpu_count = requested_worker_count
    effective_worker_count = min(
        requested_worker_count,
        max(1, int(machine_cpu_count)),
        len(tasks),
    )
    if not tasks:
        scheduler = "no_distance_tasks"
        outcomes: list[_DistanceOutcome] = []
        task_submission_order = "none"
    else:
        # Start the largest matrices first so the H34/H38 tail does not leave
        # most workers idle after all small H14/H24 conditions finish.
        scheduled_tasks = sorted(
            tasks,
            key=lambda task: (-len(task.distance_cells), task.ordinal),
        )
        task_submission_order = (
            "descending_unique_profile_count_then_declared_ordinal"
        )
    if tasks and effective_worker_count == 1:
        if all(task.distance_compute is not None for task in scheduled_tasks):
            scheduler = "serial_inline_injected_test_condition"
            outcomes = [_execute_distance_task(task) for task in scheduled_tasks]
            worker_lifecycle = "inline_injected_test_double"
        else:
            scheduler = "spawn_process_pool_fresh_conditions"
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(processes=1, maxtasksperchild=1)
            try:
                outcomes = pool.map(
                    _execute_distance_task,
                    scheduled_tasks,
                    chunksize=1,
                )
                pool.close()
                pool.join()
            except BaseException:
                pool.terminate()
                pool.join()
                raise
            worker_lifecycle = "fresh_spawn_process_per_condition"
    elif tasks:
        scheduler = "spawn_process_pool_fresh_conditions"
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(
            processes=effective_worker_count,
            maxtasksperchild=1,
        )
        try:
            outcomes = pool.map(_execute_distance_task, scheduled_tasks, chunksize=1)
            pool.close()
            pool.join()
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        worker_lifecycle = "fresh_spawn_process_per_condition"
    else:
        worker_lifecycle = "no_worker"
    outcomes.sort(key=lambda outcome: outcome.ordinal)
    expected_ordinals = [task.ordinal for task in tasks]
    if [outcome.ordinal for outcome in outcomes] != expected_ordinals:
        raise RuntimeError("Parallel distance results changed declared task order.")
    return outcomes, {
        "schema_version": DISTANCE_EXECUTION_SCHEMA_VERSION,
        "block_index": int(block_index),
        "requested_worker_count": int(requested_worker_count),
        "effective_worker_count": int(effective_worker_count),
        "machine_cpu_count": int(machine_cpu_count),
        "distance_task_count": len(tasks),
        "scheduler": scheduler,
        "worker_unit": "condition",
        "worker_lifecycle": worker_lifecycle,
        "within_condition_order_execution": "sequential_forward_then_reverse",
        "task_submission_order": task_submission_order,
        "result_collection_order": "declared_condition_order",
        "distance_batch_wall_time_ns": time.perf_counter_ns() - start,
        "record_origin": "generated",
    }


def _finish_case(
    prepared: _PreparedCase,
    outcome: _DistanceOutcome,
    *,
    distance_execution: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], DistanceMatrix]:
    if outcome.failure is not None or outcome.distance is None:
        raise RuntimeError(f"Cannot finish failed case {prepared.case_id_value}.")
    if outcome.case_id_value != prepared.case_id_value:
        raise RuntimeError("Distance result case id changed during parallel execution.")
    if outcome.distance_runtime is None:
        raise RuntimeError("Successful distance result has no resource record.")
    distance = outcome.distance
    metadata = dict(prepared.metadata_base)
    candidate_diagnostics = candidate_graph_summary(
        prepared.selected_levels,
        distance,
        radii=(2, 4, 8),
    )
    metadata.update(
        {
            "distance_summary": distance_matrix_summary(distance),
            "distance_runtime": outcome.distance_runtime,
            "distance_execution": {
                "schema_version": DISTANCE_EXECUTION_SCHEMA_VERSION,
                "requested_worker_count": int(
                    distance_execution["requested_worker_count"]
                ),
                "effective_worker_count": int(
                    distance_execution["effective_worker_count"]
                ),
                "scheduler": str(distance_execution["scheduler"]),
                "worker_unit": "condition",
                "within_condition_order_execution": (
                    "sequential_forward_then_reverse"
                ),
                "condition_task_ordinal": int(outcome.ordinal),
            },
            "candidate_graph_diagnostics_r2_r4_r8": candidate_diagnostics,
            "scale_free_and_fixed_r4_ambiguity": ambiguity_case_summary(
                prepared.selected_levels,
                metadata["generations"],
                distance,
                fixed_radius=4,
                fixed_candidate_graph=candidate_diagnostics,
            ),
        }
    )
    return prepared.input_payload, metadata, distance


def _condition_declaration(
    *,
    block_index: int,
    height: int,
    policy: str,
    schedule: Sequence[int],
    placement_seed: int | None,
    status: str,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id(block_index, height, policy),
        "block_index": int(block_index),
        "height": int(height),
        "placement_policy": policy,
        "generations": [int(value) for value in schedule],
        "placement_schedule_seed": placement_seed,
        "status": status,
        "failure": None if failure is None else dict(failure),
    }


def _new_manifest(
    *,
    config_path: Path,
    simulator_regime_id: str,
    resolved_simulator_config: Mapping[str, Any],
    block_count: int,
    base_seed: int,
    heights: Sequence[int],
    policies: Sequence[str],
    technical_preflight: bool,
    nonproduction_full_factorial_smoke: bool,
    created_at_utc: str | None,
    resources: Mapping[str, int],
) -> dict[str, Any]:
    simulator_overrides = deepcopy(
        SIMULATOR_OVERRIDES_BY_REGIME[simulator_regime_id]
    )
    if technical_preflight:
        scientific_role = "resource_preflight_not_accuracy_evidence"
        contract_mode = PREFLIGHT_CONTRACT_MODE_BY_SIMULATOR_REGIME[
            simulator_regime_id
        ]
    elif nonproduction_full_factorial_smoke:
        scientific_role = "nonproduction_technical_smoke"
        contract_mode = SMOKE_CONTRACT_MODE_BY_SIMULATOR_REGIME[
            simulator_regime_id
        ]
    else:
        scientific_role = PRODUCTION_SCIENTIFIC_ROLE_BY_SIMULATOR_REGIME[
            simulator_regime_id
        ]
        contract_mode = PRODUCTION_CONTRACT_MODE_BY_SIMULATOR_REGIME[
            simulator_regime_id
        ]
    return {
        "schema_version": BANK_SCHEMA_VERSION,
        "bank_id": BANK_ID_BY_SIMULATOR_REGIME[simulator_regime_id],
        "status": "in_progress",
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "scientific_role": scientific_role,
        "contract_mode": contract_mode,
        "simulator_regime_id": simulator_regime_id,
        "simulator_overrides": simulator_overrides,
        "resolved_simulator_config_sha256": canonical_json_digest(
            resolved_simulator_config
        ),
        "paired_seed_reference_bank_id": (
            None
            if simulator_regime_id == BASELINE_SIMULATOR_REGIME
            else DEFAULT_BANK_ID
        ),
        "paired_seed_semantics": (
            "reference_regime"
            if simulator_regime_id == BASELINE_SIMULATOR_REGIME
            else "same_coordinate_seed_map_changed_simulator_parameter"
        ),
        "base_seed": int(base_seed),
        "seed_namespace": SEED_NAMESPACE,
        "base_config_path": str(config_path),
        "base_config_sha256": _file_sha256(config_path),
        "simulator_config_path": BANK_CONFIG_NAME,
        "block_count": int(block_count),
        "heights": [int(value) for value in heights],
        "placement_policies": list(policies),
        "simulation_height": max(heights),
        "declared_condition_count": int(block_count * len(heights) * len(policies)),
        "sampling_rule": SAMPLING_RULE,
        "shortlist_arm_ids": list(SHORTLIST_ARM_IDS),
        "v2_reproduction_arm_ids": list(V2_COMPLETE_ARM_IDS),
        "selected_algorithm_arm_ids": list(SELECTED_V2_ARM_IDS),
        "independent_unit": "truth_block",
        "resource_contract": dict(resources),
        "distance_execution_semantics": DISTANCE_EXECUTION_SEMANTICS,
        "distance_execution_by_block": [],
        "completed_block_count": 0,
        "completed_condition_count": 0,
        "available_condition_count": 0,
        "unavailable_condition_count": 0,
        "condition_inventory": [],
        "cases": [],
        "failures": [],
    }


def _validate_distance_execution_prefix(
    manifest: Mapping[str, Any],
    *,
    completed_blocks: int,
) -> list[dict[str, Any]] | None:
    records = manifest.get("distance_execution_by_block")
    semantics = manifest.get("distance_execution_semantics")
    if records is None and semantics is None:
        return None
    if semantics != DISTANCE_EXECUTION_SEMANTICS:
        raise ValueError("Stored distance-execution semantics changed.")
    if not isinstance(records, list) or len(records) != completed_blocks:
        raise ValueError("Stored distance-execution history is not a block prefix.")
    normalized: list[dict[str, Any]] = []
    for block_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("Stored distance-execution record is invalid.")
        if (
            record.get("schema_version") != DISTANCE_EXECUTION_SCHEMA_VERSION
            or record.get("block_index") != block_index
        ):
            raise ValueError("Stored distance-execution block order changed.")
        requested = record.get("requested_worker_count")
        validate_positive_integer(requested, "stored requested_worker_count")
        effective = record.get("effective_worker_count")
        if (
            isinstance(effective, bool)
            or not isinstance(effective, int)
            or not 0 <= effective <= requested
        ):
            raise ValueError("Stored effective_worker_count is invalid.")
        normalized.append(dict(record))
    return normalized


def _inferred_serial_execution_prefix(
    completed_blocks: int,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": DISTANCE_EXECUTION_SCHEMA_VERSION,
            "block_index": block_index,
            "requested_worker_count": 1,
            "effective_worker_count": 1,
            "machine_cpu_count": None,
            "distance_task_count": None,
            "scheduler": "serial_condition",
            "worker_unit": "condition",
            "within_condition_order_execution": "sequential_forward_then_reverse",
            "result_collection_order": "declared_condition_order",
            "distance_batch_wall_time_ns": None,
            "record_origin": "inferred_pre_parallel_option_serial_prefix",
        }
        for block_index in range(completed_blocks)
    ]


def _resume_manifest(
    *,
    output_root: Path | str,
    expected: Mapping[str, Any],
    requested_distance_workers: int,
) -> tuple[Path, dict[str, Any], int]:
    root = Path(output_root).expanduser().resolve()
    path = root / BANK_MANIFEST_NAME
    if not root.is_dir() or not path.is_file():
        raise ValueError("A resumed shortlist bank requires an existing manifest.")
    manifest = read_json(path)
    for field in (
        "schema_version",
        "bank_id",
        "contract_mode",
        "simulator_regime_id",
        "simulator_overrides",
        "resolved_simulator_config_sha256",
        "paired_seed_reference_bank_id",
        "paired_seed_semantics",
        "base_seed",
        "seed_namespace",
        "base_config_sha256",
        "block_count",
        "heights",
        "placement_policies",
        "simulation_height",
        "declared_condition_count",
        "selected_algorithm_arm_ids",
        "resource_contract",
    ):
        if manifest.get(field) != expected.get(field):
            raise ValueError(
                f"Cannot resume because stored {field} does not match this command."
            )
    stored_config_path = root / BANK_CONFIG_NAME
    if not stored_config_path.is_file() or canonical_json_digest(
        read_json(stored_config_path)
    ) != expected.get("resolved_simulator_config_sha256"):
        raise ValueError(
            "Cannot resume because the resolved simulator config changed."
        )
    if manifest.get("status") == "complete":
        raise ValueError("A completed shortlist bank cannot be resumed.")
    if manifest.get("status") not in {"in_progress", "failure"}:
        raise ValueError("Only an interrupted or failed shortlist bank can resume.")
    completed_blocks = int(manifest.get("completed_block_count", -1))
    if not 0 <= completed_blocks <= int(manifest["block_count"]):
        raise ValueError("Stored shortlist completed-block count is invalid.")
    per_block = len(manifest["heights"]) * len(manifest["placement_policies"])
    inventory = manifest.get("condition_inventory")
    if not isinstance(inventory, list) or len(inventory) != completed_blocks * per_block:
        raise ValueError("Stored shortlist condition inventory is not a block prefix.")
    distance_execution = _validate_distance_execution_prefix(
        manifest,
        completed_blocks=completed_blocks,
    )
    execution_prefix_was_inferred = distance_execution is None
    if distance_execution is None:
        distance_execution = _inferred_serial_execution_prefix(completed_blocks)
        manifest["distance_execution_semantics"] = DISTANCE_EXECUTION_SEMANTICS
        manifest["distance_execution_by_block"] = distance_execution
    prior_status = manifest["status"]
    history = manifest.setdefault("resume_history", [])
    if not isinstance(history, list):
        raise ValueError("Stored shortlist resume history is invalid.")
    history.append(
        {
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_status": prior_status,
            "preserved_block_count": completed_blocks,
            "preserved_execution_prefix_was_inferred": (
                execution_prefix_was_inferred
            ),
            "previous_distance_worker_count": (
                distance_execution[-1]["requested_worker_count"]
                if distance_execution
                else None
            ),
            "requested_distance_worker_count": int(requested_distance_workers),
            "previous_runner_failure": manifest.pop("runner_failure", None),
        }
    )
    manifest["status"] = "in_progress"
    manifest.pop("completed_at_utc", None)
    write_json(path, manifest)
    return root, manifest, completed_blocks


def generate_bank(
    *,
    output_root: Path | str,
    base_config_path: Path | str = DEFAULT_BASE_CONFIG,
    simulator_regime_id: str = BASELINE_SIMULATOR_REGIME,
    base_seed: int = DEFAULT_BASE_SEED,
    block_count: int = DEFAULT_BLOCK_COUNT,
    heights: Sequence[int] = HEIGHTS,
    placement_policies: Sequence[str] = PLACEMENT_POLICIES,
    simulation_timeout_seconds: int = DEFAULT_SIMULATION_TIMEOUT_SECONDS,
    distance_timeout_seconds: int = DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    cnp2cnp_process_timeout_seconds: int = DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    distance_workers: int = DEFAULT_DISTANCE_WORKERS,
    distance_compute: DistanceCompute | None = None,
    simulator_factory: SimulatorFactory | None = None,
    created_at_utc: str | None = None,
    progress: bool = False,
    technical_preflight: bool = False,
    allow_nonproduction_size: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate every declared input once; failures remain typed and unreplaced."""
    simulator_regime_id = str(simulator_regime_id)
    if simulator_regime_id not in SIMULATOR_REGIMES:
        raise ValueError(
            "simulator_regime_id must be one of "
            f"{SIMULATOR_REGIMES}."
        )
    regime_heights = HEIGHTS_BY_SIMULATOR_REGIME[simulator_regime_id]
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer.")
    validate_positive_integer(block_count, "block_count")
    normalized_heights = tuple(int(value) for value in heights)
    normalized_policies = tuple(str(value) for value in placement_policies)
    if (
        not normalized_heights
        or tuple(sorted(set(normalized_heights))) != normalized_heights
        or any(value not in regime_heights for value in normalized_heights)
    ):
        raise ValueError(
            f"heights must be an ordered subset of {regime_heights} for "
            f"{simulator_regime_id}."
        )
    if (
        not normalized_policies
        or len(set(normalized_policies)) != len(normalized_policies)
        or any(value not in PLACEMENT_POLICIES for value in normalized_policies)
    ):
        raise ValueError(
            f"placement_policies must be a unique subset of {PLACEMENT_POLICIES}."
        )
    if not technical_preflight and (
        (block_count != DEFAULT_BLOCK_COUNT and not allow_nonproduction_size)
        or normalized_heights != regime_heights
        or normalized_policies != PLACEMENT_POLICIES
    ):
        raise ValueError(
            "The production shortlist bank requires exactly 100 blocks, "
            f"heights {regime_heights}, and spread/late/random policies."
        )
    if technical_preflight and not (
        normalized_heights == (regime_heights[-1],)
        and normalized_policies == ("late",)
    ):
        raise ValueError(
            "The technical resource preflight is fixed to "
            f"H{regime_heights[-1]}-late for {simulator_regime_id}."
        )
    for field, value in (
        ("simulation_timeout_seconds", simulation_timeout_seconds),
        ("distance_timeout_seconds", distance_timeout_seconds),
        ("cnp2cnp_process_timeout_seconds", cnp2cnp_process_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("capture_limit_bytes", capture_limit_bytes),
        ("distance_workers", distance_workers),
    ):
        validate_positive_integer(value, field)
    if distance_workers > MAX_DISTANCE_WORKERS:
        raise ValueError(
            f"distance_workers may not exceed {MAX_DISTANCE_WORKERS}."
        )
    if distance_workers > 1 and distance_compute is not None:
        try:
            pickle.dumps(distance_compute)
        except (pickle.PickleError, TypeError, AttributeError) as error:
            raise ValueError(
                "A parallel injected distance_compute must be picklable."
            ) from error

    config_path = Path(base_config_path).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"Base simulator config is not a file: {config_path}.")
    base_config = read_json(config_path)
    _validate_standard_base_config(base_config)
    simulation_config = dict(base_config)
    simulation_config["CNA_EVENT_PROBABILITY"] = (
        CNA_EVENT_PROBABILITY_BY_SIMULATOR_REGIME[simulator_regime_id]
    )
    simulation_config.update(
        deepcopy(SIMULATOR_OVERRIDES_BY_REGIME[simulator_regime_id])
    )
    simulation_config["NUMBER_OF_GENERATIONS"] = max(normalized_heights)
    load_simulator_inputs(simulation_config)
    resource_contract = {
        "simulation_timeout_seconds_per_block": int(simulation_timeout_seconds),
        "distance_timeout_seconds_per_condition": int(distance_timeout_seconds),
        "cnp2cnp_process_timeout_seconds": int(cnp2cnp_process_timeout_seconds),
        "rss_limit_bytes_per_stage": int(rss_limit_bytes),
        "capture_limit_bytes": int(capture_limit_bytes),
        "simulation_execution": (
            fresh_process_contract(TRUTH_BLOCK_SIMULATION_WORKER_UNIT)
            if simulator_factory is None
            else {
                "isolation": "inline_injected_test_double",
                "worker_unit": TRUTH_BLOCK_SIMULATION_WORKER_UNIT,
            }
        ),
        "production_distance_worker_lifecycle": (
            "fresh_spawn_process_per_condition"
        ),
    }
    expected = _new_manifest(
        config_path=config_path,
        simulator_regime_id=simulator_regime_id,
        resolved_simulator_config=simulation_config,
        block_count=block_count,
        base_seed=base_seed,
        heights=normalized_heights,
        policies=normalized_policies,
        technical_preflight=technical_preflight,
        nonproduction_full_factorial_smoke=(
            allow_nonproduction_size and not technical_preflight
        ),
        created_at_utc=created_at_utc,
        resources=resource_contract,
    )
    if resume:
        root, manifest, start_block = _resume_manifest(
            output_root=output_root,
            expected=expected,
            requested_distance_workers=distance_workers,
        )
    else:
        root = ensure_new_output_root(output_root)
        write_json(root / BANK_CONFIG_NAME, simulation_config)
        manifest = expected
        start_block = 0
        write_json(root / BANK_MANIFEST_NAME, manifest)

    if distance_compute is None:
        runtime_config = replace(
            load_ctbs_runtime_config(),
            cnp2cnp_timeout_seconds=float(cnp2cnp_process_timeout_seconds),
            cnp2cnp_capture_limit_bytes=int(capture_limit_bytes),
        )
    else:
        runtime_config = None
    factory = simulator_factory or _default_simulator_factory

    try:
        for block_index in range(start_block, block_count):
            if progress:
                print(
                    f"simulating block {block_index + 1}/{block_count}",
                    file=sys.stderr,
                    flush=True,
                )
            schedules = {
                (height, policy): placement_schedule(
                    policy,
                    height=height,
                    base_seed=base_seed,
                    block_index=block_index,
                )
                for height in normalized_heights
                for policy in normalized_policies
            }
            local_inventory: list[dict[str, Any]] = []
            local_cases: list[dict[str, Any]] = []
            block_distance_execution: dict[str, Any] | None = None
            simulation_seed = derived_seed("simulation", base_seed, block_index)
            if simulator_factory is None:
                with FreshSpawnPerTaskExecutor() as simulation_executor:
                    (
                        simulation_value,
                        simulation_runtime,
                        simulation_error,
                    ) = simulation_executor.run(
                        _execute_isolated_simulation_stage,
                        dict(simulation_config),
                        simulation_seed,
                        simulation_timeout_seconds,
                        rss_limit_bytes,
                        timeout_seconds=(
                            simulation_timeout_seconds
                            + SIMULATION_WORKER_GRACE_SECONDS
                        ),
                    )
            else:
                simulation_value, simulation_runtime, simulation_error = (
                    measured_stage(
                        lambda: _simulate_preserving_guard_outcome(
                            factory,
                            simulation_config,
                            simulation_seed,
                        ),
                        timeout_seconds=simulation_timeout_seconds,
                        rss_limit_bytes=rss_limit_bytes,
                    )
                )
            if simulation_error is None:
                simulation_error = _resource_audit_error(
                    simulation_runtime,
                    "simulation",
                )
            if simulation_error is not None or simulation_value is None:
                failure = _typed_error(
                    simulation_error or RuntimeError("Simulation returned no value."),
                    "simulation",
                )
                for (height, policy), (schedule, placement_seed) in schedules.items():
                    local_inventory.append(
                        _condition_declaration(
                            block_index=block_index,
                            height=height,
                            policy=policy,
                            schedule=schedule,
                            placement_seed=placement_seed,
                            status="unavailable",
                            failure=failure,
                        )
                    )
                manifest["failures"].append(
                    {"block_index": block_index, **failure}
                )
                _outcomes, block_distance_execution = _run_distance_tasks(
                    [],
                    block_index=block_index,
                    requested_worker_count=distance_workers,
                )
            else:
                simulator, guard_failure = simulation_value
                truth_tree = simulator.canonicalized_tree_by_genome()
                truth_relative = truth_path(block_index)
                write_json(
                    root / truth_relative,
                    serialize_truth(block_index, truth_tree),
                )
                failure_generation = None
                if guard_failure is not None:
                    outcome = guard_failure.get("simulation_outcome")
                    if isinstance(outcome, Mapping):
                        failure_generation = outcome.get("failure_generation")
                    manifest["failures"].append(
                        {"block_index": block_index, **guard_failure}
                    )

                usable_heights = [
                    int(height)
                    for height in normalized_heights
                    if failure_generation is None
                    or int(height) < int(failure_generation)
                ]
                truth_prefix_by_height = {}
                maximum_truth_generation = max(
                    int(attributes.get("generation", -1))
                    for _node, attributes in truth_tree.nodes(data=True)
                )
                for height in usable_heights:
                    if height >= maximum_truth_generation:
                        truth_prefix_by_height[height] = truth_tree
                    else:
                        prefix_nodes = [
                            node
                            for node, attributes in truth_tree.nodes(data=True)
                            if int(attributes.get("generation", -1)) <= height
                        ]
                        # Diagnostics are read-only; a view avoids copying the
                        # same truth prefix once for every placement policy.
                        truth_prefix_by_height[height] = truth_tree.subgraph(
                            prefix_nodes
                        )

                required_generations = sorted(
                    {
                        int(generation)
                        for (height, _policy), (schedule, _seed) in schedules.items()
                        if failure_generation is None or height < int(failure_generation)
                        for generation in schedule
                    }
                )
                sampled_by_generation: dict[
                    int, tuple[list[Genotype], dict[str, Any]]
                ] = {}
                sampling_failures: dict[int, dict[str, Any]] = {}
                for generation in required_generations:
                    try:
                        sampled_by_generation[generation] = _sample_generation(
                            simulator,
                            generation,
                            base_seed=base_seed,
                            block_index=block_index,
                        )
                    except Exception as error:
                        sampling_failures[generation] = _typed_error(error, "sampling")

                condition_slots: list[dict[str, Any]] = []
                prepared_cases: list[_PreparedCase] = []
                preparation_start = time.perf_counter_ns()
                if progress:
                    print(
                        f"preparing block {block_index + 1}/{block_count}: "
                        f"{len(schedules)} condition diagnostics",
                        file=sys.stderr,
                        flush=True,
                    )
                for (height, policy), (schedule, placement_seed) in schedules.items():
                    declaration = _condition_declaration(
                        block_index=block_index,
                        height=height,
                        policy=policy,
                        schedule=schedule,
                        placement_seed=placement_seed,
                        status="available",
                    )
                    if failure_generation is not None and height >= int(failure_generation):
                        declaration["status"] = "unavailable"
                        declaration["failure"] = dict(guard_failure or {})
                        condition_slots.append(
                            {"declaration": declaration, "prepared": None}
                        )
                        continue
                    missing = [
                        generation
                        for generation in schedule
                        if int(generation) in sampling_failures
                    ]
                    if missing:
                        declaration["status"] = "unavailable"
                        declaration["failure"] = {
                            "stage": "sampling",
                            "type": "UnavailableBiopsyGeneration",
                            "message": f"Unavailable generations: {missing}.",
                            "generation_failures": {
                                str(value): sampling_failures[int(value)]
                                for value in missing
                            },
                        }
                        condition_slots.append(
                            {"declaration": declaration, "prepared": None}
                        )
                        continue
                    try:
                        prepared = _prepare_case(
                            truth_at_height=truth_prefix_by_height[int(height)],
                            block_index=block_index,
                            height=height,
                            policy=policy,
                            generations=schedule,
                            placement_seed=placement_seed,
                            sampled_by_generation=sampled_by_generation,
                            base_seed=base_seed,
                        )
                    except Exception as error:
                        declaration["status"] = "unavailable"
                        declaration["failure"] = _typed_error(error, "case_generation")
                        condition_slots.append(
                            {"declaration": declaration, "prepared": None}
                        )
                        continue
                    prepared_cases.append(prepared)
                    condition_slots.append(
                        {"declaration": declaration, "prepared": prepared}
                    )

                if progress:
                    print(
                        f"prepared block {block_index + 1}/{block_count}: "
                        f"{len(prepared_cases)}/{len(schedules)} conditions in "
                        f"{(time.perf_counter_ns() - preparation_start) / 1e9:.3f}s",
                        file=sys.stderr,
                        flush=True,
                    )

                simulator_event_totals = _event_totals(simulator)
                del simulator, truth_tree, truth_prefix_by_height
                gc.collect()

                distance_tasks = [
                    _DistanceTask(
                        ordinal=ordinal,
                        case_id_value=prepared.case_id_value,
                        distance_cells=prepared.distance_cells,
                        runtime_config=runtime_config,
                        distance_compute=distance_compute,
                        distance_timeout_seconds=distance_timeout_seconds,
                        rss_limit_bytes=rss_limit_bytes,
                    )
                    for ordinal, prepared in enumerate(prepared_cases)
                ]
                if progress and distance_tasks:
                    print(
                        f"computing block {block_index + 1}/{block_count}: "
                        f"{len(distance_tasks)} distances with up to "
                        f"{distance_workers} workers",
                        file=sys.stderr,
                        flush=True,
                    )
                distance_outcomes, block_distance_execution = _run_distance_tasks(
                    distance_tasks,
                    block_index=block_index,
                    requested_worker_count=distance_workers,
                )
                outcome_by_case = {
                    outcome.case_id_value: outcome for outcome in distance_outcomes
                }
                if len(outcome_by_case) != len(distance_outcomes):
                    raise RuntimeError("Parallel distance results duplicated a case id.")
                if progress and condition_slots:
                    print(
                        f"summarizing block {block_index + 1}/{block_count}: "
                        f"candidate and ambiguity diagnostics",
                        file=sys.stderr,
                        flush=True,
                    )

                for slot in condition_slots:
                    declaration = slot["declaration"]
                    prepared = slot["prepared"]
                    if prepared is None:
                        local_inventory.append(declaration)
                        continue
                    outcome = outcome_by_case.get(prepared.case_id_value)
                    if outcome is None:
                        raise RuntimeError(
                            f"Missing distance result for {prepared.case_id_value}."
                        )
                    if outcome.failure is not None:
                        declaration["status"] = "unavailable"
                        declaration["failure"] = dict(outcome.failure)
                        local_inventory.append(declaration)
                        continue
                    try:
                        input_payload, metadata, distance = _finish_case(
                            prepared,
                            outcome,
                            distance_execution=block_distance_execution,
                        )
                    except Exception as error:
                        declaration["status"] = "unavailable"
                        declaration["failure"] = _typed_error(
                            error,
                            "case_generation",
                        )
                        local_inventory.append(declaration)
                        continue
                    case_height = int(declaration["height"])
                    case_policy = str(declaration["placement_policy"])
                    paths = condition_paths(block_index, case_height, case_policy)
                    metadata.update(
                        {
                            "simulation_seed": int(simulation_seed),
                            "simulation_runtime": simulation_runtime,
                            "simulation_guard_failure": guard_failure,
                            "simulator_event_totals": simulator_event_totals,
                        }
                    )
                    write_json(root / paths["input"], input_payload)
                    write_json(
                        root / paths["distance"],
                        serialize_distance(prepared.case_id_value, distance),
                    )
                    write_json(root / paths["metadata"], metadata)
                    case_record = {
                        **declaration,
                        "metadata_path": paths["metadata"],
                        "input_path": paths["input"],
                        "distance_path": paths["distance"],
                        "truth_path": truth_relative,
                    }
                    local_cases.append(case_record)
                    local_inventory.append(declaration)
                gc.collect()

            if block_distance_execution is None:
                raise RuntimeError("Block has no distance-execution record.")
            execution_history = manifest.get("distance_execution_by_block")
            if (
                not isinstance(execution_history, list)
                or len(execution_history) != block_index
            ):
                raise RuntimeError("Distance-execution history lost block-prefix order.")
            execution_history.append(block_distance_execution)
            manifest["condition_inventory"].extend(local_inventory)
            manifest["cases"].extend(local_cases)
            manifest["completed_block_count"] = block_index + 1
            manifest["available_condition_count"] = len(manifest["cases"])
            manifest["unavailable_condition_count"] = (
                len(manifest["condition_inventory"]) - len(manifest["cases"])
            )
            manifest["completed_condition_count"] = len(
                manifest["condition_inventory"]
            )
            write_json(root / BANK_MANIFEST_NAME, manifest)
            if progress:
                print(
                    f"completed block {block_index + 1}/{block_count}: "
                    f"{len(local_cases)}/{len(local_inventory)} conditions available",
                    file=sys.stderr,
                    flush=True,
                )
    except BaseException as error:
        manifest["status"] = "failure"
        manifest["runner_failure"] = _typed_error(error, "bank_generator")
        write_json(root / BANK_MANIFEST_NAME, manifest)
        raise

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / BANK_MANIFEST_NAME, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--simulator-regime",
        dest="simulator_regime_id",
        choices=SIMULATOR_REGIMES,
        default=BASELINE_SIMULATOR_REGIME,
        help=(
            "Frozen baseline or paired sensitivity regime; each sensitivity "
            "retains the baseline coordinate-derived seed map."
        ),
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--block-count", type=int, default=DEFAULT_BLOCK_COUNT)
    parser.add_argument(
        "--heights",
        type=int,
        nargs="+",
        help="Defaults to the complete height set frozen for the selected regime.",
    )
    parser.add_argument(
        "--placement-policies",
        nargs="+",
        default=list(PLACEMENT_POLICIES),
        choices=PLACEMENT_POLICIES,
    )
    parser.add_argument(
        "--simulation-timeout-seconds",
        type=int,
        default=DEFAULT_SIMULATION_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--distance-timeout-seconds",
        type=int,
        default=DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--cnp2cnp-process-timeout-seconds",
        type=int,
        default=DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument(
        "--capture-limit-bytes",
        type=int,
        default=DEFAULT_CAPTURE_LIMIT_BYTES,
    )
    parser.add_argument(
        "--distance-workers",
        type=int,
        default=DEFAULT_DISTANCE_WORKERS,
        help=(
            "Maximum concurrent condition-level distance workers; each worker "
            "retains sequential forward/reverse cnp2cnp execution."
        ),
    )
    parser.add_argument("--technical-preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = generate_bank(
        output_root=arguments.output_root,
        base_config_path=arguments.base_config,
        simulator_regime_id=arguments.simulator_regime_id,
        base_seed=arguments.base_seed,
        block_count=arguments.block_count,
        heights=(
            arguments.heights
            if arguments.heights is not None
            else HEIGHTS_BY_SIMULATOR_REGIME[arguments.simulator_regime_id]
        ),
        placement_policies=arguments.placement_policies,
        simulation_timeout_seconds=arguments.simulation_timeout_seconds,
        distance_timeout_seconds=arguments.distance_timeout_seconds,
        cnp2cnp_process_timeout_seconds=arguments.cnp2cnp_process_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        capture_limit_bytes=arguments.capture_limit_bytes,
        distance_workers=arguments.distance_workers,
        technical_preflight=arguments.technical_preflight,
        resume=arguments.resume,
        progress=arguments.progress,
    )
    print(
        "complete: "
        f"{manifest['available_condition_count']}/"
        f"{manifest['declared_condition_count']} conditions available"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
