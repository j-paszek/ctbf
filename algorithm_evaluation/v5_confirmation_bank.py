"""Generate the prospectively locked CTBF v5 confirmation input bank."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from algorithm_evaluation.paper_pipeline_contract import (
    aligned_distance_submatrix,
    file_sha256,
    json_safe,
    read_json,
)
from algorithm_evaluation.paper_pipeline_runner import (
    measured_stage,
    serialize_tree,
    validate_reconstruction_input,
)
from algorithm_evaluation.process_isolation import (
    CONDITION_DISTANCE_WORKER_UNIT,
    FreshSpawnTaskPool,
    TRUTH_BLOCK_SIMULATION_WORKER_UNIT,
    fresh_process_contract,
)
from algorithm_evaluation.simulator_reconstruction_intuition_probe import (
    _canonical_cells_at_generation,
    _event_totals,
    _profile,
    truth_sampling_diagnostics,
)
from algorithm_evaluation.v5_algorithm_development_common import truth_prefix
from algorithm_evaluation.v5_confirmation_common import (
    BANK_CHECKSUM_NAME,
    BANK_MANIFEST_NAME,
    BANK_SCHEMA_VERSION,
    BIOPSY_LOWER_BOUND,
    BLOCK_COUNT,
    CONDITIONS,
    CONFIRMATION_NAMESPACE,
    DEFAULT_MANIFEST_PATH,
    OWNER_APPROVAL_PATH,
    PAPER_ARM_IDS,
    PROJECT_ROOT,
    SAMPLING_RULE,
    SIMULATION_HEIGHT,
    SIMULATOR_CONFIG_NAME,
    TARGET_FRACTION,
    condition_paths,
    confirmation_case_id,
    derive_seed_for_namespace,
    distance_path,
    ensure_new_output_root,
    hybrid_sample_size,
    resolve_schedule,
    truth_path,
    validate_checksum_closure,
    validate_cnp2cnp_source,
    validate_internal_runtime,
    validate_manifest,
    write_checksum_closure,
    write_json,
)
from ctbs import (
    Cnp2CnpFileDistanceProvider,
    CtbsRuntimeConfig,
    DistanceMatrix,
    load_ctbs_runtime_config,
)
from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    stable_distance_label_key,
    validate_distance_label_coverage,
    validate_distance_matrix,
)
from simulator import (
    CancerCellEvolutionSimulator,
    Genotype,
    SimulationResourceLimitExceeded,
)
from simulator_config import load_simulator_inputs


SMOKE_NAMESPACE = "ctbf-v5-g2-01a-technical-smoke-v1"
OWNER_APPROVAL_SNAPSHOT_NAME = "owner_approval.snapshot.py"
TRUTH_SCHEMA_VERSION = "ctbf-v5-confirmatory-truth-v1"
BLOCK_INPUT_SCHEMA_VERSION = "ctbf-v5-confirmatory-block-input-v1"
CASE_SCHEMA_VERSION = "ctbf-v5-confirmatory-case-v1"
DISTANCE_SCHEMA_VERSION = "ctbf-v5-confirmatory-distance-v1"
DEFAULT_SIMULATION_TIMEOUT_SECONDS = 300
DEFAULT_DISTANCE_TIMEOUT_SECONDS = 1200
DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS = 540
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = 1_048_576
DEFAULT_SIMULATION_WORKERS = 4
DEFAULT_DISTANCE_WORKERS = 6
MAX_SIMULATION_WORKERS = 8
MAX_DISTANCE_WORKERS = 6
WORKER_GRACE_SECONDS = 60

DistanceCompute = Callable[[Sequence[Genotype]], DistanceMatrix]
SimulatorFactory = Callable[[Mapping[str, Any], int], CancerCellEvolutionSimulator]


def _typed_error(error: BaseException, stage: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error)[:4096],
    }
    if isinstance(error, SimulationResourceLimitExceeded):
        record["resource_limit"] = error.as_dict()
        record["simulation_outcome"] = getattr(error, "simulation_outcome", None)
    return record


def _resource_audit_error(resources: Mapping[str, Any], stage: str) -> RuntimeError | None:
    memory = resources.get("memory")
    if not isinstance(memory, Mapping) or memory.get("peak_rss_bytes") is None:
        return RuntimeError(f"{stage} has no auditable process-tree peak RSS.")
    return None


def _run_simulator(config: Mapping[str, Any], seed: int):
    simulator = CancerCellEvolutionSimulator(config, seed=seed)
    try:
        simulator.run_simulation()
    except SimulationResourceLimitExceeded as error:
        outcome = getattr(error, "simulation_outcome", None)
        if not isinstance(outcome, Mapping):
            outcome = simulator.tree.graph.get("simulation_outcome")
        return simulator, _typed_error(error, "simulation_resource_guard"), outcome
    return simulator, None, simulator.tree.graph.get("simulation_outcome")


def _sample_generation(
    simulator: CancerCellEvolutionSimulator,
    generation: int,
    *,
    namespace: str,
    block_index: int,
) -> tuple[list[Genotype], dict[str, Any]]:
    available = _canonical_cells_at_generation(simulator, generation)
    if not available:
        raise ValueError(f"Required generation {generation} is empty.")
    realized = hybrid_sample_size(len(available))
    seed = derive_seed_for_namespace(
        namespace,
        "sampling",
        block_index,
        generation,
    )
    if realized == len(available):
        chosen = list(available)
        selection_mode = "all_available"
        seed_used = False
    else:
        permutation = np.random.Generator(np.random.PCG64(seed)).permutation(
            len(available)
        )
        chosen = [available[int(index)] for index in permutation[:realized]]
        selection_mode = "seeded_without_replacement"
        seed_used = True
    chosen.sort(
        key=lambda cell: (_profile(cell), stable_distance_label_key(cell.cell_id))
    )
    return chosen, {
        "generation": generation,
        "available_distinct_state_count": len(available),
        "target_fraction": TARGET_FRACTION,
        "fractional_count": int(math.ceil(TARGET_FRACTION * len(available))),
        "biopsy_lower_bound": BIOPSY_LOWER_BOUND,
        "realized_occurrence_count": len(chosen),
        "selection_mode": selection_mode,
        "sampling_seed": int(seed),
        "sampling_seed_used": seed_used,
        "seed_scope": "namespace_truth_block_and_generation",
    }


def _case_input(
    *,
    case_id: str,
    condition_id: str,
    condition_family: str,
    schedule: Sequence[int],
    selected_levels: Sequence[Sequence[Genotype]],
) -> dict[str, Any]:
    payload = {
        "schema_version": "ctbf-v5-reconstruction-input-v1",
        "case_id": case_id,
        "condition_id": condition_id,
        "condition_family": condition_family,
        "sampling_rule": SAMPLING_RULE,
        "levels": [
            {
                "biopsy_level": level_index,
                "generation": int(generation),
                "states": [
                    {
                        "state_label": json_safe(cell.cell_id),
                        "cnp": [int(value) for value in _profile(cell)],
                    }
                    for cell in cells
                ],
            }
            for level_index, (generation, cells) in enumerate(
                zip(schedule, selected_levels)
            )
        ],
    }
    validate_reconstruction_input(payload)
    return payload


def _maximum_states(
    selected_by_generation: Mapping[int, Sequence[Genotype]],
) -> list[dict[str, Any]]:
    by_label: dict[Any, tuple[int, ...]] = {}
    for cells in selected_by_generation.values():
        for cell in cells:
            profile = _profile(cell)
            previous = by_label.setdefault(cell.cell_id, profile)
            if previous != profile:
                raise ValueError("One state label maps to multiple CNPs.")
    return [
        {"state_label": json_safe(label), "cnp": list(by_label[label])}
        for label in sorted(by_label, key=stable_distance_label_key)
    ]


def _simulate_block_task(
    output_root: str,
    config: Mapping[str, Any],
    block_index: int,
    namespace: str,
    timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, Any]:
    root = Path(output_root)
    simulation_seed = derive_seed_for_namespace(
        namespace,
        "simulation",
        block_index,
    )
    result, resources, error = measured_stage(
        lambda: _run_simulator(config, simulation_seed),
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if error is None:
        error = _resource_audit_error(resources, "simulation")
    if error is not None or result is None:
        return {
            "block_index": block_index,
            "status": "failure",
            "failure": _typed_error(
                error or RuntimeError("Simulation returned no result."),
                "simulation",
            ),
            "simulation_seed": int(simulation_seed),
            "resources": resources,
            "condition_inventory": [],
            "case_inventory": [],
        }

    simulator, guard_failure, simulation_outcome = result
    truth = simulator.canonicalized_tree_by_genome()
    maximum_generation = max(
        int(attributes.get("generation", -1))
        for _node, attributes in truth.nodes(data=True)
    )
    failure_generation = None
    if isinstance(simulation_outcome, Mapping):
        failure_generation = simulation_outcome.get("failure_generation")

    resolved_conditions = []
    required_generations: set[int] = set()
    for ordinal, condition in enumerate(CONDITIONS):
        schedule, schedule_seed = resolve_schedule(
            condition,
            block_index,
            namespace=namespace,
        )
        resolved_conditions.append((ordinal, condition, schedule, schedule_seed))
        if failure_generation is None or condition.height < int(failure_generation):
            required_generations.update(schedule)

    sampled_by_generation: dict[int, tuple[list[Genotype], dict[str, Any]]] = {}
    sampling_failures: dict[int, dict[str, Any]] = {}
    for generation in sorted(required_generations):
        try:
            sampled_by_generation[generation] = _sample_generation(
                simulator,
                generation,
                namespace=namespace,
                block_index=block_index,
            )
        except Exception as sampling_error:
            sampling_failures[generation] = _typed_error(sampling_error, "sampling")

    truth_relative = truth_path(block_index)
    write_json(
        root / truth_relative,
        {
            "schema_version": TRUTH_SCHEMA_VERSION,
            "block_index": block_index,
            "simulation_seed": int(simulation_seed),
            "tree": serialize_tree(truth),
        },
    )

    condition_inventory = []
    case_inventory = []
    selected_for_maximum: dict[int, Sequence[Genotype]] = {}
    truth_prefixes: dict[int, Any] = {}
    for ordinal, condition, schedule, schedule_seed in resolved_conditions:
        case_id = confirmation_case_id(block_index, condition.condition_id)
        declaration: dict[str, Any] = {
            "case_id": case_id,
            "block_index": block_index,
            "condition_id": condition.condition_id,
            "condition_ordinal": ordinal,
            "condition_family": condition.family,
            "height": condition.height,
            "generations": list(schedule),
            "random_schedule_seed": schedule_seed,
            "status": "available",
            "failure": None,
        }
        if failure_generation is not None and condition.height >= int(failure_generation):
            declaration["status"] = "unavailable"
            declaration["failure"] = guard_failure
            condition_inventory.append(declaration)
            continue
        missing = [generation for generation in schedule if generation in sampling_failures]
        if missing:
            declaration["status"] = "unavailable"
            declaration["failure"] = {
                "stage": "sampling",
                "type": "UnavailableBiopsyGeneration",
                "message": f"Unavailable generations: {missing}.",
                "generation_failures": {
                    str(generation): sampling_failures[generation]
                    for generation in missing
                },
            }
            condition_inventory.append(declaration)
            continue
        try:
            levels = tuple(
                tuple(sampled_by_generation[generation][0])
                for generation in schedule
            )
            if condition.height not in truth_prefixes:
                truth_prefixes[condition.height] = (
                    truth
                    if condition.height >= maximum_generation
                    else truth_prefix(truth, condition.height)
                )
            diagnostics = truth_sampling_diagnostics(
                truth_prefixes[condition.height],
                levels,
            )
            diagnostics["normalized_minimum_invented_edge_fraction"] = (
                float(
                    diagnostics[
                        "minimum_invented_edges_for_observed_only_arborescence"
                    ]
                )
                / max(1, int(diagnostics["selected_occurrence_count"]) - 1)
            )
            paths = condition_paths(block_index, condition.condition_id)
            input_payload = _case_input(
                case_id=case_id,
                condition_id=condition.condition_id,
                condition_family=condition.family,
                schedule=schedule,
                selected_levels=levels,
            )
            reconstruction_seed = derive_seed_for_namespace(
                namespace,
                "reconstruction",
                block_index,
                ordinal,
            )
            metadata = {
                "schema_version": CASE_SCHEMA_VERSION,
                **declaration,
                "sampling": [
                    sampled_by_generation[generation][1]
                    for generation in schedule
                ],
                "selected_occurrence_count": sum(len(level) for level in levels),
                "selected_unique_state_count": len(
                    {
                        cell.cell_id
                        for level in levels
                        for cell in level
                    }
                ),
                "reconstruction_seed": int(reconstruction_seed),
                "reconstruction_seed_scope": "truth_block_condition_shared_across_methods",
                "canonical_input_order": "cnp_tuple_then_stable_state_label",
                "truth_sampling_diagnostics": diagnostics,
            }
            write_json(root / paths["input"], input_payload)
            write_json(root / paths["metadata"], metadata)
            for generation in schedule:
                selected_for_maximum[generation] = sampled_by_generation[generation][0]
            case_inventory.append(
                {
                    **declaration,
                    "metadata_path": paths["metadata"],
                    "input_path": paths["input"],
                    "truth_path": truth_relative,
                    "distance_path": distance_path(block_index),
                }
            )
        except Exception as case_error:
            declaration["status"] = "unavailable"
            declaration["failure"] = _typed_error(case_error, "case_generation")
        condition_inventory.append(declaration)

    block_relative = (
        Path("blocks") / f"block_{block_index + 1:03d}.json"
    ).as_posix()
    block_payload = {
        "schema_version": BLOCK_INPUT_SCHEMA_VERSION,
        "block_index": block_index,
        "status": "success" if case_inventory else "failure",
        "simulation_seed": int(simulation_seed),
        "simulation_guard_failure": guard_failure,
        "simulation_outcome": simulation_outcome,
        "simulation_resources": resources,
        "simulator_event_totals": _event_totals(simulator),
        "truth_path": truth_relative,
        "maximum_observation_states": _maximum_states(selected_for_maximum),
        "condition_inventory": condition_inventory,
        "case_inventory": case_inventory,
    }
    write_json(root / block_relative, block_payload)
    del simulator, truth, truth_prefixes
    gc.collect()
    return {
        "block_index": block_index,
        "status": block_payload["status"],
        "failure": None if case_inventory else {
            "stage": "case_generation",
            "type": "NoAvailableConditions",
            "message": "Simulation produced no available confirmation condition.",
        },
        "simulation_seed": int(simulation_seed),
        "resources": resources,
        "block_path": block_relative,
        "truth_path": truth_relative,
        "condition_inventory": condition_inventory,
        "case_inventory": case_inventory,
    }


def _distance_block_task(
    output_root: str,
    block_record: Mapping[str, Any],
    runtime_config: CtbsRuntimeConfig | None,
    distance_compute: DistanceCompute | None,
    timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, Any]:
    root = Path(output_root)
    block_index = int(block_record["block_index"])
    if block_record.get("status") != "success":
        return {
            "block_index": block_index,
            "status": "not_run_dependency",
            "failure": {
                "stage": "dependency",
                "type": "SimulationDependencyFailure",
                "message": "No distance was run because the simulation block failed.",
            },
            "resources": None,
            "distance_path": distance_path(block_index),
        }
    block_payload = read_json(root / str(block_record["block_path"]))
    states = block_payload.get("maximum_observation_states", [])
    if not states:
        return {
            "block_index": block_index,
            "status": "failure",
            "failure": {
                "stage": "distance",
                "type": "EmptyMaximumObservationSet",
                "message": "No states were available for distance computation.",
            },
            "resources": None,
            "distance_path": distance_path(block_index),
        }
    cells = [
        Genotype(
            state["cnp"],
            state["state_label"],
            generation=0,
            cell_id=state["state_label"],
        )
        for state in states
    ]
    compute = distance_compute
    if compute is None:
        if runtime_config is None:
            raise RuntimeError("Production distance task has no runtime configuration.")
        compute = Cnp2CnpFileDistanceProvider(runtime_config).compute
    distance, resources, error = measured_stage(
        lambda: compute(cells),
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if error is None:
        error = _resource_audit_error(resources, "distance")
    if error is not None or distance is None:
        return {
            "block_index": block_index,
            "status": "failure",
            "failure": _typed_error(
                error or RuntimeError("Distance computation returned no matrix."),
                "distance",
            ),
            "resources": resources,
            "distance_path": distance_path(block_index),
        }
    try:
        validate_distance_matrix(distance.ids, distance.matrix)
        validate_distance_label_coverage(
            distance.ids,
            [state["state_label"] for state in states],
            allow_extra=False,
        )
        provenance = distance.provenance
        if not isinstance(provenance, Mapping):
            raise ValueError("Distance matrix has no provenance.")
        semantics = provenance.get(
            "semantics_version",
            provenance.get("semantic_version"),
        )
        if semantics != CNP2CNP_SEMANTICS_VERSION:
            raise ValueError("Distance semantics differ from the frozen contract.")
        for case in block_record["case_inventory"]:
            input_payload = read_json(root / str(case["input_path"]))
            labels = {
                state["state_label"]
                for level in input_payload["levels"]
                for state in level["states"]
            }
            sub_ids, submatrix = aligned_distance_submatrix(
                distance.ids,
                distance.matrix,
                labels,
            )
            validate_distance_matrix(sub_ids, submatrix)
            validate_distance_label_coverage(sub_ids, labels, allow_extra=False)
    except Exception as validation_error:
        return {
            "block_index": block_index,
            "status": "failure",
            "failure": _typed_error(validation_error, "distance_validation"),
            "resources": resources,
            "distance_path": distance_path(block_index),
        }

    relative = distance_path(block_index)
    write_json(
        root / relative,
        {
            "schema_version": DISTANCE_SCHEMA_VERSION,
            "block_index": block_index,
            "semantics_version": CNP2CNP_SEMANTICS_VERSION,
            "ids": json_safe(distance.ids),
            "matrix": json_safe(np.asarray(distance.matrix, dtype=float)),
            "provenance": json_safe(distance.provenance),
            "resources": resources,
        },
    )
    return {
        "block_index": block_index,
        "status": "success",
        "failure": None,
        "resources": resources,
        "distance_path": relative,
        "unique_state_count": len(distance.ids),
    }


def _new_bank_manifest(
    *,
    validation: Mapping[str, Any],
    namespace: str,
    block_count: int,
    technical_smoke: bool,
    resources: Mapping[str, Any],
    cnp2cnp_runtime: Mapping[str, Any],
    internal_runtime: Mapping[str, Any],
    created_at_utc: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": BANK_SCHEMA_VERSION,
        "bank_id": (
            "ctbf-v5-g2-01a-technical-smoke-bank-v1"
            if technical_smoke
            else "ctbf-v5-g2-01a-independent-confirmation-bank-v1"
        ),
        "status": "in_progress",
        "scientific_role": (
            "technical_smoke_not_accuracy_evidence"
            if technical_smoke
            else "prospectively_locked_independent_confirmation"
        ),
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "design_manifest_path": validation["manifest_path"],
        "design_manifest_sha256": validation["manifest_sha256"],
        "protocol_sha256": validation["protocol_sha256"],
        "source_lock_sha256": validation["source_lock"].get("sha256"),
        "owner_approval_sha256": validation["owner_approval"]["sha256"],
        "seed_namespace": namespace,
        "block_count": block_count,
        "simulation_height": SIMULATION_HEIGHT,
        "condition_ids": [condition.condition_id for condition in CONDITIONS],
        "arm_ids": list(PAPER_ARM_IDS),
        "declared_condition_count": block_count * len(CONDITIONS),
        "resource_contract": dict(resources),
        "cnp2cnp_runtime": dict(cnp2cnp_runtime),
        "internal_runtime": dict(internal_runtime),
        "execution": {
            "simulation": fresh_process_contract(
                TRUTH_BLOCK_SIMULATION_WORKER_UNIT
            ),
            "distance": fresh_process_contract(CONDITION_DISTANCE_WORKER_UNIT),
            "distance_scope": "one_maximal_observation_matrix_per_truth_block",
        },
        "completed_simulation_block_count": 0,
        "completed_distance_block_count": 0,
        "simulation_blocks": [],
        "distance_blocks": [],
        "condition_inventory": [],
        "cases": [],
        "available_condition_count": 0,
        "unavailable_condition_count": 0,
        "failures": [],
    }


def _resume_bank(
    root: Path,
    *,
    validation: Mapping[str, Any],
    namespace: str,
    block_count: int,
    technical_smoke: bool,
    resources: Mapping[str, Any],
    cnp2cnp_runtime: Mapping[str, Any],
    internal_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = root / BANK_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError("A resumed confirmation bank requires bank_manifest.json.")
    manifest = read_json(manifest_path)
    expected = _new_bank_manifest(
        validation=validation,
        namespace=namespace,
        block_count=block_count,
        technical_smoke=technical_smoke,
        resources=resources,
        cnp2cnp_runtime=cnp2cnp_runtime,
        internal_runtime=internal_runtime,
        created_at_utc=manifest.get("created_at_utc"),
    )
    for field in (
        "schema_version",
        "bank_id",
        "scientific_role",
        "design_manifest_sha256",
        "protocol_sha256",
        "source_lock_sha256",
        "owner_approval_sha256",
        "seed_namespace",
        "block_count",
        "simulation_height",
        "condition_ids",
        "arm_ids",
        "declared_condition_count",
        "resource_contract",
        "cnp2cnp_runtime",
        "internal_runtime",
        "execution",
    ):
        if manifest.get(field) != expected.get(field):
            raise ValueError(f"Cannot resume because stored {field} changed.")
    if manifest.get("status") not in {"in_progress", "interrupted", "failure"}:
        raise ValueError("Only an incomplete confirmation bank can resume.")
    simulation_records = manifest.get("simulation_blocks")
    distance_records = manifest.get("distance_blocks")
    if not isinstance(simulation_records, list) or not isinstance(distance_records, list):
        raise ValueError("Stored bank progress inventory is invalid.")
    if [record.get("block_index") for record in simulation_records] != list(
        range(len(simulation_records))
    ):
        raise ValueError("Simulation records are not an exact block prefix.")
    if [record.get("block_index") for record in distance_records] != list(
        range(len(distance_records))
    ):
        raise ValueError("Distance records are not an exact block prefix.")
    if len(distance_records) > len(simulation_records):
        raise ValueError("Distance progress exceeds simulation progress.")
    manifest["status"] = "in_progress"
    manifest.pop("runner_failure", None)
    manifest.setdefault("resume_history", []).append(
        {
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "preserved_simulation_blocks": len(simulation_records),
            "preserved_distance_blocks": len(distance_records),
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def generate_bank(
    *,
    output_root: Path | str,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    simulation_workers: int = DEFAULT_SIMULATION_WORKERS,
    distance_workers: int = DEFAULT_DISTANCE_WORKERS,
    simulation_timeout_seconds: int = DEFAULT_SIMULATION_TIMEOUT_SECONDS,
    distance_timeout_seconds: int = DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    cnp2cnp_process_timeout_seconds: int = DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    cnp2cnp_file: Path | str | None = None,
    technical_smoke: bool = False,
    resume: bool = False,
    progress: bool = False,
    simulator_factory: SimulatorFactory | None = None,
    distance_compute: DistanceCompute | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Generate immutable inputs; no reconstruction score is computed here."""
    if not 1 <= simulation_workers <= MAX_SIMULATION_WORKERS:
        raise ValueError(f"simulation_workers must be in [1,{MAX_SIMULATION_WORKERS}].")
    if not 1 <= distance_workers <= MAX_DISTANCE_WORKERS:
        raise ValueError(f"distance_workers must be in [1,{MAX_DISTANCE_WORKERS}].")
    for name, value in (
        ("simulation_timeout_seconds", simulation_timeout_seconds),
        ("distance_timeout_seconds", distance_timeout_seconds),
        ("cnp2cnp_process_timeout_seconds", cnp2cnp_process_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("capture_limit_bytes", capture_limit_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    validation = validate_manifest(
        manifest_path,
        require_approved=not technical_smoke,
    )
    design = validation["manifest"]
    internal_runtime = validate_internal_runtime(design)
    if simulator_factory is not None or distance_compute is not None:
        raise ValueError(
            "Injected simulator or distance implementations are not supported by "
            "the isolated confirmation bank."
        )
    runtime_config = load_ctbs_runtime_config()
    if cnp2cnp_file is not None:
        executable = Path(cnp2cnp_file).expanduser().resolve()
        runtime_config = replace(
            runtime_config,
            cnp2cnp_folder=str(executable.parent),
            cnp2cnp_file=str(executable),
        )
    runtime_config = replace(
        runtime_config,
        cnp2cnp_timeout_seconds=float(cnp2cnp_process_timeout_seconds),
        cnp2cnp_capture_limit_bytes=int(capture_limit_bytes),
    )
    cnp2cnp_runtime = validate_cnp2cnp_source(
        design,
        runtime_config.cnp2cnp_file,
    )
    block_count = 1 if technical_smoke else BLOCK_COUNT
    namespace = SMOKE_NAMESPACE if technical_smoke else CONFIRMATION_NAMESPACE
    resources = {
        "simulation_timeout_seconds": simulation_timeout_seconds,
        "distance_timeout_seconds": distance_timeout_seconds,
        "cnp2cnp_process_timeout_seconds": cnp2cnp_process_timeout_seconds,
        "rss_limit_bytes": rss_limit_bytes,
        "capture_limit_bytes": capture_limit_bytes,
        "requested_simulation_workers": simulation_workers,
        "requested_distance_workers": distance_workers,
    }
    root = Path(output_root).expanduser().resolve()
    if resume:
        if not root.is_dir():
            raise ValueError("A resumed confirmation bank requires an existing root.")
        manifest = _resume_bank(
            root,
            validation=validation,
            namespace=namespace,
            block_count=block_count,
            technical_smoke=technical_smoke,
            resources=resources,
            cnp2cnp_runtime=cnp2cnp_runtime,
            internal_runtime=internal_runtime,
        )
    else:
        root = ensure_new_output_root(root)
        shutil.copyfile(validation["manifest_path"], root / "design_manifest.snapshot.json")
        shutil.copyfile(OWNER_APPROVAL_PATH, root / OWNER_APPROVAL_SNAPSHOT_NAME)
        config_source = PROJECT_ROOT / design["simulator"]["config_path"]
        shutil.copyfile(config_source, root / SIMULATOR_CONFIG_NAME)
        manifest = _new_bank_manifest(
            validation=validation,
            namespace=namespace,
            block_count=block_count,
            technical_smoke=technical_smoke,
            resources=resources,
            cnp2cnp_runtime=cnp2cnp_runtime,
            internal_runtime=internal_runtime,
            created_at_utc=created_at_utc,
        )
        write_json(root / BANK_MANIFEST_NAME, manifest)

    config = read_json(root / SIMULATOR_CONFIG_NAME)
    load_simulator_inputs(config)
    simulation_start = len(manifest["simulation_blocks"])
    simulation_arguments = (
        (
            str(root),
            deepcopy(config),
            block_index,
            namespace,
            simulation_timeout_seconds,
            rss_limit_bytes,
        )
        for block_index in range(simulation_start, block_count)
    )
    try:
        if simulation_start < block_count:
            with FreshSpawnTaskPool(simulation_workers, buffer_factor=2) as pool:
                for record in pool.map_ordered(
                    _simulate_block_task,
                    simulation_arguments,
                    timeout_seconds=simulation_timeout_seconds + WORKER_GRACE_SECONDS,
                ):
                    manifest["simulation_blocks"].append(record)
                    manifest["completed_simulation_block_count"] = len(
                        manifest["simulation_blocks"]
                    )
                    if record["status"] != "success":
                        manifest["failures"].append(
                            {"phase": "simulation", **record}
                        )
                    write_json(root / BANK_MANIFEST_NAME, manifest)
                    if progress:
                        print(
                            f"simulation {len(manifest['simulation_blocks'])}/{block_count}: "
                            f"{record['status']}",
                            file=sys.stderr,
                            flush=True,
                        )

        distance_start = len(manifest["distance_blocks"])
        distance_arguments = (
            (
                str(root),
                manifest["simulation_blocks"][block_index],
                runtime_config,
                distance_compute,
                distance_timeout_seconds,
                rss_limit_bytes,
            )
            for block_index in range(distance_start, block_count)
        )
        if distance_start < block_count:
            with FreshSpawnTaskPool(distance_workers, buffer_factor=2) as pool:
                for record in pool.map_ordered(
                    _distance_block_task,
                    distance_arguments,
                    timeout_seconds=distance_timeout_seconds + WORKER_GRACE_SECONDS,
                ):
                    manifest["distance_blocks"].append(record)
                    manifest["completed_distance_block_count"] = len(
                        manifest["distance_blocks"]
                    )
                    if record["status"] != "success":
                        manifest["failures"].append(
                            {"phase": "distance", **record}
                        )
                    write_json(root / BANK_MANIFEST_NAME, manifest)
                    if progress:
                        print(
                            f"distance {len(manifest['distance_blocks'])}/{block_count}: "
                            f"{record['status']}",
                            file=sys.stderr,
                            flush=True,
                        )
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failure"
        manifest["runner_failure"] = _typed_error(error, "bank_runner")
        write_json(root / BANK_MANIFEST_NAME, manifest)
        raise

    condition_inventory = []
    cases = []
    distance_by_block = {
        int(record["block_index"]): record for record in manifest["distance_blocks"]
    }
    for simulation_record in manifest["simulation_blocks"]:
        block_index = int(simulation_record["block_index"])
        distance_record = distance_by_block[block_index]
        for declaration in simulation_record.get("condition_inventory", []):
            resolved = dict(declaration)
            if resolved["status"] == "available" and distance_record["status"] != "success":
                resolved["status"] = "unavailable"
                resolved["failure"] = {
                    "stage": "dependency",
                    "type": "DistanceDependencyFailure",
                    "message": "The block-level production distance failed.",
                    "distance_failure": distance_record.get("failure"),
                }
            condition_inventory.append(resolved)
        if distance_record["status"] == "success":
            cases.extend(simulation_record.get("case_inventory", []))

    manifest["condition_inventory"] = condition_inventory
    manifest["cases"] = cases
    manifest["available_condition_count"] = len(cases)
    manifest["unavailable_condition_count"] = len(condition_inventory) - len(cases)
    final_cnp2cnp_runtime = validate_cnp2cnp_source(
        design,
        runtime_config.cnp2cnp_file,
    )
    if final_cnp2cnp_runtime != cnp2cnp_runtime:
        raise RuntimeError("cnp2cnp source identity changed during bank generation.")
    if validate_internal_runtime(design) != internal_runtime:
        raise RuntimeError("Internal runtime identity changed during bank generation.")
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest.pop("runner_failure", None)
    write_json(root / BANK_MANIFEST_NAME, manifest)
    write_checksum_closure(root, BANK_CHECKSUM_NAME)
    return manifest


def deserialize_confirmation_distance(payload: Mapping[str, Any]) -> DistanceMatrix:
    if payload.get("schema_version") != DISTANCE_SCHEMA_VERSION:
        raise ValueError("Unknown confirmation distance schema.")
    if payload.get("semantics_version") != CNP2CNP_SEMANTICS_VERSION:
        raise ValueError("Confirmation distance semantics changed.")
    distance = DistanceMatrix(
        ids=list(payload["ids"]),
        matrix=np.asarray(payload["matrix"], dtype=float),
        provenance=payload.get("provenance"),
    )
    validate_distance_matrix(distance.ids, distance.matrix)
    return distance


def load_confirmation_bank(
    bank_root: Path | str,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    allow_smoke: bool = False,
) -> tuple[Path, dict[str, Any]]:
    root = Path(bank_root).expanduser().resolve()
    bank = read_json(root / BANK_MANIFEST_NAME)
    if bank.get("schema_version") != BANK_SCHEMA_VERSION or bank.get("status") != "complete":
        raise ValueError("Confirmation bank is not complete under the current schema.")
    is_smoke = bank.get("scientific_role") == "technical_smoke_not_accuracy_evidence"
    if is_smoke and not allow_smoke:
        raise ValueError("Technical-smoke bank cannot be used as confirmation evidence.")
    validation = validate_manifest(
        manifest_path,
        require_approved=not is_smoke,
    )
    if bank.get("design_manifest_sha256") != validation["manifest_sha256"]:
        raise ValueError("Bank design-manifest digest changed.")
    approval_snapshot = root / OWNER_APPROVAL_SNAPSHOT_NAME
    if (
        not approval_snapshot.is_file()
        or file_sha256(approval_snapshot) != bank.get("owner_approval_sha256")
        or bank.get("owner_approval_sha256")
        != validation["owner_approval"]["sha256"]
    ):
        raise ValueError("Bank owner-approval snapshot changed.")
    expected_blocks = 1 if is_smoke else BLOCK_COUNT
    if bank.get("block_count") != expected_blocks:
        raise ValueError("Confirmation bank block count changed.")
    if bank.get("condition_ids") != [condition.condition_id for condition in CONDITIONS]:
        raise ValueError("Confirmation bank conditions changed.")
    if bank.get("arm_ids") != list(PAPER_ARM_IDS):
        raise ValueError("Confirmation bank method declaration changed.")
    closure = validate_checksum_closure(root, BANK_CHECKSUM_NAME)
    bank["checksum_closure"] = closure
    return root, bank


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--simulation-workers", type=int, default=DEFAULT_SIMULATION_WORKERS)
    parser.add_argument("--distance-workers", type=int, default=DEFAULT_DISTANCE_WORKERS)
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
    parser.add_argument("--capture-limit-bytes", type=int, default=DEFAULT_CAPTURE_LIMIT_BYTES)
    parser.add_argument(
        "--cnp2cnp-file",
        type=Path,
        help="Path to the source-locked cnp2cnp.py entry point.",
    )
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = generate_bank(
        output_root=arguments.output_root,
        manifest_path=arguments.manifest,
        simulation_workers=arguments.simulation_workers,
        distance_workers=arguments.distance_workers,
        simulation_timeout_seconds=arguments.simulation_timeout_seconds,
        distance_timeout_seconds=arguments.distance_timeout_seconds,
        cnp2cnp_process_timeout_seconds=arguments.cnp2cnp_process_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        capture_limit_bytes=arguments.capture_limit_bytes,
        cnp2cnp_file=arguments.cnp2cnp_file,
        technical_smoke=arguments.technical_smoke,
        resume=arguments.resume,
        progress=arguments.progress,
    )
    print(
        "complete: "
        f"{result['available_condition_count']}/"
        f"{result['declared_condition_count']} conditions available"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CASE_SCHEMA_VERSION",
    "DISTANCE_SCHEMA_VERSION",
    "OWNER_APPROVAL_SNAPSHOT_NAME",
    "SMOKE_NAMESPACE",
    "TRUTH_SCHEMA_VERSION",
    "deserialize_confirmation_distance",
    "generate_bank",
    "load_confirmation_bank",
    "main",
]
