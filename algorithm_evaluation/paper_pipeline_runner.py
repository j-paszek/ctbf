"""CTBF v5 paper pipeline, disabled until a new manifest is frozen.

Reusable validation and execution mechanics are retained. Registered and smoke
execution both pass through the manifest lock, which currently refuses work
until the owner approves new CTBF v5 protocol and manifest bytes.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import importlib.metadata
from hashlib import sha256
import json
import locale
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - recorded explicitly in artifacts
    psutil = None

from algorithm_evaluation.paper_pipeline_analysis import write_analysis
from algorithm_evaluation.paper_pipeline_contract import (
    ANALYSIS_SCHEMA_VERSION,
    CASE_SCHEMA_VERSION,
    DEFAULT_MANIFEST_PATH,
    ENVIRONMENT_SCHEMA_VERSION,
    EVALUATION_RESULT_SCHEMA_VERSION,
    EXPECTED_INVENTORY_SCHEMA_VERSION,
    EXPERIMENT_STATUS_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PROJECT_ROOT,
    REGISTERED_ARM_SPECS,
    REGISTERED_CLEAN_EXPERIMENT,
    aligned_distance_submatrix,
    condition_id,
    derive_seed,
    ensure_new_empty_output_root,
    file_sha256,
    json_safe,
    read_json,
    sample_nested_observations,
    source_freeze_manifest,
    status_record,
    validate_checksum_closure,
    validate_manifest,
    validate_status_record,
    write_checksum_file,
    write_json_atomic,
)
from ctbs import (
    Cnp2CnpFileDistanceProvider,
    DistanceMatrix,
    load_ctbs_runtime_config,
    unique_cells_by_cell_id,
)
from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    DISTANCE_PROVENANCE_SCHEMA_VERSION,
    stable_distance_label_key,
    validate_distance_matrix,
)
from evaluation_contract import (
    EvaluationContractError,
    evaluate_tree_pair_result,
    evaluation_failure_result,
    validate_evaluation_result,
)
from reconstructor import build_evolution_tree
from reconstructor_registry import resolve_reconstruction_algorithm
from simulator import CancerCellEvolutionSimulator, Genotype


TRUTH_SCHEMA_VERSION = "ctbf-v5-paper-truth-v1"
RECONSTRUCTION_INPUT_SCHEMA_VERSION = "ctbf-v5-reconstruction-input-v1"
DISTANCE_RECORD_SCHEMA_VERSION = "ctbf-v5-distance-record-v1"
RECONSTRUCTION_RESULT_SCHEMA_VERSION = "ctbf-v5-reconstruction-result-v1"
RESOURCE_RECORD_SCHEMA_VERSION = "ctbf-v5-resource-record-v1"
SMOKE_EXPERIMENT_ID = "ctbf-v5-g2-01-smoke-v1"
SMOKE_CONDITION = condition_id(0.5, "L3")

ARM_BUILD_SPECS = {
    "classical_partial": ("neighbor_joining_classical", "pooled", True),
    "biopsy_guided_classical": ("neighbor_joining_classical", "ordered", False),
    "rooted_labeled_nj": ("rooted_labeled_nj", "pooled", True),
    "temporal_minimum": ("temporal_cnp_arborescence", "ordered", False),
    "temporal_minimum_no_time": (
        "temporal_cnp_arborescence_no_time",
        "ordered",
        False,
    ),
    "anticentral_parsimony": (
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
        "pooled",
        True,
    ),
}


class StageTimeoutError(TimeoutError):
    pass


class StageResourceError(RuntimeError):
    pass


class PeakProcessTreeRss:
    """Sample runner plus recursive-child RSS at the registered 10 ms interval."""

    def __init__(self, interval_seconds: float = 0.01):
        self.interval_seconds = interval_seconds
        self.peak_bytes: int | None = None
        self.sample_count = 0
        self.sampling_available = psutil is not None
        self.child_sampling_available = psutil is not None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        if psutil is None:
            return
        try:
            root = psutil.Process(os.getpid())
            try:
                children = root.children(recursive=True)
            except Exception:
                children = []
                self.child_sampling_available = False
            rss = 0
            for process in [root, *children]:
                try:
                    rss += int(process.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.sample_count += 1
            self.peak_bytes = rss if self.peak_bytes is None else max(self.peak_bytes, rss)
        except Exception:
            self.sampling_available = False

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self):
        self._sample()
        if psutil is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()

    def report(self) -> dict[str, Any]:
        if psutil is None:
            method = "unavailable_psutil_not_installed"
        elif not self.sampling_available:
            method = "unavailable_psutil_permission_denied"
        elif self.child_sampling_available:
            method = "psutil_process_tree_rss_10ms"
        else:
            method = "psutil_runner_process_rss_10ms_child_access_denied"
        return {
            "method": method,
            "peak_rss_bytes": self.peak_bytes,
            "sample_count": self.sample_count,
        }


@contextmanager
def _stage_timeout(seconds: float | int | None):
    if seconds is None:
        yield
        return
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Registered stage timeouts require a POSIX main-thread runner.")

    def handle_timeout(_signum, _frame):
        raise StageTimeoutError(f"Stage exceeded its registered {seconds}-second timeout.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def measured_stage(
    function: Callable[[], Any],
    *,
    timeout_seconds: int | None,
    rss_limit_bytes: int,
) -> tuple[Any | None, dict[str, Any], BaseException | None]:
    start = time.perf_counter_ns()
    value = None
    error: BaseException | None = None
    with PeakProcessTreeRss() as memory:
        try:
            with _stage_timeout(timeout_seconds):
                value = function()
        except Exception as exc:  # retained as a typed stage record
            error = exc
    runtime = {
        "wall_time_ns": time.perf_counter_ns() - start,
        "timeout_seconds": timeout_seconds,
        "memory": memory.report(),
        "rss_limit_bytes": int(rss_limit_bytes),
    }
    peak = runtime["memory"]["peak_rss_bytes"]
    if error is None and peak is not None and peak > rss_limit_bytes:
        error = StageResourceError(
            f"Stage peak RSS {peak} exceeded the registered limit {rss_limit_bytes}."
        )
        value = None
    return value, runtime, error


def serialize_tree(tree: nx.DiGraph) -> dict[str, Any]:
    return {
        "directed": bool(tree.is_directed()),
        "multigraph": bool(tree.is_multigraph()),
        "graph_attributes": json_safe(dict(tree.graph)),
        "nodes": [
            {"id": json_safe(node), **json_safe(dict(attributes))}
            for node, attributes in sorted(
                tree.nodes(data=True), key=lambda item: str(item[0])
            )
        ],
        "links": [
            {
                "source": json_safe(source),
                "target": json_safe(target),
                **json_safe(dict(attributes)),
            }
            for source, target, attributes in sorted(
                tree.edges(data=True), key=lambda item: (str(item[0]), str(item[1]))
            )
        ],
    }


def deserialize_tree(serialized: Mapping[str, Any]) -> nx.DiGraph:
    if serialized.get("directed") is not True or serialized.get("multigraph") is True:
        raise ValueError("Paper tree serialization must be a simple directed graph.")
    tree = nx.DiGraph()
    tree.graph.update(serialized.get("graph_attributes", {}))
    for record in serialized.get("nodes", []):
        record = dict(record)
        node = record.pop("id")
        tree.add_node(node, **record)
    for record in serialized.get("links", []):
        record = dict(record)
        source = record.pop("source")
        target = record.pop("target")
        tree.add_edge(source, target, **record)
    return tree


def _actual_root(tree: nx.DiGraph) -> Any:
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected one directed root, found {len(roots)}.")
    return roots[0]


def _canonical_cells(cells: Iterable[Genotype]) -> list[Genotype]:
    return sorted(
        cells,
        key=lambda cell: (
            tuple(int(value) for value in np.asarray(cell.genome).tolist()),
            str(cell.cell_id).strip(),
        ),
    )


def _condition_input(case_id_value: str, condition) -> dict[str, Any]:
    levels = []
    for biopsy_level, (generation, cells) in enumerate(
        zip(condition.generations, condition.cells_by_generation)
    ):
        levels.append(
            {
                "biopsy_level": biopsy_level,
                "generation": int(generation),
                "states": [
                    {
                        "state_label": json_safe(cell.cell_id),
                        "cnp": [int(value) for value in np.asarray(cell.genome).tolist()],
                    }
                    for cell in _canonical_cells(cells)
                ],
            }
        )
    payload = {
        "schema_version": RECONSTRUCTION_INPUT_SCHEMA_VERSION,
        "case_id": case_id_value,
        "condition_id": condition.condition_id,
        "fraction": condition.fraction,
        "schedule_id": condition.schedule_id,
        "levels": levels,
    }
    validate_reconstruction_input(payload)
    return payload


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def validate_reconstruction_input(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RECONSTRUCTION_INPUT_SCHEMA_VERSION:
        raise ValueError("Unknown reconstruction-input schema.")
    forbidden = {
        "node_id",
        "simulator_node_id",
        "truth",
        "truth_tree",
        "sampling_rank",
        "occurrence_correspondence",
    }
    present = forbidden & set(_walk_keys(payload))
    if present:
        raise ValueError("Truth-only fields entered reconstruction input: " + ", ".join(sorted(present)))
    levels = payload.get("levels")
    if not isinstance(levels, list) or len(levels) < 2:
        raise ValueError("Reconstruction input needs at least two observation levels.")
    labels_to_cnp = {}
    for expected_level, level in enumerate(levels):
        if level.get("biopsy_level") != expected_level:
            raise ValueError("Biopsy levels must be consecutive and ordered.")
        states = level.get("states")
        if not isinstance(states, list) or not states:
            raise ValueError("Every selected observation level must be nonempty.")
        level_labels = set()
        for state in states:
            if set(state) != {"state_label", "cnp"}:
                raise ValueError("Observation states may contain only state_label and cnp.")
            label = state["state_label"]
            cnp = tuple(state["cnp"])
            if label in level_labels:
                raise ValueError("One observation level contains a duplicate state label.")
            if label in labels_to_cnp and labels_to_cnp[label] != cnp:
                raise ValueError("One observable state label maps to multiple CNPs.")
            level_labels.add(label)
            labels_to_cnp[label] = cnp
    json.dumps(json_safe(payload), sort_keys=True, allow_nan=False)


def _cells_from_input(payload: Mapping[str, Any]) -> list[list[Genotype]]:
    validate_reconstruction_input(payload)
    return [
        [
            Genotype(
                state["cnp"],
                state["state_label"],
                generation=level["generation"],
                cell_id=state["state_label"],
            )
            for state in level["states"]
        ]
        for level in payload["levels"]
    ]


def _observed_labels(payload: Mapping[str, Any]) -> list[Any]:
    return sorted(
        {
            state["state_label"]
            for level in payload["levels"]
            for state in level["states"]
        },
        key=stable_distance_label_key,
    )


def reconstruct_arm(
    arm_id: str,
    reconstruction_input: Mapping[str, Any],
    distance_matrix: DistanceMatrix,
    *,
    reconstruction_seed: int,
) -> tuple[nx.DiGraph, dict[Any, Any], Any, dict[str, Any]]:
    """Run one registered arm from observable bytes only."""
    if arm_id not in ARM_BUILD_SPECS:
        raise ValueError(f"Unknown registered arm {arm_id!r}.")
    algorithm_name, input_mode, only_nj = ARM_BUILD_SPECS[arm_id]
    algorithm = resolve_reconstruction_algorithm(algorithm_name)
    cell_lists = _cells_from_input(reconstruction_input)
    if input_mode == "pooled":
        build_input = [[cell for cells in cell_lists for cell in cells]]
    else:
        build_input = cell_lists
    tree, levels, returned_root = build_evolution_tree(
        build_input,
        seed=int(reconstruction_seed),
        r=4,
        only_nj=only_nj,
        distance_matrix=distance_matrix,
        neighbor_joining=algorithm,
    )
    actual_root = _actual_root(tree)
    if not nx.is_arborescence(tree):
        raise ValueError("Reconstruction did not return one directed arborescence.")
    metadata = {
        "arm_id": arm_id,
        "algorithm": algorithm_name,
        "input_mode": input_mode,
        "only_nj": only_nj,
        "radius": 4,
        "reconstruction_seed": int(reconstruction_seed),
        "returned_root": json_safe(returned_root),
        "actual_root": json_safe(actual_root),
    }
    return tree, dict(levels), returned_root, metadata


def _distance_payload(
    *,
    case_id_value: str,
    distance: DistanceMatrix | None,
    provider_status: Mapping[str, Any],
    resources: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": DISTANCE_RECORD_SCHEMA_VERSION,
        "case_id": case_id_value,
        "method": method,
        "status_record": provider_status,
        "resources": resources,
    }
    if distance is not None:
        payload.update(
            {
                "ids": json_safe(distance.ids),
                "matrix": json_safe(np.asarray(distance.matrix)),
                "provenance": json_safe(distance.provenance),
            }
        )
    return payload


def _l1_smoke_distance(cells: Sequence[Genotype]) -> DistanceMatrix:
    unique = unique_cells_by_cell_id(cells)
    unique = sorted(unique, key=lambda cell: stable_distance_label_key(cell.cell_id))
    ids = [cell.cell_id for cell in unique]
    genomes = np.asarray([cell.genome for cell in unique], dtype=np.int64)
    matrix = np.abs(genomes[:, None, :] - genomes[None, :, :]).sum(axis=2).astype(float)
    return DistanceMatrix(
        ids=ids,
        matrix=matrix,
        provenance={
            "schema_version": DISTANCE_PROVENANCE_SCHEMA_VERSION,
            "semantics_version": "ctbf-g2-01-smoke-l1-only-v1",
            "metric": "cnp_l1",
            "construction": "toy_nonheldout_injected_matrix",
            "paper_evidence_allowed": False,
            "fallback_for_cnp2cnp": False,
        },
    )


def _distance_semantics_version(provenance: Mapping[str, Any]) -> Any:
    """Read the established provenance key, accepting old smoke-only spelling."""
    return provenance.get("semantics_version", provenance.get("semantic_version"))


def _validate_primary_distance_provenance(
    distance: DistanceMatrix,
    *,
    smoke: bool,
) -> None:
    provenance = distance.provenance
    if not isinstance(provenance, dict):
        raise ValueError("Distance result is missing provenance.")
    if provenance.get("schema_version") != DISTANCE_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Distance result has the wrong provenance schema.")
    if smoke:
        if (
            provenance.get("paper_evidence_allowed") is not False
            or provenance.get("fallback_for_cnp2cnp") is not False
            or provenance.get("metric") != "cnp_l1"
        ):
            raise ValueError("Smoke distance provenance does not prohibit paper use.")
        return
    if (
        _distance_semantics_version(provenance) != CNP2CNP_SEMANTICS_VERSION
        or provenance.get("metric") != "cnp2cnp"
        or provenance.get("distance_mode") != "any"
        or provenance.get("symmetrization") != "minimum_bidirectional"
        or provenance.get("formula") != "min(d_any(u,v),d_any(v,u))"
        or provenance.get("construction")
        not in {"opposite_order_matrix_mode", "trivial_singleton"}
    ):
        raise ValueError("cnp2cnp provider returned the wrong scientific provenance.")


def _generation_cells(simulator: CancerCellEvolutionSimulator, generations: Sequence[int]):
    result = {}
    for generation in generations:
        cells = [
            genotype
            for genotype in simulator.genotypes.values()
            if genotype.generation == generation
        ]
        result[generation] = simulator.canonicalize_biopsy_genotypes(cells)
    return result


def _observation_artifact(case_id_value: str, design, selected_condition_ids: Sequence[str]):
    conditions = [design.conditions[value] for value in selected_condition_ids]
    status = status_record(
        entity_type="observation_set",
        entity_id=case_id_value,
        status="success",
        stage="sampling",
        code="nested_observations_complete",
    )
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "case_id": case_id_value,
        "status_record": status,
        "maximal_generations": [
            {
                "generation": int(generation),
                "representatives": [
                    {
                        "state_label": json_safe(cell.cell_id),
                        "cnp": [int(value) for value in np.asarray(cell.genome).tolist()],
                    }
                    for cell in _canonical_cells(cells)
                ],
            }
            for generation, cells in sorted(design.maximal_cells_by_generation.items())
        ],
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "fraction": condition.fraction,
                "schedule_id": condition.schedule_id,
                "generations": list(condition.generations),
                "occurrence_count": condition.occurrence_count,
                "unique_state_count": len(condition.unique_labels),
            }
            for condition in conditions
        ],
        "nesting": "single_per_generation_permutation_then_fraction_prefix_and_literal_schedule_subset",
        "sampling_rank_stored": False,
    }


def _truth_artifact(case_id_value: str, simulator: CancerCellEvolutionSimulator):
    tree = simulator.canonicalized_tree_by_genome()
    return {
        "schema_version": TRUTH_SCHEMA_VERSION,
        "case_id": case_id_value,
        "status_record": status_record(
            entity_type="truth",
            entity_id=case_id_value,
            status="success",
            stage="simulation",
            code="truth_complete",
        ),
        "simulator_provenance": simulator.provenance(),
        "simulation_diagnostics": simulator.diagnostics_snapshot(),
        "tree": serialize_tree(tree),
        "generation_sizes": {
            str(generation): sum(
                genotype.generation == generation
                for genotype in simulator.genotypes.values()
            )
            for generation in range(simulator.num_generations + 1)
        },
    }, tree


def _dependency_evaluation(labels: Iterable[Any], dependency: str, message: str):
    error = EvaluationContractError(
        "not_run_dependency",
        message,
        stage="dependency",
        details={"dependency": dependency},
    )
    return evaluation_failure_result(error, labels)


def _write_arm_dependency(
    arm_root: Path,
    *,
    arm_id: str,
    observed_labels: Iterable[Any],
    dependency: str,
    message: str,
) -> None:
    arm_root.mkdir(parents=True, exist_ok=True)
    status = status_record(
        entity_type="reconstruction_arm",
        entity_id=arm_id,
        status="not_run_dependency",
        stage="dependency",
        code="not_run_dependency",
        dependency=dependency,
        message=message,
    )
    write_json_atomic(arm_root / "status.json", status)
    write_json_atomic(
        arm_root / "reconstruction.json",
        {
            "schema_version": RECONSTRUCTION_RESULT_SCHEMA_VERSION,
            "status": "not_run_dependency",
            "dependency": dependency,
            "message": message,
        },
    )
    evaluation = _dependency_evaluation(observed_labels, dependency, message)
    validate_evaluation_result(evaluation)
    write_json_atomic(arm_root / "evaluation.json", evaluation)
    write_json_atomic(
        arm_root / "resources.json",
        {
            "schema_version": RESOURCE_RECORD_SCHEMA_VERSION,
            "status": "not_run_dependency",
            "dependency": dependency,
            "reconstruction": None,
            "evaluation": None,
        },
    )


def _write_condition_dependency(
    condition_root: Path,
    *,
    condition_id_value: str,
    dependency: str,
    message: str,
) -> None:
    condition_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        condition_root / "input.json",
        {
            "schema_version": RECONSTRUCTION_INPUT_SCHEMA_VERSION,
            "condition_id": condition_id_value,
            "status_record": status_record(
                entity_type="condition",
                entity_id=condition_id_value,
                status="not_run_dependency",
                stage="dependency",
                code="not_run_dependency",
                dependency=dependency,
                message=message,
            ),
        },
    )
    for arm_id, _algorithm in REGISTERED_ARM_SPECS:
        _write_arm_dependency(
            condition_root / "arms" / arm_id,
            arm_id=arm_id,
            observed_labels=(),
            dependency=dependency,
            message=message,
        )


def _write_condition_arms(
    *,
    condition_root: Path,
    input_payload: Mapping[str, Any],
    maximal_distance: DistanceMatrix,
    true_tree: nx.DiGraph,
    reconstruction_seed: int,
    resources: Mapping[str, Any],
) -> dict[str, int]:
    observed_labels = _observed_labels(input_payload)
    sub_ids, submatrix = aligned_distance_submatrix(
        maximal_distance.ids,
        maximal_distance.matrix,
        observed_labels,
    )
    distance = DistanceMatrix(
        ids=sub_ids,
        matrix=submatrix,
        provenance=maximal_distance.provenance,
    )
    stored_input = deepcopy(input_payload)
    maximal_positions = {
        cell_id: index for index, cell_id in enumerate(maximal_distance.ids)
    }
    stored_input["distance"] = {
        "semantic_version": _distance_semantics_version(maximal_distance.provenance),
        "ids": json_safe(sub_ids),
        "maximal_row_indices": [maximal_positions[cell_id] for cell_id in sub_ids],
        "matrix_shape": list(submatrix.shape),
        "matrix_sha256_float64_c_order": sha256(
            np.asarray(submatrix, dtype="<f8", order="C").tobytes(order="C")
        ).hexdigest(),
        "maximal_distance_artifact": "../../distances/minimum_bidirectional.json",
        "maximal_matrix_submatrix": True,
    }
    write_json_atomic(condition_root / "input.json", stored_input)

    counts = {"success": 0, "failure": 0, "not_run_dependency": 0}
    for arm_id, declared_algorithm in REGISTERED_ARM_SPECS:
        arm_root = condition_root / "arms" / arm_id
        arm_root.mkdir(parents=True, exist_ok=True)
        reconstruction_result, reconstruction_runtime, reconstruction_error = measured_stage(
            lambda arm_id=arm_id: reconstruct_arm(
                arm_id,
                input_payload,
                distance,
                reconstruction_seed=reconstruction_seed,
            ),
            timeout_seconds=int(resources["reconstruction_timeout_seconds_per_arm"]),
            rss_limit_bytes=int(resources["aggregate_rss_limit_bytes_per_worker"]),
        )
        if reconstruction_error is not None:
            arm_status = status_record(
                entity_type="reconstruction_arm",
                entity_id=arm_id,
                status="failure",
                stage="reconstruction",
                code=(
                    "reconstruction_timeout"
                    if isinstance(reconstruction_error, StageTimeoutError)
                    else "reconstruction_failure"
                ),
                runtime=reconstruction_runtime,
                exception=reconstruction_error,
            )
            write_json_atomic(arm_root / "status.json", arm_status)
            write_json_atomic(
                arm_root / "reconstruction.json",
                {
                    "schema_version": RECONSTRUCTION_RESULT_SCHEMA_VERSION,
                    "status": "failure",
                    "algorithm": declared_algorithm,
                    "exception": {
                        "type": type(reconstruction_error).__name__,
                        "message": str(reconstruction_error)[:4096],
                    },
                },
            )
            evaluation = _dependency_evaluation(
                observed_labels,
                f"reconstruction:{arm_id}",
                "Evaluation was not run because reconstruction failed.",
            )
            write_json_atomic(arm_root / "evaluation.json", evaluation)
            write_json_atomic(
                arm_root / "resources.json",
                {
                    "schema_version": RESOURCE_RECORD_SCHEMA_VERSION,
                    "status": "failure",
                    "reconstruction": reconstruction_runtime,
                    "evaluation": None,
                },
            )
            counts["failure"] += 1
            continue

        tree, levels, _returned_root, metadata = reconstruction_result
        reconstruction_payload = {
            "schema_version": RECONSTRUCTION_RESULT_SCHEMA_VERSION,
            "status": "success",
            "metadata": metadata,
            "node_levels": [
                {"node_id": json_safe(node), "level": json_safe(level)}
                for node, level in sorted(levels.items(), key=lambda item: str(item[0]))
            ],
            "tree": serialize_tree(tree),
        }
        write_json_atomic(arm_root / "reconstruction.json", reconstruction_payload)

        evaluation, evaluation_runtime, evaluation_error = measured_stage(
            lambda: evaluate_tree_pair_result(true_tree, tree, observed_labels),
            timeout_seconds=int(resources["evaluation_timeout_seconds_per_arm"]),
            rss_limit_bytes=int(resources["aggregate_rss_limit_bytes_per_worker"]),
        )
        if evaluation_error is not None:
            evaluation = evaluation_failure_result(
                EvaluationContractError(
                    "evaluation_timeout"
                    if isinstance(evaluation_error, StageTimeoutError)
                    else "evaluation_exception",
                    str(evaluation_error),
                    stage="evaluation",
                    details={"exception_type": type(evaluation_error).__name__},
                ),
                observed_labels,
            )
        validate_evaluation_result(evaluation)
        write_json_atomic(arm_root / "evaluation.json", evaluation)

        coverage_failure = False
        if evaluation["status"] == "success":
            coverage_failure = (
                evaluation["inputs"]["observation_label_coverage"]["fraction"] < 1.0
            )
        if evaluation["status"] != "success" or coverage_failure:
            code = (
                "observation_coverage_below_one"
                if coverage_failure
                else "evaluation_failure"
            )
            arm_status = status_record(
                entity_type="reconstruction_arm",
                entity_id=arm_id,
                status="failure",
                stage="evaluation",
                code=code,
                runtime={
                    "reconstruction": reconstruction_runtime,
                    "evaluation": evaluation_runtime,
                },
                message=(
                    "Native evaluation coverage is below the registered 1.0 requirement."
                    if coverage_failure
                    else "Native schema-v1 evaluation failed."
                ),
            )
            counts["failure"] += 1
        else:
            arm_status = status_record(
                entity_type="reconstruction_arm",
                entity_id=arm_id,
                status="success",
                stage="evaluation",
                code="arm_complete",
                runtime={
                    "reconstruction": reconstruction_runtime,
                    "evaluation": evaluation_runtime,
                },
            )
            counts["success"] += 1
        write_json_atomic(arm_root / "status.json", arm_status)
        write_json_atomic(
            arm_root / "resources.json",
            {
                "schema_version": RESOURCE_RECORD_SCHEMA_VERSION,
                "status": arm_status["status"],
                "limits": {
                    "reconstruction_timeout_seconds": resources[
                        "reconstruction_timeout_seconds_per_arm"
                    ],
                    "evaluation_timeout_seconds": resources[
                        "evaluation_timeout_seconds_per_arm"
                    ],
                    "rss_limit_bytes": resources["aggregate_rss_limit_bytes_per_worker"],
                    "memory_sampling_interval_ms": resources[
                        "memory_sampling_interval_ms"
                    ],
                },
                "reconstruction": reconstruction_runtime,
                "evaluation": evaluation_runtime,
            },
        )
    return counts


def _case_paths(case_id_value: str, condition_ids: Sequence[str]):
    base = Path("cases") / case_id_value
    raw = [
        (base / "case_status.json").as_posix(),
        (base / "truth.json").as_posix(),
        (base / "observations.json").as_posix(),
        (base / "distances" / "minimum_bidirectional.json").as_posix(),
    ]
    evaluations = []
    direct_statuses = []
    nested_statuses = [
        {"path": (base / "case_status.json").as_posix(), "field": "status_record"},
        {"path": (base / "truth.json").as_posix(), "field": "status_record"},
        {"path": (base / "observations.json").as_posix(), "field": "status_record"},
        {
            "path": (base / "distances" / "minimum_bidirectional.json").as_posix(),
            "field": "status_record",
        },
    ]
    for condition in condition_ids:
        condition_base = base / "conditions" / condition
        input_path = (condition_base / "input.json").as_posix()
        raw.append(input_path)
        nested_statuses.append({"path": input_path, "field": "status_record", "optional_on_success": True})
        for arm_id, _algorithm in REGISTERED_ARM_SPECS:
            arm_base = condition_base / "arms" / arm_id
            for filename in ("status.json", "reconstruction.json", "evaluation.json", "resources.json"):
                raw.append((arm_base / filename).as_posix())
            status_path = (arm_base / "status.json").as_posix()
            direct_statuses.append(status_path)
            evaluations.append((arm_base / "evaluation.json").as_posix())
    return raw, evaluations, direct_statuses, nested_statuses


def build_expected_inventory(
    *,
    experiment_id: str,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    raw = [
        "design_manifest.snapshot.json",
        "source_manifest.json",
        "environment.json",
        "run_status.json",
        "expected_inventory.json",
    ]
    evaluations: list[str] = []
    direct_statuses = ["run_status.json"]
    nested_statuses = []
    for case in cases:
        case_raw, case_evaluations, case_statuses, case_nested = _case_paths(
            case["case_id"], case["condition_ids"]
        )
        raw.extend(case_raw)
        evaluations.extend(case_evaluations)
        direct_statuses.extend(case_statuses)
        nested_statuses.extend(case_nested)
    return {
        "schema_version": EXPECTED_INVENTORY_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "cases": [
            {
                "case_id": case["case_id"],
                "regime_id": case["regime_id"],
                "replicate": case["replicate"],
                "condition_ids": list(case["condition_ids"]),
            }
            for case in cases
        ],
        "case_count": len(cases),
        "condition_count": sum(len(case["condition_ids"]) for case in cases),
        "arm_record_count": len(evaluations),
        "raw_files": sorted(set(raw), key=lambda value: value.encode("utf-8")),
        "evaluation_files": sorted(evaluations, key=lambda value: value.encode("utf-8")),
        "status_files": sorted(direct_statuses, key=lambda value: value.encode("utf-8")),
        "nested_status_records": sorted(
            nested_statuses,
            key=lambda value: (value["path"].encode("utf-8"), value["field"]),
        ),
        "required_root_files": [
            "design_manifest.snapshot.json",
            "source_manifest.json",
            "environment.json",
            "run_status.json",
            "expected_inventory.json",
            "raw_checksums.sha256",
            "complete_checksums.sha256",
        ],
    }


def _dependency_artifact(schema_version: str, entity_type: str, entity_id: str, dependency: str, message: str):
    return {
        "schema_version": schema_version,
        "status_record": status_record(
            entity_type=entity_type,
            entity_id=entity_id,
            status="not_run_dependency",
            stage="dependency",
            code="not_run_dependency",
            dependency=dependency,
            message=message,
        ),
    }


def _write_failed_case(
    *,
    output_root: Path,
    case: Mapping[str, Any],
    condition_ids: Sequence[str],
    error: BaseException,
    stage: str,
) -> None:
    case_root = output_root / "cases" / case["case_id"]
    case_root.mkdir(parents=True, exist_ok=True)
    case_status = status_record(
        entity_type="case",
        entity_id=case["case_id"],
        status="failure",
        stage=stage,
        code=f"{stage}_failure",
        exception=error,
    )
    write_json_atomic(
        case_root / "case_status.json",
        {
            "schema_version": CASE_SCHEMA_VERSION,
            **json_safe(case),
            "status_record": case_status,
        },
    )
    dependency = f"case:{case['case_id']}:{stage}"
    message = f"Case did not reach this artifact because {stage} failed."
    if not (case_root / "truth.json").exists():
        write_json_atomic(
            case_root / "truth.json",
            _dependency_artifact(TRUTH_SCHEMA_VERSION, "truth", case["case_id"], dependency, message),
        )
    if not (case_root / "observations.json").exists():
        write_json_atomic(
            case_root / "observations.json",
            _dependency_artifact(OBSERVATION_SCHEMA_VERSION, "observation_set", case["case_id"], dependency, message),
        )
    if not (case_root / "distances" / "minimum_bidirectional.json").exists():
        write_json_atomic(
            case_root / "distances" / "minimum_bidirectional.json",
            _dependency_artifact(DISTANCE_RECORD_SCHEMA_VERSION, "distance_provider", case["case_id"], dependency, message),
        )
    for selected_condition in condition_ids:
        _write_condition_dependency(
            case_root / "conditions" / selected_condition,
            condition_id_value=selected_condition,
            dependency=dependency,
            message=message,
        )


def _case_record(
    *,
    output_root: Path,
    case: Mapping[str, Any],
    manifest: Mapping[str, Any],
    experiment_id: str,
    smoke: bool,
) -> dict[str, int]:
    case_root = output_root / "cases" / case["case_id"]
    case_root.mkdir(parents=True, exist_ok=True)
    resources = manifest["resource_limits"]
    simulation_result, simulation_runtime, simulation_error = measured_stage(
        lambda: _simulate(case),
        timeout_seconds=None,
        rss_limit_bytes=int(resources["aggregate_rss_limit_bytes_per_worker"]),
    )
    if simulation_error is not None:
        _write_failed_case(
            output_root=output_root,
            case=case,
            condition_ids=case["condition_ids"],
            error=simulation_error,
            stage="simulation",
        )
        return {"success": 0, "failure": 0, "not_run_dependency": len(case["condition_ids"]) * len(REGISTERED_ARM_SPECS)}
    simulator = simulation_result

    try:
        truth_payload, true_tree = _truth_artifact(case["case_id"], simulator)
        write_json_atomic(case_root / "truth.json", truth_payload)
    except Exception as error:
        _write_failed_case(
            output_root=output_root,
            case=case,
            condition_ids=case["condition_ids"],
            error=error,
            stage="truth_validation",
        )
        return {"success": 0, "failure": 0, "not_run_dependency": len(case["condition_ids"]) * len(REGISTERED_ARM_SPECS)}

    try:
        generation_cells = _generation_cells(
            simulator,
            manifest["shared_observation_design"]["maximal_generations"],
        )
        nested = sample_nested_observations(
            generation_cells,
            experiment_id=experiment_id,
            sampling_seed=case["sampling_seed"],
            regime_id=case["regime_id"],
            observation_contract=manifest["shared_observation_design"],
        )
        selected_conditions = [nested.conditions[value] for value in case["condition_ids"]]
        maximum_bound = int(
            resources["max_unique_states_or_occurrences_per_synthetic_reconstruction_condition"]
        )
        maximal_occurrences = sum(
            len(cells) for cells in nested.maximal_cells_by_generation.values()
        )
        if maximal_occurrences > maximum_bound:
            raise ValueError(
                f"Maximal observation union {maximal_occurrences} exceeds bound {maximum_bound}."
            )
        for condition in selected_conditions:
            if condition.occurrence_count > maximum_bound or len(condition.unique_labels) > maximum_bound:
                raise ValueError(
                    f"Condition {condition.condition_id} exceeds the registered size bound."
                )
        write_json_atomic(
            case_root / "observations.json",
            _observation_artifact(case["case_id"], nested, case["condition_ids"]),
        )
    except Exception as error:
        _write_failed_case(
            output_root=output_root,
            case=case,
            condition_ids=case["condition_ids"],
            error=error,
            stage="sampling",
        )
        return {"success": 0, "failure": 0, "not_run_dependency": len(case["condition_ids"]) * len(REGISTERED_ARM_SPECS)}

    maximal_cells = unique_cells_by_cell_id(
        [
            cell
            for cells in nested.maximal_cells_by_generation.values()
            for cell in cells
        ]
    )
    if smoke:
        provider_name = "toy_l1_injected_not_paper_evidence"
        distance_result, distance_runtime, distance_error = measured_stage(
            lambda: _l1_smoke_distance(maximal_cells),
            timeout_seconds=None,
            rss_limit_bytes=int(resources["aggregate_rss_limit_bytes_per_worker"]),
        )
    else:
        provider_name = "cnp2cnp_minimum_bidirectional"
        runtime_config = replace(
            load_ctbs_runtime_config(),
            cnp2cnp_timeout_seconds=float(
                resources["cnp2cnp_timeout_seconds_per_external_process"]
            ),
            cnp2cnp_capture_limit_bytes=int(
                resources["stdout_stderr_limit_bytes_per_process"]
            ),
        )
        provider = Cnp2CnpFileDistanceProvider(runtime_config)
        distance_result, distance_runtime, distance_error = measured_stage(
            lambda: provider.compute(maximal_cells),
            timeout_seconds=int(resources["cnp2cnp_timeout_seconds_per_external_process"]) * 2,
            rss_limit_bytes=int(resources["aggregate_rss_limit_bytes_per_worker"]),
        )
    if distance_error is None:
        try:
            _validate_primary_distance_provenance(distance_result, smoke=smoke)
        except ValueError as error:
            distance_error = error
            distance_result = None

    if distance_error is not None:
        provider_status = status_record(
            entity_type="distance_provider",
            entity_id=case["case_id"],
            status="failure",
            stage="distance",
            code="distance_failure",
            runtime=distance_runtime,
            exception=distance_error,
            attempts=(
                [distance_error.record]
                if getattr(distance_error, "record", None) is not None
                else []
            ),
        )
        write_json_atomic(
            case_root / "distances" / "minimum_bidirectional.json",
            _distance_payload(
                case_id_value=case["case_id"],
                distance=None,
                provider_status=provider_status,
                resources=distance_runtime,
                method=provider_name,
            ),
        )
        dependency = f"distance:{case['case_id']}"
        for selected_condition in case["condition_ids"]:
            _write_condition_dependency(
                case_root / "conditions" / selected_condition,
                condition_id_value=selected_condition,
                dependency=dependency,
                message="Condition was not reconstructed because maximal distance construction failed.",
            )
        write_json_atomic(
            case_root / "case_status.json",
            {
                "schema_version": CASE_SCHEMA_VERSION,
                **json_safe(case),
                "simulation_runtime": simulation_runtime,
                "status_record": status_record(
                    entity_type="case",
                    entity_id=case["case_id"],
                    status="failure",
                    stage="distance",
                    code="distance_failure",
                    dependency=dependency,
                    message=str(distance_error)[:4096],
                ),
            },
        )
        return {"success": 0, "failure": 0, "not_run_dependency": len(case["condition_ids"]) * len(REGISTERED_ARM_SPECS)}

    provider_status = status_record(
        entity_type="distance_provider",
        entity_id=case["case_id"],
        status="success",
        stage="distance",
        code="distance_complete",
        runtime=distance_runtime,
    )
    write_json_atomic(
        case_root / "distances" / "minimum_bidirectional.json",
        _distance_payload(
            case_id_value=case["case_id"],
            distance=distance_result,
            provider_status=provider_status,
            resources=distance_runtime,
            method=provider_name,
        ),
    )

    total_counts = {"success": 0, "failure": 0, "not_run_dependency": 0}
    for selected_condition in case["condition_ids"]:
        condition = nested.conditions[selected_condition]
        input_payload = _condition_input(case["case_id"], condition)
        counts = _write_condition_arms(
            condition_root=case_root / "conditions" / selected_condition,
            input_payload=input_payload,
            maximal_distance=distance_result,
            true_tree=true_tree,
            reconstruction_seed=case["reconstruction_seed"],
            resources=resources,
        )
        for key, value in counts.items():
            total_counts[key] += value

    write_json_atomic(
        case_root / "case_status.json",
        {
            "schema_version": CASE_SCHEMA_VERSION,
            **json_safe(case),
            "simulation_runtime": simulation_runtime,
            "arm_status_counts": total_counts,
            "status_record": status_record(
                entity_type="case",
                entity_id=case["case_id"],
                status="success",
                stage="case_close",
                code="case_complete",
                runtime={"simulation": simulation_runtime, "distance": distance_runtime},
            ),
        },
    )
    return total_counts


def _simulate(case: Mapping[str, Any]) -> CancerCellEvolutionSimulator:
    simulator = CancerCellEvolutionSimulator(
        str(PROJECT_ROOT / case["config_path"]),
        seed=int(case["simulation_seed"]),
    )
    simulator.run_simulation()
    return simulator


def _clean_cases(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    clean = manifest["experiments"]["clean_confirmation"]
    assets = {asset["id"]: asset for asset in manifest["input_assets"]["clean_regimes"]}
    seed_contract = next(
        record
        for record in manifest["seed_contract"]["experiments"]
        if record["experiment_id"] == clean["experiment_id"]
    )
    condition_ids = [
        condition_id(fraction, schedule)
        for fraction in manifest["shared_observation_design"]["fractions"]
        for schedule in manifest["shared_observation_design"]["level_schedules"]
    ]
    cases = []
    for replicate in range(1, int(seed_contract["replicates"]) + 1):
        seeds = {
            f"{stream}_seed": derive_seed(clean["experiment_id"], stream, replicate)
            for stream in seed_contract["streams"]
        }
        for regime_id in clean["regime_ids"]:
            cases.append(
                {
                    "case_id": f"{regime_id}-r{replicate:03d}",
                    "experiment_id": clean["experiment_id"],
                    "regime_id": regime_id,
                    "replicate": replicate,
                    "config_path": assets[regime_id]["path"],
                    "config_sha256": assets[regime_id]["sha256"],
                    "condition_ids": list(condition_ids),
                    **seeds,
                }
            )
    return cases


def _smoke_cases(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    asset = manifest["input_assets"]["clean_regimes"][0]
    return [
        {
            "case_id": "smoke-clean-balanced-r001",
            "experiment_id": SMOKE_EXPERIMENT_ID,
            "regime_id": asset["id"],
            "replicate": 1,
            "config_path": asset["path"],
            "config_sha256": asset["sha256"],
            "condition_ids": [SMOKE_CONDITION],
            "simulation_seed": derive_seed(SMOKE_EXPERIMENT_ID, "simulation", 1),
            "sampling_seed": derive_seed(SMOKE_EXPERIMENT_ID, "sampling", 1),
            "reconstruction_seed": derive_seed(SMOKE_EXPERIMENT_ID, "reconstruction", 1),
        }
    ]


def _external_git_identity(path: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(path.parent), *arguments],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout

    try:
        root = Path(git("rev-parse", "--show-toplevel").strip()).resolve()
        head = git("rev-parse", "HEAD").strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Could not freeze external-tool Git identity for {path}: {exc}") from exc
    return {
        "repository_root": str(root),
        "head": head,
        "status_porcelain_v1": status,
    }


def runtime_environment(*, workers: int, run_kind: str, cnp2cnp_path: Path) -> dict[str, Any]:
    dependency_inventory = sorted(
        {
            (distribution.metadata.get("Name") or distribution.name, distribution.version)
            for distribution in importlib.metadata.distributions()
        },
        key=lambda item: item[0].lower(),
    )
    parallel_variables = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "run_kind": run_kind,
        "python": sys.version,
        "python_executable": sys.executable,
        "dependencies": [
            {"name": name, "version": version}
            for name, version in dependency_inventory
        ],
        "numpy": {
            "version": np.__version__,
            "registered_bit_generator": "PCG64",
            "simulator_rng_stream_spawn_keys": {
                name: [index]
                for index, name in enumerate(
                    CancerCellEvolutionSimulator._RNG_STREAM_NAMES
                )
            },
        },
        "locale": {
            "preferred_encoding": locale.getpreferredencoding(False),
            "current": locale.setlocale(locale.LC_ALL, None),
        },
        "timezone": {
            "tz_environment": os.environ.get("TZ"),
            "time_tzname": list(time.tzname),
        },
        "platform": platform.platform(),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu_count": os.cpu_count(),
        "ram_bytes": None if psutil is None else int(psutil.virtual_memory().total),
        "workers": int(workers),
        "parallelism_environment": parallel_variables,
        "memory_sampling": "psutil process-tree RSS every 10 ms" if psutil else "unavailable",
        "cnp2cnp": {
            "path": str(cnp2cnp_path),
            "sha256": file_sha256(cnp2cnp_path),
            "git": _external_git_identity(cnp2cnp_path),
            "command_template": [sys.executable, str(cnp2cnp_path), "-m", "matrix", "-d", "any"],
        },
    }


def _runtime_preflight(validation: Mapping[str, Any]) -> dict[str, Any]:
    runtime_config = load_ctbs_runtime_config()
    cnp2cnp_path = Path(runtime_config.cnp2cnp_file).expanduser().resolve()
    source = source_freeze_manifest(
        external_paths={"cnp2cnp": cnp2cnp_path},
    )
    smoke_seeds = {
        derive_seed(SMOKE_EXPERIMENT_ID, stream, 1)
        for stream in ("simulation", "sampling", "reconstruction")
    }
    registered = set()
    for experiment in validation["manifest"]["seed_contract"]["experiments"]:
        for replicate in range(1, int(experiment["replicates"]) + 1):
            for stream in experiment["streams"]:
                registered.add(derive_seed(experiment["experiment_id"], stream, replicate))
    calibration = {
        derive_seed("ctbf-v5-g0-05a-calibration-v1", stream, replicate)
        for stream in ("simulation", "crucial", "wgd", "distance")
        for replicate in range(1, 33)
    }
    if smoke_seeds & (registered | calibration):
        raise ValueError("Smoke seeds collide numerically with registered/calibration seeds.")
    return {
        "source_file_count": len(source["files"]),
        "git_head": source["git"]["head"],
        "dirty_inventory_count": len(source["git"]["status_porcelain_v1"]),
        "cnp2cnp_path": str(cnp2cnp_path),
        "cnp2cnp_sha256": source["external_tools"]["cnp2cnp"]["sha256"],
        "smoke_seed_namespace": SMOKE_EXPERIMENT_ID,
        "smoke_seed_separation": True,
        "posix_stage_timeouts_available": hasattr(signal, "SIGALRM"),
    }


def _prepare_root(
    *,
    output_root: Path,
    validation: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    experiment_id: str,
    workers: int,
    run_kind: str,
) -> dict[str, Any]:
    manifest = validation["manifest"]
    runtime_config = load_ctbs_runtime_config()
    cnp2cnp_path = Path(runtime_config.cnp2cnp_file).expanduser().resolve()
    source = source_freeze_manifest(external_paths={"cnp2cnp": cnp2cnp_path})
    environment = runtime_environment(
        workers=workers,
        run_kind=run_kind,
        cnp2cnp_path=cnp2cnp_path,
    )
    inventory = build_expected_inventory(experiment_id=experiment_id, cases=cases)
    write_json_atomic(output_root / "design_manifest.snapshot.json", manifest)
    write_json_atomic(output_root / "source_manifest.json", source)
    write_json_atomic(output_root / "environment.json", environment)
    write_json_atomic(output_root / "expected_inventory.json", inventory)
    write_json_atomic(
        output_root / "run_status.json",
        status_record(
            entity_type="experiment",
            entity_id=experiment_id,
            status="success",
            stage="initialization",
            code="initialized",
            message="Raw execution has started; this record is finalized before checksum close.",
        ),
    )
    return inventory


def _verify_raw_inventory(output_root: Path, inventory: Mapping[str, Any]) -> None:
    missing = [
        relative
        for relative in inventory["raw_files"]
        if not (output_root / relative).is_file()
    ]
    if missing:
        raise ValueError(f"Expected raw artifact inventory is incomplete: {missing[:10]}")
    for relative in inventory["status_files"]:
        validate_status_record(read_json(output_root / relative))


def execute(
    *,
    manifest_path: Path | str,
    output_root: Path | str,
    experiment_id: str,
    workers: int,
    smoke: bool,
) -> dict[str, Any]:
    validation = validate_manifest(manifest_path)
    manifest = validation["manifest"]
    if workers != manifest["resource_limits"]["workers"]:
        raise ValueError("Registered execution requires exactly one worker.")
    if not smoke and experiment_id != REGISTERED_CLEAN_EXPERIMENT:
        raise ValueError(
            "G2-01-A implements only the first registered clean confirmation run; "
            f"unsupported experiment {experiment_id!r}."
        )
    output_root = ensure_new_empty_output_root(output_root)
    cases = _smoke_cases(manifest) if smoke else _clean_cases(manifest)
    effective_experiment_id = SMOKE_EXPERIMENT_ID if smoke else experiment_id
    inventory = _prepare_root(
        output_root=output_root,
        validation=validation,
        cases=cases,
        experiment_id=effective_experiment_id,
        workers=workers,
        run_kind="toy_nonheldout_smoke" if smoke else "registered_clean_confirmation",
    )

    raw_execution_start_ns = time.perf_counter_ns()
    aggregate_counts = {"success": 0, "failure": 0, "not_run_dependency": 0}
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['case_id']}", flush=True)
        counts = _case_record(
            output_root=output_root,
            case=case,
            manifest=manifest,
            experiment_id=effective_experiment_id,
            smoke=smoke,
        )
        for key, value in counts.items():
            aggregate_counts[key] += value

    write_json_atomic(
        output_root / "run_status.json",
        status_record(
            entity_type="experiment",
            entity_id=effective_experiment_id,
            status="success",
            stage="raw_close",
            code="raw_closed",
            message="Every expected raw path exists; aggregate analysis may begin only after checksum validation.",
            runtime={
                "wall_time_ns": time.perf_counter_ns() - raw_execution_start_ns,
                "arm_status_counts": aggregate_counts,
            },
        ),
    )
    _verify_raw_inventory(output_root, inventory)
    raw_entries = write_checksum_file(
        output_root,
        "raw_checksums.sha256",
        include_analysis=False,
    )
    validate_checksum_closure(
        output_root,
        "raw_checksums.sha256",
        include_analysis=False,
    )
    analysis_path = write_analysis(
        output_root,
        run_kind="toy_nonheldout_smoke" if smoke else "registered_clean_confirmation",
    )
    complete_entries = write_checksum_file(
        output_root,
        "complete_checksums.sha256",
        include_analysis=True,
    )
    validate_checksum_closure(
        output_root,
        "complete_checksums.sha256",
        include_analysis=True,
    )
    return {
        "experiment_id": effective_experiment_id,
        "output_root": str(output_root),
        "case_count": len(cases),
        "arm_status_counts": aggregate_counts,
        "raw_checksum_entry_count": len(raw_entries),
        "complete_checksum_entry_count": len(complete_entries),
        "analysis_path": str(analysis_path.relative_to(output_root)),
        "paper_evidence_allowed": not smoke,
    }


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate the frozen v5 contract without writing output.",
    )
    validate_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))

    smoke_parser = subparsers.add_parser("smoke", help="Run one non-held-out injected-distance smoke case.")
    smoke_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    smoke_parser.add_argument("--output-root", required=True)

    run_parser = subparsers.add_parser("run", help="Run one registered experiment into a new empty root.")
    run_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    run_parser.add_argument("--experiment", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.add_argument("--workers", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "validate":
        validation = validate_manifest(args.manifest)
        report = {
            "status": "valid",
            "manifest_id": validation["manifest"]["manifest_id"],
            "manifest_sha256": validation["manifest_sha256"],
            "protocol_sha256": validation["protocol_sha256"],
            "seed_validation": validation["seed_validation"],
            "arm_count": validation["arm_count"],
            "condition_count": validation["condition_count"],
            "runtime_preflight": _runtime_preflight(validation),
            "output_written": False,
        }
    elif args.command == "smoke":
        report = execute(
            manifest_path=args.manifest,
            output_root=args.output_root,
            experiment_id=SMOKE_EXPERIMENT_ID,
            workers=1,
            smoke=True,
        )
    else:
        report = execute(
            manifest_path=args.manifest,
            output_root=args.output_root,
            experiment_id=args.experiment,
            workers=args.workers,
            smoke=False,
        )
    print(json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_BUILD_SPECS",
    "SMOKE_EXPERIMENT_ID",
    "build_expected_inventory",
    "deserialize_tree",
    "execute",
    "main",
    "measured_stage",
    "reconstruct_arm",
    "serialize_tree",
    "validate_reconstruction_input",
]
