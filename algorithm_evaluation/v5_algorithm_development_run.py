"""Run selected reconstruction algorithms on an immutable CTBF v5 bank."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from algorithm_evaluation.paper_pipeline_contract import read_json
from algorithm_evaluation.paper_pipeline_runner import measured_stage
from algorithm_evaluation.process_isolation import (
    CASE_ARM_WORKER_UNIT,
    FreshSpawnPerTaskExecutor,
    fresh_process_contract,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    DEFAULT_BLOCK_COUNT,
    RUN_SCHEMA_VERSION,
    DevelopmentArmSpec,
    canonical_topology_digest,
    ensure_new_output_root,
    load_bank_manifest,
    observed_labels,
    read_case_assets,
    reconstruct_development_arm,
    resolve_arm_specs,
    tree_summary,
    write_json,
)
from ctbs import DistanceMatrix
from evaluation_contract import (
    evaluate_tree_pair_result,
    validate_evaluation_result,
)


DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS = 300
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 300
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
RESULT_NAME = "result.json"
SEMANTIC_GATE_VERSION = "ctbf-v5-algorithm-development-d0-v1"
RECORD_WORKER_GRACE_SECONDS = 120
SEMANTIC_GATE_WORKER_UNIT = "semantic_gate_arm"


def _validate_positive_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")


def _typed_error(error: BaseException, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error)[:4096],
    }


def _run_resource_contract(
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, Any]:
    return {
        "reconstruction_timeout_seconds_per_arm": reconstruction_timeout_seconds,
        "evaluation_timeout_seconds_per_arm": evaluation_timeout_seconds,
        "rss_limit_bytes_per_stage": rss_limit_bytes,
        "record_worker_outer_timeout_seconds": (
            reconstruction_timeout_seconds
            + evaluation_timeout_seconds
            + RECORD_WORKER_GRACE_SECONDS
        ),
        "record_execution": fresh_process_contract(CASE_ARM_WORKER_UNIT),
        "semantic_gate_execution": fresh_process_contract(
            SEMANTIC_GATE_WORKER_UNIT
        ),
    }


def _resume_run(
    *,
    output_root: Path | str,
    run_id: str,
    source_root: Path,
    bank: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    specs: Sequence[DevelopmentArmSpec],
    resources: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], set[tuple[str, str]]]:
    """Load a strictly matching interrupted run and return its record prefix."""
    root = Path(output_root).expanduser().resolve()
    result_path = root / RESULT_NAME
    if not root.is_dir() or not result_path.is_file():
        raise ValueError("A resumed development run requires an existing result root.")
    result = read_json(result_path)
    expected_specs = [spec.as_record() for spec in specs]
    expected_record_count = len(cases) * len(specs)
    exact_fields = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "bank_id": bank["bank_id"],
        "bank_root": str(source_root),
        "block_count": int(bank["block_count"]),
        "condition_count": int(bank["condition_count"]),
        "arm_specs": expected_specs,
        "expected_record_count": expected_record_count,
        "resources": dict(resources),
    }
    for field, expected in exact_fields.items():
        if result.get(field) != expected:
            raise ValueError(
                f"Cannot resume because stored {field} does not match this command."
            )
    previous_status = result.get("status")
    if previous_status == "complete":
        raise ValueError("A completed development run cannot be resumed.")
    if previous_status not in {"in_progress", "failure"}:
        raise ValueError("Only an interrupted or failed development run can be resumed.")
    semantic_gate = result.get("semantic_gate_by_arm")
    if not isinstance(semantic_gate, Mapping) or set(semantic_gate) != {
        spec.arm_id for spec in specs
    }:
        raise ValueError("Cannot resume a run with an incompatible D0 gate inventory.")
    records = result.get("records")
    if not isinstance(records, list) or len(records) > expected_record_count:
        raise ValueError("Cannot resume a run with an invalid record inventory.")
    expected_prefix = [
        (str(case["case_id"]), spec.arm_id)
        for case in cases
        for spec in specs
    ][: len(records)]
    observed_prefix = [
        (str(record.get("case_id")), str(record.get("arm_id")))
        for record in records
        if isinstance(record, Mapping)
    ]
    if len(observed_prefix) != len(records) or observed_prefix != expected_prefix:
        raise ValueError(
            "Cannot resume because stored records are not the exact run-order prefix."
        )

    resumed_at = datetime.now(timezone.utc).isoformat()
    history = result.setdefault("resume_history", [])
    if not isinstance(history, list):
        raise ValueError("Cannot resume a run with invalid resume history.")
    history.append(
        {
            "resumed_at_utc": resumed_at,
            "previous_status": previous_status,
            "preserved_record_count": len(records),
            "previous_runner_failure": result.pop("runner_failure", None),
        }
    )
    result["status"] = "in_progress"
    result["completed_record_count"] = len(records)
    result["completed_condition_count"] = len(records) // len(specs)
    result.pop("completed_at_utc", None)
    result.pop("success_count", None)
    result.pop("failure_count", None)
    write_json(result_path, result)
    return root, result, set(observed_prefix)


def _resource_audit_error(
    resources: Mapping[str, Any],
    stage: str,
) -> RuntimeError | None:
    memory = resources.get("memory")
    if (
        not isinstance(memory, Mapping)
        or memory.get("peak_rss_bytes") is None
    ):
        return RuntimeError(
            f"{stage} has no auditable process-tree peak-RSS measurement."
        )
    return None


def _semantic_fixture(
    fixture_id: str,
    levels: Sequence[tuple[int, Sequence[tuple[int, Sequence[int]]]]],
) -> tuple[dict[str, Any], DistanceMatrix]:
    payload = {
        "schema_version": "ctbf-v5-reconstruction-input-v1",
        "case_id": f"d0-{fixture_id}",
        "condition_id": fixture_id,
        "levels": [
            {
                "biopsy_level": level_index,
                "generation": generation,
                "states": [
                    {"state_label": label, "cnp": list(profile)}
                    for label, profile in states
                ],
            }
            for level_index, (generation, states) in enumerate(levels)
        ],
    }
    profiles_by_label = {}
    for level in payload["levels"]:
        for state in level["states"]:
            label = state["state_label"]
            profile = tuple(state["cnp"])
            previous = profiles_by_label.setdefault(label, profile)
            if previous != profile:
                raise ValueError("D0 fixture repeats one label with a different CNP.")
    ids = sorted(profiles_by_label)
    profiles = np.asarray([profiles_by_label[label] for label in ids], dtype=float)
    matrix = np.abs(profiles[:, None, :] - profiles[None, :, :]).sum(axis=2)
    return payload, DistanceMatrix(ids=ids, matrix=matrix)


def _semantic_fixtures() -> tuple[tuple[str, dict[str, Any], DistanceMatrix], ...]:
    fixtures = (
        (
            "chain_fork",
            (
                (9, ((0, (2, 2, 2, 2)), (1, (2, 2, 2, 3)))),
                (12, ((2, (2, 2, 3, 3)), (3, (1, 2, 2, 3)))),
                (14, ((4, (1, 2, 3, 3)), (5, (1, 3, 3, 3)))),
            ),
        ),
        (
            "repeated_state_copied_parent",
            (
                (9, ((0, (2, 2, 2, 2)), (1, (2, 2, 2, 3)))),
                (12, ((0, (2, 2, 2, 2)), (2, (2, 2, 3, 3)))),
                (14, ((3, (2, 3, 3, 3)), (4, (1, 2, 3, 3)))),
            ),
        ),
        (
            "missing_parent_same_time_exact_tie",
            (
                (9, ((0, (0, 2, 2, 2)), (1, (2, 0, 2, 2)))),
                (12, ((2, (1, 1, 2, 2)), (3, (2, 2, 1, 2)))),
                (12, ((4, (2, 2, 2, 1)), (5, (1, 1, 3, 2)))),
            ),
        ),
    )
    return tuple(
        (fixture_id, *_semantic_fixture(fixture_id, levels))
        for fixture_id, levels in fixtures
    )


def _assert_semantic_output(
    tree: nx.DiGraph,
    reconstruction_input: Mapping[str, Any],
    spec: DevelopmentArmSpec,
) -> None:
    required = set(observed_labels(reconstruction_input))
    output_labels = {
        attributes.get("cell_id")
        for _node, attributes in tree.nodes(data=True)
        if attributes.get("cell_id") is not None
    }
    if not required <= output_labels:
        raise ValueError("D0 output omits one or more declared observation labels.")
    if not output_labels <= required:
        raise ValueError("D0 output invents a biological state label.")
    if len(tree) != len(set(tree.nodes)):
        raise ValueError("D0 output has non-unique graph node identity.")
    unlabeled_count = sum(
        attributes.get("cell_id") is None
        for _node, attributes in tree.nodes(data=True)
    )
    if spec.problem != "partial" and unlabeled_count:
        raise ValueError(
            "A fully labeled D0 family returned an unlabeled internal node."
        )


def _gate_trial(
    spec: DevelopmentArmSpec,
    payload: Mapping[str, Any],
    distance: DistanceMatrix,
    *,
    seed: int,
) -> nx.DiGraph:
    tree, _levels, _root, _metadata = reconstruct_development_arm(
        spec,
        payload,
        distance,
        reconstruction_seed=seed,
    )
    _assert_semantic_output(tree, payload, spec)
    return tree


def _assert_fixture_specific_semantics(
    tree: nx.DiGraph,
    spec: DevelopmentArmSpec,
    fixture_id: str,
) -> None:
    if fixture_id != "repeated_state_copied_parent":
        return
    labels = [
        attributes.get("cell_id")
        for _node, attributes in tree.nodes(data=True)
        if attributes.get("cell_id") is not None
    ]
    repeated_occurrence_count = len(labels) - len(set(labels))
    if spec.problem == "inferred_copy_fully_labeled_closed_state":
        if repeated_occurrence_count < 1:
            raise ValueError("D0 inferred-copy output created no copied state occurrence.")
    if spec.problem == "occurrence_aware_fully_labeled_closed_state":
        if labels.count(0) < 2:
            raise ValueError("D0 temporal output collapsed a repeated observed occurrence.")


def _permuted_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    permuted = copy.deepcopy(payload)
    for level in permuted["levels"]:
        level["states"].reverse()
    return permuted


def _renamed_input_and_distance(
    payload: Mapping[str, Any],
    distance: DistanceMatrix,
) -> tuple[dict[str, Any], DistanceMatrix, dict[int, int]]:
    labels = observed_labels(payload)
    renamed_values = [1009 + 37 * index for index in range(len(labels))][::-1]
    mapping = dict(zip(labels, renamed_values))
    inverse = {value: key for key, value in mapping.items()}
    renamed = copy.deepcopy(payload)
    for level in renamed["levels"]:
        for state in level["states"]:
            state["state_label"] = mapping[state["state_label"]]
    renamed_distance = DistanceMatrix(
        ids=[mapping[label] for label in distance.ids],
        matrix=distance.matrix,
        provenance=distance.provenance,
    )
    return renamed, renamed_distance, inverse


def _digest_with_labels_restored(
    tree: nx.DiGraph,
    inverse_mapping: Mapping[int, int],
) -> str:
    normalized = tree.copy()
    for _node, attributes in normalized.nodes(data=True):
        label = attributes.get("cell_id")
        if label in inverse_mapping:
            attributes["cell_id"] = inverse_mapping[label]
    return canonical_topology_digest(normalized)


def _run_semantic_gate_body(spec: DevelopmentArmSpec) -> dict[str, Any]:
    fixtures = _semantic_fixtures()
    fixture_results = []
    for fixture_index, (fixture_id, payload, distance) in enumerate(fixtures):
        seed = 4100 + fixture_index
        first = _gate_trial(spec, payload, distance, seed=seed)
        _assert_fixture_specific_semantics(first, spec, fixture_id)
        repeated = _gate_trial(spec, payload, distance, seed=seed)
        first_digest = canonical_topology_digest(first)
        if canonical_topology_digest(repeated) != first_digest:
            raise ValueError(f"D0 same-seed instability on fixture {fixture_id}.")
        permuted = _gate_trial(
            spec,
            _permuted_input(payload),
            distance,
            seed=seed,
        )
        alternate_seed = _gate_trial(spec, payload, distance, seed=seed + 10_000)
        row = {
            "fixture_id": fixture_id,
            "same_seed_reproducible": True,
            "within_level_permutation_same_topology": (
                canonical_topology_digest(permuted) == first_digest
            ),
            "different_seed_same_topology": (
                canonical_topology_digest(alternate_seed) == first_digest
            ),
        }
        if fixture_index == 0:
            renamed_input, renamed_distance, inverse = _renamed_input_and_distance(
                payload,
                distance,
            )
            renamed_tree = _gate_trial(
                spec,
                renamed_input,
                renamed_distance,
                seed=seed,
            )
            row["bijective_label_opacity"] = (
                _digest_with_labels_restored(renamed_tree, inverse) == first_digest
            )
            if not row["bijective_label_opacity"]:
                raise ValueError("D0 bijective-label opacity check failed.")
        fixture_results.append(row)
    return {
        "semantic_gate_version": SEMANTIC_GATE_VERSION,
        "status": "passed",
        "truth_supplied_to_reconstruction": False,
        "fixture_results": fixture_results,
    }


def _execute_isolated_semantic_gate(
    spec: DevelopmentArmSpec,
    timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, Any]:
    value, resources, error = measured_stage(
        lambda: _run_semantic_gate_body(spec),
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if error is None:
        error = _resource_audit_error(resources, "D0 semantic gate")
    if error is None and value is not None:
        outcome = dict(value)
        outcome["resources"] = resources
    else:
        gate_error = error or RuntimeError("D0 semantic gate returned no value.")
        outcome = {
            "semantic_gate_version": SEMANTIC_GATE_VERSION,
            "status": "failed",
            "failure": _typed_error(gate_error, "d0_semantic_gate"),
            "resources": resources,
        }
    outcome["execution"] = fresh_process_contract(SEMANTIC_GATE_WORKER_UNIT)
    return outcome


def run_semantic_gate(
    specs: Sequence[DevelopmentArmSpec],
    *,
    timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, dict[str, Any]]:
    outcomes = {}
    with FreshSpawnPerTaskExecutor() as executor:
        for spec in specs:
            outcomes[spec.arm_id] = executor.run(
                _execute_isolated_semantic_gate,
                spec,
                timeout_seconds,
                rss_limit_bytes,
                timeout_seconds=timeout_seconds + RECORD_WORKER_GRACE_SECONDS,
            )
    return outcomes


def _failed_record(
    *,
    case: Mapping[str, Any],
    spec: DevelopmentArmSpec,
    error: BaseException | Mapping[str, Any],
    stage: str,
    reconstruction_runtime: Mapping[str, Any] | None,
    evaluation_runtime: Mapping[str, Any] | None,
) -> dict[str, Any]:
    failure = (
        dict(error)
        if isinstance(error, Mapping)
        else _typed_error(error, stage)
    )
    return {
        "case_id": case["case_id"],
        "block_index": int(case["block_index"]),
        "height": int(case["height"]),
        "arm_id": spec.arm_id,
        "family": spec.family,
        "problem": spec.problem,
        "primary_metric": spec.primary_metric,
        "status": "failure",
        "failure": failure,
        "metrics": None,
        "observation_coverage": None,
        "tree_summary": None,
        "resources": {
            "reconstruction": reconstruction_runtime,
            "evaluation": evaluation_runtime,
        },
    }


def _run_one(
    *,
    case: Mapping[str, Any],
    spec: DevelopmentArmSpec,
    reconstruction_input: Mapping[str, Any],
    distance,
    truth_tree,
    reconstruction_seed: int,
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, Any]:
    reconstruction, reconstruction_runtime, reconstruction_error = measured_stage(
        lambda: reconstruct_development_arm(
            spec,
            reconstruction_input,
            distance,
            reconstruction_seed=reconstruction_seed,
        ),
        timeout_seconds=reconstruction_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if reconstruction_error is None:
        reconstruction_error = _resource_audit_error(
            reconstruction_runtime,
            "reconstruction",
        )
    if reconstruction_error is not None:
        return _failed_record(
            case=case,
            spec=spec,
            error=reconstruction_error,
            stage="reconstruction",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=None,
        )
    if reconstruction is None:
        return _failed_record(
            case=case,
            spec=spec,
            error=RuntimeError("Reconstruction returned no result."),
            stage="reconstruction",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=None,
        )

    tree, _levels, _root, reconstruction_metadata = reconstruction
    labels = observed_labels(reconstruction_input)
    evaluation, evaluation_runtime, evaluation_error = measured_stage(
        lambda: evaluate_tree_pair_result(truth_tree, tree, labels),
        timeout_seconds=evaluation_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if evaluation_error is None:
        evaluation_error = _resource_audit_error(
            evaluation_runtime,
            "evaluation",
        )
    if evaluation_error is not None:
        return _failed_record(
            case=case,
            spec=spec,
            error=evaluation_error,
            stage="evaluation",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=evaluation_runtime,
        )
    if evaluation is None:
        return _failed_record(
            case=case,
            spec=spec,
            error=RuntimeError("Evaluation returned no result."),
            stage="evaluation",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=evaluation_runtime,
        )
    validate_evaluation_result(evaluation)
    if evaluation["status"] != "success":
        failure = dict(evaluation["failure"])
        failure["stage"] = f"evaluation:{failure['stage']}"
        return _failed_record(
            case=case,
            spec=spec,
            error=failure,
            stage="evaluation",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=evaluation_runtime,
        )

    coverage = evaluation["inputs"]["observation_label_coverage"]
    if coverage["fraction"] < 1.0:
        return _failed_record(
            case=case,
            spec=spec,
            error={
                "stage": "evaluation",
                "type": "ObservationCoverageError",
                "message": "Reconstructed observation-label coverage is below one.",
            },
            stage="evaluation",
            reconstruction_runtime=reconstruction_runtime,
            evaluation_runtime=evaluation_runtime,
        )

    metrics = evaluation["metrics"]
    declared_metrics = {
        name: float(metrics[name])
        for name in (spec.primary_metric, *spec.complementary_metrics)
    }
    return {
        "case_id": case["case_id"],
        "block_index": int(case["block_index"]),
        "height": int(case["height"]),
        "arm_id": spec.arm_id,
        "family": spec.family,
        "problem": spec.problem,
        "primary_metric": spec.primary_metric,
        "status": "success",
        "failure": None,
        "metrics": declared_metrics,
        "observation_coverage": {
            "required_unique_label_count": coverage["required_unique_label_count"],
            "reconstructed_unique_label_count": coverage[
                "reconstructed_unique_label_count"
            ],
            "fraction": float(coverage["fraction"]),
        },
        "tree_summary": tree_summary(tree),
        "reconstruction_metadata": reconstruction_metadata,
        "resources": {
            "reconstruction": reconstruction_runtime,
            "evaluation": evaluation_runtime,
        },
    }


def _execute_isolated_case_arm(
    source_root: str,
    case: Mapping[str, Any],
    spec: DevelopmentArmSpec,
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, Any]:
    """Load and execute exactly one case-arm record in a fresh worker."""
    reconstruction_input, distance, truth_tree, metadata = read_case_assets(
        Path(source_root),
        case,
    )
    return _run_one(
        case=case,
        spec=spec,
        reconstruction_input=reconstruction_input,
        distance=distance,
        truth_tree=truth_tree,
        reconstruction_seed=int(metadata["reconstruction_seed"]),
        reconstruction_timeout_seconds=reconstruction_timeout_seconds,
        evaluation_timeout_seconds=evaluation_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )


def run_algorithms(
    *,
    bank_root: Path | str,
    output_root: Path | str,
    run_id: str,
    arm_ids: Sequence[str] | None = None,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    evaluation_timeout_seconds: int = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    expected_block_count: int = DEFAULT_BLOCK_COUNT,
    created_at_utc: str | None = None,
    progress: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if not run_id or not run_id.strip():
        raise ValueError("run_id must be nonempty.")
    for field, value in (
        ("reconstruction_timeout_seconds", reconstruction_timeout_seconds),
        ("evaluation_timeout_seconds", evaluation_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("expected_block_count", expected_block_count),
    ):
        _validate_positive_integer(value, field)
    specs = resolve_arm_specs(arm_ids)
    source_root, bank = load_bank_manifest(
        bank_root,
        expected_block_count=expected_block_count,
    )
    cases = sorted(
        bank["cases"],
        key=lambda case: (int(case["block_index"]), int(case["height"])),
    )
    resource_contract = _run_resource_contract(
        reconstruction_timeout_seconds,
        evaluation_timeout_seconds,
        rss_limit_bytes,
    )
    if resume:
        root, result, completed_keys = _resume_run(
            output_root=output_root,
            run_id=run_id,
            source_root=source_root,
            bank=bank,
            cases=cases,
            specs=specs,
            resources=resource_contract,
        )
        semantic_gate = result["semantic_gate_by_arm"]
    else:
        root = ensure_new_output_root(output_root)
        semantic_gate = run_semantic_gate(
            specs,
            timeout_seconds=reconstruction_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )
        result = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "in_progress",
            "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
            "scientific_role": "method_development_only_not_paper_accuracy_evidence",
            "bank_id": bank["bank_id"],
            "bank_root": str(source_root),
            "block_count": int(bank["block_count"]),
            "condition_count": int(bank["condition_count"]),
            "arm_specs": [spec.as_record() for spec in specs],
            "semantic_gate_by_arm": semantic_gate,
            "expected_record_count": len(cases) * len(specs),
            "resources": resource_contract,
            "records": [],
        }
        completed_keys = set()
        write_json(root / RESULT_NAME, result)

    try:
        with FreshSpawnPerTaskExecutor() as record_executor:
            for case_index, case in enumerate(cases):
                case_keys = {
                    (str(case["case_id"]), spec.arm_id)
                    for spec in specs
                }
                if case_keys <= completed_keys:
                    continue
                for spec in specs:
                    record_key = (str(case["case_id"]), spec.arm_id)
                    if record_key in completed_keys:
                        continue
                    gate = semantic_gate[spec.arm_id]
                    if gate["status"] != "passed":
                        record = _failed_record(
                            case=case,
                            spec=spec,
                            error={
                                "stage": "d0_semantic_gate",
                                "type": "D0SemanticGateFailure",
                                "message": (
                                    "Arm was not admitted to D1 accuracy screening."
                                ),
                                "gate_failure": gate.get("failure"),
                            },
                            stage="d0_semantic_gate",
                            reconstruction_runtime=gate.get("resources"),
                            evaluation_runtime=None,
                        )
                    else:
                        record = record_executor.run(
                            _execute_isolated_case_arm,
                            str(source_root),
                            dict(case),
                            spec,
                            reconstruction_timeout_seconds,
                            evaluation_timeout_seconds,
                            rss_limit_bytes,
                            timeout_seconds=resource_contract[
                                "record_worker_outer_timeout_seconds"
                            ],
                        )
                    result["records"].append(record)
                    completed_keys.add(record_key)
                    if progress:
                        print(
                            f"{case['case_id']} {spec.arm_id}: {record['status']}",
                            file=sys.stderr,
                            flush=True,
                        )
                result["completed_condition_count"] = case_index + 1
                result["completed_record_count"] = len(result["records"])
                write_json(root / RESULT_NAME, result)
    except BaseException as error:
        result["status"] = "failure"
        result["runner_failure"] = _typed_error(error, "runner")
        result["completed_record_count"] = len(result["records"])
        write_json(root / RESULT_NAME, result)
        raise

    result["status"] = "complete"
    result["completed_record_count"] = len(result["records"])
    result["success_count"] = sum(
        record["status"] == "success" for record in result["records"]
    )
    result["failure_count"] = len(result["records"]) - result["success_count"]
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / RESULT_NAME, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["all"],
        help=(
            "Explicit arm ids; 'all' for the immutable initial 32-arm roster; "
            "or 'biopsy_guided_full' for the approved ten-arm full-output "
            "extension."
        ),
    )
    parser.add_argument(
        "--reconstruction-timeout-seconds",
        type=int,
        default=DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--evaluation-timeout-seconds",
        type=int,
        default=DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume the exact stored run-order prefix in --output-root; all "
            "bank, run, arm, and resource declarations must match."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_algorithms(
        bank_root=arguments.bank_root,
        output_root=arguments.output_root,
        run_id=arguments.run_id,
        arm_ids=arguments.arms,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        progress=arguments.progress,
        resume=arguments.resume,
    )
    print(
        f"complete: {result['completed_record_count']} arm records; "
        f"{result['failure_count']} failures"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
