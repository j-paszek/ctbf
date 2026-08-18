"""Post-registration mechanism diagnostics for temporal CNP arborescences.

The frozen-tree stage reads already closed reconstruction artifacts and never
reruns reconstruction.  The optional seed-sensitivity stage reconstructs only
from the stored observable input and stored distance matrix, using a fixed
truth-independent seed list.  Neither stage writes into the source corpus.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from algorithm_evaluation.paper_pipeline_contract import (
    EXPECTED_INVENTORY_SCHEMA_VERSION,
    PROJECT_ROOT,
    aligned_distance_submatrix,
    canonical_json_sha256,
    file_sha256,
    json_safe,
    read_json,
    validate_checksum_closure,
    validate_status_record,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import (
    DISTANCE_RECORD_SCHEMA_VERSION,
    RECONSTRUCTION_RESULT_SCHEMA_VERSION,
    deserialize_tree,
    reconstruct_arm,
    validate_reconstruction_input,
)
from ctbs import DistanceMatrix
from distance_semantics import CNP2CNP_SEMANTICS_VERSION, stable_distance_label_key
from evaluation_contract import validate_evaluation_result
from evaluator_full import normalize_cell_labels, prf1_iou, unique_ancestor_pair_set


DIAGNOSTIC_SCHEMA_VERSION = "ctbf-v5-temporal-arborescence-diagnostic-v1"
DIAGNOSTIC_SEED_NAMESPACE = "ctbf-v5-temporal-arborescence-seed-diagnostic-v1"
DIAGNOSTIC_SEEDS = (
    4224489865,
    329440355,
    884007044,
    3973510421,
    991571315,
    459553856,
    3319943489,
    3548649098,
)
ANCHOR_CONDITION = "f0p50_L3"
TEMPORAL_ARMS = ("temporal_minimum", "temporal_minimum_no_time")
AD_F1_EQUALITY_TOLERANCE = 1e-12
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "experimental_results" / "ctbf_v5_clean_confirmation_v1"


def _derived_diagnostic_seeds(count: int = 8) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(
            sha256(f"{DIAGNOSTIC_SEED_NAMESPACE}:seed:{index}".encode("utf-8")).digest()[:4],
            "big",
        )
        for index in range(1, count + 1)
    )


if DIAGNOSTIC_SEEDS != _derived_diagnostic_seeds():  # pragma: no cover - import invariant
    raise RuntimeError("Frozen temporal diagnostic seeds disagree with their namespace.")


def _stable_key(value: Any) -> tuple[str, str]:
    return f"{type(value).__module__}.{type(value).__qualname__}", repr(value)


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized) if normalized else None,
        "median": statistics.median(normalized) if normalized else None,
        "minimum": min(normalized) if normalized else None,
        "maximum": max(normalized) if normalized else None,
    }


def _fraction(count: int, total: int) -> float | None:
    return float(count / total) if total else None


def _actual_root(tree: nx.DiGraph) -> Any:
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}.")
    return roots[0]


def occurrence_identity_signature(tree: nx.DiGraph) -> str:
    records = []
    for node, attributes in tree.nodes(data=True):
        records.append(
            {
                "node_id": json_safe(node),
                "cell_id": json_safe(attributes.get("cell_id")),
                "biopsy_level": attributes.get("biopsy_level"),
                "genome": json_safe(np.asarray(attributes.get("genome")).tolist()),
            }
        )
    records.sort(key=lambda record: _stable_key(record["node_id"]))
    return canonical_json_sha256(records)


def occurrence_tree_signature(tree: nx.DiGraph) -> str:
    root = _actual_root(tree)
    edges = sorted(
        ((json_safe(parent), json_safe(child)) for parent, child in tree.edges()),
        key=lambda edge: (_stable_key(edge[0]), _stable_key(edge[1])),
    )
    return canonical_json_sha256({"root": json_safe(root), "edges": edges})


def observed_labels_from_input(payload: Mapping[str, Any]) -> list[Any]:
    labels = {
        state["state_label"]
        for level in payload["levels"]
        for state in level["states"]
    }
    return sorted(labels, key=stable_distance_label_key)


def ancestor_pair_set(tree: nx.DiGraph, observed_labels: Iterable[Any]) -> set[tuple[Any, Any]]:
    return unique_ancestor_pair_set(tree, restrict_labels=observed_labels)


def ancestor_pair_signature(pairs: Iterable[tuple[Any, Any]]) -> str:
    records = sorted(
        ([json_safe(ancestor), json_safe(descendant)] for ancestor, descendant in pairs),
        key=lambda pair: (_stable_key(pair[0]), _stable_key(pair[1])),
    )
    return canonical_json_sha256(records)


def ad_f1_from_pair_sets(
    true_pairs: set[tuple[Any, Any]],
    reconstructed_pairs: set[tuple[Any, Any]],
) -> dict[str, Any]:
    tp = len(true_pairs & reconstructed_pairs)
    fp = len(reconstructed_pairs - true_pairs)
    fn = len(true_pairs - reconstructed_pairs)
    precision, recall, f1, iou = prf1_iou(tp, fp, fn)
    return {
        "ad_f1": float(f1),
        "ad_precision": float(precision),
        "ad_recall": float(recall),
        "ad_iou": float(iou),
        "counts": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "true_unique_pair_count": len(true_pairs),
            "reconstructed_unique_pair_count": len(reconstructed_pairs),
        },
    }


def tree_scientific_diagnostics(
    tree: nx.DiGraph,
    distance: DistanceMatrix,
) -> dict[str, Any]:
    """Return the three scientific tiers and temporal-feasibility diagnostics."""
    if not nx.is_arborescence(tree):
        raise ValueError("Temporal diagnostic requires one directed arborescence.")
    ids = list(distance.ids)
    matrix = np.asarray(distance.matrix, dtype=float)
    positions = {cell_id: index for index, cell_id in enumerate(ids)}
    if len(positions) != len(ids):
        raise ValueError("Diagnostic distance ids must be unique.")

    root = _actual_root(tree)
    levels = []
    for node, attributes in tree.nodes(data=True):
        if "cell_id" not in attributes or attributes["cell_id"] not in positions:
            raise ValueError(f"Tree node {node!r} has no aligned distance state.")
        if "biopsy_level" not in attributes:
            raise ValueError(f"Tree node {node!r} has no biopsy level.")
        genome = np.asarray(attributes.get("genome"))
        if genome.ndim != 1 or not genome.size:
            raise ValueError(f"Tree node {node!r} has no valid CNP.")
        levels.append(int(attributes["biopsy_level"]))

    violation_count = 0
    backward_edges = []
    same_level_edge_count = 0
    forward_edge_count = 0
    edge_distances = []
    for parent, child, attributes in tree.edges(data=True):
        parent_attributes = tree.nodes[parent]
        child_attributes = tree.nodes[child]
        parent_genome = np.asarray(parent_attributes["genome"])
        child_genome = np.asarray(child_attributes["genome"])
        violation_count += int(np.any((parent_genome == 0) & (child_genome > 0)))
        parent_level = int(parent_attributes["biopsy_level"])
        child_level = int(child_attributes["biopsy_level"])
        if parent_level > child_level:
            backward_edges.append([json_safe(parent), json_safe(child)])
        elif parent_level == child_level:
            same_level_edge_count += 1
        else:
            forward_edge_count += 1

        expected = float(
            matrix[
                positions[parent_attributes["cell_id"]],
                positions[child_attributes["cell_id"]],
            ]
        )
        stored = attributes.get("weight")
        if stored is not None and not math.isclose(
            float(stored), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("Stored reconstruction edge weight disagrees with the matrix.")
        edge_distances.append(expected)

    root_state = tree.nodes[root]["cell_id"]
    root_position = positions[root_state]
    root_score = math.fsum(
        float(matrix[root_position, positions[attributes["cell_id"]]])
        for node, attributes in tree.nodes(data=True)
        if node != root
    )
    earliest_level = min(levels)
    root_level = int(tree.nodes[root]["biopsy_level"])
    total_distance = math.fsum(edge_distances)
    return {
        "root_node_id": json_safe(root),
        "root_cell_id": json_safe(root_state),
        "root_biopsy_level": root_level,
        "earliest_biopsy_level": earliest_level,
        "late_root": root_level > earliest_level,
        "backward_edge_count": len(backward_edges),
        "backward_edges": backward_edges,
        "same_level_edge_count": same_level_edge_count,
        "forward_edge_count": forward_edge_count,
        "constraint_active": bool(backward_edges or root_level > earliest_level),
        "plausibility_violation_edge_count": int(violation_count),
        "total_edge_distance": float(total_distance),
        "root_score": float(root_score),
        "scientific_objective_tuple": [
            int(violation_count),
            float(total_distance),
            float(root_score),
        ],
    }


def _validate_stored_ad_f1(stored: Mapping[str, Any], recomputed: Mapping[str, Any]) -> None:
    validate_evaluation_result(stored)
    if stored.get("status") != "success":
        raise ValueError("Stored temporal evaluation is not successful.")
    if not math.isclose(
        float(stored["metrics"]["ad_f1"]),
        float(recomputed["ad_f1"]),
        rel_tol=0.0,
        abs_tol=AD_F1_EQUALITY_TOLERANCE,
    ):
        raise ValueError("Stored AD-F1 disagrees with the diagnostic pair-set value.")
    stored_counts = stored["metrics"]["ad_counts"]
    if stored_counts != recomputed["counts"]:
        raise ValueError("Stored AD-F1 counts disagree with diagnostic pair sets.")


@dataclass(frozen=True)
class _CaseData:
    metadata: Mapping[str, Any]
    input_payload: Mapping[str, Any]
    distance: DistanceMatrix
    observed_labels: Sequence[Any]
    true_pairs: set[tuple[Any, Any]]
    frozen_trees: Mapping[str, nx.DiGraph]
    frozen_evaluations: Mapping[str, Mapping[str, Any]]


def _load_distance(
    case_root: Path,
    input_payload: Mapping[str, Any],
    case_id: str,
) -> DistanceMatrix:
    record = read_json(case_root / "distances" / "minimum_bidirectional.json")
    if record.get("schema_version") != DISTANCE_RECORD_SCHEMA_VERSION:
        raise ValueError("Frozen distance record has the wrong schema.")
    if record.get("case_id") != case_id:
        raise ValueError("Frozen distance record has the wrong case id.")
    validate_status_record(record["status_record"])
    if record["status_record"]["status"] != "success":
        raise ValueError("Frozen distance record is not successful.")
    provenance = record.get("provenance", {})
    if provenance.get("semantics_version") != CNP2CNP_SEMANTICS_VERSION:
        raise ValueError("Frozen distance semantics are not the approved cnp2cnp minimum.")

    requested_ids = list(input_payload["distance"]["ids"])
    sub_ids, submatrix = aligned_distance_submatrix(
        record["ids"],
        record["matrix"],
        requested_ids,
    )
    if list(sub_ids) != requested_ids:
        raise ValueError("Frozen condition distance ids are not canonically aligned.")
    metadata = input_payload["distance"]
    if list(submatrix.shape) != metadata["matrix_shape"]:
        raise ValueError("Frozen condition distance shape disagrees with its metadata.")
    digest = sha256(
        np.asarray(submatrix, dtype="<f8", order="C").tobytes(order="C")
    ).hexdigest()
    if digest != metadata["matrix_sha256_float64_c_order"]:
        raise ValueError("Frozen condition distance bytes disagree with their digest.")
    if metadata.get("semantic_version") != CNP2CNP_SEMANTICS_VERSION:
        raise ValueError("Frozen condition input declares the wrong distance semantics.")
    return DistanceMatrix(ids=sub_ids, matrix=submatrix, provenance=provenance)


def _load_successful_arm(case_root: Path, arm_id: str):
    arm_root = case_root / "conditions" / ANCHOR_CONDITION / "arms" / arm_id
    status = read_json(arm_root / "status.json")
    validate_status_record(status)
    if status["status"] != "success":
        raise ValueError(f"Frozen arm {arm_id} is not successful.")
    reconstruction = read_json(arm_root / "reconstruction.json")
    if (
        reconstruction.get("schema_version") != RECONSTRUCTION_RESULT_SCHEMA_VERSION
        or reconstruction.get("status") != "success"
    ):
        raise ValueError(f"Frozen arm {arm_id} has no successful reconstruction.")
    if reconstruction.get("metadata", {}).get("arm_id") != arm_id:
        raise ValueError(f"Frozen arm {arm_id} metadata disagree.")
    evaluation = read_json(arm_root / "evaluation.json")
    validate_evaluation_result(evaluation)
    if evaluation.get("status") != "success":
        raise ValueError(f"Frozen arm {arm_id} has no successful evaluation.")
    return deserialize_tree(reconstruction["tree"]), evaluation


def _load_case(source_root: Path, metadata: Mapping[str, Any]) -> _CaseData:
    case_id = str(metadata["case_id"])
    if ANCHOR_CONDITION not in metadata.get("condition_ids", []):
        raise ValueError(f"Case {case_id} lacks the registered anchor condition.")
    case_root = source_root / "cases" / case_id
    input_payload = read_json(
        case_root / "conditions" / ANCHOR_CONDITION / "input.json"
    )
    validate_reconstruction_input(input_payload)
    if input_payload.get("case_id") != case_id or input_payload.get("condition_id") != ANCHOR_CONDITION:
        raise ValueError("Frozen reconstruction input has the wrong identity.")
    distance = _load_distance(case_root, input_payload, case_id)
    observed_labels = observed_labels_from_input(input_payload)

    truth_record = read_json(case_root / "truth.json")
    if truth_record.get("case_id") != case_id or "tree" not in truth_record:
        raise ValueError("Frozen truth record has the wrong identity.")
    truth = deserialize_tree(truth_record["tree"])
    true_pairs = ancestor_pair_set(truth, observed_labels)

    frozen_trees = {}
    frozen_evaluations = {}
    for arm_id in TEMPORAL_ARMS:
        tree, evaluation = _load_successful_arm(case_root, arm_id)
        frozen_trees[arm_id] = tree
        frozen_evaluations[arm_id] = evaluation
    identities = {occurrence_identity_signature(tree) for tree in frozen_trees.values()}
    if len(identities) != 1:
        raise ValueError("Paired frozen arms do not use identical occurrence vertices.")
    return _CaseData(
        metadata=metadata,
        input_payload=input_payload,
        distance=distance,
        observed_labels=observed_labels,
        true_pairs=true_pairs,
        frozen_trees=frozen_trees,
        frozen_evaluations=frozen_evaluations,
    )


def frozen_case_audit(case: _CaseData) -> dict[str, Any]:
    arm_records = {}
    pair_sets = {}
    for arm_id in TEMPORAL_ARMS:
        tree = case.frozen_trees[arm_id]
        pairs = ancestor_pair_set(tree, case.observed_labels)
        ad = ad_f1_from_pair_sets(case.true_pairs, pairs)
        _validate_stored_ad_f1(case.frozen_evaluations[arm_id], ad)
        pair_sets[arm_id] = pairs
        arm_records[arm_id] = {
            "occurrence_tree_sha256": occurrence_tree_signature(tree),
            "ancestor_pair_set_sha256": ancestor_pair_signature(pairs),
            "ancestor_pair_count": len(pairs),
            "ad_f1": ad,
            "tree_diagnostics": tree_scientific_diagnostics(tree, case.distance),
        }

    temporal = arm_records["temporal_minimum"]
    no_time = arm_records["temporal_minimum_no_time"]
    same_tree = temporal["occurrence_tree_sha256"] == no_time["occurrence_tree_sha256"]
    same_pairs = pair_sets["temporal_minimum"] == pair_sets["temporal_minimum_no_time"]
    temporal_ad = float(temporal["ad_f1"]["ad_f1"])
    no_time_ad = float(no_time["ad_f1"]["ad_f1"])
    same_ad = math.isclose(
        temporal_ad,
        no_time_ad,
        rel_tol=0.0,
        abs_tol=AD_F1_EQUALITY_TOLERANCE,
    )
    return {
        "case_id": case.metadata["case_id"],
        "replicate": int(case.metadata["replicate"]),
        "regime_id": case.metadata["regime_id"],
        "occurrence_identity_sha256": occurrence_identity_signature(
            case.frozen_trees["temporal_minimum"]
        ),
        "arms": arm_records,
        "comparison": {
            "same_directed_occurrence_tree": same_tree,
            "same_unique_observed_label_ancestor_pair_set": same_pairs,
            "same_ad_f1": same_ad,
            "different_tree_same_pair_set": (not same_tree) and same_pairs,
            "different_pair_set_same_ad_f1": (not same_pairs) and same_ad,
            "temporal_minus_no_time_ad_f1": temporal_ad - no_time_ad,
            "no_time_constraint_active": no_time["tree_diagnostics"]["constraint_active"],
        },
    }


def _frozen_group_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    comparisons = [record["comparison"] for record in records]
    no_time_diagnostics = [
        record["arms"]["temporal_minimum_no_time"]["tree_diagnostics"]
        for record in records
    ]

    def count(field: str) -> int:
        return sum(bool(record[field]) for record in comparisons)

    active = count("no_time_constraint_active")
    same_tree = count("same_directed_occurrence_tree")
    same_pairs = count("same_unique_observed_label_ancestor_pair_set")
    same_ad = count("same_ad_f1")
    different_tree_same_pairs = count("different_tree_same_pair_set")
    different_pairs_same_ad = count("different_pair_set_same_ad_f1")
    late_roots = sum(record["late_root"] for record in no_time_diagnostics)
    backward_cases = sum(record["backward_edge_count"] > 0 for record in no_time_diagnostics)
    inactive_different_tree = sum(
        not record["no_time_constraint_active"]
        and not record["same_directed_occurrence_tree"]
        for record in comparisons
    )
    active_same_pairs = sum(
        record["no_time_constraint_active"]
        and record["same_unique_observed_label_ancestor_pair_set"]
        for record in comparisons
    )
    arm_objectives = {
        arm_id: [
            record["arms"][arm_id]["tree_diagnostics"]["scientific_objective_tuple"]
            for record in records
        ]
        for arm_id in TEMPORAL_ARMS
    }
    return {
        "case_count": total,
        "no_time_constraint_active": {"count": active, "fraction": _fraction(active, total)},
        "no_time_backward_edge_present": {
            "count": backward_cases,
            "fraction": _fraction(backward_cases, total),
        },
        "no_time_late_root": {"count": late_roots, "fraction": _fraction(late_roots, total)},
        "no_time_backward_edge_count": _numeric_summary(
            record["backward_edge_count"] for record in no_time_diagnostics
        ),
        "same_directed_occurrence_tree": {
            "count": same_tree,
            "fraction": _fraction(same_tree, total),
        },
        "same_unique_observed_label_ancestor_pair_set": {
            "count": same_pairs,
            "fraction": _fraction(same_pairs, total),
        },
        "same_ad_f1": {"count": same_ad, "fraction": _fraction(same_ad, total)},
        "different_tree_same_pair_set": {
            "count": different_tree_same_pairs,
            "fraction": _fraction(different_tree_same_pairs, total),
        },
        "different_pair_set_same_ad_f1": {
            "count": different_pairs_same_ad,
            "fraction": _fraction(different_pairs_same_ad, total),
        },
        "constraint_inactive_but_tree_different": {
            "count": inactive_different_tree,
            "fraction": _fraction(inactive_different_tree, total),
        },
        "constraint_active_but_pair_set_same": {
            "count": active_same_pairs,
            "fraction": _fraction(active_same_pairs, total),
        },
        "temporal_minus_no_time_ad_f1": _numeric_summary(
            record["temporal_minus_no_time_ad_f1"] for record in comparisons
        ),
        "selected_scientific_objectives": {
            arm_id: {
                "plausibility_violation_edge_count": _numeric_summary(
                    objective[0] for objective in objectives
                ),
                "total_edge_distance": _numeric_summary(
                    objective[1] for objective in objectives
                ),
                "root_score": _numeric_summary(objective[2] for objective in objectives),
            }
            for arm_id, objectives in arm_objectives.items()
        },
        "temporal_minus_no_time_scientific_objective": {
            "plausibility_violation_edge_count": _numeric_summary(
                temporal[0] - no_time[0]
                for temporal, no_time in zip(
                    arm_objectives["temporal_minimum"],
                    arm_objectives["temporal_minimum_no_time"],
                )
            ),
            "total_edge_distance": _numeric_summary(
                temporal[1] - no_time[1]
                for temporal, no_time in zip(
                    arm_objectives["temporal_minimum"],
                    arm_objectives["temporal_minimum_no_time"],
                )
            ),
            "root_score": _numeric_summary(
                temporal[2] - no_time[2]
                for temporal, no_time in zip(
                    arm_objectives["temporal_minimum"],
                    arm_objectives["temporal_minimum_no_time"],
                )
            ),
        },
    }


def summarize_frozen_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    regimes = sorted({str(record["regime_id"]) for record in records})
    return {
        "overall": _frozen_group_summary(records),
        "by_regime": {
            regime: _frozen_group_summary(
                [record for record in records if record["regime_id"] == regime]
            )
            for regime in regimes
        },
    }


def _seed_arm_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    objective_tuples = {
        tuple(run["tree_diagnostics"]["scientific_objective_tuple"])
        for run in runs
    }
    if len(objective_tuples) != 1:
        raise ValueError("Reconstruction seed changed a higher-priority scientific objective.")
    tree_hashes = {run["occurrence_tree_sha256"] for run in runs}
    pair_hashes = {run["ancestor_pair_set_sha256"] for run in runs}
    roots = {
        (
            json.dumps(run["tree_diagnostics"]["root_node_id"], sort_keys=True),
            json.dumps(run["tree_diagnostics"]["root_cell_id"], sort_keys=True),
            int(run["tree_diagnostics"]["root_biopsy_level"]),
        )
        for run in runs
    }
    ad_values = [float(run["ad_f1"]["ad_f1"]) for run in runs]
    distinct_ad_values = set(ad_values)
    return {
        "seed_count": len(runs),
        "scientific_objective_invariant": True,
        "scientific_objective_tuple": list(next(iter(objective_tuples))),
        "distinct_directed_occurrence_tree_count": len(tree_hashes),
        "distinct_unique_observed_label_ancestor_pair_set_count": len(pair_hashes),
        "distinct_root_count": len(roots),
        "distinct_ad_f1_count": len(distinct_ad_values),
        "tree_seed_sensitive": len(tree_hashes) > 1,
        "ancestor_pair_set_seed_sensitive": len(pair_hashes) > 1,
        "ad_f1_seed_sensitive": len(distinct_ad_values) > 1,
        "tree_sensitive_but_pair_set_invariant": len(tree_hashes) > 1 and len(pair_hashes) == 1,
        "pair_set_sensitive_but_ad_f1_invariant": len(pair_hashes) > 1 and len(distinct_ad_values) == 1,
        "ad_f1": {**_numeric_summary(ad_values), "range": max(ad_values) - min(ad_values)},
    }


def seed_sensitivity_case(case: _CaseData) -> dict[str, Any]:
    frozen_identity = occurrence_identity_signature(case.frozen_trees["temporal_minimum"])
    reconstructed_by_arm: dict[str, list[tuple[int, nx.DiGraph, dict[str, Any]]]] = {}

    # Reconstruction is completed from observable bytes before truth-derived
    # pair sets or AD-F1 are consulted below.
    for arm_id in TEMPORAL_ARMS:
        reconstructed = []
        for seed in DIAGNOSTIC_SEEDS:
            tree, _levels, _returned_root, _metadata = reconstruct_arm(
                arm_id,
                case.input_payload,
                case.distance,
                reconstruction_seed=seed,
            )
            if occurrence_identity_signature(tree) != frozen_identity:
                raise ValueError("Diagnostic rerun changed the frozen occurrence identity set.")
            reconstructed.append((seed, tree, tree_scientific_diagnostics(tree, case.distance)))
        reconstructed_by_arm[arm_id] = reconstructed

    arm_records = {}
    pair_sets_by_arm: dict[str, dict[int, set[tuple[Any, Any]]]] = {}
    for arm_id, reconstructed in reconstructed_by_arm.items():
        runs = []
        pair_sets_by_arm[arm_id] = {}
        for seed, tree, diagnostics in reconstructed:
            pairs = ancestor_pair_set(tree, case.observed_labels)
            pair_sets_by_arm[arm_id][seed] = pairs
            runs.append(
                {
                    "seed": seed,
                    "occurrence_tree_sha256": occurrence_tree_signature(tree),
                    "ancestor_pair_set_sha256": ancestor_pair_signature(pairs),
                    "ancestor_pair_count": len(pairs),
                    "ad_f1": ad_f1_from_pair_sets(case.true_pairs, pairs),
                    "tree_diagnostics": diagnostics,
                }
            )
        arm_records[arm_id] = {"summary": _seed_arm_summary(runs), "runs": runs}

    paired_runs = []
    for index, seed in enumerate(DIAGNOSTIC_SEEDS):
        temporal = arm_records["temporal_minimum"]["runs"][index]
        no_time = arm_records["temporal_minimum_no_time"]["runs"][index]
        if temporal["seed"] != seed or no_time["seed"] != seed:
            raise ValueError("Diagnostic seed alignment failed.")
        temporal_ad = float(temporal["ad_f1"]["ad_f1"])
        no_time_ad = float(no_time["ad_f1"]["ad_f1"])
        paired_runs.append(
            {
                "seed": seed,
                "same_directed_occurrence_tree": (
                    temporal["occurrence_tree_sha256"] == no_time["occurrence_tree_sha256"]
                ),
                "same_unique_observed_label_ancestor_pair_set": (
                    temporal["ancestor_pair_set_sha256"] == no_time["ancestor_pair_set_sha256"]
                ),
                "temporal_minus_no_time_ad_f1": temporal_ad - no_time_ad,
                "no_time_constraint_active": no_time["tree_diagnostics"]["constraint_active"],
            }
        )
    effects = [run["temporal_minus_no_time_ad_f1"] for run in paired_runs]
    signs = {
        0 if abs(value) <= AD_F1_EQUALITY_TOLERANCE else (1 if value > 0 else -1)
        for value in effects
    }
    return {
        "case_id": case.metadata["case_id"],
        "replicate": int(case.metadata["replicate"]),
        "regime_id": case.metadata["regime_id"],
        "arms": arm_records,
        "paired": {
            "runs": paired_runs,
            "temporal_minus_no_time_ad_f1": {
                **_numeric_summary(effects),
                "range": max(effects) - min(effects),
            },
            "effect_sign_seed_sensitive": len(signs - {0}) > 1 or (0 in signs and len(signs) > 1),
        },
    }


def _seed_group_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"case_count": len(records), "arms": {}}
    for arm_id in TEMPORAL_ARMS:
        summaries = [record["arms"][arm_id]["summary"] for record in records]
        total = len(summaries)

        def count(field: str) -> int:
            return sum(bool(summary[field]) for summary in summaries)

        result["arms"][arm_id] = {
            "case_count": total,
            "scientific_objective_invariant_case_count": sum(
                summary["scientific_objective_invariant"] for summary in summaries
            ),
            "tree_seed_sensitive": {
                "count": count("tree_seed_sensitive"),
                "fraction": _fraction(count("tree_seed_sensitive"), total),
            },
            "ancestor_pair_set_seed_sensitive": {
                "count": count("ancestor_pair_set_seed_sensitive"),
                "fraction": _fraction(count("ancestor_pair_set_seed_sensitive"), total),
            },
            "ad_f1_seed_sensitive": {
                "count": count("ad_f1_seed_sensitive"),
                "fraction": _fraction(count("ad_f1_seed_sensitive"), total),
            },
            "tree_sensitive_but_pair_set_invariant": {
                "count": count("tree_sensitive_but_pair_set_invariant"),
                "fraction": _fraction(count("tree_sensitive_but_pair_set_invariant"), total),
            },
            "pair_set_sensitive_but_ad_f1_invariant": {
                "count": count("pair_set_sensitive_but_ad_f1_invariant"),
                "fraction": _fraction(count("pair_set_sensitive_but_ad_f1_invariant"), total),
            },
            "distinct_directed_occurrence_tree_count": _numeric_summary(
                summary["distinct_directed_occurrence_tree_count"] for summary in summaries
            ),
            "distinct_unique_observed_label_ancestor_pair_set_count": _numeric_summary(
                summary["distinct_unique_observed_label_ancestor_pair_set_count"]
                for summary in summaries
            ),
            "distinct_root_count": _numeric_summary(
                summary["distinct_root_count"] for summary in summaries
            ),
            "ad_f1_range": _numeric_summary(summary["ad_f1"]["range"] for summary in summaries),
            "selected_plausibility_violation_edge_count": _numeric_summary(
                summary["scientific_objective_tuple"][0] for summary in summaries
            ),
            "selected_total_edge_distance": _numeric_summary(
                summary["scientific_objective_tuple"][1] for summary in summaries
            ),
            "selected_root_score": _numeric_summary(
                summary["scientific_objective_tuple"][2] for summary in summaries
            ),
        }
    paired = [record["paired"] for record in records]
    sign_sensitive = sum(record["effect_sign_seed_sensitive"] for record in paired)
    result["paired"] = {
        "effect_sign_seed_sensitive": {
            "count": sign_sensitive,
            "fraction": _fraction(sign_sensitive, len(paired)),
        },
        "effect_range": _numeric_summary(
            record["temporal_minus_no_time_ad_f1"]["range"] for record in paired
        ),
        "mean_effect_by_seed": [
            {
                "seed": seed,
                "mean": statistics.fmean(
                    record["paired"]["runs"][index]["temporal_minus_no_time_ad_f1"]
                    for record in records
                )
                if records
                else None,
            }
            for index, seed in enumerate(DIAGNOSTIC_SEEDS)
        ],
    }
    return result


def summarize_seed_sensitivity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    regimes = sorted({str(record["regime_id"]) for record in records})
    return {
        "overall": _seed_group_summary(records),
        "by_regime": {
            regime: _seed_group_summary(
                [record for record in records if record["regime_id"] == regime]
            )
            for regime in regimes
        },
    }


def _source_hashes() -> dict[str, str]:
    paths = (
        "algorithm_evaluation/temporal_arborescence_diagnostic.py",
        "algorithm_evaluation/paper_pipeline_runner.py",
        "algorithm_evaluation/paper_pipeline_contract.py",
        "reconstructor_temporal.py",
        "reconstructor_plausibility.py",
        "evaluation_contract.py",
        "evaluator_full.py",
    )
    return {relative: file_sha256(PROJECT_ROOT / relative) for relative in paths}


def run_diagnostic(
    source_root: Path | str,
    *,
    include_seed_sensitivity: bool,
    case_limit: int | None = None,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    validate_checksum_closure(source_root, "raw_checksums.sha256", include_analysis=False)
    inventory = read_json(source_root / "expected_inventory.json")
    if inventory.get("schema_version") != EXPECTED_INVENTORY_SCHEMA_VERSION:
        raise ValueError("Frozen source has the wrong expected-inventory schema.")
    cases = list(inventory.get("cases", []))
    if len(cases) != int(inventory.get("case_count", -1)):
        raise ValueError("Frozen source case inventory is incomplete.")
    if case_limit is not None:
        if case_limit <= 0 or case_limit > len(cases):
            raise ValueError("case_limit must be between one and the full case count.")
        cases = cases[:case_limit]

    frozen_records = []
    seed_records = []
    for index, metadata in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {metadata['case_id']}",
            file=sys.stderr,
            flush=True,
        )
        case = _load_case(source_root, metadata)
        frozen_records.append(frozen_case_audit(case))
        if include_seed_sensitivity:
            seed_records.append(seed_sensitivity_case(case))

    complete = len(cases) == int(inventory["case_count"])
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "analysis_kind": "post_registration_temporal_arborescence_mechanism_diagnostic",
        "evidence_role": (
            "post_registration_descriptive_complete"
            if complete
            else "software_preflight_incomplete_not_paper_evidence"
        ),
        "source": {
            "root": str(source_root),
            "raw_checksum_file_sha256": file_sha256(source_root / "raw_checksums.sha256"),
            "raw_checksum_validated_before_analysis": True,
            "source_hashes": _source_hashes(),
        },
        "scope": {
            "condition_id": ANCHOR_CONDITION,
            "arms": list(TEMPORAL_ARMS),
            "ad_f1_equality_absolute_tolerance": AD_F1_EQUALITY_TOLERANCE,
            "source_case_count": int(inventory["case_count"]),
            "analyzed_case_count": len(cases),
            "complete_source_case_set": complete,
            "simulation_run": False,
            "observation_resampling_run": False,
            "distance_recomputation_run": False,
            "source_root_modified": False,
            "truth_use": "evaluator_only_after_each_seed_reconstruction_is_complete",
        },
        "seed_contract": {
            "included": include_seed_sensitivity,
            "namespace": DIAGNOSTIC_SEED_NAMESPACE,
            "derivation": "first_32_bits_big_endian_of_sha256(namespace:seed:index), index 1..8",
            "seeds": list(DIAGNOSTIC_SEEDS),
            "truth_used_for_seed_selection": False,
            "best_seed_selection": False,
            "finite_seed_result": "lower_bound_not_enumeration_or_uniform_optimum_sample",
        },
        "frozen_tree_audit": {
            "summary": summarize_frozen_audit(frozen_records),
            "case_records": frozen_records,
        },
        "seed_sensitivity": (
            {
                "summary": summarize_seed_sensitivity(seed_records),
                "case_records": seed_records,
            }
            if include_seed_sensitivity
            else None
        ),
    }


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("frozen-only", "all"),
        default="all",
        help="Run only the stored-tree audit or both frozen and seed-sensitivity stages.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Deterministic leading-case preflight; any limited output is not paper evidence.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source_root = Path(args.source_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"Diagnostic output already exists: {output}")
    if output == source_root or source_root in output.parents:
        raise ValueError("Diagnostic output must be outside the immutable source root.")
    result = run_diagnostic(
        source_root,
        include_seed_sensitivity=args.mode == "all",
        case_limit=args.case_limit,
    )
    write_json_atomic(output, result)
    report = {
        "status": "success",
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "output": str(output),
        "output_sha256": file_sha256(output),
        "evidence_role": result["evidence_role"],
        "analyzed_case_count": result["scope"]["analyzed_case_count"],
        "seed_sensitivity_included": result["seed_contract"]["included"],
        "frozen_summary": result["frozen_tree_audit"]["summary"]["overall"],
        "seed_summary": (
            result["seed_sensitivity"]["summary"]["overall"]
            if result["seed_sensitivity"] is not None
            else None
        ),
    }
    print(json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANCHOR_CONDITION",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DIAGNOSTIC_SEEDS",
    "ad_f1_from_pair_sets",
    "ancestor_pair_set",
    "frozen_case_audit",
    "main",
    "occurrence_tree_signature",
    "run_diagnostic",
    "seed_sensitivity_case",
    "summarize_frozen_audit",
    "summarize_seed_sensitivity",
    "tree_scientific_diagnostics",
]
