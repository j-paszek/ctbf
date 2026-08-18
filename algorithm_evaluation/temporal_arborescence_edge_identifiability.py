"""Exact selected-edge identifiability audit for temporal arborescences.

The audit reads the frozen observable occurrence trees and aligned distance
matrices, but never reads truth or evaluation artifacts.  For each selected
edge it removes that edge and globally re-solves the first three registered
objective tiers.  This distinguishes choices forced by those tiers from
choices left unresolved before the seeded final tier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from algorithm_evaluation.temporal_arborescence_diagnostic import (
    ANCHOR_CONDITION,
    DEFAULT_SOURCE_ROOT,
    TEMPORAL_ARMS,
    _load_distance,
    occurrence_identity_signature,
    tree_scientific_diagnostics,
)
from algorithm_evaluation.paper_pipeline_contract import (
    EXPECTED_INVENTORY_SCHEMA_VERSION,
    PROJECT_ROOT,
    canonical_json_sha256,
    file_sha256,
    json_safe,
    read_json,
    validate_checksum_closure,
    validate_status_record,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import (
    RECONSTRUCTION_RESULT_SCHEMA_VERSION,
    deserialize_tree,
    validate_reconstruction_input,
)
from ctbs import DistanceMatrix
from reconstructor_plausibility import is_biologically_plausible_ancestor
from reconstructor_temporal import (
    _candidate_biological_edges,
    _exact_distance_units,
    _mixed_radix_coefficients,
)


EDGE_IDENTIFIABILITY_SCHEMA_VERSION = (
    "ctbf-v5-temporal-arborescence-edge-identifiability-v1"
)


@dataclass(frozen=True)
class _Occurrence:
    node_id: Any
    biopsy_level: int
    cell_id: Any
    genome: np.ndarray


@dataclass(frozen=True)
class _ObservableCase:
    metadata: Mapping[str, Any]
    input_payload: Mapping[str, Any]
    distance: DistanceMatrix
    frozen_trees: Mapping[str, nx.DiGraph]
    reconstruction_metadata: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class _CandidateProblem:
    graph: nx.DiGraph
    virtual_root: Any
    occurrences: tuple[_Occurrence, ...]
    occurrence_by_id: Mapping[Any, _Occurrence]
    distance_denominator: int


def _stable_key(value: Any) -> tuple[str, str]:
    return f"{type(value).__module__}.{type(value).__qualname__}", repr(value)


def _fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized) if normalized else None,
        "median": statistics.median(normalized) if normalized else None,
        "minimum": min(normalized) if normalized else None,
        "maximum": max(normalized) if normalized else None,
    }


def _actual_root(tree: nx.DiGraph) -> Any:
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}.")
    return roots[0]


def _occurrences_from_tree(tree: nx.DiGraph) -> tuple[_Occurrence, ...]:
    if not nx.is_arborescence(tree):
        raise ValueError("Frozen temporal result is not an arborescence.")
    occurrences = []
    for node_id, attributes in tree.nodes(data=True):
        if "cell_id" not in attributes or "biopsy_level" not in attributes:
            raise ValueError("Frozen temporal occurrence lacks identity or biopsy level.")
        genome = np.asarray(attributes.get("genome"))
        if genome.ndim != 1 or not genome.size:
            raise ValueError("Frozen temporal occurrence lacks a valid CNP.")
        occurrences.append(
            _Occurrence(
                node_id=node_id,
                biopsy_level=int(attributes["biopsy_level"]),
                cell_id=attributes["cell_id"],
                genome=np.array(genome, copy=True),
            )
        )
    occurrences.sort(key=lambda occurrence: _stable_key(occurrence.node_id))
    return tuple(occurrences)


def _input_occurrence_signature(payload: Mapping[str, Any]) -> str:
    records = [
        {
            "biopsy_level": int(level["biopsy_level"]),
            "cell_id": json_safe(state["state_label"]),
            "genome": json_safe(state["cnp"]),
        }
        for level in payload["levels"]
        for state in level["states"]
    ]
    records.sort(
        key=lambda record: (
            int(record["biopsy_level"]),
            _stable_key(record["cell_id"]),
            tuple(record["genome"]),
        )
    )
    return canonical_json_sha256(records)


def _tree_input_occurrence_signature(tree: nx.DiGraph) -> str:
    records = [
        {
            "biopsy_level": int(attributes["biopsy_level"]),
            "cell_id": json_safe(attributes["cell_id"]),
            "genome": json_safe(np.asarray(attributes["genome"]).tolist()),
        }
        for _node, attributes in tree.nodes(data=True)
    ]
    records.sort(
        key=lambda record: (
            int(record["biopsy_level"]),
            _stable_key(record["cell_id"]),
            tuple(record["genome"]),
        )
    )
    return canonical_json_sha256(records)


def _distance_denominator(matrix: np.ndarray) -> int:
    return max(float(value).as_integer_ratio()[1] for value in matrix.flat)


def _build_candidate_problem(
    tree: nx.DiGraph,
    distance: DistanceMatrix,
    *,
    use_time: bool,
) -> _CandidateProblem:
    occurrences = _occurrences_from_tree(tree)
    occurrence_by_id = {occurrence.node_id: occurrence for occurrence in occurrences}
    if len(occurrence_by_id) != len(occurrences):
        raise ValueError("Frozen temporal occurrence node ids are not unique.")

    ids = list(distance.ids)
    id_to_index = {cell_id: index for index, cell_id in enumerate(ids)}
    if len(id_to_index) != len(ids):
        raise ValueError("Frozen distance ids are not unique.")
    observed_states = {occurrence.cell_id for occurrence in occurrences}
    if observed_states != set(ids):
        raise ValueError("Frozen occurrence states do not match aligned distance ids.")

    matrix = np.asarray(distance.matrix, dtype=float)
    distance_units = _exact_distance_units(matrix)
    denominator = _distance_denominator(matrix)
    biological_edges = _candidate_biological_edges(occurrences, use_time)
    if use_time:
        earliest_level = min(occurrence.biopsy_level for occurrence in occurrences)
        root_candidates = [
            occurrence.node_id
            for occurrence in occurrences
            if occurrence.biopsy_level == earliest_level
        ]
    else:
        root_candidates = [occurrence.node_id for occurrence in occurrences]

    max_distance_unit = max(int(value) for value in distance_units.flat)
    coefficients = _mixed_radix_coefficients(
        len(occurrences),
        max_distance_unit,
        max_distance_unit,
        0,
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(occurrence_by_id)
    virtual_root = object()
    graph.add_node(virtual_root)

    for parent_id, child_id in biological_edges:
        parent = occurrence_by_id[parent_id]
        child = occurrence_by_id[child_id]
        distance_unit = int(
            distance_units[
                id_to_index[parent.cell_id],
                id_to_index[child.cell_id],
            ]
        )
        violation = int(not is_biologically_plausible_ancestor(parent, child))
        graph.add_edge(
            parent_id,
            child_id,
            objective_cost=(
                violation * coefficients["violation"]
                + distance_unit * coefficients["distance"]
            ),
            edge_kind="biological",
            violation=violation,
            distance_unit=distance_unit,
            root_score_unit=0,
        )

    for root_id in root_candidates:
        root = occurrence_by_id[root_id]
        root_index = id_to_index[root.cell_id]
        root_score_unit = sum(
            int(distance_units[root_index, id_to_index[other.cell_id]])
            for other in occurrences
            if other.node_id != root_id
        )
        graph.add_edge(
            virtual_root,
            root_id,
            objective_cost=(
                coefficients["virtual_root"]
                + root_score_unit * coefficients["root_score"]
            ),
            edge_kind="root",
            violation=0,
            distance_unit=0,
            root_score_unit=root_score_unit,
        )

    return _CandidateProblem(
        graph=graph,
        virtual_root=virtual_root,
        occurrences=occurrences,
        occurrence_by_id=occurrence_by_id,
        distance_denominator=denominator,
    )


def _solve_objective(
    problem: _CandidateProblem,
) -> tuple[tuple[int, int, int], Any] | None:
    try:
        solved = nx.minimum_spanning_arborescence(
            problem.graph,
            attr="objective_cost",
            preserve_attrs=True,
        )
    except nx.NetworkXException as exc:
        if "No minimum spanning arborescence" in str(exc):
            return None
        raise
    if not nx.is_arborescence(solved):
        raise RuntimeError("Scientific-tier solve did not return an arborescence.")
    root_edges = [
        (parent, child, attributes)
        for parent, child, attributes in solved.edges(data=True)
        if parent is problem.virtual_root
    ]
    if len(root_edges) != 1:
        raise RuntimeError("Scientific-tier solve did not select exactly one root.")
    biological = [
        attributes
        for parent, child, attributes in solved.edges(data=True)
        if parent is not problem.virtual_root and child is not problem.virtual_root
    ]
    if len(biological) != len(problem.occurrences) - 1:
        raise RuntimeError("Scientific-tier solve selected the wrong biological-edge count.")
    objective = (
        sum(int(attributes["violation"]) for attributes in biological),
        sum(int(attributes["distance_unit"]) for attributes in biological),
        int(root_edges[0][2]["root_score_unit"]),
    )
    return objective, root_edges[0][1]


def _selected_objective(
    problem: _CandidateProblem,
    tree: nx.DiGraph,
) -> tuple[int, int, int]:
    root = _actual_root(tree)
    root_edge = (problem.virtual_root, root)
    if not problem.graph.has_edge(*root_edge):
        raise ValueError("Frozen selected root is not an admissible root candidate.")
    violation_total = 0
    distance_total = 0
    for parent, child in tree.edges():
        if not problem.graph.has_edge(parent, child):
            raise ValueError("Frozen selected edge is not an admissible candidate.")
        attributes = problem.graph.edges[parent, child]
        violation_total += int(attributes["violation"])
        distance_total += int(attributes["distance_unit"])
    return (
        violation_total,
        distance_total,
        int(problem.graph.edges[root_edge]["root_score_unit"]),
    )


def _objective_record(
    objective: tuple[int, int, int],
    denominator: int,
) -> dict[str, Any]:
    return {
        "exact_units": [int(value) for value in objective],
        "distance_unit_denominator": int(denominator),
        "values": [
            int(objective[0]),
            float(objective[1] / denominator),
            float(objective[2] / denominator),
        ],
    }


def _solve_without(
    problem: _CandidateProblem,
    edge: tuple[Any, Any],
) -> tuple[tuple[int, int, int], Any] | None:
    if not problem.graph.has_edge(*edge):
        raise ValueError("Cannot exclude an edge absent from the candidate graph.")
    attributes = dict(problem.graph.edges[edge])
    problem.graph.remove_edge(*edge)
    try:
        return _solve_objective(problem)
    finally:
        problem.graph.add_edge(*edge, **attributes)


def _choice_classification(
    alternative: tuple[tuple[int, int, int], Any] | None,
    baseline: tuple[int, int, int],
    denominator: int,
) -> dict[str, Any]:
    if alternative is None:
        return {
            "classification": "scientifically_forced_infeasible_without_choice",
            "first_deciding_tier": "feasibility",
            "scientifically_forced": True,
            "non_forced_before_seeded_tier": False,
            "excluded_optimum": None,
        }
    alternative_objective, alternative_root = alternative
    if alternative_objective < baseline:
        raise RuntimeError("Exclusion found an objective better than the frozen baseline.")
    non_forced = alternative_objective == baseline
    tier_names = (
        "plausibility_violation_edge_count",
        "total_edge_distance",
        "root_score",
    )
    first_deciding_tier = "seeded_final_tier"
    if not non_forced:
        first_deciding_tier = next(
            name
            for name, alternative_value, baseline_value in zip(
                tier_names, alternative_objective, baseline
            )
            if alternative_value != baseline_value
        )
    return {
        "classification": (
            "non_forced_before_seeded_tier"
            if non_forced
            else "scientifically_forced_objective_worsens_without_choice"
        ),
        "first_deciding_tier": first_deciding_tier,
        "scientifically_forced": not non_forced,
        "non_forced_before_seeded_tier": non_forced,
        "excluded_optimum": {
            **_objective_record(alternative_objective, denominator),
            "selected_root_node_id": json_safe(alternative_root),
            "delta_exact_units": [
                int(alternative_value - baseline_value)
                for alternative_value, baseline_value in zip(
                    alternative_objective, baseline
                )
            ],
        },
    }


def audit_selected_tree(
    tree: nx.DiGraph,
    distance: DistanceMatrix,
    *,
    use_time: bool,
) -> dict[str, Any]:
    """Classify every selected biological edge and the selected root exactly."""
    problem = _build_candidate_problem(tree, distance, use_time=use_time)
    selected = _selected_objective(problem, tree)
    baseline_solution = _solve_objective(problem)
    if baseline_solution is None:
        raise RuntimeError("Complete scientific-tier candidate graph is infeasible.")
    baseline, baseline_root = baseline_solution
    if baseline != selected:
        raise RuntimeError(
            "Frozen selected tree is not optimal on the first three objective tiers."
        )

    diagnostics = tree_scientific_diagnostics(tree, distance)
    selected_values = _objective_record(selected, problem.distance_denominator)["values"]
    reported_values = diagnostics["scientific_objective_tuple"]
    if int(reported_values[0]) != int(selected_values[0]) or not all(
        math.isclose(float(reported), float(exact), rel_tol=0.0, abs_tol=1e-12)
        for reported, exact in zip(reported_values[1:], selected_values[1:])
    ):
        raise RuntimeError("Exact selected objective disagrees with tree diagnostics.")

    edge_records = []
    for parent, child in sorted(
        tree.edges(), key=lambda edge: (_stable_key(edge[0]), _stable_key(edge[1]))
    ):
        parent_occurrence = problem.occurrence_by_id[parent]
        child_occurrence = problem.occurrence_by_id[child]
        classification = _choice_classification(
            _solve_without(problem, (parent, child)),
            baseline,
            problem.distance_denominator,
        )
        edge_records.append(
            {
                "parent_node_id": json_safe(parent),
                "child_node_id": json_safe(child),
                "parent_cell_id": json_safe(parent_occurrence.cell_id),
                "child_cell_id": json_safe(child_occurrence.cell_id),
                "parent_biopsy_level": parent_occurrence.biopsy_level,
                "child_biopsy_level": child_occurrence.biopsy_level,
                "selected_violation": int(
                    problem.graph.edges[parent, child]["violation"]
                ),
                "selected_distance_unit": int(
                    problem.graph.edges[parent, child]["distance_unit"]
                ),
                **classification,
            }
        )

    selected_root = _actual_root(tree)
    root_occurrence = problem.occurrence_by_id[selected_root]
    root_classification = _choice_classification(
        _solve_without(problem, (problem.virtual_root, selected_root)),
        baseline,
        problem.distance_denominator,
    )
    non_forced_edges = sum(
        record["non_forced_before_seeded_tier"] for record in edge_records
    )
    return {
        "use_time": bool(use_time),
        "occurrence_count": len(problem.occurrences),
        "selected_biological_edge_count": len(edge_records),
        "baseline_scientific_objective": {
            **_objective_record(baseline, problem.distance_denominator),
            "one_baseline_optimum_root_node_id": json_safe(baseline_root),
        },
        "selected_edge_summary": {
            "scientifically_forced_count": len(edge_records) - non_forced_edges,
            "non_forced_before_seeded_tier_count": non_forced_edges,
            "non_forced_before_seeded_tier_fraction": _fraction(
                non_forced_edges, len(edge_records)
            ),
        },
        "selected_root": {
            "node_id": json_safe(selected_root),
            "cell_id": json_safe(root_occurrence.cell_id),
            "biopsy_level": root_occurrence.biopsy_level,
            **root_classification,
        },
        "selected_edges": edge_records,
    }


def _load_frozen_tree(
    case_root: Path,
    arm_id: str,
) -> tuple[nx.DiGraph, Mapping[str, Any]]:
    arm_root = case_root / "conditions" / ANCHOR_CONDITION / "arms" / arm_id
    status = read_json(arm_root / "status.json")
    validate_status_record(status)
    if status.get("status") != "success":
        raise ValueError(f"Frozen arm {arm_id} is not successful.")
    reconstruction = read_json(arm_root / "reconstruction.json")
    if (
        reconstruction.get("schema_version") != RECONSTRUCTION_RESULT_SCHEMA_VERSION
        or reconstruction.get("status") != "success"
    ):
        raise ValueError(f"Frozen arm {arm_id} has no successful reconstruction.")
    metadata = reconstruction.get("metadata", {})
    if metadata.get("arm_id") != arm_id:
        raise ValueError(f"Frozen arm {arm_id} metadata disagree.")
    tree = deserialize_tree(reconstruction["tree"])
    if not nx.is_arborescence(tree):
        raise ValueError(f"Frozen arm {arm_id} is not an arborescence.")
    return tree, metadata


def _load_observable_case(
    source_root: Path,
    metadata: Mapping[str, Any],
) -> _ObservableCase:
    case_id = str(metadata["case_id"])
    if ANCHOR_CONDITION not in metadata.get("condition_ids", []):
        raise ValueError(f"Case {case_id} lacks the registered anchor condition.")
    case_root = source_root / "cases" / case_id
    input_payload = read_json(
        case_root / "conditions" / ANCHOR_CONDITION / "input.json"
    )
    validate_reconstruction_input(input_payload)
    if (
        input_payload.get("case_id") != case_id
        or input_payload.get("condition_id") != ANCHOR_CONDITION
    ):
        raise ValueError("Frozen reconstruction input has the wrong identity.")
    distance = _load_distance(case_root, input_payload, case_id)

    trees = {}
    reconstruction_metadata = {}
    for arm_id in TEMPORAL_ARMS:
        tree, arm_metadata = _load_frozen_tree(case_root, arm_id)
        trees[arm_id] = tree
        reconstruction_metadata[arm_id] = arm_metadata
    if len({occurrence_identity_signature(tree) for tree in trees.values()}) != 1:
        raise ValueError("Frozen temporal arms do not share occurrence identities.")
    input_signature = _input_occurrence_signature(input_payload)
    for tree in trees.values():
        if _tree_input_occurrence_signature(tree) != input_signature:
            raise ValueError("Frozen temporal occurrences disagree with observable input.")
    return _ObservableCase(
        metadata=metadata,
        input_payload=input_payload,
        distance=distance,
        frozen_trees=trees,
        reconstruction_metadata=reconstruction_metadata,
    )


def audit_case(case: _ObservableCase) -> dict[str, Any]:
    arms = {}
    for arm_id in TEMPORAL_ARMS:
        arm_metadata = case.reconstruction_metadata[arm_id]
        arms[arm_id] = {
            "frozen_reconstruction_seed": int(arm_metadata["reconstruction_seed"]),
            **audit_selected_tree(
                case.frozen_trees[arm_id],
                case.distance,
                use_time=arm_id == "temporal_minimum",
            ),
        }
    return {
        "case_id": case.metadata["case_id"],
        "replicate": int(case.metadata["replicate"]),
        "regime_id": case.metadata["regime_id"],
        "arms": arms,
    }


def _group_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"case_count": len(records), "arms": {}}
    for arm_id in TEMPORAL_ARMS:
        arm_records = [record["arms"][arm_id] for record in records]
        selected_edges = sum(
            record["selected_biological_edge_count"] for record in arm_records
        )
        non_forced_edges = sum(
            record["selected_edge_summary"]["non_forced_before_seeded_tier_count"]
            for record in arm_records
        )
        forced_edges = selected_edges - non_forced_edges
        non_forced_roots = sum(
            record["selected_root"]["non_forced_before_seeded_tier"]
            for record in arm_records
        )
        cases_with_non_forced_edge = sum(
            record["selected_edge_summary"]["non_forced_before_seeded_tier_count"] > 0
            for record in arm_records
        )
        result["arms"][arm_id] = {
            "selected_biological_edge_count": selected_edges,
            "scientifically_forced_selected_edge_count": forced_edges,
            "non_forced_before_seeded_tier_selected_edge_count": non_forced_edges,
            "non_forced_before_seeded_tier_selected_edge_fraction": _fraction(
                non_forced_edges, selected_edges
            ),
            "cases_with_at_least_one_non_forced_selected_edge": {
                "count": cases_with_non_forced_edge,
                "fraction": _fraction(cases_with_non_forced_edge, len(arm_records)),
            },
            "non_forced_before_seeded_tier_selected_edge_count_per_case": (
                _numeric_summary(
                    record["selected_edge_summary"][
                        "non_forced_before_seeded_tier_count"
                    ]
                    for record in arm_records
                )
            ),
            "non_forced_before_seeded_tier_selected_edge_fraction_per_case": (
                _numeric_summary(
                    record["selected_edge_summary"][
                        "non_forced_before_seeded_tier_fraction"
                    ]
                    for record in arm_records
                )
            ),
            "non_forced_before_seeded_tier_selected_root": {
                "count": non_forced_roots,
                "fraction": _fraction(non_forced_roots, len(arm_records)),
            },
            "selected_edge_first_deciding_tier_counts": {
                tier: sum(
                    edge["first_deciding_tier"] == tier
                    for record in arm_records
                    for edge in record["selected_edges"]
                )
                for tier in (
                    "feasibility",
                    "plausibility_violation_edge_count",
                    "total_edge_distance",
                    "root_score",
                    "seeded_final_tier",
                )
            },
        }
    return result


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    regimes = sorted({str(record["regime_id"]) for record in records})
    return {
        "overall": _group_summary(records),
        "by_regime": {
            regime: _group_summary(
                [record for record in records if record["regime_id"] == regime]
            )
            for regime in regimes
        },
    }


def _source_hashes() -> dict[str, str]:
    paths = (
        "algorithm_evaluation/temporal_arborescence_edge_identifiability.py",
        "algorithm_evaluation/temporal_arborescence_diagnostic.py",
        "algorithm_evaluation/paper_pipeline_runner.py",
        "algorithm_evaluation/paper_pipeline_contract.py",
        "reconstructor_temporal.py",
        "reconstructor_plausibility.py",
    )
    return {relative: file_sha256(PROJECT_ROOT / relative) for relative in paths}


def run_diagnostic(
    source_root: Path | str,
    *,
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
    records = []
    for index, metadata in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {metadata['case_id']}",
            file=sys.stderr,
            flush=True,
        )
        records.append(audit_case(_load_observable_case(source_root, metadata)))

    complete = len(cases) == int(inventory["case_count"])
    return {
        "schema_version": EDGE_IDENTIFIABILITY_SCHEMA_VERSION,
        "analysis_kind": "post_registration_temporal_arborescence_edge_identifiability",
        "evidence_role": (
            "post_registration_descriptive_complete"
            if complete
            else "software_preflight_incomplete_not_paper_evidence"
        ),
        "source": {
            "root": str(source_root),
            "raw_checksum_file_sha256": file_sha256(
                source_root / "raw_checksums.sha256"
            ),
            "raw_checksum_validated_before_analysis": True,
            "source_hashes": _source_hashes(),
        },
        "scope": {
            "condition_id": ANCHOR_CONDITION,
            "arms": list(TEMPORAL_ARMS),
            "source_case_count": int(inventory["case_count"]),
            "analyzed_case_count": len(cases),
            "complete_source_case_set": complete,
            "objective_tiers_before_seeded_rank": [
                "plausibility_violation_edge_count",
                "total_edge_distance",
                "root_score",
            ],
            "classification_test": (
                "remove_each_selected_choice_and_globally_resolve_exact_scientific_tiers"
            ),
            "truth_read": False,
            "evaluation_read": False,
            "simulation_run": False,
            "distance_recomputation_run": False,
            "source_root_modified": False,
        },
        "edge_identifiability": {
            "summary": summarize(records),
            "case_records": records,
        },
    }


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Deterministic leading-case preflight; limited output is not paper evidence.",
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
    result = run_diagnostic(source_root, case_limit=args.case_limit)
    write_json_atomic(output, result)
    print(
        json.dumps(
            json_safe(
                {
                    "status": "success",
                    "schema_version": EDGE_IDENTIFIABILITY_SCHEMA_VERSION,
                    "output": str(output),
                    "output_sha256": file_sha256(output),
                    "evidence_role": result["evidence_role"],
                    "analyzed_case_count": result["scope"]["analyzed_case_count"],
                    "summary": result["edge_identifiability"]["summary"]["overall"],
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EDGE_IDENTIFIABILITY_SCHEMA_VERSION",
    "audit_case",
    "audit_selected_tree",
    "main",
    "run_diagnostic",
    "summarize",
]
