"""Bounded non-paper reconstruction probe for the CTBF v5 simulator.

The probe asks whether the approved v5 truth process and capped longitudinal
sampling produce a meaningful reconstruction problem before paper heights are
frozen.  It runs the six established reconstruction arms on H14/H24/H34 using
the owner-approved 60%/80%/100% schedule and at most six distinct genotype
states per biopsy.  It writes compact summaries only: no CNP vectors, truth or
reconstructed trees, distance matrices, or simulator node identities.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from algorithm_evaluation.hypothesis_height_ambiguity_trend import (
    CASE_METRICS as AMBIGUITY_CASE_METRICS,
    _distance_identity,
    _sha256_json,
    ambiguity_case_summary,
)
from algorithm_evaluation.hypothesis_trend_distance_preflight import (
    _number_summary,
    _validate_production_distance,
    distance_matrix_summary,
)
from algorithm_evaluation.paper_pipeline_contract import (
    PROJECT_ROOT,
    REGISTERED_ARM_SPECS,
    canonical_json_sha256,
    json_safe,
    read_json,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import (
    ARM_BUILD_SPECS,
    RECONSTRUCTION_INPUT_SCHEMA_VERSION,
    measured_stage,
    reconstruct_arm,
    validate_reconstruction_input,
)
from algorithm_evaluation.simulator_growth_probe import (
    _file_sha256,
    _validate_standard_base_config,
)
from ctbs import (
    Cnp2CnpFileDistanceProvider,
    DistanceMatrix,
    load_ctbs_runtime_config,
    unique_cells_by_cell_id,
)
from distance_semantics import (
    CNP2CNP_SEMANTICS_VERSION,
    stable_distance_label_key,
)
from evaluation_contract import evaluate_tree_pair_result
from simulator import CancerCellEvolutionSimulator, Genotype


SCHEMA_VERSION = "ctbf-v5-simulator-reconstruction-intuition-probe-v1"
ANALYSIS_ROLE = "nonpaper_simulator_reconstruction_intuition_probe"
SEED_NAMESPACE = "ctbf-v5-nonpaper-reconstruction-intuition-v1"
SAMPLING_VERSION = "rule-y-ceiling-capped-distinct-state-v1"

DEFAULT_BASE_CONFIG = PROJECT_ROOT / "simulator_examples" / "default.json"
APPROVED_SCHEDULES = {
    14: (9, 12, 14),
    24: (15, 20, 24),
    34: (21, 28, 34),
}
DEFAULT_REPLICATES = 12
DEFAULT_BASE_SEED = 20260812
MAX_REPLICATES = 12
MAX_STATES_PER_BIOPSY = 6
MAX_OBSERVED_OCCURRENCES_PER_CASE = 18
DEFAULT_FIXED_RADIUS = 4.0
DEFAULT_SIMULATION_TIMEOUT_SECONDS = 180
DEFAULT_DISTANCE_TIMEOUT_SECONDS = 180
DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS = 60
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 60
DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS = 120
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_CAPTURE_LIMIT_BYTES = 1_048_576
DEFAULT_MAX_CASE_DEPENDENCY_FAILURES = 6


DistanceCompute = Callable[[Sequence[Any]], DistanceMatrix]


ARM_ENDPOINTS = {
    "classical_partial": {
        "problem": "partial",
        "declared_metrics": ("grf",),
    },
    "biopsy_guided_classical": {
        "problem": "partial",
        "declared_metrics": ("grf",),
    },
    "rooted_labeled_nj": {
        "problem": "fully_labeled_closed_state",
        "declared_metrics": ("ad_f1", "grf"),
    },
    "temporal_minimum": {
        "problem": "occurrence_aware_fully_labeled_closed_state",
        "declared_metrics": ("ad_f1", "grf"),
    },
    "temporal_minimum_no_time": {
        "problem": "occurrence_aware_fully_labeled_closed_state",
        "declared_metrics": ("ad_f1", "grf"),
    },
    "anticentral_parsimony": {
        "problem": "inferred_copy_fully_labeled_closed_state",
        "declared_metrics": ("ad_f1", "grf"),
    },
}

ARM_IDS = tuple(arm_id for arm_id, _algorithm in REGISTERED_ARM_SPECS)
if set(ARM_IDS) != set(ARM_ENDPOINTS):  # pragma: no cover - import invariant
    raise RuntimeError("Probe endpoints do not cover the registered arm portfolio.")
for _arm_id, _algorithm_name in REGISTERED_ARM_SPECS:  # pragma: no cover
    if ARM_BUILD_SPECS.get(_arm_id, (None,))[0] != _algorithm_name:
        raise RuntimeError("Probe arm identity differs from the registered portfolio.")


class EmptyRequiredBiopsyError(ValueError):
    """A declared observation generation contains no retained state."""


def rule_y_schedule(height: int) -> tuple[int, int, int]:
    """Return ceil(60% H), ceil(80% H), H with strict level separation."""
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer.")
    schedule = (math.ceil(0.6 * height), math.ceil(0.8 * height), height)
    if tuple(sorted(set(schedule))) != schedule:
        raise ValueError(f"Rule Y collapses at height {height}: {schedule}.")
    return schedule


def _derived_seed(
    stream: str,
    base_seed: int,
    replicate_index: int,
    *coordinates: int,
) -> int:
    if not stream:
        raise ValueError("seed stream must be nonempty.")
    for field, value in (
        ("base_seed", base_seed),
        ("replicate_index", replicate_index),
        *(("coordinate", value) for value in coordinates),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer.")
    material = "\0".join(
        [
            SEED_NAMESPACE,
            stream,
            str(base_seed),
            str(replicate_index),
            *(str(value) for value in coordinates),
        ]
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)


def _simulation_seed(base_seed: int, replicate_index: int) -> int:
    # Height is deliberately excluded: shorter truths are paired prefixes.
    return _derived_seed("simulation", base_seed, replicate_index)


def _sampling_seed(
    base_seed: int,
    replicate_index: int,
    height: int,
    generation: int,
) -> int:
    return _derived_seed(
        "sampling",
        base_seed,
        replicate_index,
        height,
        generation,
    )


def _reconstruction_seed(
    base_seed: int,
    replicate_index: int,
    height: int,
) -> int:
    # Every arm in one case receives the same prospectively derived seed.
    return _derived_seed("reconstruction", base_seed, replicate_index, height)


def _profile(cell: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(cell.genome).tolist())


def _canonical_cells_at_generation(
    simulator: CancerCellEvolutionSimulator,
    generation: int,
) -> list[Genotype]:
    cells = [
        genotype
        for genotype in simulator.genotypes.values()
        if genotype.generation == generation
    ]
    return sorted(
        simulator.canonicalize_biopsy_genotypes(cells),
        key=lambda cell: (_profile(cell), stable_distance_label_key(cell.cell_id)),
    )


def select_capped_levels(
    available_levels: Sequence[Sequence[Genotype]],
    generations: Sequence[int],
    *,
    base_seed: int,
    replicate_index: int,
    height: int,
    cap: int = MAX_STATES_PER_BIOPSY,
) -> tuple[list[list[Genotype]], list[dict[str, Any]]]:
    """Select all states below the cap and a seeded subset above the cap."""
    if len(available_levels) != len(generations):
        raise ValueError("Available levels and generations must align.")
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= 6:
        raise ValueError("The biopsy-state cap must be in [1,6].")

    selected_levels: list[list[Genotype]] = []
    rows = []
    for generation, available_values in zip(generations, available_levels):
        available = list(available_values)
        if not available:
            raise EmptyRequiredBiopsyError(
                f"Required generation {generation} contains no retained state."
            )
        seed = _sampling_seed(
            base_seed,
            replicate_index,
            height,
            int(generation),
        )
        if len(available) <= cap:
            chosen = available
            mode = "all_available"
            seed_used = False
        else:
            permutation = np.random.Generator(np.random.PCG64(seed)).permutation(
                len(available)
            )
            chosen = [available[int(index)] for index in permutation[:cap]]
            mode = "seeded_without_replacement_to_cap"
            seed_used = True
        chosen = sorted(
            chosen,
            key=lambda cell: (
                _profile(cell),
                stable_distance_label_key(cell.cell_id),
            ),
        )
        selected_levels.append(chosen)
        rows.append(
            {
                "generation": int(generation),
                "available_distinct_state_count": len(available),
                "configured_state_cap": int(cap),
                "realized_occurrence_count": len(chosen),
                "selection_mode": mode,
                "sampling_seed": seed,
                "sampling_seed_used": seed_used,
            }
        )
    return selected_levels, rows


def _truth_prefix_sha256(tree: nx.DiGraph, through_generation: int) -> str:
    """Hash a raw truth prefix without exposing its states or identities."""
    included = {
        node
        for node, attributes in tree.nodes(data=True)
        if int(attributes["generation"]) <= int(through_generation)
    }
    nodes = [
        {
            "id": json_safe(node),
            "generation": int(tree.nodes[node]["generation"]),
            "cell_id": json_safe(tree.nodes[node].get("cell_id")),
            "genome": json_safe(tree.nodes[node].get("genome")),
        }
        for node in sorted(included, key=lambda value: (str(type(value)), repr(value)))
    ]
    links = [
        {
            "source": json_safe(parent),
            "target": json_safe(child),
            "attributes": json_safe(dict(attributes)),
        }
        for parent, child, attributes in sorted(
            tree.edges(data=True),
            key=lambda value: (repr(value[0]), repr(value[1])),
        )
        if parent in included and child in included
    ]
    return canonical_json_sha256({"nodes": nodes, "links": links})


def _fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def truth_sampling_diagnostics(
    truth_tree: nx.DiGraph,
    selected_levels: Sequence[Sequence[Genotype]],
) -> dict[str, Any]:
    """Describe occurrence-level identifiability using evaluator-only identity.

    The truth is an arborescence, so all pair classifications can be counted
    from one preorder and one postorder traversal.  In particular, do not call
    NetworkX's generic LCA helper per selected pair: that helper validates the
    complete DAG on every call and turns dense percentage biopsies into an
    avoidable O(S^2 V) operation.
    """
    if not nx.is_arborescence(truth_tree):
        raise ValueError("Truth diagnostics require one directed arborescence.")
    occurrences = [cell for level in selected_levels for cell in level]
    if not occurrences:
        raise ValueError("Truth diagnostics require at least one occurrence.")
    selected_nodes = [cell.node_id for cell in occurrences]
    if len(set(selected_nodes)) != len(selected_nodes):
        raise ValueError("One simulator occurrence was selected more than once.")
    missing = set(selected_nodes) - set(truth_tree)
    if missing:
        raise ValueError("A selected occurrence is absent from the truth tree.")

    selected_set = set(selected_nodes)
    cell_by_node = {cell.node_id: cell for cell in occurrences}
    roots = [node for node, indegree in truth_tree.in_degree() if indegree == 0]
    if len(roots) != 1:  # defensive clarity beyond nx.is_arborescence
        raise ValueError("Truth diagnostics require exactly one root.")

    # One depth-first traversal counts every selected ancestor/descendant pair
    # exactly when its descendant is entered.  The active label counts provide
    # the same-state subset without enumerating all selected-node pairs.
    comparable_count = 0
    comparable_same_state_count = 0
    active_selected_count = 0
    active_label_counts: Counter[Any] = Counter()
    preorder: list[Any] = []
    depths: dict[Any, int] = {}
    nearest_selected_ancestor: dict[Any, Any | None] = {}
    # Event fields are (entering, node, depth, nearest strict selected ancestor).
    stack = [(True, roots[0], 0, None)]
    while stack:
        entering, node, depth, nearest = stack.pop()
        is_selected = node in selected_set
        if not entering:
            if is_selected:
                label = cell_by_node[node].cell_id
                active_selected_count -= 1
                active_label_counts[label] -= 1
                if active_label_counts[label] == 0:
                    del active_label_counts[label]
            continue

        preorder.append(node)
        depths[node] = depth
        if is_selected:
            label = cell_by_node[node].cell_id
            comparable_count += active_selected_count
            comparable_same_state_count += active_label_counts.get(label, 0)
            nearest_selected_ancestor[node] = nearest
            active_selected_count += 1
            active_label_counts[label] += 1

        stack.append((False, node, depth, nearest))
        child_nearest = node if is_selected else nearest
        children = list(truth_tree.successors(node))
        stack.extend(
            (True, child, depth + 1, child_nearest)
            for child in reversed(children)
        )

    labels = [cell.cell_id for cell in occurrences]
    label_counts = Counter(labels)
    same_state_pair_count = sum(
        math.comb(count, 2) for count in label_counts.values()
    )
    pair_count = math.comb(len(selected_nodes), 2)
    incomparable_count = pair_count - comparable_count
    incomparable_same_state_count = (
        same_state_pair_count - comparable_same_state_count
    )

    # Every incomparable pair belongs to exactly one fork at its LCA.  Count
    # cross-child selected pairs bottom-up instead of finding an LCA per pair.
    subtree_selected_counts: dict[Any, int] = {}
    hidden_fork_count = 0
    sampled_lca_fork_count = 0
    for node in reversed(preorder):
        descendant_count = 0
        cross_child_pair_count = 0
        for child in truth_tree.successors(node):
            child_count = subtree_selected_counts[child]
            cross_child_pair_count += descendant_count * child_count
            descendant_count += child_count
        if node in selected_set:
            sampled_lca_fork_count += cross_child_pair_count
            descendant_count += 1
        else:
            hidden_fork_count += cross_child_pair_count
        subtree_selected_counts[node] = descendant_count
    if hidden_fork_count + sampled_lca_fork_count != incomparable_count:
        raise RuntimeError("Truth fork accounting did not partition incomparable pairs.")

    later_nodes = [
        cell.node_id
        for level in selected_levels[1:]
        for cell in level
    ]
    nearest_edge_lengths = []
    hidden_internal_counts = []
    nearest_same_state_count = 0
    direct_parent_sampled_count = 0
    later_with_sampled_ancestor = 0
    for node in later_nodes:
        nearest = nearest_selected_ancestor[node]
        if nearest is None:
            continue
        later_with_sampled_ancestor += 1
        edge_length = depths[node] - depths[nearest]
        nearest_edge_lengths.append(edge_length)
        hidden_internal_counts.append(max(0, edge_length - 1))
        direct_parent_sampled_count += int(edge_length == 1)
        nearest_same_state_count += int(
            cell_by_node[nearest].cell_id == cell_by_node[node].cell_id
        )

    minimal_nodes = [
        node for node in selected_nodes if nearest_selected_ancestor[node] is None
    ]
    unique_labels = set(labels)
    truth_labels = {
        attributes.get("cell_id")
        for _node, attributes in truth_tree.nodes(data=True)
        if attributes.get("cell_id") is not None
    }
    representable = len(minimal_nodes) == 1
    return {
        "truth_node_count": truth_tree.number_of_nodes(),
        "truth_edge_count": truth_tree.number_of_edges(),
        "truth_unique_state_label_count": len(truth_labels),
        "selected_occurrence_count": len(selected_nodes),
        "selected_unique_state_label_count": len(unique_labels),
        "repeated_state_occurrence_count": len(labels) - len(unique_labels),
        "observation_to_truth_node_ratio": (
            len(selected_nodes) / truth_tree.number_of_nodes()
        ),
        "unique_state_label_coverage_fraction": _fraction(
            len(unique_labels), len(truth_labels)
        ),
        "selected_unordered_pair_count": pair_count,
        "comparable_ancestor_descendant_pair_count": comparable_count,
        "comparable_ancestor_descendant_pair_fraction": _fraction(
            comparable_count, pair_count
        ),
        "incomparable_pair_count": incomparable_count,
        "incomparable_pair_fraction": _fraction(incomparable_count, pair_count),
        "hidden_fork_pair_count": hidden_fork_count,
        "hidden_fork_pair_fraction": _fraction(hidden_fork_count, pair_count),
        "sampled_lca_fork_pair_count": sampled_lca_fork_count,
        "same_state_pair_count": same_state_pair_count,
        "comparable_same_state_pair_count": comparable_same_state_count,
        "incomparable_same_state_pair_count": incomparable_same_state_count,
        "later_observation_occurrence_count": len(later_nodes),
        "later_with_sampled_ancestor_count": later_with_sampled_ancestor,
        "sampled_ancestor_coverage_fraction": _fraction(
            later_with_sampled_ancestor, len(later_nodes)
        ),
        "nearest_sampled_ancestor_edge_distance": _number_summary(
            nearest_edge_lengths
        ),
        "hidden_internal_nodes_to_nearest_sampled_ancestor": _number_summary(
            hidden_internal_counts
        ),
        "direct_truth_parent_sampled_count": direct_parent_sampled_count,
        "nearest_sampled_ancestor_same_state_count": nearest_same_state_count,
        "minimal_sampled_occurrence_count": len(minimal_nodes),
        "minimum_invented_edges_for_observed_only_arborescence": max(
            0, len(minimal_nodes) - 1
        ),
        "observed_only_occurrence_arborescence_representable": representable,
    }


def _reconstruction_input(
    case_key: str,
    height: int,
    generations: Sequence[int],
    selected_levels: Sequence[Sequence[Genotype]],
) -> dict[str, Any]:
    payload = {
        "schema_version": RECONSTRUCTION_INPUT_SCHEMA_VERSION,
        "case_id": case_key,
        "condition_id": f"H{height}_rule_y_cap6",
        "sampling_rule": SAMPLING_VERSION,
        "levels": [
            {
                "biopsy_level": level_index,
                "generation": int(generation),
                "states": [
                    {
                        "state_label": json_safe(cell.cell_id),
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


def _observed_labels(payload: Mapping[str, Any]) -> list[Any]:
    return sorted(
        {
            state["state_label"]
            for level in payload["levels"]
            for state in level["states"]
        },
        key=stable_distance_label_key,
    )


def _event_totals(simulator: CancerCellEvolutionSimulator) -> dict[str, int]:
    totals = simulator.diagnostics_snapshot().get("totals", {})
    fields = (
        "attempted_children",
        "proposed_segmental_event_records",
        "effective_segmental_event_applications",
        "retained_segmental_gain_events",
        "retained_segmental_loss_events",
        "representative_collisions",
        "cross_parent_representative_collisions",
        "viability_rejections",
        "crucial_bin_zero_rejections",
        "all_zero_genome_rejections",
    )
    return {field: int(totals.get(field, 0)) for field in fields}


def _prepare_case(
    *,
    config_path: Path,
    simulation_seed: int,
    base_seed: int,
    replicate_index: int,
    height: int,
    generations: Sequence[int],
) -> dict[str, Any]:
    simulator = CancerCellEvolutionSimulator(str(config_path), seed=simulation_seed)
    simulator.run_simulation()
    available_levels = [
        _canonical_cells_at_generation(simulator, generation)
        for generation in generations
    ]
    selected_levels, sampling_rows = select_capped_levels(
        available_levels,
        generations,
        base_seed=base_seed,
        replicate_index=replicate_index,
        height=height,
    )
    occurrences = [cell for level in selected_levels for cell in level]
    distance_cells = sorted(
        unique_cells_by_cell_id(occurrences),
        key=lambda cell: stable_distance_label_key(cell.cell_id),
    )
    if len(occurrences) > MAX_OBSERVED_OCCURRENCES_PER_CASE:
        raise ValueError("Capped sampling exceeded the 18-occurrence bound.")
    if not distance_cells:
        raise EmptyRequiredBiopsyError("Capped sampling produced no distance state.")

    case_key = f"v5-intuition-H{height}-R{replicate_index + 1:03d}"
    true_tree = simulator.canonicalized_tree_by_genome()
    truth_diagnostics = truth_sampling_diagnostics(true_tree, selected_levels)
    input_payload = _reconstruction_input(
        case_key,
        height,
        generations,
        selected_levels,
    )
    prefix_heights = [
        prefix_height
        for prefix_height in APPROVED_SCHEDULES
        if prefix_height <= height
    ]
    prefix_hashes = {
        str(prefix_height): _truth_prefix_sha256(
            simulator.tree,
            prefix_height,
        )
        for prefix_height in prefix_heights
    }
    summary = {
        "truth_node_count": simulator.tree.number_of_nodes(),
        "truth_edge_count": simulator.tree.number_of_edges(),
        "sampling": sampling_rows,
        "selected_occurrence_count": len(occurrences),
        "selected_unique_state_count": len(distance_cells),
        "repeated_state_across_biopsy_occurrence_count": (
            len(occurrences) - len(distance_cells)
        ),
        "distance_matrix_profile_count": len(distance_cells),
        "distance_matrix_cell_count": len(distance_cells) ** 2,
        "bidirectional_ordered_pair_bound": (
            len(distance_cells) * (len(distance_cells) - 1)
        ),
        "low_information_flags": {
            "one_unique_state": len(distance_cells) == 1,
            "fewer_than_three_unique_states": len(distance_cells) < 3,
            "fewer_than_six_total_occurrences": len(occurrences) < 6,
        },
        "truth_sampling_diagnostics": truth_diagnostics,
        "simulator_event_totals": _event_totals(simulator),
        "truth_prefix_sha256_by_height": prefix_hashes,
    }
    return {
        "levels": selected_levels,
        "distance_cells": distance_cells,
        "reconstruction_input": input_payload,
        "true_tree": true_tree,
        "summary": summary,
    }


def _error_fingerprint(error_type: str, message: str) -> dict[str, Any]:
    encoded = message.encode("utf-8", errors="replace")
    return {
        "type": error_type,
        "message_character_count": len(message),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _typed_error(error: BaseException | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return _error_fingerprint(type(error).__name__, str(error))


def _tree_summary(tree: nx.DiGraph) -> dict[str, Any]:
    if not nx.is_arborescence(tree):
        raise ValueError("Reconstruction summary requires one arborescence.")
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    if len(roots) != 1:
        raise ValueError("Reconstruction summary requires one root.")
    depths = nx.single_source_shortest_path_length(tree, roots[0])
    labels = [
        attributes.get("cell_id")
        for _node, attributes in tree.nodes(data=True)
        if attributes.get("cell_id") is not None
    ]
    return {
        "node_count": tree.number_of_nodes(),
        "edge_count": tree.number_of_edges(),
        "leaf_count": sum(tree.out_degree(node) == 0 for node in tree),
        "maximum_depth": max(depths.values(), default=0),
        "labeled_occurrence_count": len(labels),
        "unlabeled_node_count": tree.number_of_nodes() - len(labels),
        "unique_state_label_count": len(set(labels)),
        "repeated_state_label_occurrence_count": len(labels) - len(set(labels)),
    }


def _run_arm(
    *,
    arm_id: str,
    reconstruction_input: Mapping[str, Any],
    distance: DistanceMatrix,
    true_tree: nx.DiGraph,
    reconstruction_seed: int,
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, Any]:
    algorithm_name, input_mode, only_nj = ARM_BUILD_SPECS[arm_id]
    record = {
        "arm_id": arm_id,
        "algorithm": algorithm_name,
        "problem": ARM_ENDPOINTS[arm_id]["problem"],
        "declared_metrics": list(ARM_ENDPOINTS[arm_id]["declared_metrics"]),
        "input_mode": input_mode,
        "only_nj": bool(only_nj),
        "status": None,
        "error": None,
        "reconstruction_runtime": None,
        "evaluation_runtime": None,
        "reconstruction_summary": None,
        "evaluation": None,
    }
    reconstructed, runtime, error = measured_stage(
        lambda: reconstruct_arm(
            arm_id,
            reconstruction_input,
            distance,
            reconstruction_seed=reconstruction_seed,
        ),
        timeout_seconds=reconstruction_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    record["reconstruction_runtime"] = runtime
    if error is not None:
        record["status"] = "reconstruction_failure"
        record["error"] = _typed_error(error)
        return record

    reconstructed_tree = reconstructed[0]
    record["reconstruction_summary"] = _tree_summary(reconstructed_tree)
    observed_labels = _observed_labels(reconstruction_input)
    evaluation, evaluation_runtime, evaluation_error = measured_stage(
        lambda: evaluate_tree_pair_result(
            true_tree,
            reconstructed_tree,
            observed_labels,
        ),
        timeout_seconds=evaluation_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    record["evaluation_runtime"] = evaluation_runtime
    if evaluation_error is not None:
        record["status"] = "evaluation_failure"
        record["error"] = _typed_error(evaluation_error)
        return record
    if evaluation.get("status") != "success":
        record["status"] = "evaluation_failure"
        failure_message = str(
            evaluation.get("failure", {}).get("message", "evaluation failure")
        )
        record["error"] = _error_fingerprint(
            "EvaluationContractFailure",
            failure_message,
        )
        record["evaluation"] = {
            "status": "failure",
        }
        return record

    metrics = evaluation["metrics"]
    declared_metrics = ARM_ENDPOINTS[arm_id]["declared_metrics"]
    metric_payload = {}
    if "ad_f1" in declared_metrics:
        metric_payload.update(
            {
                "ad_f1": metrics["ad_f1"],
                "ad_precision": metrics["ad_precision"],
                "ad_recall": metrics["ad_recall"],
                "ad_iou": metrics["ad_iou"],
                "ad_counts": metrics["ad_counts"],
                "ad_f1_degenerate": metrics["ad_f1_degenerate"],
                "ad_f1_degeneracy": metrics["ad_f1_degeneracy"],
            }
        )
    if "grf" in declared_metrics:
        metric_payload["grf"] = metrics["grf"]
    record["status"] = "success"
    record["evaluation"] = {
        "status": "success",
        "required_observation_label_count": evaluation["inputs"][
            "observation_label_coverage"
        ]["required_unique_label_count"],
        "reconstructed_observation_label_count": evaluation["inputs"][
            "observation_label_coverage"
        ]["reconstructed_unique_label_count"],
        "observation_label_coverage_fraction": evaluation["inputs"][
            "observation_label_coverage"
        ]["fraction"],
        "metrics": metric_payload,
    }
    return record


def _run_case(
    *,
    base_config: Mapping[str, Any],
    height: int,
    replicate_index: int,
    base_seed: int,
    distance_compute: DistanceCompute,
    injected_distance: bool,
    fixed_radius: float,
    simulation_timeout_seconds: int,
    distance_timeout_seconds: int,
    reconstruction_timeout_seconds: int,
    evaluation_timeout_seconds: int,
    rss_limit_bytes: int,
) -> dict[str, Any]:
    generations = APPROVED_SCHEDULES[height]
    simulation_seed = _simulation_seed(base_seed, replicate_index)
    reconstruction_seed = _reconstruction_seed(base_seed, replicate_index, height)
    case_key = f"v5-intuition-H{height}-R{replicate_index + 1:03d}"
    config = dict(base_config)
    config["NUMBER_OF_GENERATIONS"] = height
    with tempfile.TemporaryDirectory(prefix="ctbf-v5-intuition-") as directory:
        config_path = Path(directory) / "case.json"
        write_json_atomic(config_path, config)
        prepared, simulation_runtime, simulation_error = measured_stage(
            lambda: _prepare_case(
                config_path=config_path,
                simulation_seed=simulation_seed,
                base_seed=base_seed,
                replicate_index=replicate_index,
                height=height,
                generations=generations,
            ),
            timeout_seconds=simulation_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )

    record = {
        "case_key": case_key,
        "height": height,
        "replicate_index": replicate_index,
        "simulation_seed": simulation_seed,
        "reconstruction_seed": reconstruction_seed,
        "generations": list(generations),
        "status": None,
        "error": None,
        "simulation_runtime": simulation_runtime,
        "simulation_summary": prepared["summary"] if prepared is not None else None,
        "distance": None,
        "ambiguity": None,
        "arms": [],
    }
    if simulation_error is not None:
        record["status"] = (
            "empty_required_biopsy"
            if isinstance(simulation_error, EmptyRequiredBiopsyError)
            else "simulation_failure"
        )
        record["error"] = _typed_error(simulation_error)
        return record

    distance, distance_runtime, distance_error = measured_stage(
        lambda: distance_compute(prepared["distance_cells"]),
        timeout_seconds=distance_timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if distance_error is None:
        try:
            _validate_production_distance(distance)
        except Exception as error:  # explicit typed provider boundary
            distance_error = error
            distance = None
    if distance_error is not None:
        record["status"] = "distance_failure"
        record["error"] = _typed_error(distance_error)
        record["distance"] = {"runtime": distance_runtime}
        return record

    try:
        ambiguity = ambiguity_case_summary(
            prepared["levels"],
            generations,
            distance,
            fixed_radius=fixed_radius,
        )
    except Exception as error:
        record["status"] = "diagnostic_failure"
        record["error"] = _typed_error(error)
        return record

    identity = _distance_identity(distance.provenance)
    record["distance"] = {
        "runtime": distance_runtime,
        "summary": distance_matrix_summary(distance),
        "identity": identity,
        "identity_sha256": _sha256_json(identity),
        "external_process_count": distance.provenance.get("external_process_count"),
        "injected_distance_for_test": injected_distance,
    }
    record["ambiguity"] = ambiguity
    record["arms"] = [
        _run_arm(
            arm_id=arm_id,
            reconstruction_input=prepared["reconstruction_input"],
            distance=distance,
            true_tree=prepared["true_tree"],
            reconstruction_seed=reconstruction_seed,
            reconstruction_timeout_seconds=reconstruction_timeout_seconds,
            evaluation_timeout_seconds=evaluation_timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )
        for arm_id in ARM_IDS
    ]
    record["status"] = "complete"
    del prepared
    gc.collect()
    return record


def _wins_ties_losses(
    values: Sequence[float],
    tolerance: float = 1e-12,
) -> dict[str, int]:
    wins = sum(value > tolerance for value in values)
    losses = sum(value < -tolerance for value in values)
    return {
        "wins": wins,
        "ties": len(values) - wins - losses,
        "losses": losses,
    }


def _successful_arm(record: Mapping[str, Any], arm_id: str) -> Mapping[str, Any] | None:
    for arm in record.get("arms", []):
        if arm.get("arm_id") == arm_id and arm.get("status") == "success":
            return arm
    return None


def _arm_metric(record: Mapping[str, Any], arm_id: str, metric: str) -> float | None:
    arm = _successful_arm(record, arm_id)
    if arm is None:
        return None
    value = arm["evaluation"]["metrics"].get(metric)
    return float(value) if value is not None else None


TRUTH_AGGREGATE_METRICS = (
    "truth_node_count",
    "truth_unique_state_label_count",
    "repeated_state_occurrence_count",
    "unique_state_label_coverage_fraction",
    "observation_to_truth_node_ratio",
    "comparable_ancestor_descendant_pair_fraction",
    "incomparable_pair_fraction",
    "hidden_fork_pair_fraction",
    "sampled_ancestor_coverage_fraction",
    "minimal_sampled_occurrence_count",
    "minimum_invented_edges_for_observed_only_arborescence",
    "same_state_pair_count",
    "incomparable_same_state_pair_count",
)

RECONSTRUCTION_AGGREGATE_METRICS = (
    "node_count",
    "edge_count",
    "leaf_count",
    "maximum_depth",
    "labeled_occurrence_count",
    "unlabeled_node_count",
    "unique_state_label_count",
    "repeated_state_label_occurrence_count",
)


def _height_block(
    height: int,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    complete = [record for record in records if record.get("status") == "complete"]
    sampling_metrics = {}
    for metric in (
        "selected_occurrence_count",
        "selected_unique_state_count",
        "distance_matrix_cell_count",
        "bidirectional_ordered_pair_bound",
    ):
        sampling_metrics[metric] = _number_summary(
            [float(record["simulation_summary"][metric]) for record in complete]
        )
    biopsy_levels = []
    for biopsy_level, generation in enumerate(APPROVED_SCHEDULES[height]):
        rows = [
            record["simulation_summary"]["sampling"][biopsy_level]
            for record in complete
        ]
        biopsy_levels.append(
            {
                "biopsy_level": biopsy_level,
                "generation": generation,
                "configured_state_cap": MAX_STATES_PER_BIOPSY,
                "available_distinct_state_count": _number_summary(
                    [float(row["available_distinct_state_count"]) for row in rows]
                ),
                "realized_occurrence_count": _number_summary(
                    [float(row["realized_occurrence_count"]) for row in rows]
                ),
                "below_cap_case_count": sum(
                    row["available_distinct_state_count"]
                    < MAX_STATES_PER_BIOPSY
                    for row in rows
                ),
                "selection_mode_counts": dict(
                    sorted(Counter(row["selection_mode"] for row in rows).items())
                ),
            }
        )
    truth_metrics = {}
    for metric in TRUTH_AGGREGATE_METRICS:
        values = [
            record["simulation_summary"]["truth_sampling_diagnostics"].get(metric)
            for record in complete
        ]
        truth_metrics[metric] = _number_summary(
            [float(value) for value in values if value is not None]
        )
    for source_metric, aggregate_metric in (
        (
            "nearest_sampled_ancestor_edge_distance",
            "case_mean_nearest_sampled_ancestor_edge_distance",
        ),
        (
            "hidden_internal_nodes_to_nearest_sampled_ancestor",
            "case_mean_hidden_internal_nodes_to_nearest_sampled_ancestor",
        ),
    ):
        values = [
            record["simulation_summary"]["truth_sampling_diagnostics"][
                source_metric
            ]["mean"]
            for record in complete
        ]
        truth_metrics[aggregate_metric] = _number_summary(
            [float(value) for value in values if value is not None]
        )
    truth_metrics["observed_only_occurrence_arborescence_representable_fraction"] = (
        _fraction(
            sum(
                bool(
                    record["simulation_summary"]["truth_sampling_diagnostics"][
                        "observed_only_occurrence_arborescence_representable"
                    ]
                )
                for record in complete
            ),
            len(complete),
        )
    )

    ambiguity_metrics = {
        metric: _number_summary(
            [
                float(record["ambiguity"]["case_metrics"][metric])
                for record in complete
                if record["ambiguity"]["case_metrics"].get(metric) is not None
            ]
        )
        for metric in AMBIGUITY_CASE_METRICS
    }
    low_information_flag_counts = {
        flag: sum(
            bool(record["simulation_summary"]["low_information_flags"][flag])
            for record in complete
        )
        for flag in (
            "one_unique_state",
            "fewer_than_three_unique_states",
            "fewer_than_six_total_occurrences",
        )
    }
    distance_runtime_seconds = _number_summary(
        [
            record["distance"]["runtime"]["wall_time_ns"] / 1_000_000_000
            for record in complete
        ]
    )
    distance_external_process_count = _number_summary(
        [
            float(record["distance"]["external_process_count"])
            for record in complete
            if record["distance"]["external_process_count"] is not None
        ]
    )
    arms = {}
    for arm_id in ARM_IDS:
        arm_records = [
            arm
            for record in complete
            for arm in record["arms"]
            if arm["arm_id"] == arm_id
        ]
        arms[arm_id] = {
            "problem": ARM_ENDPOINTS[arm_id]["problem"],
            "declared_metrics": list(ARM_ENDPOINTS[arm_id]["declared_metrics"]),
            "status_counts": dict(
                sorted(Counter(arm["status"] for arm in arm_records).items())
            ),
            "error_type_counts": dict(
                sorted(
                    Counter(
                        arm["error"]["type"]
                        for arm in arm_records
                        if arm.get("error") is not None
                    ).items()
                )
            ),
            "declared_metric_summaries": {
                metric: _number_summary(
                    [
                        float(arm["evaluation"]["metrics"][metric])
                        for arm in arm_records
                        if arm["status"] == "success"
                    ]
                )
                for metric in ARM_ENDPOINTS[arm_id]["declared_metrics"]
            },
            "ad_f1_degenerate_count": (
                sum(
                    arm["status"] == "success"
                    and arm["evaluation"]["metrics"]["ad_f1_degenerate"]
                    for arm in arm_records
                )
                if "ad_f1" in ARM_ENDPOINTS[arm_id]["declared_metrics"]
                else None
            ),
            "reconstruction_summaries": {
                metric: _number_summary(
                    [
                        float(arm["reconstruction_summary"][metric])
                        for arm in arm_records
                        if arm["reconstruction_summary"] is not None
                    ]
                )
                for metric in RECONSTRUCTION_AGGREGATE_METRICS
            },
            "reconstruction_wall_time_seconds": _number_summary(
                [
                    arm["reconstruction_runtime"]["wall_time_ns"] / 1_000_000_000
                    for arm in arm_records
                ]
            ),
            "evaluation_wall_time_seconds": _number_summary(
                [
                    arm["evaluation_runtime"]["wall_time_ns"] / 1_000_000_000
                    for arm in arm_records
                    if arm["evaluation_runtime"] is not None
                ]
            ),
        }

    contrasts = {}
    for name, left, right, metric in (
        (
            "temporal_minus_no_time_ad_f1",
            "temporal_minimum",
            "temporal_minimum_no_time",
            "ad_f1",
        ),
        (
            "temporal_minus_no_time_grf",
            "temporal_minimum",
            "temporal_minimum_no_time",
            "grf",
        ),
        (
            "biopsy_guided_minus_classical_partial_grf",
            "biopsy_guided_classical",
            "classical_partial",
            "grf",
        ),
    ):
        differences = []
        for record in complete:
            left_value = _arm_metric(record, left, metric)
            right_value = _arm_metric(record, right, metric)
            if left_value is not None and right_value is not None:
                differences.append(left_value - right_value)
        contrasts[name] = {
            "left_arm": left,
            "right_arm": right,
            "metric": metric,
            "difference": _number_summary(differences),
            "wins_ties_losses": _wins_ties_losses(differences),
        }
    return {
        "attempted_case_count": len(records),
        "status_counts": dict(
            sorted(Counter(record["status"] for record in records).items())
        ),
        "error_type_counts": dict(
            sorted(
                Counter(
                    record["error"]["type"]
                    for record in records
                    if record.get("error") is not None
                ).items()
            )
        ),
        "sampling_metrics": sampling_metrics,
        "biopsy_levels": biopsy_levels,
        "truth_metrics": truth_metrics,
        "ambiguity_metrics": ambiguity_metrics,
        "low_information_flag_counts": low_information_flag_counts,
        "distance_wall_time_seconds": distance_runtime_seconds,
        "distance_external_process_count": distance_external_process_count,
        "arms": arms,
        "within_height_contrasts": contrasts,
    }


def _paired_height_differences(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_replicate = {
        replicate_index: {
            record["height"]: record
            for record in cases
            if record["replicate_index"] == replicate_index
            and record["status"] == "complete"
        }
        for replicate_index in sorted({record["replicate_index"] for record in cases})
    }
    low_height = min(APPROVED_SCHEDULES)
    high_height = max(APPROVED_SCHEDULES)
    truth = {}
    for metric in TRUTH_AGGREGATE_METRICS:
        differences = []
        for block in by_replicate.values():
            if low_height not in block or high_height not in block:
                continue
            low = block[low_height]["simulation_summary"][
                "truth_sampling_diagnostics"
            ].get(metric)
            high = block[high_height]["simulation_summary"][
                "truth_sampling_diagnostics"
            ].get(metric)
            if low is not None and high is not None:
                differences.append(float(high) - float(low))
        truth[metric] = {
            "difference": _number_summary(differences),
            "wins_ties_losses": _wins_ties_losses(differences),
        }
    arms = {}
    for arm_id in ARM_IDS:
        arms[arm_id] = {}
        for metric in ARM_ENDPOINTS[arm_id]["declared_metrics"]:
            differences = []
            for block in by_replicate.values():
                if low_height not in block or high_height not in block:
                    continue
                low = _arm_metric(block[low_height], arm_id, metric)
                high = _arm_metric(block[high_height], arm_id, metric)
                if low is not None and high is not None:
                    differences.append(high - low)
            arms[arm_id][metric] = {
                "difference": _number_summary(differences),
                "wins_ties_losses": _wins_ties_losses(differences),
            }
    return {
        "contrast": f"H{high_height}_minus_H{low_height}",
        "truth_metrics": truth,
        "arm_metrics": arms,
    }


def _prefix_consistency(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    planned_comparison_count = 0
    fully_available_comparison_count = 0
    evaluable_comparison_count = 0
    evaluable_failure_count = 0
    for replicate_index in sorted({record["replicate_index"] for record in cases}):
        block = {
            record["height"]: record
            for record in cases
            if record["replicate_index"] == replicate_index
            and record.get("simulation_summary") is not None
        }
        comparisons = []
        for prefix_height, containing_heights in (
            (14, (14, 24, 34)),
            (24, (24, 34)),
        ):
            planned_comparison_count += 1
            hashes = [
                block[height]["simulation_summary"][
                    "truth_prefix_sha256_by_height"
                ][str(prefix_height)]
                for height in containing_heights
                if height in block
            ]
            fully_available = len(hashes) == len(containing_heights)
            evaluable = len(hashes) >= 2
            passed = len(set(hashes)) == 1 if evaluable else None
            fully_available_comparison_count += int(fully_available)
            evaluable_comparison_count += int(evaluable)
            evaluable_failure_count += int(evaluable and not passed)
            comparisons.append(
                {
                    "prefix_height": prefix_height,
                    "containing_heights": list(containing_heights),
                    "all_containing_heights_available": fully_available,
                    "comparison_evaluable": evaluable,
                    "passed": passed,
                }
            )
        rows.append(
            {
                "replicate_index": replicate_index,
                "comparisons": comparisons,
            }
        )
    return {
        "planned_comparison_count": planned_comparison_count,
        "fully_available_comparison_count": fully_available_comparison_count,
        "evaluable_comparison_count": evaluable_comparison_count,
        "evaluable_failure_count": evaluable_failure_count,
        "all_evaluable_common_seed_prefix_checks_passed": (
            evaluable_failure_count == 0
        ),
        "all_planned_common_seed_prefix_checks_available_and_passed": (
            fully_available_comparison_count == planned_comparison_count
            and evaluable_failure_count == 0
        ),
        "by_replicate": rows,
    }


def aggregate_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "by_height": {
            str(height): _height_block(
                height,
                [record for record in cases if record["height"] == height]
            )
            for height in APPROVED_SCHEDULES
        },
        "paired_endpoint_differences": _paired_height_differences(cases),
        "common_seed_prefix_consistency": _prefix_consistency(cases),
    }


def _validate_positive_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")


def run_probe(
    *,
    base_config_path: Path | str = DEFAULT_BASE_CONFIG,
    replicates: int = DEFAULT_REPLICATES,
    base_seed: int = DEFAULT_BASE_SEED,
    fixed_radius: float = DEFAULT_FIXED_RADIUS,
    simulation_timeout_seconds: int = DEFAULT_SIMULATION_TIMEOUT_SECONDS,
    distance_timeout_seconds: int = DEFAULT_DISTANCE_TIMEOUT_SECONDS,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    evaluation_timeout_seconds: int = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
    cnp2cnp_process_timeout_seconds: int = DEFAULT_CNP2CNP_PROCESS_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    max_case_dependency_failures: int = DEFAULT_MAX_CASE_DEPENDENCY_FAILURES,
    distance_compute: DistanceCompute | None = None,
    created_at_utc: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    base_config_path = Path(base_config_path).expanduser().resolve()
    if not base_config_path.is_file():
        raise ValueError(f"Base config is not a file: {base_config_path}.")
    base_config = read_json(base_config_path)
    _validate_standard_base_config(base_config)
    if any(
        rule_y_schedule(height) != schedule
        for height, schedule in APPROVED_SCHEDULES.items()
    ):
        raise RuntimeError("Approved schedules disagree with Rule Y.")
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise ValueError("replicates must be an integer.")
    if not 1 <= replicates <= MAX_REPLICATES:
        raise ValueError(f"replicates must be in [1,{MAX_REPLICATES}].")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer.")
    if fixed_radius != DEFAULT_FIXED_RADIUS:
        raise ValueError(f"The probe requires fixed radius {DEFAULT_FIXED_RADIUS}.")
    for field, value in (
        ("simulation_timeout_seconds", simulation_timeout_seconds),
        ("distance_timeout_seconds", distance_timeout_seconds),
        ("reconstruction_timeout_seconds", reconstruction_timeout_seconds),
        ("evaluation_timeout_seconds", evaluation_timeout_seconds),
        ("cnp2cnp_process_timeout_seconds", cnp2cnp_process_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("capture_limit_bytes", capture_limit_bytes),
        ("max_case_dependency_failures", max_case_dependency_failures),
    ):
        _validate_positive_integer(value, field)

    if distance_compute is None:
        runtime_config = replace(
            load_ctbs_runtime_config(),
            cnp2cnp_timeout_seconds=float(cnp2cnp_process_timeout_seconds),
            cnp2cnp_capture_limit_bytes=int(capture_limit_bytes),
        )
        compute = Cnp2CnpFileDistanceProvider(runtime_config).compute
    else:
        compute = distance_compute

    cases = []
    dependency_failure_count = 0
    stopped_early = False
    for replicate_index in range(replicates):
        for height in APPROVED_SCHEDULES:
            record = _run_case(
                base_config=base_config,
                height=height,
                replicate_index=replicate_index,
                base_seed=base_seed,
                distance_compute=compute,
                injected_distance=distance_compute is not None,
                fixed_radius=fixed_radius,
                simulation_timeout_seconds=simulation_timeout_seconds,
                distance_timeout_seconds=distance_timeout_seconds,
                reconstruction_timeout_seconds=reconstruction_timeout_seconds,
                evaluation_timeout_seconds=evaluation_timeout_seconds,
                rss_limit_bytes=rss_limit_bytes,
            )
            cases.append(record)
            if progress:
                arm_counts = Counter(arm["status"] for arm in record.get("arms", []))
                print(
                    json.dumps(
                        {
                            "case_key": record["case_key"],
                            "status": record["status"],
                            "arm_status_counts": dict(sorted(arm_counts.items())),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            if record["status"] != "complete":
                dependency_failure_count += 1
                if dependency_failure_count >= max_case_dependency_failures:
                    stopped_early = True
                    break
        if stopped_early:
            break

    aggregate = aggregate_cases(cases)
    evaluable_prefixes_passed = aggregate["common_seed_prefix_consistency"][
        "all_evaluable_common_seed_prefix_checks_passed"
    ]
    if not evaluable_prefixes_passed:
        raise ValueError("Common-seed height prefixes are inconsistent.")

    identities: dict[str, dict[str, Any]] = {}
    for record in cases:
        if record.get("distance", {}).get("identity_sha256"):
            identity_hash = record["distance"]["identity_sha256"]
            identities.setdefault(
                identity_hash,
                {"case_count": 0, "identity": record["distance"]["identity"]},
            )["case_count"] += 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "status": "stopped_at_dependency_failure_cap" if stopped_early else "complete",
        "scientific_role": {
            "paper_evidence_allowed": False,
            "discovery_only": True,
            "simulation_run": True,
            "cnp2cnp_run": distance_compute is None,
            "injected_distance_for_test": distance_compute is not None,
            "reconstruction_run": True,
            "evaluation_run": True,
            "selects_simulator_parameters_from_accuracy": False,
            "freezes_paper_height_set": False,
        },
        "question": {
            "primary": (
                "Do H14/H24/H34 with Rule-Y capped-six sampling form a "
                "nontrivial but nondegenerate reconstruction gradient under CTBF v5?"
            ),
            "height_estimand": (
                "end-to-end latent-depth plus naturally available observation regime"
            ),
            "not_an_isolated_fixed_sample_size_height_effect": True,
            "no_cross_output_problem_algorithm_ranking": True,
            "no_significance_testing": True,
        },
        "input": {
            "base_config": (
                base_config_path.relative_to(PROJECT_ROOT).as_posix()
                if base_config_path.is_relative_to(PROJECT_ROOT)
                else str(base_config_path)
            ),
            "base_config_sha256": _file_sha256(base_config_path),
            "heights": list(APPROVED_SCHEDULES),
            "rule_y_positions": [0.6, 0.8, 1.0],
            "rounding": "ceiling",
            "schedule_by_height": {
                str(height): list(schedule)
                for height, schedule in APPROVED_SCHEDULES.items()
            },
            "sampling_version": SAMPLING_VERSION,
            "maximum_distinct_states_per_biopsy": MAX_STATES_PER_BIOPSY,
            "below_cap_policy": "take_all_available_nonempty_states",
            "replicates": replicates,
            "base_seed": base_seed,
            "seed_namespace": SEED_NAMESPACE,
            "common_simulation_seed_within_replicate_across_heights": True,
            "fresh_from_growth_probe_by_namespace": True,
            "reconstruction_seed_shared_across_arms_within_case": True,
            "fixed_radius": fixed_radius,
            "distance_semantics": CNP2CNP_SEMANTICS_VERSION,
            "arm_portfolio": [
                {
                    "arm_id": arm_id,
                    "algorithm": algorithm,
                    **{
                        "problem": ARM_ENDPOINTS[arm_id]["problem"],
                        "declared_metrics": list(
                            ARM_ENDPOINTS[arm_id]["declared_metrics"]
                        ),
                    },
                }
                for arm_id, algorithm in REGISTERED_ARM_SPECS
            ],
        },
        "resource_bound": {
            "planned_case_count": replicates * len(APPROVED_SCHEDULES),
            "attempted_case_count": len(cases),
            "maximum_observed_occurrences_per_case": MAX_OBSERVED_OCCURRENCES_PER_CASE,
            "maximum_unique_distance_states_per_case": (
                MAX_OBSERVED_OCCURRENCES_PER_CASE
            ),
            "maximum_distance_matrix_cells_per_case": (
                MAX_OBSERVED_OCCURRENCES_PER_CASE**2
            ),
            "maximum_bidirectional_ordered_pairs_per_case": (
                MAX_OBSERVED_OCCURRENCES_PER_CASE
                * (MAX_OBSERVED_OCCURRENCES_PER_CASE - 1)
            ),
            "arm_count_per_complete_case": len(ARM_IDS),
            "sequential_execution": True,
            "simulation_timeout_seconds": simulation_timeout_seconds,
            "distance_timeout_seconds": distance_timeout_seconds,
            "reconstruction_timeout_seconds_per_arm": reconstruction_timeout_seconds,
            "evaluation_timeout_seconds_per_arm": evaluation_timeout_seconds,
            "cnp2cnp_process_timeout_seconds": cnp2cnp_process_timeout_seconds,
            "rss_limit_bytes_per_stage": rss_limit_bytes,
            "capture_limit_bytes": capture_limit_bytes,
            "max_case_dependency_failures": max_case_dependency_failures,
        },
        "cases": cases,
        "aggregate": aggregate,
        "distance_identity_counts": [
            {"identity_sha256": identity_hash, **identities[identity_hash]}
            for identity_hash in sorted(identities)
        ],
        "provenance": {
            "source_module": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "source_module_sha256": _file_sha256(Path(__file__)),
            "reads_existing_result_corpus": False,
            "writes_raw_profiles": False,
            "writes_truth_or_reconstructed_trees": False,
            "writes_distance_matrices": False,
            "writes_simulator_node_identities": False,
            "replaces_failures_or_low_information_cases": False,
        },
    }
    validate_report(report)
    return report


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown reconstruction-intuition schema.")
    if report.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("Reconstruction-intuition report has the wrong role.")
    if report.get("status") not in {
        "complete",
        "stopped_at_dependency_failure_cap",
    }:
        raise ValueError("Reconstruction-intuition report has an unknown status.")
    role = report.get("scientific_role")
    if not isinstance(role, Mapping):
        raise ValueError("Reconstruction-intuition report lacks its scientific role.")
    expected_role = {
        "paper_evidence_allowed": False,
        "discovery_only": True,
        "simulation_run": True,
        "reconstruction_run": True,
        "evaluation_run": True,
        "selects_simulator_parameters_from_accuracy": False,
        "freezes_paper_height_set": False,
    }
    for field, expected in expected_role.items():
        if role.get(field) is not expected:
            raise ValueError(f"Scientific role has invalid {field}.")
    allowed_case_statuses = {
        "complete",
        "simulation_failure",
        "empty_required_biopsy",
        "distance_failure",
        "diagnostic_failure",
    }
    for case in report.get("cases", []):
        if case.get("status") not in allowed_case_statuses:
            raise ValueError("Reconstruction-intuition case has an unknown status.")
        if case["status"] == "complete":
            arm_ids = tuple(arm.get("arm_id") for arm in case.get("arms", []))
            if arm_ids != ARM_IDS:
                raise ValueError("A complete case has the wrong arm portfolio.")
            if case.get("simulation_summary") is None or case.get("distance") is None:
                raise ValueError("A complete case lacks a required dependency summary.")
            for row in case["simulation_summary"]["sampling"]:
                realized = row["realized_occurrence_count"]
                available = row["available_distinct_state_count"]
                if realized != min(MAX_STATES_PER_BIOPSY, available):
                    raise ValueError("Capped biopsy realization is inconsistent.")
                if realized <= 0:
                    raise ValueError("A complete case contains an empty biopsy.")
            for arm in case["arms"]:
                if arm.get("status") not in {
                    "success",
                    "reconstruction_failure",
                    "evaluation_failure",
                }:
                    raise ValueError("A reconstruction arm has an unknown status.")
                if arm["status"] != "success":
                    continue
                expected_metric_fields = {"grf"}
                if "ad_f1" in ARM_ENDPOINTS[arm["arm_id"]]["declared_metrics"]:
                    expected_metric_fields.update(
                        {
                            "ad_f1",
                            "ad_precision",
                            "ad_recall",
                            "ad_iou",
                            "ad_counts",
                            "ad_f1_degenerate",
                            "ad_f1_degeneracy",
                        }
                    )
                actual_metric_fields = set(arm["evaluation"]["metrics"])
                if actual_metric_fields != expected_metric_fields:
                    raise ValueError(
                        "A successful arm exposes metrics outside its declared "
                        "output-problem contract."
                    )
    forbidden_raw_keys = {
        "cnp",
        "genome",
        "tree",
        "matrix",
        "node_id",
        "nodes",
        "links",
    }
    present = forbidden_raw_keys & set(_walk_keys(report))
    if present:
        raise ValueError(
            "Compact report contains forbidden raw fields: "
            + ", ".join(sorted(present))
        )
    json.dumps(report, sort_keys=True, allow_nan=False)


def compact_summary(report: Mapping[str, Any], output: Path) -> dict[str, Any]:
    by_height = {}
    for height, block in report["aggregate"]["by_height"].items():
        by_height[height] = {
            "status_counts": block["status_counts"],
            "error_type_counts": block["error_type_counts"],
            "sampling_metrics": block["sampling_metrics"],
            "biopsy_levels": block["biopsy_levels"],
            "truth_metrics": block["truth_metrics"],
            "ambiguity_metrics": block["ambiguity_metrics"],
            "low_information_flag_counts": block[
                "low_information_flag_counts"
            ],
            "distance_wall_time_seconds": block["distance_wall_time_seconds"],
            "distance_external_process_count": block[
                "distance_external_process_count"
            ],
            "arms": {
                arm_id: {
                    "problem": arm["problem"],
                    "declared_metrics": arm["declared_metrics"],
                    "status_counts": arm["status_counts"],
                    "error_type_counts": arm["error_type_counts"],
                    "declared_metric_means": {
                        metric: summary["mean"]
                        for metric, summary in arm[
                            "declared_metric_summaries"
                        ].items()
                    },
                    "reconstruction_summary_means": {
                        metric: summary["mean"]
                        for metric, summary in arm[
                            "reconstruction_summaries"
                        ].items()
                    },
                    "mean_reconstruction_wall_time_seconds": arm[
                        "reconstruction_wall_time_seconds"
                    ]["mean"],
                    "mean_evaluation_wall_time_seconds": arm[
                        "evaluation_wall_time_seconds"
                    ]["mean"],
                }
                for arm_id, arm in block["arms"].items()
            },
            "within_height_contrasts": block["within_height_contrasts"],
        }
    return {
        "schema_version": report["schema_version"],
        "analysis_role": report["analysis_role"],
        "status": report["status"],
        "output": str(output.resolve()),
        "planned_case_count": report["resource_bound"]["planned_case_count"],
        "attempted_case_count": report["resource_bound"]["attempted_case_count"],
        "common_seed_prefix_consistency": report["aggregate"][
            "common_seed_prefix_consistency"
        ],
        "by_height": by_height,
        "paired_endpoint_differences": report["aggregate"][
            "paired_endpoint_differences"
        ],
        "next_stage": "owner_and_agent_review_before_any_paper_height_freeze",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded non-paper H14/H24/H34 CTBF v5 reconstruction "
            "intuition probe with all six established arms."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
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
        "--reconstruction-timeout-seconds",
        type=int,
        default=DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--evaluation-timeout-seconds",
        type=int,
        default=DEFAULT_EVALUATION_TIMEOUT_SECONDS,
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
        "--max-case-dependency-failures",
        type=int,
        default=DEFAULT_MAX_CASE_DEPENDENCY_FAILURES,
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new path or pass --overwrite."
        )
    report = run_probe(
        base_config_path=arguments.base_config,
        replicates=arguments.replicates,
        base_seed=arguments.base_seed,
        simulation_timeout_seconds=arguments.simulation_timeout_seconds,
        distance_timeout_seconds=arguments.distance_timeout_seconds,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        evaluation_timeout_seconds=arguments.evaluation_timeout_seconds,
        cnp2cnp_process_timeout_seconds=arguments.cnp2cnp_process_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        capture_limit_bytes=arguments.capture_limit_bytes,
        max_case_dependency_failures=arguments.max_case_dependency_failures,
        progress=arguments.progress,
    )
    write_json_atomic(output, report)
    print(json.dumps(compact_summary(report, output), indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
