"""Truth-only nested sampling-fraction diagnostic for the CTBF v5 simulator.

The probe reuses the H14/H24/H34 Rule-Y truth blocks from the completed
reconstruction-intuition study.  For each nonempty biopsy generation with N
distinct representative genotype states, it selects

    min(N, max(K, ceil(p * N)))

states for K=6 and p in {0, 5%, 10%, 25%, 50%}.  One seeded ordering is used
per case and generation, so the selections are nested.  The probe runs no
distance provider, reconstruction algorithm, or evaluator and writes no raw
profiles, trees, matrices, or simulator node identities.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from algorithm_evaluation.hypothesis_trend_distance_preflight import (
    _number_summary,
)
from algorithm_evaluation.paper_pipeline_contract import (
    PROJECT_ROOT,
    canonical_json_sha256,
    read_json,
    write_json_atomic,
)
from algorithm_evaluation.paper_pipeline_runner import measured_stage
from algorithm_evaluation.simulator_growth_probe import (
    _file_sha256,
    _validate_standard_base_config,
)
from algorithm_evaluation.simulator_reconstruction_intuition_probe import (
    APPROVED_SCHEDULES,
    DEFAULT_BASE_CONFIG,
    DEFAULT_BASE_SEED,
    EmptyRequiredBiopsyError,
    SCHEMA_VERSION as REFERENCE_SCHEMA_VERSION,
    SEED_NAMESPACE as REUSED_SEED_NAMESPACE,
    _canonical_cells_at_generation,
    _prefix_consistency,
    _profile,
    _sampling_seed,
    _simulation_seed,
    _truth_prefix_sha256,
    truth_sampling_diagnostics,
    validate_report as validate_reference_report,
)
from distance_semantics import stable_distance_label_key
from simulator import CancerCellEvolutionSimulator, Genotype


SCHEMA_VERSION = "ctbf-v5-simulator-sampling-fraction-truth-probe-v1"
ANALYSIS_ROLE = "nonpaper_simulator_sampling_fraction_truth_probe"
SAMPLING_VERSION = "nested-fraction-with-lower-bound-v1"

DEFAULT_REPLICATES = 12
MAX_REPLICATES = 12
DEFAULT_LOWER_BOUND = 6
DEFAULT_SIMULATION_ANALYSIS_TIMEOUT_SECONDS = 300
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
DEFAULT_MAX_CASE_FAILURES = 6
MAX_UNORDERED_TRUTH_PAIRS_PER_CONDITION = 500_000

CONDITIONS = (
    ("capped_six_control", 0.0),
    ("fraction_05", 0.05),
    ("fraction_10", 0.10),
    ("fraction_25", 0.25),
    ("fraction_50", 0.50),
)
CONDITION_IDS = tuple(condition_id for condition_id, _fraction in CONDITIONS)


class TruthPairAnalysisLimitExceeded(RuntimeError):
    """Typed abort before an unexpectedly large pairwise truth analysis."""


def hybrid_sample_size(
    generation_size: int,
    biopsy_fraction: float,
    biopsy_lower_bound: int = DEFAULT_LOWER_BOUND,
) -> int:
    """Return min(N, max(K, ceil(pN))) with strict input validation."""
    if (
        isinstance(generation_size, bool)
        or not isinstance(generation_size, int)
        or generation_size < 0
    ):
        raise ValueError("generation_size must be a nonnegative integer.")
    if (
        isinstance(biopsy_lower_bound, bool)
        or not isinstance(biopsy_lower_bound, int)
        or biopsy_lower_bound < 1
    ):
        raise ValueError("biopsy_lower_bound must be a positive integer.")
    if (
        isinstance(biopsy_fraction, bool)
        or not isinstance(biopsy_fraction, (int, float))
        or not math.isfinite(float(biopsy_fraction))
        or not 0.0 <= float(biopsy_fraction) <= 1.0
    ):
        raise ValueError("biopsy_fraction must be finite and in [0,1].")
    fractional_count = math.ceil(float(biopsy_fraction) * generation_size)
    available_lower_bound = min(biopsy_lower_bound, generation_size)
    return min(generation_size, max(fractional_count, available_lower_bound))


def _sample_size_driver(
    generation_size: int,
    fractional_count: int,
    lower_bound: int,
    realized_count: int,
) -> str:
    if realized_count == generation_size:
        return "all_available"
    if fractional_count > lower_bound:
        return "fraction"
    if fractional_count < lower_bound:
        return "lower_bound"
    return "fraction_lower_bound_tie"


def select_nested_fraction_levels(
    available_levels: Sequence[Sequence[Genotype]],
    generations: Sequence[int],
    *,
    base_seed: int,
    replicate_index: int,
    height: int,
    lower_bound: int = DEFAULT_LOWER_BOUND,
) -> tuple[
    dict[str, list[list[Genotype]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Select nested prefixes of one seeded ordering at every generation."""
    if len(available_levels) != len(generations):
        raise ValueError("Available levels and generations must align.")

    selected_by_condition = {condition_id: [] for condition_id in CONDITION_IDS}
    rows_by_condition = {condition_id: [] for condition_id in CONDITION_IDS}
    nesting_rows = []
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
        permutation = np.random.Generator(np.random.PCG64(seed)).permutation(
            len(available)
        )
        ordered = [available[int(index)] for index in permutation]
        previous_nodes: set[Any] | None = None
        previous_condition_id = None
        previous_count = None
        for condition_id, biopsy_fraction in CONDITIONS:
            fractional_count = math.ceil(biopsy_fraction * len(available))
            realized_count = hybrid_sample_size(
                len(available),
                biopsy_fraction,
                lower_bound,
            )
            chosen = sorted(
                ordered[:realized_count],
                key=lambda cell: (
                    _profile(cell),
                    stable_distance_label_key(cell.cell_id),
                ),
            )
            chosen_nodes = {cell.node_id for cell in chosen}
            if len(chosen_nodes) != len(chosen):
                raise ValueError("One selected occurrence was duplicated.")
            if previous_nodes is not None and not previous_nodes <= chosen_nodes:
                raise ValueError("Nested sampling prefixes are inconsistent.")

            selected_by_condition[condition_id].append(chosen)
            rows_by_condition[condition_id].append(
                {
                    "generation": int(generation),
                    "available_distinct_state_count": len(available),
                    "target_fraction": biopsy_fraction,
                    "fractional_ceiling_count": fractional_count,
                    "configured_lower_bound": lower_bound,
                    "available_lower_bound_count": min(lower_bound, len(available)),
                    "realized_occurrence_count": realized_count,
                    "realized_fraction": realized_count / len(available),
                    "additional_over_capped_control_count": (
                        realized_count - min(lower_bound, len(available))
                    ),
                    "sample_size_driver": _sample_size_driver(
                        len(available),
                        fractional_count,
                        lower_bound,
                        realized_count,
                    ),
                    "sampling_seed": seed,
                }
            )
            if previous_nodes is not None:
                nesting_rows.append(
                    {
                        "generation": int(generation),
                        "smaller_condition_id": previous_condition_id,
                        "larger_condition_id": condition_id,
                        "nested": previous_nodes <= chosen_nodes,
                        "identical": previous_nodes == chosen_nodes,
                        "added_occurrence_count": realized_count - previous_count,
                    }
                )
            previous_nodes = chosen_nodes
            previous_condition_id = condition_id
            previous_count = realized_count

    return selected_by_condition, rows_by_condition, nesting_rows


def _fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def cross_biopsy_relation_diagnostics(
    truth_tree: nx.DiGraph,
    selected_levels: Sequence[Sequence[Genotype]],
) -> dict[str, Any]:
    """Measure ancestry specifically across ordered biopsy generations."""
    if len(selected_levels) < 2:
        raise ValueError("Cross-biopsy diagnostics require at least two levels.")
    nodes_by_level = [[cell.node_id for cell in level] for level in selected_levels]
    selected_nodes = [node for level in nodes_by_level for node in level]
    if len(set(selected_nodes)) != len(selected_nodes):
        raise ValueError("Selected truth occurrences must be unique.")
    ancestors = {node: nx.ancestors(truth_tree, node) for node in selected_nodes}

    within_pair_count = sum(math.comb(len(level), 2) for level in nodes_by_level)
    cross_pair_count = 0
    cross_ancestor_pair_count = 0
    linked_nodes: set[Any] = set()
    for earlier_index, later_index in itertools.combinations(
        range(len(nodes_by_level)),
        2,
    ):
        earlier = nodes_by_level[earlier_index]
        later = nodes_by_level[later_index]
        cross_pair_count += len(earlier) * len(later)
        for later_node in later:
            related = set(earlier) & ancestors[later_node]
            cross_ancestor_pair_count += len(related)
            linked_nodes.update(related)
            if related:
                linked_nodes.add(later_node)

    transitions = []
    adjacent_later_count = 0
    adjacent_covered_count = 0
    any_earlier_later_count = 0
    any_earlier_covered_count = 0
    adjacent_edge_distances = []
    any_earlier_edge_distances = []
    earlier_union: set[Any] = set(nodes_by_level[0])
    for later_index in range(1, len(nodes_by_level)):
        previous = set(nodes_by_level[later_index - 1])
        later = nodes_by_level[later_index]
        adjacent_covered = 0
        any_earlier_covered = 0
        transition_adjacent_distances = []
        transition_any_distances = []
        adjacent_relation_count = 0
        any_earlier_relation_count = 0
        for later_node in later:
            adjacent_candidates = previous & ancestors[later_node]
            any_candidates = earlier_union & ancestors[later_node]
            adjacent_covered += int(bool(adjacent_candidates))
            any_earlier_covered += int(bool(any_candidates))
            adjacent_relation_count += len(adjacent_candidates)
            any_earlier_relation_count += len(any_candidates)
            if adjacent_candidates:
                nearest = max(
                    adjacent_candidates,
                    key=lambda node: int(truth_tree.nodes[node]["generation"]),
                )
                distance = int(
                    truth_tree.nodes[later_node]["generation"]
                    - truth_tree.nodes[nearest]["generation"]
                )
                transition_adjacent_distances.append(distance)
                adjacent_edge_distances.append(distance)
            if any_candidates:
                nearest = max(
                    any_candidates,
                    key=lambda node: int(truth_tree.nodes[node]["generation"]),
                )
                distance = int(
                    truth_tree.nodes[later_node]["generation"]
                    - truth_tree.nodes[nearest]["generation"]
                )
                transition_any_distances.append(distance)
                any_earlier_edge_distances.append(distance)

        adjacent_later_count += len(later)
        adjacent_covered_count += adjacent_covered
        any_earlier_later_count += len(later)
        any_earlier_covered_count += any_earlier_covered
        transitions.append(
            {
                "earlier_biopsy_level": later_index - 1,
                "later_biopsy_level": later_index,
                "previous_occurrence_count": len(previous),
                "all_earlier_occurrence_count": len(earlier_union),
                "later_occurrence_count": len(later),
                "later_with_previous_biopsy_ancestor_count": adjacent_covered,
                "previous_biopsy_ancestor_coverage_fraction": _fraction(
                    adjacent_covered,
                    len(later),
                ),
                "later_with_any_earlier_biopsy_ancestor_count": (
                    any_earlier_covered
                ),
                "any_earlier_biopsy_ancestor_coverage_fraction": _fraction(
                    any_earlier_covered,
                    len(later),
                ),
                "previous_to_later_ancestor_relation_count": (
                    adjacent_relation_count
                ),
                "previous_to_later_ancestor_pair_fraction": _fraction(
                    adjacent_relation_count,
                    len(previous) * len(later),
                ),
                "all_earlier_to_later_ancestor_relation_count": (
                    any_earlier_relation_count
                ),
                "all_earlier_to_later_ancestor_pair_fraction": _fraction(
                    any_earlier_relation_count,
                    len(earlier_union) * len(later),
                ),
                "nearest_previous_biopsy_ancestor_edge_distance": (
                    _number_summary(transition_adjacent_distances)
                ),
                "nearest_any_earlier_biopsy_ancestor_edge_distance": (
                    _number_summary(transition_any_distances)
                ),
            }
        )
        earlier_union.update(later)

    minimal_nodes = [
        node
        for node in selected_nodes
        if not (set(selected_nodes) & ancestors[node])
    ]
    invented_edges = max(0, len(minimal_nodes) - 1)
    return {
        "selected_occurrence_count": len(selected_nodes),
        "within_biopsy_unordered_pair_count": within_pair_count,
        "cross_biopsy_unordered_pair_count": cross_pair_count,
        "cross_biopsy_ancestor_pair_count": cross_ancestor_pair_count,
        "cross_biopsy_ancestor_pair_fraction": _fraction(
            cross_ancestor_pair_count,
            cross_pair_count,
        ),
        "cross_biopsy_incomparable_pair_count": (
            cross_pair_count - cross_ancestor_pair_count
        ),
        "cross_biopsy_incomparable_pair_fraction": _fraction(
            cross_pair_count - cross_ancestor_pair_count,
            cross_pair_count,
        ),
        "adjacent_later_occurrence_count": adjacent_later_count,
        "adjacent_sampled_ancestor_coverage_count": adjacent_covered_count,
        "adjacent_sampled_ancestor_coverage_fraction": _fraction(
            adjacent_covered_count,
            adjacent_later_count,
        ),
        "any_earlier_later_occurrence_count": any_earlier_later_count,
        "any_earlier_sampled_ancestor_coverage_count": any_earlier_covered_count,
        "any_earlier_sampled_ancestor_coverage_fraction": _fraction(
            any_earlier_covered_count,
            any_earlier_later_count,
        ),
        "nearest_adjacent_sampled_ancestor_edge_distance": _number_summary(
            adjacent_edge_distances
        ),
        "nearest_any_earlier_sampled_ancestor_edge_distance": _number_summary(
            any_earlier_edge_distances
        ),
        "lineage_linked_occurrence_count": len(linked_nodes),
        "lineage_linked_occurrence_fraction": _fraction(
            len(linked_nodes),
            len(selected_nodes),
        ),
        "minimal_sampled_occurrence_count": len(minimal_nodes),
        "minimal_sampled_occurrence_fraction": _fraction(
            len(minimal_nodes),
            len(selected_nodes),
        ),
        "minimum_invented_edges_for_observed_only_arborescence": invented_edges,
        "minimum_invented_edge_fraction": _fraction(
            invented_edges,
            max(0, len(selected_nodes) - 1),
        ),
        "observed_only_occurrence_arborescence_representable": (
            len(minimal_nodes) == 1
        ),
        "transitions": transitions,
    }


def _condition_summary(
    truth_tree: nx.DiGraph,
    selected_levels: Sequence[Sequence[Genotype]],
    sampling_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    occurrences = [cell for level in selected_levels for cell in level]
    unordered_pair_count = math.comb(len(occurrences), 2)
    if unordered_pair_count > MAX_UNORDERED_TRUTH_PAIRS_PER_CONDITION:
        raise TruthPairAnalysisLimitExceeded(
            "Truth-only pair count exceeds the operational diagnostic limit."
        )
    unique_state_count = len({cell.cell_id for cell in occurrences})
    general = truth_sampling_diagnostics(truth_tree, selected_levels)
    cross = cross_biopsy_relation_diagnostics(truth_tree, selected_levels)
    total_available = sum(
        int(row["available_distinct_state_count"]) for row in sampling_rows
    )
    selected_count = len(occurrences)
    hidden_mean = general[
        "hidden_internal_nodes_to_nearest_sampled_ancestor"
    ]["mean"]
    scalar_metrics = {
        "selected_occurrence_count": selected_count,
        "selected_unique_state_count": unique_state_count,
        "repeated_state_occurrence_count": selected_count - unique_state_count,
        "realized_overall_state_fraction": selected_count / total_available,
        "projected_distance_matrix_cell_count": unique_state_count**2,
        "projected_bidirectional_ordered_pair_count": (
            unique_state_count * (unique_state_count - 1)
        ),
        "all_pair_incomparable_fraction": general["incomparable_pair_fraction"],
        "all_pair_hidden_fork_fraction": general["hidden_fork_pair_fraction"],
        "cross_biopsy_ancestor_pair_fraction": cross[
            "cross_biopsy_ancestor_pair_fraction"
        ],
        "cross_biopsy_incomparable_pair_fraction": cross[
            "cross_biopsy_incomparable_pair_fraction"
        ],
        "adjacent_sampled_ancestor_coverage_fraction": cross[
            "adjacent_sampled_ancestor_coverage_fraction"
        ],
        "any_earlier_sampled_ancestor_coverage_fraction": cross[
            "any_earlier_sampled_ancestor_coverage_fraction"
        ],
        "lineage_linked_occurrence_fraction": cross[
            "lineage_linked_occurrence_fraction"
        ],
        "mean_hidden_internal_nodes_to_nearest_sampled_ancestor": hidden_mean,
        "minimal_sampled_occurrence_count": cross[
            "minimal_sampled_occurrence_count"
        ],
        "minimal_sampled_occurrence_fraction": cross[
            "minimal_sampled_occurrence_fraction"
        ],
        "minimum_invented_edges_for_observed_only_arborescence": cross[
            "minimum_invented_edges_for_observed_only_arborescence"
        ],
        "minimum_invented_edge_fraction": cross[
            "minimum_invented_edge_fraction"
        ],
    }
    return {
        "sampling": list(sampling_rows),
        "selected_occurrence_count": selected_count,
        "selected_unique_state_count": unique_state_count,
        "repeated_state_occurrence_count": selected_count - unique_state_count,
        "projected_distance_matrix_cell_count": unique_state_count**2,
        "projected_bidirectional_ordered_pair_count": (
            unique_state_count * (unique_state_count - 1)
        ),
        "truth_sampling_diagnostics": general,
        "cross_biopsy_relation_diagnostics": cross,
        "scalar_metrics": scalar_metrics,
    }


def _prepare_case(
    *,
    config_path: Path,
    simulation_seed: int,
    base_seed: int,
    replicate_index: int,
    height: int,
    generations: Sequence[int],
    lower_bound: int,
) -> dict[str, Any]:
    simulator = CancerCellEvolutionSimulator(str(config_path), seed=simulation_seed)
    simulator.run_simulation()
    available_levels = [
        _canonical_cells_at_generation(simulator, generation)
        for generation in generations
    ]
    selected, sampling, nesting = select_nested_fraction_levels(
        available_levels,
        generations,
        base_seed=base_seed,
        replicate_index=replicate_index,
        height=height,
        lower_bound=lower_bound,
    )
    truth_tree = simulator.canonicalized_tree_by_genome()
    conditions = {
        condition_id: {
            "condition_id": condition_id,
            "target_fraction": biopsy_fraction,
            "summary": _condition_summary(
                truth_tree,
                selected[condition_id],
                sampling[condition_id],
            ),
        }
        for condition_id, biopsy_fraction in CONDITIONS
    }
    prefix_hashes = {
        str(prefix_height): _truth_prefix_sha256(simulator.tree, prefix_height)
        for prefix_height in APPROVED_SCHEDULES
        if prefix_height <= height
    }
    return {
        "truth_node_count": truth_tree.number_of_nodes(),
        "truth_edge_count": truth_tree.number_of_edges(),
        "available_distinct_state_count_by_generation": [
            {
                "generation": int(generation),
                "count": len(level),
            }
            for generation, level in zip(generations, available_levels)
        ],
        "truth_prefix_sha256_by_height": prefix_hashes,
        "conditions": conditions,
        "nesting": nesting,
        "all_nested_checks_passed": all(row["nested"] for row in nesting),
    }


def _error_fingerprint(error: BaseException | None) -> dict[str, Any] | None:
    if error is None:
        return None
    message = str(error)
    return {
        "type": type(error).__name__,
        "message_character_count": len(message),
        "message_sha256": hashlib.sha256(
            message.encode("utf-8", errors="replace")
        ).hexdigest(),
    }


def _reference_case_check(
    prepared: Mapping[str, Any],
    reference_case: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if reference_case is None:
        return {"performed": False, "passed": None, "mismatched_fields": []}
    mismatches = []
    reference_summary = reference_case["simulation_summary"]
    control = prepared["conditions"]["capped_six_control"]["summary"]
    comparisons = {
        "truth_prefix_sha256_by_height": (
            prepared["truth_prefix_sha256_by_height"],
            reference_summary["truth_prefix_sha256_by_height"],
        ),
        "available_distinct_state_counts": (
            [
                row["count"]
                for row in prepared[
                    "available_distinct_state_count_by_generation"
                ]
            ],
            [
                row["available_distinct_state_count"]
                for row in reference_summary["sampling"]
            ],
        ),
        "capped_control_realized_counts": (
            [row["realized_occurrence_count"] for row in control["sampling"]],
            [
                row["realized_occurrence_count"]
                for row in reference_summary["sampling"]
            ],
        ),
        "capped_control_selected_occurrence_count": (
            control["selected_occurrence_count"],
            reference_summary["selected_occurrence_count"],
        ),
        "capped_control_selected_unique_state_count": (
            control["selected_unique_state_count"],
            reference_summary["selected_unique_state_count"],
        ),
        "capped_control_truth_diagnostics_sha256": (
            canonical_json_sha256(control["truth_sampling_diagnostics"]),
            canonical_json_sha256(
                reference_summary["truth_sampling_diagnostics"]
            ),
        ),
    }
    for field, (observed, expected) in comparisons.items():
        if observed != expected:
            mismatches.append(field)
    return {
        "performed": True,
        "passed": not mismatches,
        "mismatched_fields": mismatches,
    }


def _run_case(
    *,
    base_config: Mapping[str, Any],
    height: int,
    replicate_index: int,
    base_seed: int,
    lower_bound: int,
    timeout_seconds: int,
    rss_limit_bytes: int,
    reference_case: Mapping[str, Any] | None,
) -> dict[str, Any]:
    generations = APPROVED_SCHEDULES[height]
    simulation_seed = _simulation_seed(base_seed, replicate_index)
    config = dict(base_config)
    config["NUMBER_OF_GENERATIONS"] = height
    with tempfile.TemporaryDirectory(prefix="ctbf-v5-fraction-truth-") as directory:
        config_path = Path(directory) / "case.json"
        write_json_atomic(config_path, config)
        prepared, runtime, error = measured_stage(
            lambda: _prepare_case(
                config_path=config_path,
                simulation_seed=simulation_seed,
                base_seed=base_seed,
                replicate_index=replicate_index,
                height=height,
                generations=generations,
                lower_bound=lower_bound,
            ),
            timeout_seconds=timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
        )

    record = {
        "case_key": f"v5-fraction-truth-H{height}-R{replicate_index + 1:03d}",
        "height": height,
        "replicate_index": replicate_index,
        "simulation_seed": simulation_seed,
        "generations": list(generations),
        "status": None,
        "error": _error_fingerprint(error),
        "runtime": runtime,
        "simulation_summary": prepared,
        "reference_check": None,
    }
    if error is not None:
        record["status"] = (
            "empty_required_biopsy"
            if isinstance(error, EmptyRequiredBiopsyError)
            else "analysis_pair_limit_exceeded"
            if isinstance(error, TruthPairAnalysisLimitExceeded)
            else "simulation_or_truth_analysis_failure"
        )
        return record

    reference_check = _reference_case_check(prepared, reference_case)
    record["reference_check"] = reference_check
    if reference_check["performed"] and not reference_check["passed"]:
        record["status"] = "reference_mismatch"
        return record
    if not prepared["all_nested_checks_passed"]:
        record["status"] = "nesting_invariant_failure"
        return record
    record["status"] = "complete"
    del prepared
    gc.collect()
    return record


SCALAR_METRICS = (
    "selected_occurrence_count",
    "selected_unique_state_count",
    "repeated_state_occurrence_count",
    "realized_overall_state_fraction",
    "projected_distance_matrix_cell_count",
    "projected_bidirectional_ordered_pair_count",
    "all_pair_incomparable_fraction",
    "all_pair_hidden_fork_fraction",
    "cross_biopsy_ancestor_pair_fraction",
    "cross_biopsy_incomparable_pair_fraction",
    "adjacent_sampled_ancestor_coverage_fraction",
    "any_earlier_sampled_ancestor_coverage_fraction",
    "lineage_linked_occurrence_fraction",
    "mean_hidden_internal_nodes_to_nearest_sampled_ancestor",
    "minimal_sampled_occurrence_count",
    "minimal_sampled_occurrence_fraction",
    "minimum_invented_edges_for_observed_only_arborescence",
    "minimum_invented_edge_fraction",
)


def _metric_value(
    case: Mapping[str, Any],
    condition_id: str,
    metric: str,
) -> float | None:
    value = case["simulation_summary"]["conditions"][condition_id]["summary"][
        "scalar_metrics"
    ][metric]
    return None if value is None else float(value)


def _condition_aggregate(
    records: Sequence[Mapping[str, Any]],
    condition_id: str,
) -> dict[str, Any]:
    summaries = [
        record["simulation_summary"]["conditions"][condition_id]["summary"]
        for record in records
    ]
    target_fraction = dict(CONDITIONS)[condition_id]
    scalar = {}
    for metric in SCALAR_METRICS:
        values = [summary["scalar_metrics"][metric] for summary in summaries]
        scalar[metric] = _number_summary(
            [float(value) for value in values if value is not None]
        )
    biopsy_levels = []
    for biopsy_level, generation in enumerate(records[0]["generations"]):
        rows = [summary["sampling"][biopsy_level] for summary in summaries]
        biopsy_levels.append(
            {
                "biopsy_level": biopsy_level,
                "generation": generation,
                "available_distinct_state_count": _number_summary(
                    [float(row["available_distinct_state_count"]) for row in rows]
                ),
                "realized_occurrence_count": _number_summary(
                    [float(row["realized_occurrence_count"]) for row in rows]
                ),
                "realized_fraction": _number_summary(
                    [float(row["realized_fraction"]) for row in rows]
                ),
                "sample_size_driver_counts": dict(
                    sorted(Counter(row["sample_size_driver"] for row in rows).items())
                ),
                "case_count_with_addition_over_capped_control": sum(
                    row["additional_over_capped_control_count"] > 0 for row in rows
                ),
            }
        )
    representable_count = sum(
        summary["cross_biopsy_relation_diagnostics"][
            "observed_only_occurrence_arborescence_representable"
        ]
        for summary in summaries
    )
    return {
        "condition_id": condition_id,
        "target_fraction": target_fraction,
        "case_count": len(summaries),
        "scalar_metrics": scalar,
        "biopsy_levels": biopsy_levels,
        "observed_only_occurrence_arborescence_representable_count": (
            representable_count
        ),
        "observed_only_occurrence_arborescence_representable_fraction": (
            representable_count / len(summaries)
        ),
    }


def _paired_difference(
    records: Sequence[Mapping[str, Any]],
    left_condition_id: str,
    right_condition_id: str,
    metric: str,
) -> dict[str, Any]:
    differences = []
    for record in records:
        left = _metric_value(record, left_condition_id, metric)
        right = _metric_value(record, right_condition_id, metric)
        if left is not None and right is not None:
            differences.append(left - right)
    tolerance = 1e-12
    return {
        "left_condition_id": left_condition_id,
        "right_condition_id": right_condition_id,
        "metric": metric,
        "difference": _number_summary(differences),
        "wins_ties_losses": {
            "wins": sum(value > tolerance for value in differences),
            "ties": sum(abs(value) <= tolerance for value in differences),
            "losses": sum(value < -tolerance for value in differences),
        },
    }


def aggregate_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_height = {}
    complete_cases = [case for case in cases if case["status"] == "complete"]
    for height in APPROVED_SCHEDULES:
        records = [case for case in complete_cases if case["height"] == height]
        conditions = (
            {
                condition_id: _condition_aggregate(records, condition_id)
                for condition_id in CONDITION_IDS
            }
            if records
            else {}
        )
        by_height[str(height)] = {
            "attempted_case_count": sum(case["height"] == height for case in cases),
            "complete_case_count": len(records),
            "status_counts": dict(
                sorted(
                    Counter(
                        case["status"] for case in cases if case["height"] == height
                    ).items()
                )
            ),
            "conditions": conditions,
            "paired_vs_capped_control": {
                condition_id: {
                    metric: _paired_difference(
                        records,
                        condition_id,
                        "capped_six_control",
                        metric,
                    )
                    for metric in SCALAR_METRICS
                }
                for condition_id in CONDITION_IDS[1:]
            },
            "successive_condition_differences": {
                f"{left}_minus_{right}": {
                    metric: _paired_difference(records, left, right, metric)
                    for metric in SCALAR_METRICS
                }
                for right, left in zip(CONDITION_IDS, CONDITION_IDS[1:])
            },
        }
    return {
        "by_height": by_height,
        "common_seed_prefix_consistency": _prefix_consistency(cases),
    }


def _validate_positive_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")


def _load_reference(
    path: Path | None,
    *,
    base_config_sha256: str,
    base_seed: int,
    replicates: int,
) -> tuple[dict[tuple[int, int], Mapping[str, Any]], dict[str, Any]]:
    if path is None:
        return {}, {"provided": False, "path": None, "sha256": None}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Reference report is not a file: {resolved}.")
    report = read_json(resolved)
    validate_reference_report(report)
    if report["schema_version"] != REFERENCE_SCHEMA_VERSION:
        raise ValueError("Reference report has the wrong schema.")
    expected = {
        "status": "complete",
        "base_config_sha256": base_config_sha256,
        "base_seed": base_seed,
        "heights": list(APPROVED_SCHEDULES),
        "schedule_by_height": {
            str(height): list(schedule)
            for height, schedule in APPROVED_SCHEDULES.items()
        },
        "maximum_distinct_states_per_biopsy": DEFAULT_LOWER_BOUND,
    }
    observed = {
        "status": report.get("status"),
        "base_config_sha256": report.get("input", {}).get("base_config_sha256"),
        "base_seed": report.get("input", {}).get("base_seed"),
        "heights": report.get("input", {}).get("heights"),
        "schedule_by_height": report.get("input", {}).get("schedule_by_height"),
        "maximum_distinct_states_per_biopsy": report.get("input", {}).get(
            "maximum_distinct_states_per_biopsy"
        ),
    }
    if observed != expected:
        raise ValueError("Reference report does not match the fraction-probe design.")
    reference_replicates = report.get("input", {}).get("replicates")
    if (
        isinstance(reference_replicates, bool)
        or not isinstance(reference_replicates, int)
        or reference_replicates < replicates
    ):
        raise ValueError("Reference report has too few replicate blocks.")
    if report["scientific_role"].get("injected_distance_for_test"):
        raise ValueError("Injected-distance output cannot serve as the reference.")
    index = {
        (int(case["replicate_index"]), int(case["height"])): case
        for case in report["cases"]
        if int(case["replicate_index"]) < replicates
    }
    if len(index) != replicates * len(APPROVED_SCHEDULES):
        raise ValueError("Reference report does not contain every planned case.")
    return index, {
        "provided": True,
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
        "schema_version": report["schema_version"],
        "available_replicates": reference_replicates,
        "used_replicates": replicates,
    }


def run_probe(
    *,
    base_config_path: Path | str = DEFAULT_BASE_CONFIG,
    reference_report_path: Path | str | None = None,
    replicates: int = DEFAULT_REPLICATES,
    base_seed: int = DEFAULT_BASE_SEED,
    timeout_seconds: int = DEFAULT_SIMULATION_ANALYSIS_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    max_case_failures: int = DEFAULT_MAX_CASE_FAILURES,
    created_at_utc: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    base_config_path = Path(base_config_path).expanduser().resolve()
    if not base_config_path.is_file():
        raise ValueError(f"Base config is not a file: {base_config_path}.")
    base_config = read_json(base_config_path)
    _validate_standard_base_config(base_config)
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise ValueError("replicates must be an integer.")
    if not 1 <= replicates <= MAX_REPLICATES:
        raise ValueError(f"replicates must be in [1,{MAX_REPLICATES}].")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer.")
    lower_bound = DEFAULT_LOWER_BOUND
    for field, value in (
        ("timeout_seconds", timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
        ("max_case_failures", max_case_failures),
    ):
        _validate_positive_integer(value, field)

    reference_index, reference_metadata = _load_reference(
        None if reference_report_path is None else Path(reference_report_path),
        base_config_sha256=_file_sha256(base_config_path),
        base_seed=base_seed,
        replicates=replicates,
    )

    cases = []
    failure_count = 0
    stopped_early = False
    for replicate_index in range(replicates):
        for height in APPROVED_SCHEDULES:
            record = _run_case(
                base_config=base_config,
                height=height,
                replicate_index=replicate_index,
                base_seed=base_seed,
                lower_bound=lower_bound,
                timeout_seconds=timeout_seconds,
                rss_limit_bytes=rss_limit_bytes,
                reference_case=reference_index.get((replicate_index, height)),
            )
            cases.append(record)
            if progress:
                print(
                    json.dumps(
                        {
                            "case_key": record["case_key"],
                            "status": record["status"],
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            if record["status"] != "complete":
                failure_count += 1
                if failure_count >= max_case_failures:
                    stopped_early = True
                    break
        if stopped_early:
            break

    aggregate = aggregate_cases(cases)
    prefix = aggregate["common_seed_prefix_consistency"]
    if not prefix["all_evaluable_common_seed_prefix_checks_passed"]:
        raise ValueError("Common-seed height prefixes are inconsistent.")
    reference_checks = [
        case["reference_check"]
        for case in cases
        if case.get("reference_check", {}).get("performed")
    ]
    reference_validation_passed = (
        all(check["passed"] for check in reference_checks)
        if reference_metadata["provided"]
        else None
    )
    if reference_metadata["provided"] and not reference_validation_passed:
        status = "failed_reference_validation"
    elif stopped_early:
        status = "stopped_at_failure_cap"
    elif all(case["status"] == "complete" for case in cases):
        status = "complete"
    else:
        status = "complete_with_typed_failures"

    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scientific_role": {
            "paper_evidence_allowed": False,
            "discovery_only": True,
            "simulation_run": True,
            "truth_diagnostics_run": True,
            "cnp2cnp_run": False,
            "reconstruction_run": False,
            "evaluation_run": False,
            "selects_sampling_conditions_for_later_probe": True,
            "selects_simulator_parameters_from_accuracy": False,
            "freezes_paper_sampling_design": False,
        },
        "question": {
            "primary": (
                "Does nested percentage sampling improve observed genealogical "
                "linkage enough to justify its projected quadratic distance cost?"
            ),
            "sampling_unit": "distinct_representative_genotype_state",
            "not_a_physical_cell_or_abundance_fraction": True,
            "not_a_reconstruction_accuracy_test": True,
        },
        "input": {
            "base_config": (
                base_config_path.relative_to(PROJECT_ROOT).as_posix()
                if base_config_path.is_relative_to(PROJECT_ROOT)
                else str(base_config_path)
            ),
            "base_config_sha256": _file_sha256(base_config_path),
            "heights": list(APPROVED_SCHEDULES),
            "schedule_by_height": {
                str(height): list(schedule)
                for height, schedule in APPROVED_SCHEDULES.items()
            },
            "sampling_version": SAMPLING_VERSION,
            "sample_size_formula": "min(N,max(K,ceil(p*N)))",
            "biopsy_lower_bound": lower_bound,
            "conditions": [
                {"condition_id": condition_id, "target_fraction": fraction}
                for condition_id, fraction in CONDITIONS
            ],
            "rounding": "ceiling",
            "nested_prefix_sampling": True,
            "replicates": replicates,
            "base_seed": base_seed,
            "reused_seed_namespace": REUSED_SEED_NAMESPACE,
            "same_truth_and_capped_control_as_reconstruction_probe": True,
        },
        "reference": {
            **reference_metadata,
            "performed_case_count": len(reference_checks),
            "all_performed_checks_passed": reference_validation_passed,
        },
        "resource_bound": {
            "planned_truth_case_count": replicates * len(APPROVED_SCHEDULES),
            "attempted_truth_case_count": len(cases),
            "condition_count_per_complete_truth": len(CONDITIONS),
            "distance_process_count": 0,
            "reconstruction_run_count": 0,
            "maximum_unordered_truth_pairs_per_condition": (
                MAX_UNORDERED_TRUTH_PAIRS_PER_CONDITION
            ),
            "timeout_seconds_per_simulation_and_truth_analysis": timeout_seconds,
            "rss_limit_bytes": rss_limit_bytes,
            "max_case_failures": max_case_failures,
        },
        "cases": cases,
        "aggregate": aggregate,
        "provenance": {
            "source_module": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
            "source_module_sha256": _file_sha256(Path(__file__)),
            "writes_raw_profiles": False,
            "writes_truth_trees": False,
            "writes_distance_matrices": False,
            "writes_simulator_node_identities": False,
            "replaces_failures_or_cases": False,
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
        raise ValueError("Unknown fraction-truth-probe schema.")
    if report.get("analysis_role") != ANALYSIS_ROLE:
        raise ValueError("Fraction-truth report has the wrong analysis role.")
    if report.get("status") not in {
        "complete",
        "complete_with_typed_failures",
        "stopped_at_failure_cap",
        "failed_reference_validation",
    }:
        raise ValueError("Fraction-truth report has an unknown status.")
    role = report.get("scientific_role", {})
    expected_role = {
        "paper_evidence_allowed": False,
        "discovery_only": True,
        "simulation_run": True,
        "truth_diagnostics_run": True,
        "cnp2cnp_run": False,
        "reconstruction_run": False,
        "evaluation_run": False,
        "selects_simulator_parameters_from_accuracy": False,
        "freezes_paper_sampling_design": False,
    }
    for field, expected in expected_role.items():
        if role.get(field) is not expected:
            raise ValueError(f"Scientific role has invalid {field}.")
    allowed_case_statuses = {
        "complete",
        "empty_required_biopsy",
        "analysis_pair_limit_exceeded",
        "simulation_or_truth_analysis_failure",
        "reference_mismatch",
        "nesting_invariant_failure",
    }
    for case in report.get("cases", []):
        if case.get("status") not in allowed_case_statuses:
            raise ValueError("Fraction-truth case has an unknown status.")
        if case["status"] != "complete":
            continue
        summary = case.get("simulation_summary")
        if summary is None or not summary.get("all_nested_checks_passed"):
            raise ValueError("A complete case lacks passing nested selections.")
        if set(summary["conditions"]) != set(CONDITION_IDS):
            raise ValueError("A complete case has the wrong condition order.")
        for condition_id, fraction in CONDITIONS:
            condition = summary["conditions"][condition_id]
            if condition["target_fraction"] != fraction:
                raise ValueError("A condition has the wrong target fraction.")
            for row in condition["summary"]["sampling"]:
                expected = hybrid_sample_size(
                    row["available_distinct_state_count"],
                    fraction,
                    row["configured_lower_bound"],
                )
                if row["realized_occurrence_count"] != expected:
                    raise ValueError("A hybrid sample size is inconsistent.")
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
            "Compact fraction-truth report contains forbidden raw fields: "
            + ", ".join(sorted(present))
        )
    json.dumps(report, sort_keys=True, allow_nan=False)


COMPACT_METRICS = (
    "selected_occurrence_count",
    "selected_unique_state_count",
    "realized_overall_state_fraction",
    "projected_bidirectional_ordered_pair_count",
    "cross_biopsy_ancestor_pair_fraction",
    "adjacent_sampled_ancestor_coverage_fraction",
    "any_earlier_sampled_ancestor_coverage_fraction",
    "mean_hidden_internal_nodes_to_nearest_sampled_ancestor",
    "minimal_sampled_occurrence_fraction",
    "minimum_invented_edge_fraction",
)


def compact_summary(report: Mapping[str, Any], output: Path) -> dict[str, Any]:
    by_height = {}
    for height, block in report["aggregate"]["by_height"].items():
        by_height[height] = {
            "status_counts": block["status_counts"],
            "conditions": {
                condition_id: {
                    "target_fraction": condition["target_fraction"],
                    "metric_means": {
                        metric: condition["scalar_metrics"][metric]["mean"]
                        for metric in COMPACT_METRICS
                    },
                    "metric_maxima": {
                        metric: condition["scalar_metrics"][metric]["maximum"]
                        for metric in (
                            "selected_occurrence_count",
                            "selected_unique_state_count",
                            "projected_bidirectional_ordered_pair_count",
                        )
                    },
                    "representable_fraction": condition[
                        "observed_only_occurrence_arborescence_representable_fraction"
                    ],
                    "biopsy_levels": condition["biopsy_levels"],
                }
                for condition_id, condition in block["conditions"].items()
            },
            "paired_vs_capped_control": {
                condition_id: {
                    metric: comparison["difference"]["mean"]
                    for metric, comparison in comparisons.items()
                    if metric in COMPACT_METRICS
                }
                for condition_id, comparisons in block[
                    "paired_vs_capped_control"
                ].items()
            },
        }
    return {
        "schema_version": report["schema_version"],
        "analysis_role": report["analysis_role"],
        "status": report["status"],
        "output": str(output.resolve()),
        "reference": report["reference"],
        "common_seed_prefix_consistency": report["aggregate"][
            "common_seed_prefix_consistency"
        ],
        "by_height": by_height,
        "next_stage": "owner_and_agent_review_before_any_reconstruction_fraction",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CTBF v5 truth-only nested 0/5/10/25/50-percent "
            "representative-state sampling diagnostic."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_SIMULATION_ANALYSIS_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument(
        "--max-case-failures",
        type=int,
        default=DEFAULT_MAX_CASE_FAILURES,
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
        reference_report_path=arguments.reference_report,
        replicates=arguments.replicates,
        base_seed=arguments.base_seed,
        timeout_seconds=arguments.timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        max_case_failures=arguments.max_case_failures,
        progress=arguments.progress,
    )
    write_json_atomic(output, report)
    print(json.dumps(compact_summary(report, output), indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
