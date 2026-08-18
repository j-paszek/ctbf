"""Compact mechanism audit for the immutable CTBF v5 development bank.

The audit explains three results from the initial algorithm screen without
regenerating truth, observations, or distances:

* aggregate the stored radius-2/4/8 biopsy candidate-graph diagnostics;
* rerun the plausible-parsimony incumbent with behavior-preserving orientation
  tracing; and
* rerun the exact temporal/no-time pair and summarize when the time constraint
  changes its occurrence tree and unique-label ancestry score.

No reconstructed tree, CNP profile, distance matrix, or truth tree is written
to the audit output.  The full-bank run is development evidence only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import networkx as nx
import numpy as np

from algorithm_evaluation.paper_pipeline_runner import measured_stage
from algorithm_evaluation.process_isolation import (
    FreshSpawnPerTaskExecutor,
    fresh_process_contract,
)
from algorithm_evaluation.temporal_arborescence_diagnostic import (
    ad_f1_from_pair_sets,
    ancestor_pair_set,
    occurrence_identity_signature,
    occurrence_tree_signature,
    tree_scientific_diagnostics,
)
from algorithm_evaluation.v5_algorithm_development_common import (
    ARM_SPEC_BY_ID,
    DEFAULT_BLOCK_COUNT,
    INFERRED_COPY_INCUMBENT_ID,
    ensure_new_output_root,
    load_bank_manifest,
    numeric_summary,
    observed_labels,
    read_case_assets,
    reconstruct_development_arm,
    write_json,
)
from reconstructor_ancestor_selection import (
    make_plausible_parsimony_parent_selector,
)
from reconstructor_anticentral import configure_anticentral_v3_state
from reconstructor_distance_update import (
    ANTICENTRAL_V3_CONTEXT_KEY,
    anticentral_v3_distance_update,
)
from reconstructor_engine import run_agglomerative_reconstruction
from reconstructor_merge import anticentral_weighted_copy_parent_node
from reconstructor_pair_selection import (
    make_anticentral_adaptive_v3_pair_selector,
)
from reconstructor_plausibility import is_biologically_plausible_ancestor


MECHANISM_AUDIT_SCHEMA_VERSION = (
    "ctbf-v5-algorithm-development-mechanism-audit-v3"
)
RESULT_NAME = "mechanism_audit.json"
REPORT_NAME = "report.md"
DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS = 300
DEFAULT_RSS_LIMIT_BYTES = 4 * 1024**3
AD_F1_TIE_TOLERANCE = 1e-12
MECHANISM_WORKER_UNIT = "mechanism_audit_case_arm_reconstruction"
MECHANISM_WORKER_GRACE_SECONDS = 60

TEMPORAL_ARM_IDS = ("temporal_minimum", "temporal_minimum_no_time")
ORIENTATION_COUNT_FIELDS = (
    "merge_count",
    "one_way_plausible_count",
    "both_plausible_count",
    "both_plausible_parsimony_decided_count",
    "both_plausible_parsimony_tied_count",
    "neither_plausible_count",
    "neither_plausible_parsimony_would_decide_count",
    "neither_plausible_parsimony_tied_count",
    "neither_plausible_parsimony_would_change_orientation_count",
    "centrality_fallback_count",
    "centrality_fallback_exact_tie_count",
)


def _validate_positive_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")


def _fraction(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _typed_error(error: BaseException, stage: str, case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error)[:4096],
    }


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


def _resource_scalar(resources: Mapping[str, Any]) -> dict[str, Any]:
    memory = resources.get("memory")
    peak = memory.get("peak_rss_bytes") if isinstance(memory, Mapping) else None
    wall_time_ns = resources.get("wall_time_ns")
    return {
        "runtime_seconds": (
            float(wall_time_ns) / 1_000_000_000
            if wall_time_ns is not None
            else None
        ),
        "peak_rss_bytes": int(peak) if peak is not None else None,
        "memory_method": memory.get("method") if isinstance(memory, Mapping) else None,
    }


def _deviation_from_baseline(node: Any, baseline_cn: int) -> float:
    genome = np.asarray(node.genome, dtype=float)
    return float(np.sum(np.abs(genome - baseline_cn)))


def _empty_orientation_trace() -> dict[str, int]:
    return {field: 0 for field in ORIENTATION_COUNT_FIELDS}


def _validate_orientation_trace(trace: Mapping[str, int]) -> None:
    missing = set(ORIENTATION_COUNT_FIELDS) - set(trace)
    if missing:
        raise ValueError(f"Orientation trace is missing fields: {sorted(missing)}.")
    if any(
        isinstance(trace[field], bool)
        or not isinstance(trace[field], int)
        or trace[field] < 0
        for field in ORIENTATION_COUNT_FIELDS
    ):
        raise ValueError("Orientation trace counts must be nonnegative integers.")
    if trace["merge_count"] != (
        trace["one_way_plausible_count"]
        + trace["both_plausible_count"]
        + trace["neither_plausible_count"]
    ):
        raise ValueError("Orientation plausibility categories do not cover all merges.")
    if trace["both_plausible_count"] != (
        trace["both_plausible_parsimony_decided_count"]
        + trace["both_plausible_parsimony_tied_count"]
    ):
        raise ValueError("Both-plausible parsimony categories are inconsistent.")
    if trace["neither_plausible_count"] != (
        trace["neither_plausible_parsimony_would_decide_count"]
        + trace["neither_plausible_parsimony_tied_count"]
    ):
        raise ValueError("Neither-plausible parsimony categories are inconsistent.")
    if trace["centrality_fallback_count"] != (
        trace["both_plausible_parsimony_tied_count"]
        + trace["neither_plausible_count"]
    ):
        raise ValueError("Centrality-fallback count is inconsistent.")
    if (
        trace["neither_plausible_parsimony_would_change_orientation_count"]
        > trace["neither_plausible_parsimony_would_decide_count"]
    ):
        raise ValueError("A proposed orientation change was counted without a decision.")
    if trace["centrality_fallback_exact_tie_count"] > trace["centrality_fallback_count"]:
        raise ValueError("Centrality exact ties exceed centrality fallbacks.")


def make_traced_plausible_parsimony_incumbent(
    trace: MutableMapping[str, int],
):
    """Return the exact incumbent with an external, non-scientific trace."""

    trace.clear()
    trace.update(_empty_orientation_trace())

    def algorithm(
        dist_matrix,
        cells,
        max_id,
        seed=None,
        existing_tree=None,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        baseline_cn: int = 2,
    ):
        base_parent_selector = make_plausible_parsimony_parent_selector(
            baseline_cn
        )

        def traced_parent_selector(state, pair):
            i = pair.i
            j = pair.j
            a = state.node_list[i]
            b = state.node_list[j]
            can_a_parent_b = is_biologically_plausible_ancestor(a, b)
            can_b_parent_a = is_biologically_plausible_ancestor(b, a)
            dev_a = _deviation_from_baseline(a, baseline_cn)
            dev_b = _deviation_from_baseline(b, baseline_cn)

            trace["merge_count"] += 1
            if can_a_parent_b != can_b_parent_a:
                trace["one_way_plausible_count"] += 1
            elif can_a_parent_b:
                trace["both_plausible_count"] += 1
                if dev_a == dev_b:
                    trace["both_plausible_parsimony_tied_count"] += 1
                    trace["centrality_fallback_count"] += 1
                else:
                    trace["both_plausible_parsimony_decided_count"] += 1
            else:
                trace["neither_plausible_count"] += 1
                trace["centrality_fallback_count"] += 1
                if dev_a == dev_b:
                    trace["neither_plausible_parsimony_tied_count"] += 1
                else:
                    trace[
                        "neither_plausible_parsimony_would_decide_count"
                    ] += 1

            if can_a_parent_b == can_b_parent_a:
                centrality = state.context[ANTICENTRAL_V3_CONTEXT_KEY]
                if centrality[i] == centrality[j] and (
                    not can_a_parent_b or dev_a == dev_b
                ):
                    trace["centrality_fallback_exact_tie_count"] += 1

            orientation = base_parent_selector(state, pair)
            if not can_a_parent_b and not can_b_parent_a and dev_a != dev_b:
                proposed_parent = i if dev_a < dev_b else j
                if orientation.parent_idx != proposed_parent:
                    trace[
                        "neither_plausible_parsimony_would_change_orientation_count"
                    ] += 1
            return orientation

        return run_agglomerative_reconstruction(
            dist_matrix,
            cells,
            max_id,
            seed=seed,
            existing_tree=existing_tree,
            pair_selector=make_anticentral_adaptive_v3_pair_selector(
                alpha,
                beta,
                gamma,
            ),
            ancestor_selector=traced_parent_selector,
            merge_strategy=anticentral_weighted_copy_parent_node,
            distance_update=anticentral_v3_distance_update,
            configure_state=configure_anticentral_v3_state,
        )

    algorithm.__name__ = f"{INFERRED_COPY_INCUMBENT_ID}_trace_only"
    return algorithm


def reconstruct_incumbent_with_orientation_trace(
    reconstruction_input: Mapping[str, Any],
    distance,
    *,
    reconstruction_seed: int,
) -> tuple[nx.DiGraph, dict[str, int]]:
    trace: dict[str, int] = {}
    algorithm = make_traced_plausible_parsimony_incumbent(trace)
    tree, _levels, _root, _metadata = reconstruct_development_arm(
        ARM_SPEC_BY_ID[INFERRED_COPY_INCUMBENT_ID],
        reconstruction_input,
        distance,
        reconstruction_seed=reconstruction_seed,
        algorithm_override=algorithm,
    )
    _validate_orientation_trace(trace)
    return tree, dict(trace)


def _degree_summary_from_records(
    records: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    histogram: Counter[int] = Counter()
    for record in records:
        summary = record[field]
        local = Counter(
            {
                int(value): int(count)
                for value, count in summary.get("histogram", {}).items()
            }
        )
        if sum(local.values()) != int(summary["count"]):
            raise ValueError(f"Stored {field} histogram has the wrong count.")
        histogram.update(local)
    count = sum(histogram.values())
    weighted_sum = sum(value * frequency for value, frequency in histogram.items())
    return {
        "count": count,
        "minimum": min(histogram) if histogram else None,
        "mean": float(weighted_sum / count) if count else None,
        "maximum": max(histogram) if histogram else None,
        "zero_count": histogram.get(0, 0),
        "zero_fraction": _fraction(histogram.get(0, 0), count),
        "one_count": histogram.get(1, 0),
        "one_fraction": _fraction(histogram.get(1, 0), count),
        "multiple_count": sum(
            frequency for value, frequency in histogram.items() if value > 1
        ),
        "multiple_fraction": _fraction(
            sum(frequency for value, frequency in histogram.items() if value > 1),
            count,
        ),
        "histogram": {
            str(value): histogram[value] for value in sorted(histogram)
        },
    }


def _summarize_candidate_layers(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("Candidate-layer summary requires at least one record.")
    child_count = sum(int(record["child_count"]) for record in records)
    missing_parent_count = sum(int(record["missing_parent_count"]) for record in records)
    same_state_count = sum(
        int(record["same_state_priority_count"]) for record in records
    )
    four_cycle_values = [
        float(record["plausible_radius_four_cycle_count"]) for record in records
    ]
    return {
        "transition_record_count": len(records),
        "summed_parent_level_size": sum(
            int(record["parent_count"]) for record in records
        ),
        "summed_child_level_size": child_count,
        "raw_radius_child_degree": _degree_summary_from_records(
            records,
            "raw_radius_child_degree",
        ),
        "plausible_radius_child_degree": _degree_summary_from_records(
            records,
            "plausible_radius_child_degree",
        ),
        "minimum_parent_child_degree": _degree_summary_from_records(
            records,
            "minimum_parent_child_degree",
        ),
        "same_state_priority_count": same_state_count,
        "same_state_priority_fraction": _fraction(same_state_count, child_count),
        "missing_parent_count": missing_parent_count,
        "missing_parent_fraction": _fraction(missing_parent_count, child_count),
        "plausible_radius_four_cycle_count": {
            "total": int(sum(four_cycle_values)),
            "per_transition": numeric_summary(four_cycle_values),
        },
    }


def summarize_radius_candidate_graph(
    case_metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    flattened: list[dict[str, Any]] = []
    for metadata in case_metadata:
        value = metadata.get("candidate_graph_diagnostics_r2_r4_r8")
        if not isinstance(value, Mapping):
            raise ValueError("Case metadata has no candidate-graph diagnostics.")
        transitions = value.get("transitions")
        if not isinstance(transitions, list) or len(transitions) != int(
            value.get("transition_count", -1)
        ):
            raise ValueError("Stored candidate-graph transition inventory changed.")
        for transition in transitions:
            radii = transition.get("radii")
            if not isinstance(radii, list):
                raise ValueError("Stored candidate-graph radii are invalid.")
            observed_radii = {float(record["radius"]) for record in radii}
            if observed_radii != {2.0, 4.0, 8.0}:
                raise ValueError("Stored candidate-graph radii changed.")
            for radius_record in radii:
                flattened.append(
                    {
                        "case_id": metadata["case_id"],
                        "height": int(metadata["height"]),
                        "parent_level": int(transition["parent_level"]),
                        "child_level": int(transition["child_level"]),
                        **radius_record,
                    }
                )

    heights = sorted({int(record["height"]) for record in flattened})
    radii = (2.0, 4.0, 8.0)
    by_height = {
        str(height): {
            str(int(radius)): _summarize_candidate_layers(
                [
                    record
                    for record in flattened
                    if record["height"] == height and record["radius"] == radius
                ]
            )
            for radius in radii
        }
        for height in heights
    }
    by_transition = []
    transition_keys = sorted(
        {
            (
                int(record["height"]),
                int(record["parent_level"]),
                int(record["child_level"]),
            )
            for record in flattened
        }
    )
    for height, parent_level, child_level in transition_keys:
        by_transition.append(
            {
                "height": height,
                "parent_level": parent_level,
                "child_level": child_level,
                "radii": {
                    str(int(radius)): _summarize_candidate_layers(
                        [
                            record
                            for record in flattened
                            if record["height"] == height
                            and record["parent_level"] == parent_level
                            and record["child_level"] == child_level
                            and record["radius"] == radius
                        ]
                    )
                    for radius in radii
                },
            }
        )
    return {
        "case_count": len(case_metadata),
        "stored_layer_radius_record_count": len(flattened),
        "combined": {
            str(int(radius)): _summarize_candidate_layers(
                [record for record in flattened if record["radius"] == radius]
            )
            for radius in radii
        },
        "by_height": by_height,
        "by_transition": by_transition,
    }


def _pooled_mean_range_from_summaries(
    summaries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    values = [summary for summary in summaries if int(summary["count"]) > 0]
    count = sum(int(summary["count"]) for summary in values)
    if not count:
        return {"count": 0, "minimum": None, "mean": None, "maximum": None}
    return {
        "count": count,
        "minimum": min(float(summary["minimum"]) for summary in values),
        "mean": float(
            sum(float(summary["mean"]) * int(summary["count"]) for summary in values)
            / count
        ),
        "maximum": max(float(summary["maximum"]) for summary in values),
    }


def _summarize_truth_sampling_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = sum(int(row["selected_occurrence_count"]) for row in rows)
    pair_count = sum(int(row["selected_unordered_pair_count"]) for row in rows)
    comparable = sum(
        int(row["comparable_ancestor_descendant_pair_count"]) for row in rows
    )
    later = sum(int(row["later_observation_occurrence_count"]) for row in rows)
    with_ancestor = sum(int(row["later_with_sampled_ancestor_count"]) for row in rows)
    direct_parent = sum(int(row["direct_truth_parent_sampled_count"]) for row in rows)
    representable = sum(
        bool(row["observed_only_occurrence_arborescence_representable"])
        for row in rows
    )
    return {
        "case_count": len(rows),
        "selected_occurrence_count": selected,
        "comparable_ancestor_descendant_pair_count": comparable,
        "comparable_ancestor_descendant_pair_fraction": _fraction(
            comparable,
            pair_count,
        ),
        "later_observation_occurrence_count": later,
        "later_with_sampled_ancestor_count": with_ancestor,
        "sampled_ancestor_coverage_fraction": _fraction(with_ancestor, later),
        "direct_truth_parent_sampled_count": direct_parent,
        "direct_truth_parent_sampled_fraction_of_later": _fraction(
            direct_parent,
            later,
        ),
        "nearest_sampled_ancestor_edge_distance": (
            _pooled_mean_range_from_summaries(
                row["nearest_sampled_ancestor_edge_distance"] for row in rows
            )
        ),
        "hidden_internal_nodes_to_nearest_sampled_ancestor": (
            _pooled_mean_range_from_summaries(
                row["hidden_internal_nodes_to_nearest_sampled_ancestor"]
                for row in rows
            )
        ),
        "minimal_sampled_occurrence_count": numeric_summary(
            float(row["minimal_sampled_occurrence_count"]) for row in rows
        ),
        "minimum_invented_edges_for_observed_only_arborescence": numeric_summary(
            float(row["minimum_invented_edges_for_observed_only_arborescence"])
            for row in rows
        ),
        "observed_only_occurrence_arborescence_representable_case_count": (
            representable
        ),
        "observed_only_occurrence_arborescence_representable_case_fraction": (
            _fraction(representable, len(rows))
        ),
    }


def summarize_truth_sampling(
    case_metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for metadata in case_metadata:
        diagnostics = metadata.get("truth_sampling_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("Case metadata has no truth-sampling diagnostics.")
        rows.append(
            {
                "height": int(metadata["height"]),
                **diagnostics,
            }
        )
    heights = sorted({int(row["height"]) for row in rows})
    return {
        "combined": _summarize_truth_sampling_rows(rows),
        "by_height": {
            str(height): _summarize_truth_sampling_rows(
                [row for row in rows if row["height"] == height]
            )
            for height in heights
        },
    }


def _summarize_orientation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    totals = {
        field: sum(int(row["trace"][field]) for row in rows)
        for field in ORIENTATION_COUNT_FIELDS
    }
    merge_count = totals["merge_count"]
    cases_with_neither = sum(row["trace"]["neither_plausible_count"] > 0 for row in rows)
    cases_with_change = sum(
        row["trace"][
            "neither_plausible_parsimony_would_change_orientation_count"
        ]
        > 0
        for row in rows
    )
    return {
        "case_count": len(rows),
        "counts": totals,
        "fractions_of_merges": {
            field.removesuffix("_count"): _fraction(totals[field], merge_count)
            for field in ORIENTATION_COUNT_FIELDS
            if field != "merge_count"
        },
        "cases_with_neither_plausible_merge_count": cases_with_neither,
        "cases_with_neither_plausible_merge_fraction": _fraction(
            cases_with_neither,
            len(rows),
        ),
        "cases_where_proposed_fallback_would_change_orientation_count": (
            cases_with_change
        ),
        "cases_where_proposed_fallback_would_change_orientation_fraction": (
            _fraction(cases_with_change, len(rows))
        ),
        "proposed_fallback_change_case_ids": sorted(
            row["case_id"]
            for row in rows
            if row["trace"][
                "neither_plausible_parsimony_would_change_orientation_count"
            ]
            > 0
        ),
    }


def summarize_orientation_traces(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    heights = sorted({int(row["height"]) for row in rows})
    return {
        "combined": _summarize_orientation_rows(rows),
        "by_height": {
            str(height): _summarize_orientation_rows(
                [row for row in rows if row["height"] == height]
            )
            for height in heights
        },
    }


def temporal_case_diagnostic(
    *,
    case_id: str,
    block_index: int,
    height: int,
    reconstruction_input: Mapping[str, Any],
    distance,
    truth_tree: nx.DiGraph,
    temporal_tree: nx.DiGraph,
    no_time_tree: nx.DiGraph,
) -> dict[str, Any]:
    if occurrence_identity_signature(temporal_tree) != occurrence_identity_signature(
        no_time_tree
    ):
        raise ValueError("Temporal and no-time outputs changed occurrence identity.")
    temporal_diagnostics = tree_scientific_diagnostics(temporal_tree, distance)
    no_time_diagnostics = tree_scientific_diagnostics(no_time_tree, distance)
    if temporal_diagnostics["constraint_active"]:
        raise ValueError("The time-constrained tree violates its temporal constraint.")

    labels = observed_labels(reconstruction_input)
    true_pairs = ancestor_pair_set(truth_tree, labels)
    temporal_pairs = ancestor_pair_set(temporal_tree, labels)
    no_time_pairs = ancestor_pair_set(no_time_tree, labels)
    temporal_ad = ad_f1_from_pair_sets(true_pairs, temporal_pairs)
    no_time_ad = ad_f1_from_pair_sets(true_pairs, no_time_pairs)

    def compact_tree_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            field: value[field]
            for field in (
                "root_biopsy_level",
                "earliest_biopsy_level",
                "late_root",
                "backward_edge_count",
                "same_level_edge_count",
                "forward_edge_count",
                "constraint_active",
                "plausibility_violation_edge_count",
                "total_edge_distance",
                "root_score",
            )
        }

    return {
        "case_id": case_id,
        "block_index": int(block_index),
        "height": int(height),
        "same_directed_occurrence_tree": (
            occurrence_tree_signature(temporal_tree)
            == occurrence_tree_signature(no_time_tree)
        ),
        "same_unique_observed_label_ancestor_pair_set": (
            temporal_pairs == no_time_pairs
        ),
        "temporal": {
            "ad": temporal_ad,
            "tree": compact_tree_diagnostics(temporal_diagnostics),
        },
        "no_time": {
            "ad": no_time_ad,
            "tree": compact_tree_diagnostics(no_time_diagnostics),
        },
        "temporal_minus_no_time": {
            "ad_f1": float(temporal_ad["ad_f1"] - no_time_ad["ad_f1"]),
            "ad_precision": float(
                temporal_ad["ad_precision"] - no_time_ad["ad_precision"]
            ),
            "ad_recall": float(
                temporal_ad["ad_recall"] - no_time_ad["ad_recall"]
            ),
            "plausibility_violation_edge_count": int(
                temporal_diagnostics["plausibility_violation_edge_count"]
                - no_time_diagnostics["plausibility_violation_edge_count"]
            ),
            "total_edge_distance": float(
                temporal_diagnostics["total_edge_distance"]
                - no_time_diagnostics["total_edge_distance"]
            ),
            "root_score": float(
                temporal_diagnostics["root_score"]
                - no_time_diagnostics["root_score"]
            ),
        },
    }


def _summarize_temporal_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    case_count = len(rows)
    constraint_active = sum(row["no_time"]["tree"]["constraint_active"] for row in rows)
    backward = sum(row["no_time"]["tree"]["backward_edge_count"] > 0 for row in rows)
    late_root = sum(row["no_time"]["tree"]["late_root"] for row in rows)
    same_tree = sum(row["same_directed_occurrence_tree"] for row in rows)
    same_pairs = sum(
        row["same_unique_observed_label_ancestor_pair_set"] for row in rows
    )
    deltas = [float(row["temporal_minus_no_time"]["ad_f1"]) for row in rows]
    return {
        "case_count": case_count,
        "no_time_constraint_active_count": constraint_active,
        "no_time_constraint_active_fraction": _fraction(constraint_active, case_count),
        "no_time_backward_edge_present_count": backward,
        "no_time_backward_edge_present_fraction": _fraction(backward, case_count),
        "no_time_late_root_count": late_root,
        "no_time_late_root_fraction": _fraction(late_root, case_count),
        "no_time_backward_edge_count": numeric_summary(
            float(row["no_time"]["tree"]["backward_edge_count"]) for row in rows
        ),
        "same_directed_occurrence_tree_count": same_tree,
        "same_directed_occurrence_tree_fraction": _fraction(same_tree, case_count),
        "same_unique_observed_label_ancestor_pair_set_count": same_pairs,
        "same_unique_observed_label_ancestor_pair_set_fraction": _fraction(
            same_pairs,
            case_count,
        ),
        "temporal_ad_f1_better_count": sum(
            value > AD_F1_TIE_TOLERANCE for value in deltas
        ),
        "temporal_ad_f1_tied_count": sum(
            abs(value) <= AD_F1_TIE_TOLERANCE for value in deltas
        ),
        "temporal_ad_f1_worse_count": sum(
            value < -AD_F1_TIE_TOLERANCE for value in deltas
        ),
        "temporal_minus_no_time": {
            field: numeric_summary(
                float(row["temporal_minus_no_time"][field]) for row in rows
            )
            for field in (
                "ad_f1",
                "ad_precision",
                "ad_recall",
                "plausibility_violation_edge_count",
                "total_edge_distance",
                "root_score",
            )
        },
        "constraint_strata": {
            str(active).lower(): {
                "case_count": sum(
                    bool(row["no_time"]["tree"]["constraint_active"]) is active
                    for row in rows
                ),
                "temporal_minus_no_time_ad_f1": numeric_summary(
                    float(row["temporal_minus_no_time"]["ad_f1"])
                    for row in rows
                    if bool(row["no_time"]["tree"]["constraint_active"])
                    is active
                ),
            }
            for active in (False, True)
        },
        "largest_no_time_ad_f1_advantage_case_ids": [
            row["case_id"]
            for row in sorted(
                rows,
                key=lambda row: (
                    float(row["temporal_minus_no_time"]["ad_f1"]),
                    row["case_id"],
                ),
            )[:5]
        ],
    }


def summarize_temporal_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    heights = sorted({int(row["height"]) for row in rows})
    return {
        "combined": _summarize_temporal_rows(rows),
        "by_height": {
            str(height): _summarize_temporal_rows(
                [row for row in rows if row["height"] == height]
            )
            for height in heights
        },
    }


def _format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# CTBF v5 development-bank mechanism audit",
        "",
        "This is method-development evidence. It does not select an algorithm or "
        "provide confirmatory paper accuracy.",
        "",
        "## Radius candidate graph",
        "",
        "| Height | Radius | Missing-parent fraction | Plausible multi-parent "
        "fraction | Minimum-distance tie fraction | Mean plausible degree | "
        "Mean four-cycles/transition |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    radius = result["radius_candidate_graph"]
    for height, height_summary in radius["by_height"].items():
        for radius_value, summary in height_summary.items():
            lines.append(
                "| {height} | {radius} | {missing} | {plausible_multi} | "
                "{minimum_multi} | {plausible_mean} | {cycles} |".format(
                    height=height,
                    radius=radius_value,
                    missing=_format_number(summary["missing_parent_fraction"]),
                    plausible_multi=_format_number(
                        summary["plausible_radius_child_degree"]["multiple_fraction"]
                    ),
                    minimum_multi=_format_number(
                        summary["minimum_parent_child_degree"]["multiple_fraction"]
                    ),
                    plausible_mean=_format_number(
                        summary["plausible_radius_child_degree"]["mean"]
                    ),
                    cycles=_format_number(
                        summary["plausible_radius_four_cycle_count"][
                            "per_transition"
                        ]["mean"]
                    ),
                )
            )

    lines.extend(
        [
            "",
            "## Incumbent orientation branches",
            "",
            "The proposed fallback column counts merges where applying diploid-"
            "baseline parsimony to a neither-direction-plausible pair would change "
            "the incumbent's current centrality orientation.",
            "",
            "| Height | Cases | Merges | One-way plausible | Both plausible | "
            "Neither plausible | Proposed fallback changes |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    orientation = result["incumbent_orientation_trace"]
    for height, summary in orientation["by_height"].items():
        counts = summary["counts"]
        lines.append(
            "| {height} | {cases} | {merges} | {one_way} | {both} | {neither} | "
            "{changes} |".format(
                height=height,
                cases=summary["case_count"],
                merges=counts["merge_count"],
                one_way=counts["one_way_plausible_count"],
                both=counts["both_plausible_count"],
                neither=counts["neither_plausible_count"],
                changes=counts[
                    "neither_plausible_parsimony_would_change_orientation_count"
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Temporal versus no-time mechanism",
            "",
            "| Height | Cases | Constraint active | Backward edge present | Late "
            "root | Same occurrence tree | Time minus no-time AD-F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    temporal = result["temporal_no_time_diagnostic"]
    for height, summary in temporal["by_height"].items():
        lines.append(
            "| {height} | {cases} | {active} | {backward} | {late} | {same} | "
            "{delta} |".format(
                height=height,
                cases=summary["case_count"],
                active=summary["no_time_constraint_active_count"],
                backward=summary["no_time_backward_edge_present_count"],
                late=summary["no_time_late_root_count"],
                same=summary["same_directed_occurrence_tree_count"],
                delta=_format_number(
                    summary["temporal_minus_no_time"]["ad_f1"]["mean"]
                ),
            )
        )
    lines.extend(
        [
            "",
            f"Failures: {result['failure_count']}.",
            "",
        ]
    )
    return "\n".join(lines)


def _execute_isolated_orientation_trace(
    source_root: str,
    case: Mapping[str, Any],
    timeout_seconds: int,
    rss_limit_bytes: int,
):
    reconstruction_input, distance, _truth_tree, metadata = read_case_assets(
        Path(source_root),
        case,
    )
    value, resources, error = measured_stage(
        lambda: reconstruct_incumbent_with_orientation_trace(
            reconstruction_input,
            distance,
            reconstruction_seed=int(metadata["reconstruction_seed"]),
        ),
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if value is not None:
        _tree, trace = value
        value = trace
    return value, resources, error


def _execute_isolated_temporal_arm(
    source_root: str,
    case: Mapping[str, Any],
    arm_id: str,
    timeout_seconds: int,
    rss_limit_bytes: int,
):
    reconstruction_input, distance, _truth_tree, metadata = read_case_assets(
        Path(source_root),
        case,
    )
    value, resources, error = measured_stage(
        lambda: reconstruct_development_arm(
            ARM_SPEC_BY_ID[arm_id],
            reconstruction_input,
            distance,
            reconstruction_seed=int(metadata["reconstruction_seed"]),
        ),
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
    )
    if value is not None:
        value = value[0]
    return value, resources, error


def run_mechanism_audit(
    *,
    bank_root: Path | str,
    output_root: Path | str,
    expected_block_count: int = DEFAULT_BLOCK_COUNT,
    reconstruction_timeout_seconds: int = DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    rss_limit_bytes: int = DEFAULT_RSS_LIMIT_BYTES,
    case_limit: int | None = None,
    created_at_utc: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    for field, value in (
        ("expected_block_count", expected_block_count),
        ("reconstruction_timeout_seconds", reconstruction_timeout_seconds),
        ("rss_limit_bytes", rss_limit_bytes),
    ):
        _validate_positive_integer(value, field)
    if case_limit is not None:
        _validate_positive_integer(case_limit, "case_limit")

    source_root, bank = load_bank_manifest(
        bank_root,
        expected_block_count=expected_block_count,
    )
    output_path = Path(output_root).expanduser().resolve()
    if output_path == source_root or source_root in output_path.parents:
        raise ValueError("Mechanism-audit output must be outside the immutable bank.")
    root = ensure_new_output_root(output_path)

    all_cases = sorted(
        bank["cases"],
        key=lambda case: (int(case["block_index"]), int(case["height"])),
    )
    if case_limit is not None:
        if case_limit > len(all_cases):
            raise ValueError("case_limit exceeds the bank condition count.")
        cases = all_cases[:case_limit]
    else:
        cases = all_cases

    result: dict[str, Any] = {
        "schema_version": MECHANISM_AUDIT_SCHEMA_VERSION,
        "status": "in_progress",
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "scientific_role": (
            "method_development_mechanism_audit_not_paper_accuracy_evidence"
            if len(cases) == len(all_cases)
            else "implementation_smoke_only"
        ),
        "bank_id": bank["bank_id"],
        "bank_root": str(source_root),
        "bank_condition_count": len(all_cases),
        "requested_case_count": len(cases),
        "complete_bank_audit": len(cases) == len(all_cases),
        "truth_supplied_to_reconstruction": False,
        "writes_reconstructed_trees_or_profiles": False,
        "resources": {
            "reconstruction_timeout_seconds_per_arm": (
                reconstruction_timeout_seconds
            ),
            "rss_limit_bytes_per_stage": rss_limit_bytes,
            "reconstruction_execution": fresh_process_contract(
                MECHANISM_WORKER_UNIT
            ),
            "worker_outer_timeout_seconds": (
                reconstruction_timeout_seconds + MECHANISM_WORKER_GRACE_SECONDS
            ),
        },
        "case_records": [],
        "failures": [],
    }
    write_json(root / RESULT_NAME, result)

    metadata_records = []
    orientation_rows = []
    temporal_rows = []
    try:
        for case_index, case in enumerate(cases):
            reconstruction_input, distance, truth_tree, metadata = read_case_assets(
                source_root,
                case,
            )
            metadata_records.append(metadata)
            case_record: dict[str, Any] = {
                "case_id": case["case_id"],
                "block_index": int(case["block_index"]),
                "height": int(case["height"]),
                "incumbent_orientation": None,
                "temporal_no_time": None,
                "resources": {},
            }

            with FreshSpawnPerTaskExecutor() as reconstruction_executor:
                traced_value, traced_resources, traced_error = (
                    reconstruction_executor.run(
                        _execute_isolated_orientation_trace,
                        str(source_root),
                        dict(case),
                        reconstruction_timeout_seconds,
                        rss_limit_bytes,
                        timeout_seconds=(
                            reconstruction_timeout_seconds
                            + MECHANISM_WORKER_GRACE_SECONDS
                        ),
                    )
                )
            if traced_error is None:
                traced_error = _resource_audit_error(
                    traced_resources,
                    "incumbent orientation trace",
                )
            case_record["resources"]["incumbent_orientation"] = _resource_scalar(
                traced_resources
            )
            if traced_error is None and traced_value is not None:
                trace = traced_value
                orientation_row = {
                    "case_id": case["case_id"],
                    "block_index": int(case["block_index"]),
                    "height": int(case["height"]),
                    "trace": trace,
                }
                orientation_rows.append(orientation_row)
                case_record["incumbent_orientation"] = {
                    "status": "success",
                    "trace": trace,
                }
            else:
                error = traced_error or RuntimeError(
                    "Incumbent orientation trace returned no value."
                )
                failure = _typed_error(
                    error,
                    "incumbent_orientation_trace",
                    case["case_id"],
                )
                result["failures"].append(failure)
                case_record["incumbent_orientation"] = {
                    "status": "failure",
                    "failure": failure,
                }

            temporal_values: dict[str, nx.DiGraph] = {}
            temporal_failed = False
            for arm_id in TEMPORAL_ARM_IDS:
                with FreshSpawnPerTaskExecutor() as reconstruction_executor:
                    value, resources, error = reconstruction_executor.run(
                        _execute_isolated_temporal_arm,
                        str(source_root),
                        dict(case),
                        arm_id,
                        reconstruction_timeout_seconds,
                        rss_limit_bytes,
                        timeout_seconds=(
                            reconstruction_timeout_seconds
                            + MECHANISM_WORKER_GRACE_SECONDS
                        ),
                    )
                if error is None:
                    error = _resource_audit_error(resources, arm_id)
                case_record["resources"][arm_id] = _resource_scalar(resources)
                if error is not None or value is None:
                    stage_error = error or RuntimeError(
                        f"{arm_id} returned no reconstruction."
                    )
                    failure = _typed_error(
                        stage_error,
                        arm_id,
                        case["case_id"],
                    )
                    result["failures"].append(failure)
                    temporal_failed = True
                else:
                    temporal_values[arm_id] = value

            if not temporal_failed:
                try:
                    temporal_record = temporal_case_diagnostic(
                        case_id=case["case_id"],
                        block_index=int(case["block_index"]),
                        height=int(case["height"]),
                        reconstruction_input=reconstruction_input,
                        distance=distance,
                        truth_tree=truth_tree,
                        temporal_tree=temporal_values["temporal_minimum"],
                        no_time_tree=temporal_values[
                            "temporal_minimum_no_time"
                        ],
                    )
                except Exception as error:
                    failure = _typed_error(
                        error,
                        "temporal_no_time_diagnostic",
                        case["case_id"],
                    )
                    result["failures"].append(failure)
                    case_record["temporal_no_time"] = {
                        "status": "failure",
                        "failure": failure,
                    }
                else:
                    temporal_rows.append(temporal_record)
                    case_record["temporal_no_time"] = {
                        "status": "success",
                        **temporal_record,
                    }
            else:
                case_record["temporal_no_time"] = {
                    "status": "failure",
                    "failure": {
                        "case_id": case["case_id"],
                        "stage": "temporal_no_time_diagnostic",
                        "type": "DependencyFailure",
                        "message": "At least one matched temporal reconstruction failed.",
                    },
                }

            result["case_records"].append(case_record)
            result["completed_case_count"] = case_index + 1
            write_json(root / RESULT_NAME, result)
            if progress:
                print(
                    f"{case['case_id']}: "
                    f"orientation={case_record['incumbent_orientation']['status']} "
                    f"temporal={case_record['temporal_no_time']['status']}",
                    file=sys.stderr,
                    flush=True,
                )
            gc.collect()

        result["radius_candidate_graph"] = summarize_radius_candidate_graph(
            metadata_records
        )
        result["truth_sampling"] = summarize_truth_sampling(metadata_records)
        result["incumbent_orientation_trace"] = summarize_orientation_traces(
            orientation_rows
        )
        result["temporal_no_time_diagnostic"] = summarize_temporal_diagnostics(
            temporal_rows
        )
    except BaseException as error:
        result["status"] = "failure"
        result["runner_failure"] = _typed_error(
            error,
            "mechanism_audit_runner",
            "bank",
        )
        result["completed_case_count"] = len(result["case_records"])
        write_json(root / RESULT_NAME, result)
        raise

    result["status"] = "complete"
    result["completed_case_count"] = len(result["case_records"])
    result["orientation_success_count"] = len(orientation_rows)
    result["temporal_pair_success_count"] = len(temporal_rows)
    result["failure_count"] = len(result["failures"])
    result["failed_case_count"] = len(
        {failure["case_id"] for failure in result["failures"]}
    )
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(root / RESULT_NAME, result)
    (root / REPORT_NAME).write_text(_render_report(result), encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reconstruction-timeout-seconds",
        type=int,
        default=DEFAULT_RECONSTRUCTION_TIMEOUT_SECONDS,
    )
    parser.add_argument("--rss-limit-bytes", type=int, default=DEFAULT_RSS_LIMIT_BYTES)
    parser.add_argument(
        "--case-limit",
        type=int,
        help="Implementation-smoke prefix only; omit for the approved full audit.",
    )
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_mechanism_audit(
        bank_root=arguments.bank_root,
        output_root=arguments.output_root,
        reconstruction_timeout_seconds=arguments.reconstruction_timeout_seconds,
        rss_limit_bytes=arguments.rss_limit_bytes,
        case_limit=arguments.case_limit,
        progress=arguments.progress,
    )
    print(
        f"complete: {result['completed_case_count']} cases; "
        f"{result['failure_count']} failures"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MECHANISM_AUDIT_SCHEMA_VERSION",
    "make_traced_plausible_parsimony_incumbent",
    "reconstruct_incumbent_with_orientation_trace",
    "run_mechanism_audit",
    "summarize_orientation_traces",
    "summarize_radius_candidate_graph",
    "summarize_temporal_diagnostics",
    "summarize_truth_sampling",
    "temporal_case_diagnostic",
]
