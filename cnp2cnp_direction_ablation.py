"""Pure helpers for the bounded G0-03-A cnp2cnp ablation.

This module does not sample truth, mutate genotypes, or select observations.
Truth ancestry, when audited, is an explicit evaluator-only input.
"""

from collections import Counter
import time

import numpy as np

from distance_semantics import (
    DirectedDistanceBundle,
    stable_distance_label_key,
    stable_row_order_digest,
    validate_distance_matrix,
)
from evaluator import grf_tree
from evaluator_full import evaluate_4
from reconstructor_temporal import (
    temporal_cnp_arborescence,
    temporal_cnp_arborescence_directed,
    temporal_cnp_arborescence_directed_no_time,
    temporal_cnp_arborescence_no_time,
)


DIRECTION_AUDIT_SCHEMA_VERSION = "ctbf-cnp2cnp-direction-audit-v1"


def timed_distance_compute(provider, cells):
    """Compute one distance input and return its distance-only run record."""
    start = time.perf_counter_ns()
    distance_input = provider.compute(cells)
    elapsed = time.perf_counter_ns() - start
    provenance = distance_input.provenance
    return distance_input, {
        "distance_wall_time_ns": elapsed,
        "distance_provenance": provenance,
        "external_process_count": (
            None if provenance is None else provenance.get("external_process_count")
        ),
        "directional_transformation_count": (
            None
            if provenance is None
            else provenance.get("directional_transformation_count")
        ),
    }


def canonical_fast_row_order(ids):
    """Return the biopsy-order-independent label order used by fast mode."""
    return tuple(sorted(ids, key=stable_distance_label_key))


def ordered_triangle_fast_view(directed_bundle, row_order=None):
    """Derive the one-triangle mirrored matrix for one explicit row order."""
    if not isinstance(directed_bundle, DirectedDistanceBundle):
        raise ValueError("ordered_triangle_fast_view requires a DirectedDistanceBundle.")

    bundle_ids = list(directed_bundle.ids)
    row_order = list(
        canonical_fast_row_order(bundle_ids)
        if row_order is None
        else row_order
    )
    try:
        if len(row_order) != len(set(row_order)):
            raise ValueError("Fast row order contains duplicate ids.")
        same_ids = set(row_order) == set(bundle_ids)
    except TypeError as exc:
        raise ValueError("Fast row-order ids must be hashable.") from exc
    if not same_ids:
        raise ValueError("Fast row order must contain every bundle id exactly once.")

    bundle_index = {cell_id: index for index, cell_id in enumerate(bundle_ids)}
    alignment = [bundle_index[cell_id] for cell_id in row_order]
    ordered_directed = directed_bundle.directed_matrix[np.ix_(alignment, alignment)]
    fast = np.triu(ordered_directed, k=1)
    fast = fast + fast.T
    return validate_distance_matrix(row_order, fast)


def _plausibility_flags(left_genome, right_genome):
    left = np.asarray(left_genome)
    right = np.asarray(right_genome)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("Pair genomes must be aligned one-dimensional profiles.")
    left_parent = not np.any((left == 0) & (right > 0))
    right_parent = not np.any((right == 0) & (left > 0))
    return bool(left_parent), bool(right_parent)


def _plausibility_stratum(left_parent, right_parent):
    if left_parent and right_parent:
        return "both_plausible"
    if left_parent:
        return "left_only_plausible"
    if right_parent:
        return "right_only_plausible"
    return "neither_plausible"


def _pair_key(left, right):
    return tuple(sorted((left, right), key=stable_distance_label_key))


def audit_directed_distances(
    directed_bundle,
    genomes_by_id,
    *,
    truth_directions=(),
    time_decided_pairs=(),
):
    """Summarize asymmetry and evaluator-only truth-direction sign accuracy.

    ``truth_directions`` contains explicit ``(ancestor_state, descendant_state)``
    pairs. ``time_decided_pairs`` contains unordered pairs for which biopsy time
    already fixes orientation; those pairs are excluded from residual numerical
    sign accuracy. Recurrent state occurrences should be omitted unless their
    state-level ancestry direction is unambiguous in the caller's truth view.
    """
    if not isinstance(directed_bundle, DirectedDistanceBundle):
        raise ValueError("Direction audit requires a DirectedDistanceBundle.")

    ids = list(directed_bundle.ids)
    if set(genomes_by_id) != set(ids):
        raise ValueError("genomes_by_id must contain exactly the bundle ids.")
    time_decided = {_pair_key(*pair) for pair in time_decided_pairs}
    pair_records = {}
    stratum_totals = Counter()
    stratum_asymmetric = Counter()
    all_positive_totals = Counter()
    magnitudes = []

    for left_index, left_id in enumerate(ids):
        for right_index in range(left_index + 1, len(ids)):
            right_id = ids[right_index]
            left_to_right = float(
                directed_bundle.directed_matrix[left_index, right_index]
            )
            right_to_left = float(
                directed_bundle.directed_matrix[right_index, left_index]
            )
            difference = left_to_right - right_to_left
            left_parent, right_parent = _plausibility_flags(
                genomes_by_id[left_id],
                genomes_by_id[right_id],
            )
            stratum = _plausibility_stratum(left_parent, right_parent)
            all_positive = bool(
                np.all(np.asarray(genomes_by_id[left_id]) > 0)
                and np.all(np.asarray(genomes_by_id[right_id]) > 0)
            )
            asymmetric = difference != 0
            stratum_totals[stratum] += 1
            stratum_asymmetric[stratum] += int(asymmetric)
            all_positive_totals[
                "all_positive" if all_positive else "contains_zero"
            ] += 1
            if asymmetric:
                magnitudes.append(abs(difference))
            pair_records[_pair_key(left_id, right_id)] = {
                "left": left_id,
                "right": right_id,
                "left_to_right": left_to_right,
                "right_to_left": right_to_left,
                "difference": difference,
                "stratum": stratum,
                "all_positive": all_positive,
            }

    truth_counts = Counter()
    truth_margins = []
    truth_by_margin = {}
    seen_truth = set()
    id_to_index = {cell_id: index for index, cell_id in enumerate(ids)}
    for ancestor, descendant in truth_directions:
        if ancestor == descendant:
            raise ValueError("Truth-direction pairs must contain distinct states.")
        if ancestor not in id_to_index or descendant not in id_to_index:
            raise ValueError("Truth-direction pair contains an id outside the bundle.")
        directed_pair = (ancestor, descendant)
        if directed_pair in seen_truth:
            raise ValueError(f"Duplicate truth-direction pair {directed_pair!r}.")
        seen_truth.add(directed_pair)

        key = _pair_key(ancestor, descendant)
        record = pair_records[key]
        truth_counts["provided"] += 1
        if key in time_decided:
            truth_counts["excluded_time_decided"] += 1
            continue
        ancestor_parent, descendant_parent = _plausibility_flags(
            genomes_by_id[ancestor],
            genomes_by_id[descendant],
        )
        if ancestor_parent != descendant_parent:
            truth_counts["excluded_plausibility_decided"] += 1
            continue
        if not ancestor_parent and not descendant_parent:
            truth_counts["excluded_neither_plausible"] += 1
            continue

        forward = float(
            directed_bundle.directed_matrix[
                id_to_index[ancestor],
                id_to_index[descendant],
            ]
        )
        reverse = float(
            directed_bundle.directed_matrix[
                id_to_index[descendant],
                id_to_index[ancestor],
            ]
        )
        truth_counts["eligible_both_plausible"] += 1
        if forward == reverse:
            truth_counts["ties"] += 1
            continue
        truth_counts["sign_informative"] += 1
        correct = forward < reverse
        truth_counts["correct"] += int(correct)
        truth_counts["incorrect"] += int(not correct)
        margin = abs(forward - reverse)
        truth_margins.append(margin)
        margin_record = truth_by_margin.setdefault(
            str(margin),
            {"pairs": 0, "correct": 0},
        )
        margin_record["pairs"] += 1
        margin_record["correct"] += int(correct)

    unordered_count = len(pair_records)
    asymmetric_count = len(magnitudes)
    informative = truth_counts["sign_informative"]
    eligible = truth_counts["eligible_both_plausible"]
    calibrated = {
        margin: {
            **record,
            "accuracy": record["correct"] / record["pairs"],
        }
        for margin, record in sorted(
            truth_by_margin.items(),
            key=lambda item: float(item[0]),
        )
    }
    return {
        "schema_version": DIRECTION_AUDIT_SCHEMA_VERSION,
        "bundle_provenance": directed_bundle.provenance,
        "unordered_pair_count": unordered_count,
        "asymmetric_pair_count": asymmetric_count,
        "asymmetric_fraction": (
            asymmetric_count / unordered_count if unordered_count else 0.0
        ),
        "asymmetry_magnitude": {
            "minimum": min(magnitudes) if magnitudes else None,
            "median": float(np.median(magnitudes)) if magnitudes else None,
            "maximum": max(magnitudes) if magnitudes else None,
        },
        "plausibility_strata": {
            name: {
                "pairs": stratum_totals[name],
                "asymmetric_pairs": stratum_asymmetric[name],
            }
            for name in (
                "both_plausible",
                "left_only_plausible",
                "right_only_plausible",
                "neither_plausible",
            )
        },
        "profile_strata": dict(all_positive_totals),
        "truth_direction": {
            **dict(truth_counts),
            "sign_accuracy": (
                truth_counts["correct"] / informative if informative else None
            ),
            "false_direction_rate": (
                truth_counts["incorrect"] / informative if informative else None
            ),
            "sign_coverage": informative / eligible if eligible else None,
            "tie_fraction": truth_counts["ties"] / eligible if eligible else None,
            "informative_margin_median": (
                float(np.median(truth_margins)) if truth_margins else None
            ),
            "accuracy_by_absolute_difference": calibrated,
        },
    }


def audit_fast_row_order_sensitivity(directed_bundle, row_orders):
    """Compare fast matrices after realigning every requested order by id."""
    row_orders = [tuple(order) for order in row_orders]
    if not row_orders:
        raise ValueError("At least one fast row order is required.")

    reference_ids = list(directed_bundle.ids)
    aligned_matrices = []
    records = []
    for row_order in row_orders:
        ids, matrix = ordered_triangle_fast_view(directed_bundle, row_order)
        order_index = {cell_id: index for index, cell_id in enumerate(ids)}
        alignment = [order_index[cell_id] for cell_id in reference_ids]
        aligned = matrix[np.ix_(alignment, alignment)]
        aligned_matrices.append(aligned)
        records.append({
            "row_order": list(ids),
            "row_order_sha256": stable_row_order_digest(ids),
        })

    baseline = aligned_matrices[0]
    upper = np.triu_indices(len(reference_ids), k=1)
    for record, aligned in zip(records, aligned_matrices):
        record["changed_unordered_pairs_vs_first"] = int(
            np.count_nonzero(aligned[upper] != baseline[upper])
        )
    return {
        "reference_ids": reference_ids,
        "orders": records,
        "any_matrix_change": any(
            record["changed_unordered_pairs_vs_first"] > 0
            for record in records[1:]
        ),
    }


def audit_fast_reconstruction_sensitivity(
    cell_lists,
    directed_bundle,
    row_orders,
    *,
    seed=7,
    use_time=True,
):
    """Count fast-tree changes across prespecified row orders."""
    row_orders = [tuple(order) for order in row_orders]
    if not row_orders:
        raise ValueError("At least one fast row order is required.")
    algorithm = (
        temporal_cnp_arborescence
        if use_time
        else temporal_cnp_arborescence_no_time
    )
    records = []
    signatures = []
    for row_order in row_orders:
        ids, matrix = ordered_triangle_fast_view(directed_bundle, row_order)
        tree, _new_nodes, root = algorithm(
            matrix,
            cell_lists,
            ids,
            seed=seed,
        )
        signature = (root, tuple(sorted(tree.edges())))
        signatures.append(signature)
        records.append({
            "row_order": list(ids),
            "row_order_sha256": stable_row_order_digest(ids),
        })

    baseline = signatures[0]
    for record, signature in zip(records, signatures):
        record["topology_changed_vs_first"] = signature != baseline
    return {
        "seed": seed,
        "use_time": bool(use_time),
        "orders": records,
        "changed_order_count_vs_first": sum(
            record["topology_changed_vs_first"]
            for record in records[1:]
        ),
    }


def build_three_arm_temporal_trees(
    cell_lists,
    directed_bundle,
    *,
    seed=7,
    fast_row_order=None,
    use_time=True,
):
    """Build matched A/B/C trees without inspecting or modifying truth."""
    fast_ids, fast_matrix = ordered_triangle_fast_view(
        directed_bundle,
        fast_row_order,
    )
    symmetric_ids = list(directed_bundle.ids)
    symmetric = np.array(directed_bundle.minimum_matrix, copy=True)

    if use_time:
        minimum_algorithm = temporal_cnp_arborescence
        directed_algorithm = temporal_cnp_arborescence_directed
    else:
        minimum_algorithm = temporal_cnp_arborescence_no_time
        directed_algorithm = temporal_cnp_arborescence_directed_no_time

    fast_tree = minimum_algorithm(fast_matrix, cell_lists, fast_ids, seed=seed)
    minimum_tree = minimum_algorithm(
        symmetric,
        cell_lists,
        symmetric_ids,
        seed=seed,
    )
    directed_tree = directed_algorithm(
        symmetric,
        cell_lists,
        symmetric_ids,
        seed=seed,
        directed_distance_bundle=directed_bundle,
    )
    return {
        "ordered_triangle_fast": fast_tree,
        "minimum_bidirectional": minimum_tree,
        "minimum_with_directed": directed_tree,
    }


def evaluate_three_arm_temporal_trees(
    true_tree,
    true_root,
    arm_results,
    *,
    observed_labels,
):
    """Evaluate matched arm outputs; truth enters only at this boundary."""
    reports = {}
    for arm_name, (tree, _new_nodes, returned_root) in arm_results.items():
        roots = [node for node, indegree in tree.in_degree() if indegree == 0]
        if len(roots) != 1 or roots[0] != returned_root:
            raise ValueError(f"Arm {arm_name!r} does not expose one consistent root.")
        ancestry = evaluate_4(
            true_tree,
            tree,
            restrict_labels=set(observed_labels),
        )
        reports[arm_name] = {
            "ad_f1": ancestry["ancestors_unique_restricted"]["F1"],
            "grf": grf_tree(true_tree, true_root, tree, returned_root),
            "evaluate_4": ancestry,
        }
    return reports


__all__ = [
    "DIRECTION_AUDIT_SCHEMA_VERSION",
    "audit_directed_distances",
    "audit_fast_reconstruction_sensitivity",
    "audit_fast_row_order_sensitivity",
    "build_three_arm_temporal_trees",
    "canonical_fast_row_order",
    "evaluate_three_arm_temporal_trees",
    "ordered_triangle_fast_view",
    "timed_distance_compute",
]
