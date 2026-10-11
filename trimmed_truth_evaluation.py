"""State-restricted truth projection, separate from native full-truth GRF.

Retain *every* occurrence of an observed state, not just sampled vertices.
This evaluator-only projection never changes a reconstruction or the native
evaluation contract. It discards nonmetric attributes such as genomes, depths
and edge weights, since a contracted edge can represent an original path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx

from evaluation_contract import (
    EvaluationContractError,
    ValidatedTree,
    evaluate_tree_pair_result,
    validate_evaluation_result,
    validate_rooted_labeled_tree,
)
from evaluator_full import normalize_cell_labels, unique_ancestor_pair_set


TRIMMED_TRUTH_SCHEMA = "ctbf-state-restricted-truth-evaluation-v1"
TRIMMED_TRUTH_SEMANTICS = "ctbf-all-observed-state-occurrences-trim-v1"


def truth_target_descriptor() -> dict:
    return {
        "semantics": TRIMMED_TRUTH_SEMANTICS,
        "retained_labels": "biopsy_state_union",
        "retained_occurrences": "all_true_occurrences_of_retained_labels",
        "unlabeled_terminal_branches": "prune",
        "unlabeled_nonroot_unary_nodes": "suppress",
        "unlabeled_unary_root": "promote_its_child_repeatedly",
        "labeled_unary_nodes": "retain",
        "repeated_labels": "retain_all_occurrences_without_merging",
        "reconstructed_tree": "unchanged",
        "native_full_truth_results": "read_only",
    }


def _error(code: str, message: str, **details: Any) -> EvaluationContractError:
    return EvaluationContractError(code, message, stage="trimmed_truth", details=details)


def _labels(values: Iterable[Any]) -> tuple[str, ...]:
    try:
        labels = normalize_cell_labels(values)
    except (TypeError, ValueError) as error:
        raise _error("invalid_observation_labels", "Invalid observation-label set.") from error
    if not labels:
        raise _error("observation_labels_empty", "Trimming requires nonempty observed states.")
    return tuple(sorted(labels))


def _project(validated: ValidatedTree, observed: tuple[str, ...]) -> nx.DiGraph:
    context = validated.context
    allowed = frozenset(observed)
    missing = allowed.difference(context.labels.values())
    if missing:
        raise _error("observation_labels_missing_from_truth",
                     "Observed states must occur in the true tree.", labels=sorted(missing))

    # Iterative postorder and one surviving frontier per original vertex.
    # No repeated NetworkX scans, ancestor-path sets or recursive depth limit.
    order, stack = [], [validated.root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(context.children.get(node, ()))
    result = nx.DiGraph()
    frontier = {}
    absent = object()
    for node in reversed(order):
        children = [frontier.pop(child) for child in context.children.get(node, ())]
        children = [child for child in children if child is not absent]
        label = context.labels.get(node)
        if label in allowed or len(children) >= 2:
            result.add_node(node, cell_id=label if label in allowed else None)
            result.add_edges_from((node, child) for child in children)
            frontier[node] = node
        else:
            frontier[node] = children[0] if children else absent
    return result


def trim_true_tree(true_tree: Any, observation_labels: Iterable[Any]) -> nx.DiGraph:
    """Return a new metric-only tree in O(V+E) time, preserving original node ids.

    Unlabeled branching nodes survive. Labeled unary nodes and unsampled
    occurrences of observed states survive. Inputs and their attributes are
    never mutated. Labels follow the native string/strip normalization.
    """
    return _project(validate_rooted_labeled_tree(true_tree, "true_tree"),
                    _labels(observation_labels))


@dataclass(frozen=True)
class PreparedTrimmedTruth:
    original: ValidatedTree
    trimmed: ValidatedTree
    observation_labels: tuple[str, ...]
    true_ad_pair_count: int


def prepare_trimmed_truth(true_tree: Any, observation_labels: Iterable[Any]) -> PreparedTrimmedTruth:
    """Prepare once per condition and verify exact state-ancestry invariance."""
    labels = _labels(observation_labels)
    original = validate_rooted_labeled_tree(true_tree, "true_tree")
    trimmed = validate_rooted_labeled_tree(_project(original, labels), "trimmed_true_tree")
    original_pairs = unique_ancestor_pair_set(original.context, restrict_labels=labels)
    trimmed_pairs = unique_ancestor_pair_set(trimmed.context, restrict_labels=labels)
    if original_pairs != trimmed_pairs:
        raise _error("trimming_changed_ancestry", "Trimming changed observed-state ancestry.")
    return PreparedTrimmedTruth(original, trimmed, labels, len(original_pairs))


def evaluate_prepared_trimmed_truth(
    truth: PreparedTrimmedTruth, reconstructed_tree: Any, frozen_full_evaluation: dict,
) -> dict:
    """Evaluate unchanged reconstruction and retain the checksum-bound old score.

    The caller must bind the frozen result to this exact stored reconstruction
    and original truth using source checksums. Full GRF is deliberately not
    recomputed. Metadata and all AD fields are checked against the old result.
    """
    validate_evaluation_result(frozen_full_evaluation)
    if frozen_full_evaluation["status"] != "success":
        raise _error("source_evaluation_failed", "A failed source result is not rescored.")
    if frozen_full_evaluation["observation_labels"] != list(truth.observation_labels):
        raise _error("source_observations_changed", "Source observation labels differ.")
    if frozen_full_evaluation["inputs"]["true_tree"] != truth.original.metadata:
        raise _error("source_truth_changed", "Original true-tree metadata differs.")
    result = evaluate_tree_pair_result(
        truth.trimmed.context, reconstructed_tree, truth.observation_labels)
    validate_evaluation_result(result)
    if result["status"] != "success":
        raise _error("trimmed_evaluation_failed", "Native evaluator rejected the tree pair.",
                     native_failure=result["failure"])
    if result["inputs"]["observation_label_coverage"]["fraction"] != 1.0:
        raise _error("incomplete_reconstruction", "Reconstruction omits observed states.")
    for field in ("reconstructed_tree", "observation_label_coverage"):
        if result["inputs"][field] != frozen_full_evaluation["inputs"][field]:
            raise _error("source_reconstruction_changed", "Reconstructed input metadata differs.")
    for field, value in result["metrics"].items():
        if field not in {"grf", "ext_grf"} and value != frozen_full_evaluation["metrics"][field]:
            raise _error("ad_f1_invariance_failed", "An AD-F1 field differs from its frozen value.",
                         field=field)
    if result["metrics"]["ad_counts"]["true_unique_pair_count"] != truth.true_ad_pair_count:
        raise _error("ad_pair_count_changed", "Native AD pair count differs from the verified set.")
    return {
        "schema_version": TRIMMED_TRUTH_SCHEMA,
        "status": "success",
        "truth_target": truth_target_descriptor(),
        "observation_labels": list(truth.observation_labels),
        "original_truth": truth.original.metadata,
        "trimmed_truth": truth.trimmed.metadata,
        "grf_full": frozen_full_evaluation["metrics"]["grf"],
        "grf_trim": result["metrics"]["grf"],
        "ad_f1": result["metrics"]["ad_f1"],
        "ad_relation_set_invariant": True,
        "ad_fields_match_frozen": True,
        "full_grf_recomputed": False,
        "trimmed_native_evaluation": result,
    }
