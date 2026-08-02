"""Versioned, strict paper-facing tree-evaluation contract.

Compatibility evaluator helpers remain in :mod:`evaluator` and
:mod:`evaluator_full`. New experimental evidence should cross this boundary so
that label, tree, metric-direction, empty-case, and failure semantics travel
with every result.
"""

from collections import Counter
from dataclasses import dataclass
import json
import math
from typing import Any, Dict, Iterable, Optional, Tuple

import networkx as nx

from evaluator import cluster_evaluation_context, ext_grf_from_cluster_counts
from evaluator_full import (
    TreeEvaluationContext,
    adf1_restricted_metrics_from_contexts,
    ensure_tree_evaluation_context,
    normalize_cell_labels,
    prf1_iou,
)


EVALUATION_RESULT_SCHEMA_VERSION = "ctbf-tree-evaluation-result-v1"
LABEL_NORMALIZATION_VERSION = "ctbf-cell-label-string-strip-v1"
AD_F1_SEMANTICS_VERSION = "ctbf-ad-f1-truth-restricted-unique-pairs-v1"
GRF_SEMANTICS_VERSION = "ctbf-rooted-generalized-rf-cluster-multiset-v1"
TREE_VALIDATION_VERSION = "ctbf-rooted-directed-arborescence-v1"

class EvaluationContractError(ValueError):
    """A declared evaluator-input or result-schema failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str = "evaluation",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.details = details or {}


@dataclass(frozen=True)
class ValidatedTree:
    context: TreeEvaluationContext
    root: Any
    metadata: Dict[str, Any]


def metric_contract_descriptor() -> Dict[str, Any]:
    """Return the complete semantic descriptor embedded in result v1."""
    return {
        "label_normalization": LABEL_NORMALIZATION_VERSION,
        "tree_validation": TREE_VALIDATION_VERSION,
        "ad_f1": {
            "semantics": AD_F1_SEMANTICS_VERSION,
            "direction": "higher_is_better",
            "truth_pair_restriction": "both_endpoints_in_observation_labels",
            "reconstructed_pair_restriction": "none",
            "reconstructed_label_policy": "non_null_labels_must_be_observed",
            "pair_multiplicity": "unique_label_pairs",
            "strict_ancestor_only": True,
            "recurrent_self_label_pair_allowed": True,
            "empty_empty_value": 0.0,
        },
        "grf": {
            "semantics": GRF_SEMANTICS_VERSION,
            "direction": "higher_is_better",
            "relationship": "grf = 1 - ext_grf",
            "label_multiplicity": "retained_within_clusters",
            "equal_cluster_multiplicity": "retained_across_nodes",
            "observation_restriction": "none",
        },
        "ext_grf": {
            "semantics": GRF_SEMANTICS_VERSION,
            "direction": "lower_is_better",
            "relationship": "ext_grf = 1 - grf",
        },
    }


def _contract_error(
    code: str,
    message: str,
    *,
    stage: str,
    **details: Any,
) -> EvaluationContractError:
    return EvaluationContractError(code, message, stage=stage, details=details)


def _validate_node_link_shape(data: Dict[str, Any], role: str) -> None:
    if data.get("directed", True) is False:
        raise _contract_error(
            "tree_not_directed",
            f"{role} tree must be directed parent-to-child.",
            stage=role,
        )
    if data.get("multigraph", False) is True:
        raise _contract_error(
            "tree_is_multigraph",
            f"{role} must be a simple directed graph.",
            stage=role,
        )

    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        raise _contract_error(
            "invalid_tree_input",
            f"{role} node-link input must contain a nodes list.",
            stage=role,
        )
    node_ids = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or "id" not in node:
            raise _contract_error(
                "invalid_tree_input",
                f"{role} node-link node {index} has no id.",
                stage=role,
            )
        node_ids.append(node["id"])
    try:
        unique_node_ids = set(node_ids)
    except TypeError as exc:
        raise _contract_error(
            "invalid_tree_input",
            f"{role} node ids must be hashable.",
            stage=role,
        ) from exc
    if len(unique_node_ids) != len(node_ids):
        raise _contract_error(
            "duplicate_node_id",
            f"{role} node-link input contains duplicate node ids.",
            stage=role,
        )

    edges = data.get("links", data.get("edges", []))
    if not isinstance(edges, list):
        raise _contract_error(
            "invalid_tree_input",
            f"{role} node-link edges must be a list.",
            stage=role,
        )
    edge_keys = set()
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict) or "source" not in edge or "target" not in edge:
            raise _contract_error(
                "invalid_tree_input",
                f"{role} node-link edge {index} needs source and target.",
                stage=role,
            )
        source, target = edge["source"], edge["target"]
        try:
            endpoints_declared = source in unique_node_ids and target in unique_node_ids
            key = (source, target)
            duplicate = key in edge_keys
        except TypeError as exc:
            raise _contract_error(
                "invalid_tree_input",
                f"{role} edge endpoints must be hashable node ids.",
                stage=role,
            ) from exc
        if not endpoints_declared:
            raise _contract_error(
                "unknown_edge_endpoint",
                f"{role} edge {index} refers to an undeclared node.",
                stage=role,
            )
        if duplicate:
            raise _contract_error(
                "duplicate_edge",
                f"{role} contains duplicate edge {source!r}->{target!r}.",
                stage=role,
            )
        edge_keys.add(key)


def _tree_context(tree: Any, role: str) -> TreeEvaluationContext:
    if isinstance(tree, nx.MultiDiGraph):
        raise _contract_error(
            "tree_is_multigraph",
            f"{role} must be a simple directed graph.",
            stage=role,
        )
    if isinstance(tree, nx.Graph) and not tree.is_directed():
        raise _contract_error(
            "tree_not_directed",
            f"{role} tree must be directed parent-to-child.",
            stage=role,
        )
    if isinstance(tree, dict):
        _validate_node_link_shape(tree, role)
    if isinstance(tree, str):
        raise _contract_error(
            "unsupported_tree_input",
            f"{role} must be a DiGraph, TreeEvaluationContext, or node-link mapping; parse Newick first.",
            stage=role,
        )
    try:
        return ensure_tree_evaluation_context(tree)
    except EvaluationContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise _contract_error(
            "invalid_tree_input",
            f"Could not construct {role} evaluation context: {exc}",
            stage=role,
        ) from exc


def validate_rooted_labeled_tree(tree: Any, role: str) -> ValidatedTree:
    """Validate the v1 rooted-arborescence contract without mutating a tree."""
    context = _tree_context(tree, role)
    nodes = set(context.parents) | set(context.children) | set(context.labels)
    for children in context.children.values():
        nodes.update(children)
    if not nodes:
        raise _contract_error(
            "empty_tree",
            f"{role} tree is empty.",
            stage=role,
        )
    if not context.labels:
        raise _contract_error(
            "tree_has_no_labels",
            f"{role} tree has no non-null canonical cell labels.",
            stage=role,
        )

    incoming = {node: [] for node in nodes}
    edges = set()
    for source in nodes:
        for target in context.children.get(source, ()):
            if target not in nodes:
                raise _contract_error(
                    "unknown_edge_endpoint",
                    f"{role} contains an edge to an unknown node.",
                    stage=role,
                )
            edge = (source, target)
            if edge in edges:
                raise _contract_error(
                    "duplicate_edge",
                    f"{role} contains a duplicate edge.",
                    stage=role,
                )
            if source == target:
                raise _contract_error(
                    "tree_cycle",
                    f"{role} contains a self-loop.",
                    stage=role,
                )
            edges.add(edge)
            incoming[target].append(source)

    multiple_parent_nodes = sum(1 for parents in incoming.values() if len(parents) > 1)
    if multiple_parent_nodes:
        raise _contract_error(
            "multiple_parents",
            f"{role} has {multiple_parent_nodes} node(s) with multiple parents.",
            stage=role,
            node_count=multiple_parent_nodes,
        )

    roots = [node for node, parents in incoming.items() if not parents]
    if len(roots) != 1:
        raise _contract_error(
            "root_count",
            f"{role} must have exactly one root; found {len(roots)}.",
            stage=role,
            root_count=len(roots),
        )
    root = roots[0]

    for node in nodes:
        expected_parent = incoming[node][0] if incoming[node] else None
        if context.parents.get(node) != expected_parent:
            raise _contract_error(
                "inconsistent_tree_context",
                f"{role} parent and child maps disagree.",
                stage=role,
            )
    if len(context.roots) != 1 or context.roots[0] != root:
        raise _contract_error(
            "inconsistent_tree_context",
            f"{role} stored roots disagree with its edges.",
            stage=role,
        )

    reached = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in reached:
            raise _contract_error(
                "tree_cycle",
                f"{role} contains a directed cycle.",
                stage=role,
            )
        reached.add(node)
        stack.extend(context.children.get(node, ()))
    if reached != nodes:
        raise _contract_error(
            "tree_not_connected",
            f"{role} is not a single rooted connected tree.",
            stage=role,
            reached_node_count=len(reached),
            node_count=len(nodes),
        )
    if len(edges) != len(nodes) - 1:
        raise _contract_error(
            "invalid_tree_edge_count",
            f"{role} must have n-1 edges.",
            stage=role,
            edge_count=len(edges),
            node_count=len(nodes),
        )

    label_counts = Counter(context.labels.values())
    metadata = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "labeled_node_count": len(context.labels),
        "unlabeled_node_count": len(nodes) - len(context.labels),
        "unique_label_count": len(label_counts),
        "repeated_label_occurrence_count": sum(label_counts.values()) - len(label_counts),
        "root_label": context.labels.get(root),
    }
    return ValidatedTree(context=context, root=root, metadata=metadata)


def _observation_labels(values: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    try:
        labels = normalize_cell_labels(values)
    except (TypeError, ValueError) as exc:
        raise _contract_error(
            "invalid_observation_labels",
            "Observation labels must be an iterable of canonicalizable values.",
            stage="observation_labels",
        ) from exc
    if labels is None:
        raise _contract_error(
            "observation_labels_required",
            "Versioned evaluation requires an explicit observation-label set.",
            stage="observation_labels",
        )
    if not labels:
        raise _contract_error(
            "observation_labels_empty",
            "Versioned evaluation requires at least one canonical observation label.",
            stage="observation_labels",
        )
    return tuple(sorted(labels))


def _ad_f1_degeneracy(true_pair_count: int, rec_pair_count: int) -> str:
    if true_pair_count == 0 and rec_pair_count == 0:
        return "empty_truth_and_reconstruction"
    if true_pair_count == 0:
        return "empty_truth"
    if rec_pair_count == 0:
        return "empty_reconstruction"
    return "none"


def evaluate_tree_pair(
    true_tree: Any,
    reconstructed_tree: Any,
    observation_labels: Iterable[Any],
) -> Dict[str, Any]:
    """Return a strict successful result or raise :class:`EvaluationContractError`."""
    observed = _observation_labels(observation_labels)
    observed_set = frozenset(observed)
    true_validated = validate_rooted_labeled_tree(true_tree, "true_tree")
    rec_validated = validate_rooted_labeled_tree(reconstructed_tree, "reconstructed_tree")

    true_labels = frozenset(true_validated.context.labels.values())
    reconstructed_labels = frozenset(rec_validated.context.labels.values())
    missing_from_truth = observed_set - true_labels
    if missing_from_truth:
        raise _contract_error(
            "observation_labels_missing_from_truth",
            "Observation labels must occur in the true tree.",
            stage="observation_labels",
            labels=sorted(missing_from_truth),
        )
    outside_reconstructed = reconstructed_labels - observed_set
    if outside_reconstructed:
        raise _contract_error(
            "reconstructed_labels_outside_observation_set",
            "Every non-null reconstructed label must belong to the observation-label set.",
            stage="reconstructed_tree",
            labels=sorted(outside_reconstructed),
        )

    ad = adf1_restricted_metrics_from_contexts(
        true_validated.context,
        rec_validated.context,
        restrict_labels=observed_set,
    )
    true_clusters = cluster_evaluation_context(
        true_validated.context,
        true_validated.root,
    )
    rec_clusters = cluster_evaluation_context(
        rec_validated.context,
        rec_validated.root,
    )
    ext_grf = float(
        ext_grf_from_cluster_counts(true_clusters.counts, rec_clusters.counts)
    )
    grf = 1.0 - ext_grf
    missing_reconstructed = sorted(observed_set - reconstructed_labels)
    degeneracy = _ad_f1_degeneracy(
        ad["num_unique_pairs_true"],
        ad["num_unique_pairs_rec"],
    )

    result = {
        "schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
        "status": "success",
        "metric_contract": metric_contract_descriptor(),
        "observation_labels": list(observed),
        "inputs": {
            "true_tree": true_validated.metadata,
            "reconstructed_tree": rec_validated.metadata,
            "observation_label_coverage": {
                "required_unique_label_count": len(observed),
                "reconstructed_unique_label_count": len(observed_set & reconstructed_labels),
                "fraction": len(observed_set & reconstructed_labels) / len(observed),
                "missing_labels": missing_reconstructed,
            },
        },
        "metrics": {
            "ad_f1": float(ad["F1"]),
            "ad_precision": float(ad["precision"]),
            "ad_recall": float(ad["recall"]),
            "ad_iou": float(ad["IoU"]),
            "ad_counts": {
                "tp": int(ad["TP"]),
                "fp": int(ad["FP"]),
                "fn": int(ad["FN"]),
                "true_unique_pair_count": int(ad["num_unique_pairs_true"]),
                "reconstructed_unique_pair_count": int(ad["num_unique_pairs_rec"]),
            },
            "ad_f1_degenerate": degeneracy != "none",
            "ad_f1_degeneracy": degeneracy,
            "grf": grf,
            "ext_grf": ext_grf,
        },
    }
    validate_evaluation_result(result)
    return result


def _failure_observation_labels(values: Optional[Iterable[Any]]) -> list:
    try:
        normalized = normalize_cell_labels(values)
    except (TypeError, ValueError):
        return []
    return sorted(normalized or ())


def evaluation_failure_result(
    error: EvaluationContractError,
    observation_labels: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    result = {
        "schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
        "status": "failure",
        "metric_contract": metric_contract_descriptor(),
        "observation_labels": _failure_observation_labels(observation_labels),
        "failure": {
            "code": error.code,
            "stage": error.stage,
            "message": str(error),
            "details": error.details,
        },
    }
    validate_evaluation_result(result)
    return result


def evaluate_tree_pair_result(
    true_tree: Any,
    reconstructed_tree: Any,
    observation_labels: Optional[Iterable[Any]],
) -> Dict[str, Any]:
    """Return one status-bearing v1 result for success or declared failure."""
    captured_labels = observation_labels
    if observation_labels is not None and not isinstance(observation_labels, str):
        try:
            captured_labels = tuple(observation_labels)
        except TypeError:
            error = _contract_error(
                "invalid_observation_labels",
                "Observation labels must be an iterable of canonicalizable values.",
                stage="observation_labels",
            )
            return evaluation_failure_result(error)
    try:
        return evaluate_tree_pair(true_tree, reconstructed_tree, captured_labels)
    except EvaluationContractError as error:
        return evaluation_failure_result(error, captured_labels)


def _schema_error(message: str, **details: Any) -> EvaluationContractError:
    return _contract_error(
        "invalid_evaluation_result",
        message,
        stage="result_schema",
        **details,
    )


def _finite_unit_interval(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _schema_error(f"{field} must be numeric.", field=field)
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise _schema_error(f"{field} must be finite and in [0, 1].", field=field)
    return numeric


def validate_evaluation_result(result: Dict[str, Any]) -> None:
    """Raise if a payload is not an interpretable result-schema-v1 object."""
    if not isinstance(result, dict):
        raise _schema_error("Evaluation result must be a mapping.")
    if result.get("schema_version") != EVALUATION_RESULT_SCHEMA_VERSION:
        raise _schema_error("Unknown or missing evaluation schema version.")
    if result.get("metric_contract") != metric_contract_descriptor():
        raise _schema_error("Metric contract descriptor is missing or altered.")
    labels = result.get("observation_labels")
    if not isinstance(labels, list):
        raise _schema_error("Observation labels must be a list.")
    if any(not isinstance(label, str) or not label for label in labels):
        raise _schema_error("Observation labels must be non-empty canonical strings.")
    if labels != sorted(set(labels)):
        raise _schema_error("Observation labels must be sorted and unique.")

    status = result.get("status")
    if status == "failure":
        failure = result.get("failure")
        if not isinstance(failure, dict):
            raise _schema_error("Failure result must contain a failure mapping.")
        for field in ("code", "stage", "message"):
            if not isinstance(failure.get(field), str) or not failure[field]:
                raise _schema_error(f"Failure field {field} must be a non-empty string.")
        if not isinstance(failure.get("details"), dict):
            raise _schema_error("Failure details must be a mapping.")
        try:
            json.dumps(result, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise _schema_error(f"Failure result is not strict JSON: {exc}") from exc
        return
    if status != "success":
        raise _schema_error("Result status must be success or failure.")
    if not labels:
        raise _schema_error("Successful result requires observation labels.")

    metrics = result.get("metrics")
    inputs = result.get("inputs")
    if not isinstance(metrics, dict) or not isinstance(inputs, dict):
        raise _schema_error("Successful result requires inputs and metrics mappings.")
    numeric_fields = (
        "ad_f1",
        "ad_precision",
        "ad_recall",
        "ad_iou",
        "grf",
        "ext_grf",
    )
    numeric = {
        field: _finite_unit_interval(metrics.get(field), f"metrics.{field}")
        for field in numeric_fields
    }
    if not math.isclose(
        numeric["grf"] + numeric["ext_grf"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise _schema_error("grf and ext_grf must be exact complements within tolerance.")

    counts = metrics.get("ad_counts")
    if not isinstance(counts, dict):
        raise _schema_error("metrics.ad_counts must be a mapping.")
    count_fields = ("tp", "fp", "fn", "true_unique_pair_count", "reconstructed_unique_pair_count")
    for field in count_fields:
        value = counts.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _schema_error(f"metrics.ad_counts.{field} must be a nonnegative integer.")
    if counts["tp"] + counts["fn"] != counts["true_unique_pair_count"]:
        raise _schema_error("AD-F1 truth pair count disagrees with TP+FN.")
    if counts["tp"] + counts["fp"] != counts["reconstructed_unique_pair_count"]:
        raise _schema_error("AD-F1 reconstructed pair count disagrees with TP+FP.")
    precision, recall, f1, iou = prf1_iou(counts["tp"], counts["fp"], counts["fn"])
    for field, expected in (
        ("ad_precision", precision),
        ("ad_recall", recall),
        ("ad_f1", f1),
        ("ad_iou", iou),
    ):
        if not math.isclose(numeric[field], expected, rel_tol=0.0, abs_tol=1e-12):
            raise _schema_error(f"metrics.{field} disagrees with AD confusion counts.")
    degeneracy = _ad_f1_degeneracy(
        counts["true_unique_pair_count"],
        counts["reconstructed_unique_pair_count"],
    )
    if metrics.get("ad_f1_degeneracy") != degeneracy:
        raise _schema_error("AD-F1 degeneracy label disagrees with pair counts.")
    if metrics.get("ad_f1_degenerate") is not (degeneracy != "none"):
        raise _schema_error("AD-F1 degeneracy flag disagrees with pair counts.")

    for role in ("true_tree", "reconstructed_tree"):
        tree_metadata = inputs.get(role)
        if not isinstance(tree_metadata, dict):
            raise _schema_error(f"inputs.{role} must be a mapping.")
        for field in ("node_count", "edge_count", "labeled_node_count", "unlabeled_node_count", "unique_label_count", "repeated_label_occurrence_count"):
            value = tree_metadata.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise _schema_error(f"inputs.{role}.{field} must be a nonnegative integer.")
        if tree_metadata["edge_count"] != tree_metadata["node_count"] - 1:
            raise _schema_error(f"inputs.{role} violates the rooted-tree edge count.")
        if tree_metadata["labeled_node_count"] + tree_metadata["unlabeled_node_count"] != tree_metadata["node_count"]:
            raise _schema_error(f"inputs.{role} labeled/unlabeled counts disagree with node count.")
        if tree_metadata["unique_label_count"] > tree_metadata["labeled_node_count"]:
            raise _schema_error(f"inputs.{role} unique-label count exceeds labeled nodes.")
        if tree_metadata["repeated_label_occurrence_count"] != tree_metadata["labeled_node_count"] - tree_metadata["unique_label_count"]:
            raise _schema_error(f"inputs.{role} repeated-label count is inconsistent.")
        root_label = tree_metadata.get("root_label")
        if root_label is not None and (not isinstance(root_label, str) or not root_label):
            raise _schema_error(f"inputs.{role}.root_label must be null or a canonical string.")

    coverage = inputs.get("observation_label_coverage")
    if not isinstance(coverage, dict):
        raise _schema_error("Observation-label coverage metadata is missing.")
    required = coverage.get("required_unique_label_count")
    present = coverage.get("reconstructed_unique_label_count")
    if required != len(labels) or isinstance(present, bool) or not isinstance(present, int):
        raise _schema_error("Observation-label coverage counts are invalid.")
    if present < 0 or present > required:
        raise _schema_error("Reconstructed observation-label coverage count is out of range.")
    fraction = _finite_unit_interval(coverage.get("fraction"), "inputs.observation_label_coverage.fraction")
    if not math.isclose(
        fraction,
        present / required,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise _schema_error("Observation-label coverage fraction disagrees with counts.")
    missing = coverage.get("missing_labels")
    if not isinstance(missing, list) or missing != sorted(set(missing)):
        raise _schema_error("Missing observation labels must be a sorted unique list.")
    if any(label not in labels for label in missing) or len(missing) != required - present:
        raise _schema_error("Missing observation labels disagree with coverage counts.")

    try:
        json.dumps(result, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise _schema_error(f"Success result is not strict JSON: {exc}") from exc
