from collections import Counter

import networkx as nx
import pytest
from networkx.readwrite import json_graph

from evaluator_full import (
    RestrictedAdf1Cache,
    ancestors_unique_restricted_metrics,
    adf1_restricted_metrics_from_contexts,
    evaluate_4,
    label_edge_multiset,
    label_multiset_ancestor_pairs,
    multiset_confusion_simple,
    prf1_iou,
    tree_evaluation_context,
    tree_evaluation_context_from_node_link,
    unique_ancestor_pair_id_set,
    unique_ancestor_pair_set,
)


def _example_tree():
    tree = nx.DiGraph()
    labels = {
        "root": " A ",
        "internal": None,
        "left": "B",
        "right": "B",
        "branch": "C",
        "leaf": "D",
    }
    for node, label in labels.items():
        tree.add_node(node, cell_id=label)
    tree.add_edges_from([
        ("root", "internal"),
        ("internal", "left"),
        ("internal", "right"),
        ("root", "branch"),
        ("branch", "leaf"),
    ])
    return tree


def _reference_label(tree, node):
    label = tree.nodes[node].get("cell_id")
    if label is None:
        return None
    label = str(label).strip()
    return label if label else None


def _reference_parent(tree, node):
    parents = list(tree.predecessors(node))
    return parents[0] if parents else None


def _reference_ancestor_pairs(tree, restrict_labels=None):
    allowed = set(restrict_labels) if restrict_labels is not None else None
    nodes = [node for node in tree.nodes if _reference_label(tree, node) is not None]
    if allowed is not None:
        nodes = [node for node in nodes if _reference_label(tree, node) in allowed]

    pairs = Counter()
    for descendant in nodes:
        descendant_label = _reference_label(tree, descendant)
        current = _reference_parent(tree, descendant)
        while current is not None:
            ancestor_label = _reference_label(tree, current)
            if ancestor_label is not None and (allowed is None or ancestor_label in allowed):
                pairs[(ancestor_label, descendant_label)] += 1
            current = _reference_parent(tree, current)
    return pairs


def _reference_edge_multiset(tree, restrict_labels=None):
    allowed = set(restrict_labels) if restrict_labels is not None else None
    edges = Counter()
    for node in tree.nodes:
        parent = _reference_parent(tree, node)
        if parent is None:
            continue
        parent_label = _reference_label(tree, parent)
        child_label = _reference_label(tree, node)
        if parent_label is None or child_label is None:
            continue
        if allowed is not None and (parent_label not in allowed or child_label not in allowed):
            continue
        edges[(parent_label, child_label)] += 1
    return edges


def _positive_set(counter):
    return {key for key, value in counter.items() if value > 0}


def _reference_set_confusion(true_counter, rec_counter, restrict_labels=None):
    true_set = _positive_set(true_counter)
    if restrict_labels is not None:
        allowed = set(restrict_labels)
        true_set = {(x, y) for x, y in true_set if x in allowed and y in allowed}
    rec_set = _positive_set(rec_counter)
    return len(true_set & rec_set), len(rec_set - true_set), len(true_set - rec_set), true_set, rec_set


def _mode_from_counts(tp, fp, fn, true_size_key, true_size, rec_size_key, rec_size):
    precision, recall, f1, iou = prf1_iou(tp, fp, fn)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IoU": iou,
        true_size_key: true_size,
        rec_size_key: rec_size,
    }


def _reference_evaluate_4(true_tree, rec_tree, restrict_labels=None):
    true_pairs = _reference_ancestor_pairs(true_tree)
    rec_pairs = _reference_ancestor_pairs(rec_tree)
    tp1, fp1, fn1 = multiset_confusion_simple(true_pairs, rec_pairs)
    tp2, fp2, fn2, true_pair_set, rec_pair_set = _reference_set_confusion(
        true_pairs,
        rec_pairs,
    )
    tp0, fp0, fn0, restricted_true_set, restricted_rec_set = _reference_set_confusion(
        true_pairs,
        rec_pairs,
        restrict_labels=restrict_labels,
    )

    true_edges = _reference_edge_multiset(true_tree, restrict_labels)
    rec_edges = _reference_edge_multiset(rec_tree, restrict_labels)
    tp3, fp3, fn3 = multiset_confusion_simple(true_edges, rec_edges)
    tp4, fp4, fn4, true_edge_set, rec_edge_set = _reference_set_confusion(
        true_edges,
        rec_edges,
    )
    return {
        "ancestors_multiset": _mode_from_counts(
            tp1, fp1, fn1, "num_pairs_true", sum(true_pairs.values()), "num_pairs_rec", sum(rec_pairs.values())
        ),
        "ancestors_unique": _mode_from_counts(
            tp2,
            fp2,
            fn2,
            "num_unique_pairs_true",
            len(true_pair_set),
            "num_unique_pairs_rec",
            len(rec_pair_set),
        ),
        "edges_multiset": _mode_from_counts(
            tp3, fp3, fn3, "num_edges_true", sum(true_edges.values()), "num_edges_rec", sum(rec_edges.values())
        ),
        "edges_unique": _mode_from_counts(
            tp4,
            fp4,
            fn4,
            "num_unique_edges_true",
            len(true_edge_set),
            "num_unique_edges_rec",
            len(rec_edge_set),
        ),
        "ancestors_unique_restricted": _mode_from_counts(
            tp0,
            fp0,
            fn0,
            "num_unique_pairs_true",
            len(restricted_true_set),
            "num_unique_pairs_rec",
            len(restricted_rec_set),
        ),
    }


@pytest.mark.parametrize("restrict_labels", [None, {"A", "B"}, {"B", "C", "D"}])
def test_context_counters_match_parent_chain_reference(restrict_labels):
    tree = _example_tree()
    context = tree_evaluation_context(tree)

    assert label_multiset_ancestor_pairs(tree, restrict_labels) == _reference_ancestor_pairs(
        tree,
        restrict_labels,
    )
    assert label_multiset_ancestor_pairs(context, restrict_labels) == _reference_ancestor_pairs(
        tree,
        restrict_labels,
    )
    assert label_edge_multiset(tree, restrict_labels) == _reference_edge_multiset(
        tree,
        restrict_labels,
    )
    assert label_edge_multiset(context, restrict_labels) == _reference_edge_multiset(
        tree,
        restrict_labels,
    )


def test_evaluate_4_context_matches_reference_semantics():
    true_tree = _example_tree()
    rec_tree = _example_tree()
    rec_tree.remove_edge("branch", "leaf")
    rec_tree.add_edge("internal", "leaf")
    restricted_labels = {"A", "B", "D"}

    expected = _reference_evaluate_4(true_tree, rec_tree, restrict_labels=restricted_labels)
    actual = evaluate_4(
        tree_evaluation_context(true_tree),
        tree_evaluation_context(rec_tree),
        restrict_labels=restricted_labels,
    )

    assert actual == expected


def test_node_link_context_matches_networkx_context_and_adf1_path():
    true_tree = _example_tree()
    rec_tree = _example_tree()
    rec_tree.remove_edge("branch", "leaf")
    rec_tree.add_edge("internal", "leaf")
    restricted_labels = {"A", "B", "D"}

    true_payload = json_graph.node_link_data(true_tree, edges="links")
    rec_payload = json_graph.node_link_data(rec_tree, edges="links")
    true_context = tree_evaluation_context_from_node_link(true_payload)
    rec_context = tree_evaluation_context_from_node_link(rec_payload)

    assert true_context.labels == tree_evaluation_context(true_tree).labels
    assert rec_context.children == tree_evaluation_context(rec_tree).children
    assert unique_ancestor_pair_set(true_context, restricted_labels) == {
        pair
        for pair in _positive_set(_reference_ancestor_pairs(true_tree))
        if pair[0] in restricted_labels and pair[1] in restricted_labels
    }
    assert ancestors_unique_restricted_metrics(
        true_context,
        rec_context,
        restrict_labels=restricted_labels,
    ) == _reference_evaluate_4(
        true_tree,
        rec_tree,
        restrict_labels=restricted_labels,
    )["ancestors_unique_restricted"]


def test_unique_ancestor_pair_set_can_intern_labels_without_public_semantic_change():
    true_tree = _example_tree()
    context = tree_evaluation_context(true_tree)
    restricted_labels = {"A", "B", "D"}
    public_pairs = unique_ancestor_pair_set(context, restricted_labels)

    label_to_id = {}
    interned_pairs = unique_ancestor_pair_set(
        context,
        restricted_labels,
        label_to_id=label_to_id,
    )

    assert public_pairs == {
        pair
        for pair in _positive_set(_reference_ancestor_pairs(true_tree))
        if pair[0] in restricted_labels and pair[1] in restricted_labels
    }
    assert interned_pairs == {
        (label_to_id[ancestor], label_to_id[descendant])
        for ancestor, descendant in public_pairs
    }
    assert ancestors_unique_restricted_metrics(
        context,
        context,
        restrict_labels=restricted_labels,
    ) == _reference_evaluate_4(
        true_tree,
        true_tree,
        restrict_labels=restricted_labels,
    )["ancestors_unique_restricted"]


def _reference_pair_id(ancestor_id, descendant_id):
    pair_sum = ancestor_id + descendant_id
    return (pair_sum * (pair_sum + 1) // 2) + descendant_id


def test_adf1_pair_id_path_and_cache_match_public_restricted_semantics():
    true_tree = _example_tree()
    rec_tree = _example_tree()
    rec_tree.remove_edge("branch", "leaf")
    rec_tree.add_edge("internal", "leaf")
    restricted_labels = {"A", "B", "D"}
    true_context = tree_evaluation_context(true_tree)
    rec_context = tree_evaluation_context(rec_tree)

    label_to_id = {}
    public_pairs = unique_ancestor_pair_set(true_context, restricted_labels)
    pair_ids = unique_ancestor_pair_id_set(
        true_context,
        restricted_labels,
        label_to_id=label_to_id,
    )

    assert pair_ids == {
        _reference_pair_id(label_to_id[ancestor], label_to_id[descendant])
        for ancestor, descendant in public_pairs
    }

    cache = RestrictedAdf1Cache()
    expected = _reference_evaluate_4(
        true_tree,
        rec_tree,
        restrict_labels=restricted_labels,
    )["ancestors_unique_restricted"]
    first = adf1_restricted_metrics_from_contexts(
        true_context,
        rec_context,
        restrict_labels=restricted_labels,
        cache=cache,
    )
    cached_pair_set = next(iter(cache.true_pair_ids_by_restricted_labels.values()))
    second = adf1_restricted_metrics_from_contexts(
        true_context,
        rec_context,
        restrict_labels=["D", "B", "A"],
        cache=cache,
    )

    assert first == expected
    assert second == expected
    assert len(cache.true_pair_ids_by_restricted_labels) == 1
    assert next(iter(cache.true_pair_ids_by_restricted_labels.values())) is cached_pair_set


def test_adf1_pair_id_path_preserves_same_label_ancestor_descendant_pair():
    tree = nx.DiGraph()
    tree.add_node("root", cell_id="A")
    tree.add_node("child", cell_id="A")
    tree.add_node("leaf", cell_id="B")
    tree.add_edges_from([("root", "child"), ("child", "leaf")])
    context = tree_evaluation_context(tree)
    restricted_labels = {"A", "B"}

    assert adf1_restricted_metrics_from_contexts(
        context,
        context,
        restrict_labels=restricted_labels,
        cache=RestrictedAdf1Cache(),
    ) == _reference_evaluate_4(
        tree,
        tree,
        restrict_labels=restricted_labels,
    )["ancestors_unique_restricted"]
