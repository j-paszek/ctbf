from collections import Counter
import random

import networkx as nx
import pytest

import evaluator
from evaluator import (
    EXT_GRF_HIGHER_IS_BETTER,
    EXT_GRF_METRIC_KIND,
    EXT_GRF_METRIC_NAME,
    cluster_comparison_work,
    cluster_evaluation_context,
    compute_all_clusters,
    exact_and_legacy_grf_from_cluster_contexts,
    ext_grf,
    ext_grf_from_cluster_counts,
    ext_grf_tree,
    grf_comparison_metadata,
    grf,
    grf_tree,
    legacy_set_grf_similarity_from_cluster_contexts,
    parse_newick_to_nx,
    weighted_jaccard_distance_sum,
)
from evaluator_full import tree_evaluation_context


LEAF_COUNTS = (5, 10, 20, 50, 100)
CASES_PER_LEAF_COUNT = 20
UNEQUAL_EDGE_CASES = 20


def _reference_clusters(tree, root):
    clusters = []

    def dfs(node):
        labels = Counter()
        cell_id = tree.nodes[node].get("cell_id")
        if cell_id is not None:
            labels[cell_id] += 1
        for child in tree.successors(node):
            labels += dfs(child)
        clusters.append(tuple(sorted(labels.items())))
        return labels

    dfs(root)
    return clusters


def _reference_jaccard_distance(left, right):
    left_counts = dict(left)
    right_counts = dict(right)
    labels = set(left_counts) | set(right_counts)
    union = sum(max(left_counts.get(label, 0), right_counts.get(label, 0)) for label in labels)
    if union == 0:
        return 0.0
    intersection = sum(
        min(left_counts.get(label, 0), right_counts.get(label, 0)) for label in labels
    )
    return 1 - (intersection / union)


def _reference_difference_counts(left_counts, right_counts):
    return Counter(
        {
            item: count - right_counts.get(item, 0)
            for item, count in left_counts.items()
            if count > right_counts.get(item, 0)
        }
    )


def _reference_union_size(left_counts, right_counts):
    return sum(
        max(left_counts.get(item, 0), right_counts.get(item, 0))
        for item in set(left_counts) | set(right_counts)
    )


def _reference_weighted_sum(left_counts, right_counts):
    return sum(
        left_count * right_count * _reference_jaccard_distance(left_cluster, right_cluster)
        for left_cluster, left_count in left_counts.items()
        for right_cluster, right_count in right_counts.items()
    )


def _reference_ext_grf_tree(left_tree, left_root, right_tree, right_root):
    left_counts = Counter(_reference_clusters(left_tree, left_root))
    right_counts = Counter(_reference_clusters(right_tree, right_root))
    left_size = sum(left_counts.values())
    right_size = sum(right_counts.values())
    union_size = _reference_union_size(left_counts, right_counts)
    if union_size == 0:
        return 0.0
    if left_size == 0 or right_size == 0:
        return 1.0

    right_minus_left = _reference_difference_counts(right_counts, left_counts)
    left_minus_right = _reference_difference_counts(left_counts, right_counts)
    numerator_1 = _reference_weighted_sum(left_counts, right_minus_left)
    numerator_2 = _reference_weighted_sum(left_minus_right, right_counts)
    return (numerator_1 / (left_size * union_size)) + (
        numerator_2 / (right_size * union_size)
    )


def _legacy_set_based_distance(left_tree, left_root, right_tree, right_root):
    left = _reference_clusters(left_tree, left_root)
    right = _reference_clusters(right_tree, right_root)
    left_set = set(left)
    right_set = set(right)
    union_size = len(left_set | right_set)
    numerator_1 = sum(
        _reference_jaccard_distance(left_cluster, right_cluster)
        for left_cluster in left
        for right_cluster in right
        if right_cluster not in left_set
    )
    numerator_2 = sum(
        _reference_jaccard_distance(right_cluster, left_cluster)
        for right_cluster in right
        for left_cluster in left
        if left_cluster not in right_set
    )
    return (numerator_1 / (len(left) * union_size)) + (
        numerator_2 / (len(right) * union_size)
    )


def _make_recurrent_binary_tree(leaf_count, seed, label_mod=None):
    rng = random.Random(seed)
    tree = nx.DiGraph()
    active = []
    label_count = label_mod or max(2, min(leaf_count, max(3, leaf_count // 4)))

    for leaf_index in range(leaf_count):
        node_id = f"leaf_{leaf_index}"
        tree.add_node(node_id, cell_id=f"L{(leaf_index + seed) % label_count}")
        active.append(node_id)

    next_internal = 0
    while len(active) > 1:
        left_index = rng.randrange(len(active))
        left = active.pop(left_index)
        right_index = rng.randrange(len(active))
        right = active.pop(right_index)

        parent = f"internal_{next_internal}"
        next_internal += 1
        if rng.random() < 0.35:
            label = None
        else:
            label = f"L{rng.randrange(label_count)}"
        tree.add_node(parent, cell_id=label)
        tree.add_edge(parent, left)
        tree.add_edge(parent, right)
        active.append(parent)

    return tree, active[0]


def _leaf_count(tree):
    return sum(1 for node in tree.nodes if tree.out_degree(node) == 0)


def _make_same_size_cases():
    cases = []
    for leaf_count in LEAF_COUNTS:
        for index in range(CASES_PER_LEAF_COUNT):
            left_tree, left_root = _make_recurrent_binary_tree(
                leaf_count, seed=10_000 + leaf_count * 100 + index
            )
            right_tree, right_root = _make_recurrent_binary_tree(
                leaf_count, seed=20_000 + leaf_count * 100 + index
            )
            distance = _reference_ext_grf_tree(left_tree, left_root, right_tree, right_root)
            cases.append(
                {
                    "id": f"same_size_n{leaf_count}_{index:02d}",
                    "leaf_count": leaf_count,
                    "left_tree": left_tree,
                    "left_root": left_root,
                    "right_tree": right_tree,
                    "right_root": right_root,
                    "ext_grf": distance,
                    "grf": 1 - distance,
                }
            )
    return cases


def _make_unequal_edge_cases():
    cases = []
    for index in range(UNEQUAL_EDGE_CASES):
        left_tree, left_root = _make_recurrent_binary_tree(
            7, seed=30_000 + index, label_mod=3
        )
        right_tree, right_root = _make_recurrent_binary_tree(
            10, seed=40_000 + index, label_mod=3
        )
        distance = _reference_ext_grf_tree(left_tree, left_root, right_tree, right_root)
        cases.append(
            {
                "id": f"unequal_recurrent_7_vs_10_{index:02d}",
                "left_leaf_count": 7,
                "right_leaf_count": 10,
                "left_tree": left_tree,
                "left_root": left_root,
                "right_tree": right_tree,
                "right_root": right_root,
                "ext_grf": distance,
                "grf": 1 - distance,
            }
        )
    return cases


SAME_SIZE_CASES = _make_same_size_cases()
UNEQUAL_CASES = _make_unequal_edge_cases()
BUG_EXPOSING_NEWICK_CASES = [
    {
        "id": "three_repeated_leaves",
        "left": "(a,a,a);",
        "right": "(a,a,b);",
        "ext_grf": 11 / 24,
        "legacy_distance": 7 / 12,
    },
    {
        "id": "nested_repeated_left_leaf",
        "left": "((a,a),b);",
        "right": "((a,b),b);",
        "ext_grf": 29 / 60,
        "legacy_distance": 5 / 12,
    },
    {
        "id": "asymmetric_repeated_subtree",
        "left": "(a,(a,a));",
        "right": "(a,(a,b));",
        "ext_grf": 39 / 80,
        "legacy_distance": 26 / 45,
    },
    {
        "id": "recurrent_internal_label",
        "left": "((a,a)a,b);",
        "right": "((a,b)a,b);",
        "ext_grf": 119 / 240,
        "legacy_distance": 19 / 45,
    },
    {
        "id": "two_repeated_subtrees_with_internal_labels",
        "left": "((a,a)c,(b,b)d)e;",
        "right": "((a,b)c,(b,b)d)e;",
        "ext_grf": 1592 / 3675,
        "legacy_distance": 2129 / 5145,
    },
]


def test_ext_grf_metadata_identifies_distance_direction():
    assert EXT_GRF_METRIC_NAME == "ext_grf"
    assert EXT_GRF_METRIC_KIND == "distance"
    assert EXT_GRF_HIGHER_IS_BETTER is False


def test_recurrent_internal_labels_are_counted_in_clusters():
    tree, root = parse_newick_to_nx("(a,a)a;")

    clusters = Counter(compute_all_clusters(tree, root))

    assert clusters[(("a", 1),)] == 2
    assert clusters[(("a", 3),)] == 1


def test_ext_grf_uses_outer_multiset_counts_for_recurrent_labels():
    left_tree, left_root = parse_newick_to_nx("(a,a);", prefix="left")
    right_tree, right_root = parse_newick_to_nx("(a,b);", prefix="right")

    assert ext_grf_tree(left_tree, left_root, right_tree, right_root) == pytest.approx(5 / 9)
    assert grf_tree(left_tree, left_root, right_tree, right_root) == pytest.approx(4 / 9)
    assert ext_grf("(a,a);", "(a,b);") == pytest.approx(5 / 9)
    assert grf("(a,a);", "(a,b);") == pytest.approx(4 / 9)
    assert _legacy_set_based_distance(left_tree, left_root, right_tree, right_root) == pytest.approx(
        41 / 72
    )


def test_weighted_jaccard_sum_uses_compact_views_without_changing_values():
    left_counts = Counter({
        (("a", 2), ("b", 1)): 3,
        (("c", 4),): 2,
        tuple(): 1,
    })
    right_counts = Counter({
        (("a", 1), ("b", 3)): 5,
        (("d", 2),): 7,
        tuple(): 2,
    })

    assert weighted_jaccard_distance_sum(left_counts, right_counts) == pytest.approx(
        _reference_weighted_sum(left_counts, right_counts)
    )
    assert weighted_jaccard_distance_sum(
        left_counts,
        right_counts,
        work=cluster_comparison_work(left_counts, right_counts),
    ) == pytest.approx(_reference_weighted_sum(left_counts, right_counts))


def test_dense_weighted_jaccard_kernel_matches_reference(monkeypatch):
    def cluster(offset):
        labels = ((f"L{(offset + step) % 7}", step + 1) for step in range(3))
        return tuple(sorted(labels))

    left_counts = Counter({cluster(index): (index % 3) + 1 for index in range(12)})
    right_counts = Counter({cluster(index + 4): (index % 5) + 1 for index in range(14)})

    monkeypatch.setattr(evaluator, "_DENSE_PAIR_THRESHOLD", 1)

    assert weighted_jaccard_distance_sum(left_counts, right_counts) == pytest.approx(
        _reference_weighted_sum(left_counts, right_counts)
    )
    assert weighted_jaccard_distance_sum(
        left_counts,
        right_counts,
        work=cluster_comparison_work(left_counts, right_counts),
    ) == pytest.approx(_reference_weighted_sum(left_counts, right_counts))


def test_ext_grf_uses_adaptive_weighted_jaccard_sum(monkeypatch):
    left_counts = Counter({
        (("a", 2),): 2,
        (("a", 1), ("b", 1)): 1,
    })
    right_counts = Counter({
        (("a", 1),): 3,
        (("b", 2),): 1,
    })

    monkeypatch.setattr(evaluator, "_DENSE_PAIR_THRESHOLD", 1)

    union_size = _reference_union_size(left_counts, right_counts)
    expected = (
        _reference_weighted_sum(left_counts, _reference_difference_counts(right_counts, left_counts))
        / (sum(left_counts.values()) * union_size)
    ) + (
        _reference_weighted_sum(_reference_difference_counts(left_counts, right_counts), right_counts)
        / (sum(right_counts.values()) * union_size)
    )

    assert ext_grf_from_cluster_counts(left_counts, right_counts) == pytest.approx(expected)


def test_context_cluster_construction_matches_networkx_reference_after_counter_reuse():
    tree, root = parse_newick_to_nx("((a,a)c,(b,(a,b)c)d)e;", prefix="ctx")
    context = tree_evaluation_context(tree)

    assert Counter(evaluator.compute_all_clusters_from_context(context, root)) == Counter(
        _reference_clusters(tree, root)
    )


def test_shared_grf_work_matches_exact_and_legacy_references(monkeypatch):
    left_tree, left_root = parse_newick_to_nx("((a,a)c,(b,b)d)e;", prefix="left")
    right_tree, right_root = parse_newick_to_nx("((a,b)c,(b,b)d)e;", prefix="right")
    left_context = cluster_evaluation_context(left_tree, left_root)
    right_context = cluster_evaluation_context(right_tree, right_root)
    work = cluster_comparison_work(left_context.counts, right_context.counts)
    metadata = grf_comparison_metadata(
        left_context.counts,
        right_context.counts,
        left_context.cluster_set,
        right_context.cluster_set,
    )

    monkeypatch.setattr(evaluator, "_DENSE_PAIR_THRESHOLD", 1)

    ext_value = ext_grf_from_cluster_counts(
        left_context.counts,
        right_context.counts,
        work=work,
        metadata=metadata,
        jaccard_cache={},
    )
    legacy_similarity = legacy_set_grf_similarity_from_cluster_contexts(
        left_context,
        right_context,
        work=work,
        metadata=metadata,
        jaccard_cache={},
    )
    combined_ext, combined_legacy = exact_and_legacy_grf_from_cluster_contexts(
        left_context,
        right_context,
        jaccard_cache={},
    )

    assert ext_value == pytest.approx(_reference_ext_grf_tree(left_tree, left_root, right_tree, right_root))
    assert legacy_similarity == pytest.approx(
        1 - _legacy_set_based_distance(left_tree, left_root, right_tree, right_root)
    )
    assert combined_ext == pytest.approx(ext_value)
    assert combined_legacy == pytest.approx(legacy_similarity)


@pytest.mark.parametrize("case", BUG_EXPOSING_NEWICK_CASES, ids=lambda case: case["id"])
def test_ext_grf_bug_exposing_examples_use_outer_multiset_counts(case):
    left_tree, left_root = parse_newick_to_nx(case["left"], prefix="left")
    right_tree, right_root = parse_newick_to_nx(case["right"], prefix="right")

    exact_distance = ext_grf_tree(left_tree, left_root, right_tree, right_root)
    legacy_distance = _legacy_set_based_distance(left_tree, left_root, right_tree, right_root)

    assert exact_distance == pytest.approx(case["ext_grf"])
    assert grf_tree(left_tree, left_root, right_tree, right_root) == pytest.approx(
        1 - case["ext_grf"]
    )
    assert legacy_distance == pytest.approx(case["legacy_distance"])
    assert abs(legacy_distance - exact_distance) > 1e-12


@pytest.mark.parametrize("case", SAME_SIZE_CASES, ids=lambda case: case["id"])
def test_ext_grf_generated_same_size_cases(case):
    left_tree = case["left_tree"]
    right_tree = case["right_tree"]
    left_root = case["left_root"]
    right_root = case["right_root"]

    assert _leaf_count(left_tree) == case["leaf_count"]
    assert _leaf_count(right_tree) == case["leaf_count"]
    assert ext_grf_tree(left_tree, left_root, right_tree, right_root) == pytest.approx(
        case["ext_grf"]
    )
    assert ext_grf_tree(right_tree, right_root, left_tree, left_root) == pytest.approx(
        case["ext_grf"]
    )
    assert grf_tree(left_tree, left_root, right_tree, right_root) == pytest.approx(case["grf"])
    assert 0 <= case["ext_grf"] <= 1
    assert 0 <= case["grf"] <= 1


@pytest.mark.parametrize("case", UNEQUAL_CASES, ids=lambda case: case["id"])
def test_ext_grf_generated_unequal_recurrent_label_cases(case):
    left_tree = case["left_tree"]
    right_tree = case["right_tree"]
    left_root = case["left_root"]
    right_root = case["right_root"]

    assert _leaf_count(left_tree) == case["left_leaf_count"]
    assert _leaf_count(right_tree) == case["right_leaf_count"]
    assert ext_grf_tree(left_tree, left_root, right_tree, right_root) == pytest.approx(
        case["ext_grf"]
    )
    assert ext_grf_tree(right_tree, right_root, left_tree, left_root) == pytest.approx(
        case["ext_grf"]
    )
    assert grf_tree(left_tree, left_root, right_tree, right_root) == pytest.approx(case["grf"])
    assert 0 <= case["ext_grf"] <= 1
    assert 0 <= case["grf"] <= 1
