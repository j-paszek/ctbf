from collections import Counter
from copy import deepcopy
import random

import networkx as nx
import pytest

from evaluation_contract import EvaluationContractError, evaluate_tree_pair, metric_contract_descriptor
from evaluator_full import unique_ancestor_pair_set
from trimmed_truth_evaluation import (
    evaluate_prepared_trimmed_truth,
    prepare_trimmed_truth,
    trim_true_tree,
)


def tree(labels, edges):
    result = nx.DiGraph()
    result.add_nodes_from((node, {"cell_id": label}) for node, label in labels.items())
    result.add_edges_from(edges)
    return result


def owner_uxz():
    return tree(
        {0: "u", 1: "y", 2: "u", 3: "x", 4: "z", 5: "v", 6: "u"},
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
    )


def owner_bda():
    return tree(
        {0: "a", 1: "b", 2: "a", 3: "c", 4: "d", 5: "a",
         6: "c", 7: "e", 8: "d", 9: "a"},
        [(0, 1), (0, 2), (1, 3), (3, 6), (3, 7), (2, 4), (2, 5), (4, 8), (5, 9)],
    )


def signature(graph):
    return dict(graph.nodes(data="cell_id")), set(graph.edges)


def test_owner_uxz_keeps_all_u_and_unlabeled_y_fork():
    source = owner_uxz()
    result = trim_true_tree(source, {"u", "x", "z"})
    assert set(result) == {0, 1, 2, 3, 4, 6}
    assert set(result.edges) == {(0, 1), (0, 2), (1, 3), (1, 4), (2, 6)}
    assert result.nodes[1]["cell_id"] is None
    assert Counter(nx.get_node_attributes(result, "cell_id").values()) == {
        "u": 3, "x": 1, "z": 1, None: 1}
    assert source.nodes[1]["cell_id"] == "y" and 5 in source


def test_owner_bda_keeps_unsampled_repetitions_and_labeled_unary_paths():
    result = trim_true_tree(owner_bda(), {"b", "d", "a"})
    assert set(result) == {0, 1, 2, 4, 5, 8, 9}
    assert set(result.edges) == {(0, 1), (0, 2), (2, 4), (2, 5), (4, 8), (5, 9)}
    assert Counter(nx.get_node_attributes(result, "cell_id").values()) == {"a": 4, "d": 2, "b": 1}


def test_unlabeled_root_is_promoted_repeatedly_and_unary_paths_suppressed():
    source = tree({0: "outside", 1: None, 2: "a", 3: "hidden", 4: "a", 5: "gone"},
                  [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)])
    result = trim_true_tree(source, ["a"])
    assert signature(result) == ({2: "a", 4: "a"}, {(2, 4)})


def test_unlabeled_branching_root_is_not_promoted_and_leaf_root_is_valid():
    source = tree({0: "hidden", 1: "a", 2: "b"}, [(0, 1), (0, 2)])
    both = trim_true_tree(source, ["a", "b"])
    assert both.nodes[0]["cell_id"] is None and both.out_degree(0) == 2
    assert signature(trim_true_tree(source, ["a"])) == ({1: "a"}, set())


def test_label_normalization_no_magic_empty_state_and_no_mutation():
    source = tree({0: 7, 1: " 7 ", 2: "-", 3: " "}, [(0, 1), (1, 2), (1, 3)])
    source.nodes[0]["genome"] = [2, 2]
    source.edges[0, 1]["weight"] = 5
    source.graph["metadata"] = {"unchanged": [1]}
    before = deepcopy(source)
    result = trim_true_tree(source, (value for value in [" 7", 7, "-", None]))
    assert signature(result) == ({0: "7", 1: "7", 2: "-"}, {(0, 1), (1, 2)})
    assert nx.utils.graphs_equal(source, before)
    assert all(set(data) == {"cell_id"} for _, data in result.nodes(data=True))
    result.nodes[0]["cell_id"] = "changed"
    assert nx.utils.graphs_equal(source, before)


@pytest.mark.parametrize("observed", [None, [], [None, " "]])
def test_empty_observation_set_is_rejected(observed):
    with pytest.raises(EvaluationContractError, match="nonempty"):
        trim_true_tree(tree({1: "a"}, []), observed)


def test_missing_labels_and_invalid_trees_are_not_silently_repaired():
    with pytest.raises(EvaluationContractError) as error:
        trim_true_tree(tree({1: "a"}, []), ["b"])
    assert error.value.code == "observation_labels_missing_from_truth"
    invalid = [nx.DiGraph(), tree({0: "a", 1: "b"}, []),
               tree({0: "a", 1: "b"}, [(0, 1), (1, 0)]),
               nx.MultiDiGraph(tree({0: "a"}, []))]
    for graph in invalid:
        with pytest.raises(EvaluationContractError):
            trim_true_tree(graph, ["a"])


def test_deep_tree_is_iterative():
    size = 10000
    source = nx.DiGraph()
    source.add_nodes_from((i, {"cell_id": "a" if i == size - 1 else "hidden"}) for i in range(size))
    source.add_edges_from((i, i + 1) for i in range(size - 1))
    assert signature(trim_true_tree(source, ["a"])) == ({size - 1: "a"}, set())


@pytest.mark.parametrize("example,observed", [(owner_uxz, ["u", "x", "z"]), (owner_bda, ["b", "d", "a"])])
def test_scoring_reuses_full_score_and_preserves_reconstruction_and_ad(example, observed):
    source = example()
    reconstructed = trim_true_tree(source, observed)
    # An extra unlabeled unary root must remain in the evaluated reconstruction.
    root = next(node for node, degree in reconstructed.in_degree() if degree == 0)
    reconstructed.add_node("extra", cell_id=None)
    reconstructed.add_edge("extra", root)
    before = deepcopy(reconstructed)
    frozen = evaluate_tree_pair(source, reconstructed, observed)
    frozen_before = deepcopy(frozen)
    prepared = prepare_trimmed_truth(source, observed)
    result = evaluate_prepared_trimmed_truth(prepared, reconstructed, frozen)
    assert result["grf_full"] == frozen["metrics"]["grf"]
    assert result["grf_trim"] == evaluate_tree_pair(prepared.trimmed.context, reconstructed, observed)["metrics"]["grf"]
    assert result["ad_fields_match_frozen"] and result["ad_relation_set_invariant"]
    assert not result["full_grf_recomputed"]
    assert frozen == frozen_before and nx.utils.graphs_equal(reconstructed, before)
    assert result["trimmed_native_evaluation"]["inputs"]["reconstructed_tree"]["node_count"] == len(before)
    assert result["trimmed_native_evaluation"]["metric_contract"] == metric_contract_descriptor()


def test_random_small_trees_keep_exact_occurrences_ancestry_and_are_idempotent():
    rng = random.Random(511)
    for size in range(1, 40):
        source = tree({i: rng.choice("abcd") for i in range(size)},
                      [(rng.randrange(i), i) for i in range(1, size)])
        labels = list(set(nx.get_node_attributes(source, "cell_id").values()))
        observed = sorted(labels)[::2]
        result = trim_true_tree(source, observed)
        retained = {node for node, label in source.nodes(data="cell_id") if label in observed}
        assert {node for node, label in result.nodes(data="cell_id") if label is not None} == retained
        for u in retained:
            assert nx.descendants(source, u) & retained == nx.descendants(result, u) & retained
        assert unique_ancestor_pair_set(source, observed) == unique_ancestor_pair_set(result, observed)
        assert signature(trim_true_tree(result, observed)) == signature(result)
        permuted = tree(dict(reversed(list(source.nodes(data="cell_id")))), list(reversed(list(source.edges))))
        assert signature(trim_true_tree(permuted, observed)) == signature(result)
        assert all(label is not None or result.out_degree(node) >= 2 for node, label in result.nodes(data="cell_id"))


def test_unchanged_truth_and_source_identity_mismatch():
    source = tree({0: "a", 1: "a"}, [(0, 1)])
    prepared = prepare_trimmed_truth(source, ["a"])
    frozen = evaluate_tree_pair(source, source, ["a"])
    result = evaluate_prepared_trimmed_truth(prepared, source, frozen)
    assert result["grf_trim"] == result["grf_full"] == 1.0
    assert result["ad_f1"] == 1.0
    changed = tree({0: "a"}, [])
    with pytest.raises(EvaluationContractError):
        evaluate_prepared_trimmed_truth(prepared, changed, frozen)


def test_degenerate_ad_case_retains_native_empty_empty_zero():
    source = tree({0: "a"}, [])
    result = evaluate_prepared_trimmed_truth(prepare_trimmed_truth(source, ["a"]),
                                            source, evaluate_tree_pair(source, source, ["a"]))
    assert result["ad_f1"] == 0.0
    assert result["trimmed_native_evaluation"]["metrics"]["ad_f1_degenerate"]
