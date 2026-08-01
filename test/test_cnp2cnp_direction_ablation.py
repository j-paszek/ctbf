import networkx as nx
import numpy as np

from cnp2cnp_direction_ablation import (
    DIRECTION_AUDIT_SCHEMA_VERSION,
    audit_directed_distances,
    audit_fast_reconstruction_sensitivity,
    audit_fast_row_order_sensitivity,
    build_three_arm_temporal_trees,
    canonical_fast_row_order,
    evaluate_three_arm_temporal_trees,
    ordered_triangle_fast_view,
    timed_distance_compute,
)
from distance_semantics import DirectedDistanceBundle
from simulator import Genotype


def _cell(genome, node_id, cell_id):
    return Genotype(genome, node_id=node_id, cell_id=cell_id)


def _label_edges(tree):
    return {
        (tree.nodes[parent]["cell_id"], tree.nodes[child]["cell_id"])
        for parent, child in tree.edges()
    }


def test_fast_view_is_canonical_by_default_and_sensitive_to_recorded_order():
    bundle = DirectedDistanceBundle(
        [2, 1, 3],
        np.array([
            [0.0, 3.0, 4.0],
            [7.0, 0.0, 2.0],
            [6.0, 5.0, 0.0],
        ]),
    )

    assert canonical_fast_row_order(bundle.ids) == (1, 2, 3)
    canonical_ids, canonical = ordered_triangle_fast_view(bundle)
    reverse_ids, reverse = ordered_triangle_fast_view(bundle, [3, 2, 1])
    sensitivity = audit_fast_row_order_sensitivity(
        bundle,
        [canonical_ids, reverse_ids],
    )

    assert canonical_ids == [1, 2, 3]
    assert np.array_equal(
        canonical,
        np.array([
            [0.0, 7.0, 2.0],
            [7.0, 0.0, 4.0],
            [2.0, 4.0, 0.0],
        ]),
    )
    assert not np.array_equal(canonical, reverse)
    assert sensitivity["any_matrix_change"] is True
    assert sensitivity["orders"][1]["changed_unordered_pairs_vs_first"] == 3

    tree_sensitivity = audit_fast_reconstruction_sensitivity(
        [[
            _cell([3], 20, 2),
            _cell([2], 10, 1),
            _cell([4], 30, 3),
        ]],
        bundle,
        [canonical_ids, reverse_ids],
        seed=5,
        use_time=False,
    )
    assert len(tree_sensitivity["orders"]) == 2
    assert tree_sensitivity["changed_order_count_vs_first"] in {0, 1}


def test_direction_audit_stratifies_zeros_and_excludes_decided_truth_pairs():
    bundle = DirectedDistanceBundle(
        [1, 2, 3, 4],
        np.array([
            [0.0, 1.0, 4.0, 5.0],
            [3.0, 0.0, 2.0, 5.0],
            [0.0, 6.0, 0.0, 1.0],
            [4.0, 4.0, 2.0, 0.0],
        ]),
        provenance={"semantics_version": "test"},
    )
    genomes = {
        1: [2, 2],
        2: [3, 2],
        3: [0, 2],
        4: [1, 0],
    }

    audit = audit_directed_distances(
        bundle,
        genomes,
        truth_directions=[(1, 2), (1, 3), (2, 4)],
        time_decided_pairs=[(2, 4)],
    )

    assert audit["schema_version"] == DIRECTION_AUDIT_SCHEMA_VERSION
    assert audit["unordered_pair_count"] == 6
    assert audit["asymmetric_pair_count"] == 6
    assert audit["profile_strata"] == {"all_positive": 1, "contains_zero": 5}
    assert audit["plausibility_strata"]["both_plausible"]["pairs"] >= 1
    assert audit["plausibility_strata"]["neither_plausible"]["pairs"] >= 1
    truth = audit["truth_direction"]
    assert truth["provided"] == 3
    assert truth["excluded_time_decided"] == 1
    assert truth["excluded_plausibility_decided"] == 1
    assert truth["sign_informative"] == 1
    assert truth["correct"] == 1
    assert truth["sign_accuracy"] == 1.0
    assert truth["false_direction_rate"] == 0.0
    assert truth["sign_coverage"] == 1.0
    assert truth["accuracy_by_absolute_difference"] == {
        "2.0": {"pairs": 1, "correct": 1, "accuracy": 1.0}
    }

    class FixedProvider:
        def compute(self, cells):
            assert cells == ["cell"]
            return bundle

    timed_bundle, timing = timed_distance_compute(FixedProvider(), ["cell"])
    assert timed_bundle is bundle
    assert timing["distance_wall_time_ns"] >= 0
    assert timing["distance_provenance"] == {"semantics_version": "test"}


def test_three_arm_builder_keeps_truth_outside_reconstruction_inputs():
    cell_lists = [[
        _cell([2], 100, 1),
        _cell([3], 200, 2),
    ]]
    original_node_ids = [cell.node_id for cell in cell_lists[0]]
    bundle = DirectedDistanceBundle([1, 2], [[0.0, 9.0], [1.0, 0.0]])

    arms = build_three_arm_temporal_trees(
        cell_lists,
        bundle,
        seed=0,
        fast_row_order=[1, 2],
        use_time=False,
    )

    assert set(arms) == {
        "ordered_triangle_fast",
        "minimum_bidirectional",
        "minimum_with_directed",
    }
    assert all(nx.is_arborescence(result[0]) for result in arms.values())
    assert _label_edges(arms["minimum_bidirectional"][0]) == {(1, 2)}
    assert _label_edges(arms["minimum_with_directed"][0]) == {(2, 1)}
    assert [cell.node_id for cell in cell_lists[0]] == original_node_ids
    assert all(
        "truth_node_id" not in data and "observation_key" not in data
        for result in arms.values()
        for _, data in result[0].nodes(data=True)
    )

    truth = nx.DiGraph()
    truth.add_node(10, cell_id=1, genome=np.array([2]))
    truth.add_node(20, cell_id=2, genome=np.array([3]))
    truth.add_edge(10, 20)
    reports = evaluate_three_arm_temporal_trees(
        truth,
        10,
        arms,
        observed_labels={"1", "2"},
    )
    assert reports["minimum_bidirectional"]["ad_f1"] == 1.0
    assert reports["minimum_with_directed"]["ad_f1"] < 1.0
    assert all("grf" in report for report in reports.values())
