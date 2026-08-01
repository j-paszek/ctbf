import itertools

import networkx as nx
import numpy as np
import pytest

from reconstructor_nj import (
    neighbor_joining_classical,
    neighbor_joining_standard,
    rooted_labeled_nj,
)
from simulator import Genotype


ADDITIVE_FOUR_TAXON_MATRIX = np.array(
    [
        [0.0, 5.0, 9.0, 9.0],
        [5.0, 0.0, 10.0, 10.0],
        [9.0, 10.0, 0.0, 8.0],
        [9.0, 10.0, 8.0, 0.0],
    ]
)


def _cells(labels=("A", "B", "C", "D")):
    return [
        Genotype([2, index], node_id=index, cell_id=label)
        for index, label in enumerate(labels)
    ]


def _root_nodes(tree):
    return [node for node in tree if tree.in_degree(node) == 0]


def _nontrivial_unrooted_label_splits(tree):
    undirected = tree.to_undirected()
    all_labels = {
        data["cell_id"]
        for _, data in tree.nodes(data=True)
        if data.get("cell_id") is not None
    }
    splits = set()
    for left, right in undirected.edges:
        without_edge = undirected.copy()
        without_edge.remove_edge(left, right)
        side_nodes = nx.node_connected_component(without_edge, left)
        side_labels = {
            tree.nodes[node].get("cell_id")
            for node in side_nodes
            if tree.nodes[node].get("cell_id") is not None
        }
        other_labels = all_labels - side_labels
        if len(side_labels) > 1 and len(other_labels) > 1:
            splits.add(frozenset((frozenset(side_labels), frozenset(other_labels))))
    return splits


def _weighted_edges(tree):
    return sorted(
        (parent, child, float(data["weight"]))
        for parent, child, data in tree.edges(data=True)
    )


def test_classical_nj_recovers_known_additive_split_and_limb_lengths():
    cells = _cells()

    tree, new_nodes, root = neighbor_joining_classical(
        ADDITIVE_FOUR_TAXON_MATRIX,
        cells,
        max_id=3,
    )

    assert root == -1
    assert _root_nodes(tree) == [root]
    assert nx.is_arborescence(tree)
    assert _nontrivial_unrooted_label_splits(tree) == {
        frozenset((frozenset(("A", "B")), frozenset(("C", "D"))))
    }
    assert tree.edges[next(tree.predecessors(0)), 0]["weight"] == pytest.approx(2.0)
    assert tree.edges[next(tree.predecessors(1)), 1]["weight"] == pytest.approx(3.0)
    assert tree.edges[next(tree.predecessors(2)), 2]["weight"] == pytest.approx(4.0)
    assert tree.edges[next(tree.predecessors(3)), 3]["weight"] == pytest.approx(4.0)
    assert all(
        tree.nodes[node]["cell_id"] is None
        for node in tree
        if node not in {0, 1, 2, 3}
    )
    assert len(new_nodes) == 3


@pytest.mark.parametrize("permutation", list(itertools.permutations(range(4))))
def test_classical_nj_additive_split_is_permutation_equivalent(permutation):
    cells = _cells()
    permuted_cells = [cells[index] for index in permutation]
    permuted_matrix = ADDITIVE_FOUR_TAXON_MATRIX[np.ix_(permutation, permutation)]

    tree, _, _ = neighbor_joining_classical(permuted_matrix, permuted_cells, max_id=3)

    assert _nontrivial_unrooted_label_splits(tree) == {
        frozenset((frozenset(("A", "B")), frozenset(("C", "D"))))
    }


def test_classical_nj_handles_zero_distance_matrix_without_selecting_diagonal():
    cells = _cells(("A", "B", "C"))
    matrix = np.zeros((3, 3), dtype=float)

    tree, _, root = neighbor_joining_classical(matrix, cells, max_id=2)

    assert root in tree
    assert nx.is_arborescence(tree)
    assert tree.number_of_nodes() == 5
    assert all(float(data["weight"]) == 0.0 for _, _, data in tree.edges(data=True))


def test_classical_nj_compatibility_name_has_identical_ordinary_output():
    cells = _cells()

    standard_tree, _, standard_root = neighbor_joining_standard(
        ADDITIVE_FOUR_TAXON_MATRIX,
        cells,
        max_id=3,
    )
    classical_tree, _, classical_root = neighbor_joining_classical(
        ADDITIVE_FOUR_TAXON_MATRIX,
        cells,
        max_id=3,
    )

    assert classical_root == standard_root
    assert _weighted_edges(classical_tree) == _weighted_edges(standard_tree)
    assert set(classical_tree) == set(standard_tree)
    for node in classical_tree:
        assert classical_tree.nodes[node]["cell_id"] == standard_tree.nodes[node]["cell_id"]
        np.testing.assert_array_equal(
            classical_tree.nodes[node]["genome"],
            standard_tree.nodes[node]["genome"],
        )


def test_classical_nj_uses_available_synthetic_root_id_on_collision():
    cells = [
        Genotype([2, 2], node_id=-1, cell_id="A"),
        Genotype([2, 1], node_id=0, cell_id="B"),
    ]
    matrix = np.array([[0.0, 2.0], [2.0, 0.0]])

    tree, _, root = neighbor_joining_classical(matrix, cells, max_id=0)

    assert root == 1
    assert tree.nodes[-1]["cell_id"] == "A"
    assert tree.nodes[root]["cell_id"] is None
    assert nx.is_arborescence(tree)


def test_classical_nj_rejects_empty_or_misaligned_inputs():
    with pytest.raises(ValueError, match="at least one cell"):
        neighbor_joining_classical(np.zeros((0, 0)), [], max_id=0)

    with pytest.raises(ValueError, match="must match"):
        neighbor_joining_classical(np.zeros((2, 2)), _cells(("A",)), max_id=0)


def test_classical_nj_does_not_mutate_input_matrix():
    cells = _cells()
    matrix = ADDITIVE_FOUR_TAXON_MATRIX.copy()
    original = matrix.copy()

    neighbor_joining_classical(matrix, cells, max_id=3)

    np.testing.assert_array_equal(matrix, original)


Q_DIFFERS_FROM_RAW_MINIMUM_MATRIX = np.array(
    [
        [0.0, 18.0, 13.0, 2.0, 1.0],
        [18.0, 0.0, 7.0, 17.0, 20.0],
        [13.0, 7.0, 0.0, 5.0, 3.0],
        [2.0, 17.0, 5.0, 0.0, 11.0],
        [1.0, 20.0, 3.0, 11.0, 0.0],
    ]
)


def test_rooted_labeled_nj_uses_q_pair_and_global_row_sum_orientation():
    cells = _cells(("A", "B", "C", "D", "E"))

    tree, new_nodes, root = rooted_labeled_nj(
        Q_DIFFERS_FROM_RAW_MINIMUM_MATRIX,
        cells,
        max_id=4,
        seed=7,
    )

    # Raw distance first favors A/E (distance 1); Q uniquely favors B/C.
    # C has the smaller original row sum, so the retained directed edge is C -> B.
    assert tree.has_edge(2, 1)
    assert tree.edges[2, 1]["weight"] == pytest.approx(7.0)
    assert root == 2
    assert new_nodes == {}
    assert set(tree) == {0, 1, 2, 3, 4}
    assert tree.number_of_edges() == 4
    assert nx.is_arborescence(tree)
    assert _root_nodes(tree) == [root]
    assert {tree.nodes[node]["cell_id"] for node in tree} == {"A", "B", "C", "D", "E"}


def test_rooted_labeled_nj_preserves_repeated_labels_as_distinct_occurrences():
    cells = [
        Genotype([2, 2], node_id=10, cell_id="A"),
        Genotype([2, 2], node_id=11, cell_id="A"),
        Genotype([2, 1], node_id=12, cell_id="B"),
    ]
    matrix = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    tree, new_nodes, root = rooted_labeled_nj(matrix, cells, max_id=12, seed=11)

    assert root in tree
    assert new_nodes == {}
    assert set(tree) == {10, 11, 12}
    assert [tree.nodes[node]["cell_id"] for node in (10, 11, 12)] == ["A", "A", "B"]
    assert nx.is_arborescence(tree)


def test_rooted_labeled_nj_exact_ties_are_seed_reproducible_and_inputs_are_owned():
    cells = _cells()
    matrix = np.zeros((4, 4), dtype=float)
    original = matrix.copy()

    first_tree, _, first_root = rooted_labeled_nj(matrix, cells, max_id=3, seed=19)
    second_tree, _, second_root = rooted_labeled_nj(matrix, cells, max_id=3, seed=19)

    assert first_root == second_root
    assert _weighted_edges(first_tree) == _weighted_edges(second_tree)
    np.testing.assert_array_equal(matrix, original)
