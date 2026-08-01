from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstructor import build_evolution_tree
from reconstructor_plausibility import is_biologically_plausible_ancestor
from reconstructor_temporal import (
    temporal_cnp_arborescence,
    temporal_cnp_arborescence_no_time,
    uses_ordered_occurrence_input,
)
from simulator import Genotype


def _cell(genome, node_id, cell_id, generation=None):
    return Genotype(
        genome,
        node_id=node_id,
        generation=generation,
        cell_id=cell_id,
    )


def _node_for_state(tree, cell_id, level=None):
    matches = [
        node
        for node, data in tree.nodes(data=True)
        if data["cell_id"] == cell_id
        and (level is None or data["biopsy_level"] == level)
    ]
    assert len(matches) == 1
    return matches[0]


def _tree_signature(tree):
    nodes = sorted(
        (
            node,
            data["cell_id"],
            data["biopsy_level"],
            tuple(np.asarray(data["genome"]).tolist()),
        )
        for node, data in tree.nodes(data=True)
    )
    edges = sorted(
        (parent, child, data["weight"])
        for parent, child, data in tree.edges(data=True)
    )
    return tuple(nodes), tuple(edges)


def _label_topology(tree, root):
    return (
        tree.nodes[root]["cell_id"],
        sorted(
            (tree.nodes[parent]["cell_id"], tree.nodes[child]["cell_id"])
            for parent, child in tree.edges()
        ),
    )


def test_temporal_arborescence_creates_one_vertex_per_level_state_record():
    duplicate = _cell([2, 2], 101, 5, generation=4)
    cell_lists = [
        [
            _cell([2, 2], 100, 5, generation=4),
            duplicate,
            _cell([3, 2], 200, 7, generation=4),
        ],
        [
            _cell([2, 2], 300, 5, generation=8),
            _cell([3, 3], 400, 9, generation=8),
        ],
    ]
    matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    tree, new_nodes, root = temporal_cnp_arborescence(
        matrix,
        cell_lists,
        [5, 7, 9],
        seed=11,
    )

    assert nx.is_arborescence(tree)
    assert root in tree
    assert new_nodes == {}
    assert tree.number_of_nodes() == 4
    assert tree.number_of_edges() == 3
    assert sorted(
        (data["biopsy_level"], data["cell_id"])
        for _, data in tree.nodes(data=True)
    ) == [(0, 5), (0, 7), (1, 5), (1, 9)]
    assert sum(data["cell_id"] == 5 for _, data in tree.nodes(data=True)) == 2
    assert duplicate.node_id == 101


def test_single_occurrence_uses_the_earliest_nonempty_level_as_root():
    tree, new_nodes, root = temporal_cnp_arborescence(
        np.array([[0.0]]),
        [[], [_cell([2, 2], 800, 5, generation=8)]],
        [5],
        seed=3,
    )

    assert nx.is_arborescence(tree)
    assert tree.number_of_nodes() == 1
    assert tree.number_of_edges() == 0
    assert tree.nodes[root]["biopsy_level"] == 1
    assert tree.nodes[root]["cell_id"] == 5
    assert new_nodes == {}


def test_ordered_and_no_time_modes_use_identical_vertices_but_different_constraints():
    cell_lists = [
        [_cell([0], 100, 1, generation=4)],
        [_cell([2], 900, 2, generation=8)],
    ]
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])

    ordered, _, ordered_root = temporal_cnp_arborescence(
        matrix,
        cell_lists,
        [1, 2],
        seed=7,
    )
    ablated, _, ablated_root = temporal_cnp_arborescence_no_time(
        matrix,
        cell_lists,
        [1, 2],
        seed=7,
    )

    assert list(ordered.nodes(data=True)) == list(ablated.nodes(data=True))
    early = _node_for_state(ordered, 1, level=0)
    late = _node_for_state(ordered, 2, level=1)
    assert ordered_root == early
    assert list(ordered.edges()) == [(early, late)]
    assert ablated_root == late
    assert list(ablated.edges()) == [(late, early)]


def test_plausibility_violations_dominate_arbitrarily_shorter_distance():
    cell_lists = [[
        _cell([2], 1, 1),
        _cell([0], 2, 2),
        _cell([1], 3, 3),
    ]]
    matrix = np.array(
        [
            [0.0, 0.1, 100.0],
            [0.1, 0.0, 0.1],
            [100.0, 0.1, 0.0],
        ]
    )

    tree, _, _ = temporal_cnp_arborescence(
        matrix,
        cell_lists,
        [1, 2, 3],
        seed=5,
    )

    assert sum(data["weight"] for _, _, data in tree.edges(data=True)) > 50
    for parent, child in tree.edges():
        parent_cell = _cell(
            tree.nodes[parent]["genome"],
            parent,
            tree.nodes[parent]["cell_id"],
        )
        child_cell = _cell(
            tree.nodes[child]["genome"],
            child,
            tree.nodes[child]["cell_id"],
        )
        assert is_biologically_plausible_ancestor(parent_cell, child_cell)


def test_distance_then_root_score_selects_the_central_root():
    cell_lists = [[
        _cell([2], 10, 1),
        _cell([3], 20, 2),
        _cell([4], 30, 3),
    ]]
    matrix = np.array(
        [
            [0.0, 1.0, 10.0],
            [1.0, 0.0, 1.0],
            [10.0, 1.0, 0.0],
        ]
    )

    tree, _, root = temporal_cnp_arborescence_no_time(
        matrix,
        cell_lists,
        [1, 2, 3],
        seed=5,
    )

    assert tree.nodes[root]["cell_id"] == 2
    assert sum(data["weight"] for _, _, data in tree.edges(data=True)) == 2.0
    assert {
        (tree.nodes[parent]["cell_id"], tree.nodes[child]["cell_id"])
        for parent, child in tree.edges()
    } == {(2, 1), (2, 3)}


def test_seeded_ties_are_reproducible_and_same_level_input_order_is_irrelevant():
    cells = [
        _cell([2], 100, 1),
        _cell([2], 200, 2),
        _cell([2], 300, 3),
        _cell([2], 400, 4),
    ]
    matrix = np.zeros((4, 4), dtype=float)

    first, _, _ = temporal_cnp_arborescence(matrix, [cells], [1, 2, 3, 4], seed=19)
    second, _, _ = temporal_cnp_arborescence(
        matrix,
        [list(reversed(cells))],
        [1, 2, 3, 4],
        seed=19,
    )
    signatures = {
        _tree_signature(
            temporal_cnp_arborescence(matrix, [cells], [1, 2, 3, 4], seed=seed)[0]
        )
        for seed in range(8)
    }

    assert _tree_signature(first) == _tree_signature(second)
    assert len(signatures) > 1
    assert [cell.node_id for cell in cells] == [100, 200, 300, 400]


def test_no_time_topology_does_not_depend_on_distinct_state_level_assignment():
    a = _cell([2], 100, 1)
    b = _cell([3], 200, 2)
    c = _cell([4], 300, 3)
    matrix = np.array(
        [
            [0.0, 1.0, 5.0],
            [1.0, 0.0, 2.0],
            [5.0, 2.0, 0.0],
        ]
    )

    first, _, first_root = temporal_cnp_arborescence_no_time(
        matrix,
        [[a], [b, c]],
        [1, 2, 3],
        seed=23,
    )
    second, _, second_root = temporal_cnp_arborescence_no_time(
        matrix,
        [[c], [b, a]],
        [1, 2, 3],
        seed=23,
    )

    assert _label_topology(first, first_root) == _label_topology(second, second_root)


def test_facade_dispatch_preserves_occurrences_and_rejects_only_nj_pooling():
    cell_lists = [
        [_cell([2], 100, 5)],
        [_cell([2], 900, 5), _cell([3], 700, 7)],
    ]
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])

    tree, node_levels, root = build_evolution_tree(
        cell_lists,
        inids=[5, 7],
        indm=matrix,
        neighbor_joining=temporal_cnp_arborescence,
        seed=13,
    )

    assert nx.is_arborescence(tree)
    assert root in tree
    assert tree.number_of_nodes() == 3
    assert node_levels == nx.get_node_attributes(tree, "biopsy_level")
    assert [cell.node_id for level in cell_lists for cell in level] == [100, 900, 700]

    with pytest.raises(ValueError, match="does not use only_nj pooling"):
        build_evolution_tree(
            cell_lists,
            inids=[5, 7],
            indm=matrix,
            only_nj=True,
            neighbor_joining=temporal_cnp_arborescence,
        )


def test_temporal_algorithm_contract_metadata_is_explicit():
    assert uses_ordered_occurrence_input(temporal_cnp_arborescence)
    assert uses_ordered_occurrence_input(temporal_cnp_arborescence_no_time)
    assert temporal_cnp_arborescence.ctbf_use_time is True
    assert temporal_cnp_arborescence_no_time.ctbf_use_time is False
    assert temporal_cnp_arborescence.ctbf_order_ablation is (
        temporal_cnp_arborescence_no_time
    )


@pytest.mark.parametrize(
    ("matrix", "ids", "match"),
    [
        (np.array([[0.0, -1.0], [-1.0, 0.0]]), [1, 2], "nonnegative"),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), [1, 2], "exactly symmetric"),
        (np.array([[1.0, 1.0], [1.0, 0.0]]), [1, 2], "diagonal"),
        (np.array([[0.0, 1.0], [1.0, 0.0]]), [1, 1], "Duplicate"),
        (np.array([[0.0, 1.0], [1.0, 0.0]]), [1, 3], "match the observed"),
    ],
)
def test_temporal_arborescence_rejects_invalid_distance_input(matrix, ids, match):
    cells = [[_cell([2], 10, 1), _cell([3], 20, 2)]]

    with pytest.raises(ValueError, match=match):
        temporal_cnp_arborescence(matrix, cells, ids)


def test_temporal_arborescence_rejects_empty_or_inconsistent_occurrences():
    with pytest.raises(ValueError, match="at least one observation"):
        temporal_cnp_arborescence(np.empty((0, 0)), [[]], [])

    inconsistent = [
        [_cell([2, 2], 10, 1)],
        [_cell([2, 3], 20, 1)],
    ]
    with pytest.raises(ValueError, match="inconsistent genomes"):
        temporal_cnp_arborescence(np.array([[0.0]]), inconsistent, [1])
