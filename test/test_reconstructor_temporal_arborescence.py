from pathlib import Path
import random
import sys

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstructor import build_evolution_tree
from distance_semantics import DirectedDistanceBundle
from reconstructor_plausibility import is_biologically_plausible_ancestor
from reconstructor_temporal import (
    TEMPORAL_ARBORESCENCE_SOLVER_VERSION,
    _WorkingEdge,
    _compact_minimum_spanning_arborescence,
    _compact_seeded_tie_ranks,
    _edge_tie_key_index,
    _seeded_tie_ranks,
    temporal_cnp_arborescence,
    temporal_cnp_arborescence_directed,
    temporal_cnp_arborescence_directed_no_time,
    temporal_cnp_arborescence_no_time,
    uses_directed_distance_input,
    uses_ordered_occurrence_input,
)
from simulator import Genotype
from ctbs import run_single_test


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


def _working_edges(edge_specs):
    return [
        _WorkingEdge(
            edge_id=edge_id,
            original_source=source,
            original_target=target,
            source=source,
            target=target,
            weight=weight,
        )
        for edge_id, (source, target, weight) in enumerate(edge_specs)
    ]


@pytest.mark.parametrize("occurrence_count", [1, 2, 5, 9])
@pytest.mark.parametrize("seed", [0, 7, 29])
def test_compact_tie_ranks_exactly_match_the_legacy_tuple_universe(
    occurrence_count,
    seed,
):
    biological_edges = [
        (parent, child)
        for parent in range(occurrence_count)
        for child in range(occurrence_count)
        if parent != child
    ]
    legacy = _seeded_tie_ranks(
        biological_edges,
        list(range(occurrence_count)),
        seed,
    )
    compact = _compact_seeded_tie_ranks(occurrence_count, seed)
    for parent, child in biological_edges:
        assert legacy[("edge", parent, child)] == compact[
            _edge_tie_key_index(parent, child, occurrence_count)
        ]
    root_offset = occurrence_count * (occurrence_count - 1)
    for root in range(occurrence_count):
        assert legacy[("root", root)] == compact[root_offset + root]


@pytest.mark.parametrize("seed", range(24))
def test_compact_edmonds_exactly_matches_networkx_reference(seed):
    biological_count = 3 + seed % 5
    virtual_root = biological_count
    rng = random.Random(seed)
    edge_specs = [
        (source, target, rng.randrange(9))
        for source in range(biological_count)
        for target in range(biological_count)
        if source != target
    ]
    edge_specs.extend(
        (virtual_root, target, 20 + rng.randrange(9))
        for target in range(biological_count)
    )

    reference_graph = nx.DiGraph()
    reference_graph.add_nodes_from(range(biological_count + 1))
    for edge_id, (source, target, weight) in enumerate(edge_specs):
        reference_graph.add_edge(
            source,
            target,
            objective_cost=weight,
            edge_id=edge_id,
        )
    reference = nx.minimum_spanning_arborescence(
        reference_graph,
        attr="objective_cost",
        preserve_attrs=True,
    )
    reference_edge_ids = {
        data["edge_id"]
        for _, _, data in reference.edges(data=True)
    }

    selected_edge_ids = _compact_minimum_spanning_arborescence(
        biological_count + 1,
        virtual_root,
        _working_edges(edge_specs),
    )

    assert selected_edge_ids == reference_edge_ids


def test_compact_edmonds_expands_nested_cycle_contractions():
    virtual_root = 3
    edge_specs = [
        (0, 1, 1),
        (0, 2, 1),
        (1, 0, 1),
        (2, 0, 2),
        (virtual_root, 0, 10),
        (virtual_root, 1, 11),
        (virtual_root, 2, 12),
    ]
    reference_graph = nx.DiGraph()
    reference_graph.add_nodes_from(range(4))
    for edge_id, (source, target, weight) in enumerate(edge_specs):
        reference_graph.add_edge(
            source,
            target,
            objective_cost=weight,
            edge_id=edge_id,
        )
    reference = nx.minimum_spanning_arborescence(
        reference_graph,
        attr="objective_cost",
        preserve_attrs=True,
    )

    selected_edge_ids = _compact_minimum_spanning_arborescence(
        4,
        virtual_root,
        _working_edges(edge_specs),
    )

    assert selected_edge_ids == {
        data["edge_id"]
        for _, _, data in reference.edges(data=True)
    }


def test_production_temporal_solver_does_not_call_networkx_edmonds(monkeypatch):
    def reject_networkx_backend(*_args, **_kwargs):
        raise AssertionError("The production temporal solver used NetworkX Edmonds.")

    monkeypatch.setattr(
        nx,
        "minimum_spanning_arborescence",
        reject_networkx_backend,
    )
    cells = [_cell([2, index + 1], index, index) for index in range(40)]
    matrix = np.abs(
        np.subtract.outer(np.arange(len(cells)), np.arange(len(cells)))
    ).astype(float)

    tree, _, _ = temporal_cnp_arborescence_no_time(
        matrix,
        [cells],
        list(range(len(cells))),
        seed=31,
    )

    assert nx.is_arborescence(tree)
    assert tree.number_of_nodes() == 40
    assert tree.number_of_edges() == 39


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
    assert uses_directed_distance_input(temporal_cnp_arborescence) is False
    assert uses_directed_distance_input(temporal_cnp_arborescence_directed) is True
    assert temporal_cnp_arborescence_directed.ctbf_order_ablation is (
        temporal_cnp_arborescence_directed_no_time
    )
    assert {
        algorithm.ctbf_solver_version
        for algorithm in (
            temporal_cnp_arborescence,
            temporal_cnp_arborescence_no_time,
            temporal_cnp_arborescence_directed,
            temporal_cnp_arborescence_directed_no_time,
        )
    } == {TEMPORAL_ARBORESCENCE_SOLVER_VERSION}


def test_directed_variant_changes_only_both_plausible_numeric_edge_tier():
    cells = [[_cell([2], 10, 1), _cell([3], 20, 2)]]
    bundle = DirectedDistanceBundle(
        [1, 2],
        np.array([[0.0, 9.0], [1.0, 0.0]]),
    )

    minimum_tree, _, _ = temporal_cnp_arborescence(
        bundle.minimum_matrix,
        cells,
        [1, 2],
        seed=0,
    )
    directed_tree, _, directed_root = temporal_cnp_arborescence_directed(
        bundle.minimum_matrix,
        cells,
        [1, 2],
        seed=0,
        directed_distance_bundle=bundle,
    )

    assert _label_topology(minimum_tree, 0) == (1, [(1, 2)])
    assert _label_topology(directed_tree, directed_root) == (2, [(2, 1)])
    assert next(iter(directed_tree.edges(data=True)))[2]["weight"] == 1.0


def test_directed_variant_keeps_no_regain_above_a_cheaper_impossible_direction():
    lost = _cell([0], 10, 1)
    present = _cell([1], 20, 2)
    bundle = DirectedDistanceBundle(
        [1, 2],
        np.array([[0.0, 0.0], [100.0, 0.0]]),
    )

    tree, _, root = temporal_cnp_arborescence_directed_no_time(
        bundle.minimum_matrix,
        [[lost, present]],
        [1, 2],
        seed=0,
        directed_distance_bundle=bundle,
    )

    assert _label_topology(tree, root) == (2, [(2, 1)])


def test_directed_variant_keeps_biopsy_time_above_a_cheaper_reverse_direction():
    early = _cell([2], 10, 1)
    late = _cell([3], 20, 2)
    bundle = DirectedDistanceBundle(
        [1, 2],
        np.array([[0.0, 9.0], [1.0, 0.0]]),
    )

    ordered, _, ordered_root = temporal_cnp_arborescence_directed(
        bundle.minimum_matrix,
        [[early], [late]],
        [1, 2],
        seed=0,
        directed_distance_bundle=bundle,
    )
    ablated, _, ablated_root = temporal_cnp_arborescence_directed_no_time(
        bundle.minimum_matrix,
        [[early], [late]],
        [1, 2],
        seed=0,
        directed_distance_bundle=bundle,
    )

    assert _label_topology(ordered, ordered_root) == (1, [(1, 2)])
    assert _label_topology(ablated, ablated_root) == (2, [(2, 1)])


def test_directed_variant_matches_minimum_variant_when_counts_tie():
    cells = [[
        _cell([2], 10, 1),
        _cell([3], 20, 2),
        _cell([4], 30, 3),
    ]]
    symmetric = np.array([
        [0.0, 1.0, 4.0],
        [1.0, 0.0, 2.0],
        [4.0, 2.0, 0.0],
    ])
    bundle = DirectedDistanceBundle([1, 2, 3], symmetric)

    minimum, _, _ = temporal_cnp_arborescence_no_time(
        symmetric,
        cells,
        [1, 2, 3],
        seed=17,
    )
    directed, _, _ = temporal_cnp_arborescence_directed_no_time(
        symmetric,
        cells,
        [1, 2, 3],
        seed=17,
        directed_distance_bundle=bundle,
    )

    assert _tree_signature(directed) == _tree_signature(minimum)


def test_facade_passes_directed_bundle_only_to_declared_algorithm():
    cells = [[_cell([2], 10, 1), _cell([3], 20, 2)]]
    bundle = DirectedDistanceBundle([1, 2], [[0.0, 9.0], [1.0, 0.0]])

    tree, _, _ = build_evolution_tree(
        cells,
        distance_matrix=bundle,
        neighbor_joining=temporal_cnp_arborescence_directed,
        seed=0,
    )
    assert nx.is_arborescence(tree)

    with pytest.raises(ValueError, match="explicitly declares"):
        build_evolution_tree(
            cells,
            distance_matrix=bundle,
            neighbor_joining=temporal_cnp_arborescence,
        )
    with pytest.raises(ValueError, match="requires a DirectedDistanceBundle"):
        build_evolution_tree(
            cells,
            inids=[1, 2],
            indm=bundle.minimum_matrix,
            neighbor_joining=temporal_cnp_arborescence_directed,
        )


def test_run_single_test_carries_directed_provider_to_ordered_and_no_time_pair():
    class TwoObservationSimulator:
        def __init__(self):
            self.tree = nx.DiGraph()
            self.tree.add_node(0, genome=[2], generation=0, cell_id=0)
            self.tree.add_node(5, genome=[2], generation=1, cell_id=1)
            self.tree.add_node(7, genome=[3], generation=2, cell_id=2)
            self.tree.add_edge(0, 5, events="")
            self.tree.add_edge(5, 7, events="gain")

        def perform_biopsy(
            self,
            generation,
            biopsy_size=0,
            biopsy_size_scalable=None,
            seed=None,
        ):
            if generation == 1:
                return [_cell([2], 5, 1)]
            if generation == 2:
                return [_cell([3], 7, 2)]
            return []

    bundle = DirectedDistanceBundle(
        [1, 2],
        [[0.0, 9.0], [1.0, 0.0]],
        provenance={"semantics_version": "test"},
    )

    class FixedDirectedProvider:
        def compute(self, cells):
            assert {cell.cell_id for cell in cells} == {1, 2}
            return bundle

    result = run_single_test(
        seed=0,
        simulator_with_loaded_tree=TwoObservationSimulator(),
        biopsy_generations=[1, 2],
        biopsy_size_scalable=1.0,
        reconstruction_algorithm=temporal_cnp_arborescence_directed,
        distance_provider=FixedDirectedProvider(),
    )

    assert result is not None
    _truth, ordered, no_time = result
    assert _label_topology(ordered, 0) == (1, [(1, 2)])
    no_time_root = next(node for node, indegree in no_time.in_degree() if indegree == 0)
    assert _label_topology(no_time, no_time_root) == (2, [(2, 1)])


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
