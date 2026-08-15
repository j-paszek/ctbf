import itertools
from collections import defaultdict

import networkx as nx
import numpy as np

from reconstructor_biopsy_blocks import (
    BiopsyGuidedConfig,
    BiopsyGuidedDecisionAudit,
    BiopsySubtreeConfig,
    _minimum_distance_pair_selector,
    assign_compatible_node_ids,
    build_final_distance_matrix,
    copy_missing_parent_to_upper,
    default_biopsy_guided_config,
    deduplicate_cells_by_cell_id,
    extend_biopsy_levels,
    filter_plausible_ancestor_candidates,
    find_radius_candidates,
    initialize_biopsy_tree,
    make_binarized_group_attachment_strategy,
    make_id_to_index,
    normalize_biopsy_guided_config,
    normalize_biopsy_subtree_config,
    reconstruct_biopsy_layers,
    select_anticentral_candidate,
    select_biopsy_parent,
    select_central_candidate,
    select_closest_candidate,
    select_deferred_candidate,
    select_diploid_parsimony_candidate,
    select_first_candidate,
    select_same_id_candidate,
)
from reconstructor import build_evolution_tree
from reconstructor_ancestor_selection import (
    keep_pair_order_parent_selector,
    more_central_parent_selector_left_tie,
)
from reconstructor_anticentral import configure_anticentral_v3_state
from reconstructor_engine import initialize_reconstruction_state
from reconstructor_pair_selection import (
    make_anticentral_adaptive_v3_pair_selector,
    make_hybrid_opt_v2_pair_selector,
)
from reconstructor_biopsy_presets import (
    make_anticentral_binarized_biopsy_guided_config,
    make_anticentral_tie_binarized_biopsy_guided_config,
)
from simulator import Genotype


def _cells():
    parent_same_id = Genotype([2, 0, 2], 1)
    parent_same_id.node_id = 1
    parent_plausible = Genotype([2, 1, 2], 2)
    parent_plausible.node_id = 2
    parent_implausible = Genotype([2, 0, 2], 3)
    parent_implausible.node_id = 3
    child = Genotype([2, 1, 2], 1)
    child.node_id = 10
    other_child = Genotype([2, 1, 2], 4)
    other_child.node_id = 4
    return parent_same_id, parent_plausible, parent_implausible, child, other_child


def test_extend_biopsy_levels_copies_missing_intermediate_observation():
    first = Genotype([2, 2], 7)
    first.node_id = 7
    last = Genotype([2, 2], 7)
    last.node_id = 7
    cell_lists = [[first], [], [last]]

    extended = extend_biopsy_levels(cell_lists)

    assert extended is cell_lists
    assert [cell.cell_id for cell in extended[1]] == [7]
    assert extended[1][0].node_id == 7
    assert extended[1][0] is not first


def test_build_evolution_tree_does_not_mutate_caller_biopsy_inputs():
    top = Genotype([2, 2], 7)
    top.node_id = 700
    middle = []
    bottom = Genotype([2, 2], 7)
    bottom.node_id = 701
    cell_lists = [[top], middle, [bottom]]
    ids = [7]
    distances = np.array([[0.0]])

    first_tree, first_levels, first_root = build_evolution_tree(
        cell_lists,
        inids=ids,
        indm=distances,
        r=0,
        seed=7,
    )
    second_tree, second_levels, second_root = build_evolution_tree(
        cell_lists,
        inids=ids,
        indm=distances,
        r=0,
        seed=7,
    )

    assert top.node_id == 700
    assert bottom.node_id == 701
    assert middle == []
    assert list(first_tree.edges(data="weight")) == list(second_tree.edges(data="weight"))
    assert dict(first_levels) == dict(second_levels)
    assert first_root == second_root


def test_biopsy_parent_selection_prefers_same_cell_id_before_plausibility():
    same_id, plausible, _implausible, child, _other_child = _cells()
    ids = [1, 2]
    id_to_index = make_id_to_index(ids)
    distances = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    assert select_biopsy_parent(child, [plausible, same_id], distances, id_to_index, radius=2) is same_id


def test_biopsy_parent_selection_filters_plausibility_then_chooses_closest():
    _same_id, plausible, implausible, _child, other_child = _cells()
    closer_plausible = Genotype([2, 1, 2], 5)
    closer_plausible.node_id = 5
    ids = [2, 3, 4, 5]
    id_to_index = make_id_to_index(ids)
    distances = np.array(
        [
            [0.0, 1.0, 3.0, 1.0],
            [1.0, 0.0, 2.0, 1.0],
            [3.0, 2.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )

    candidates = find_radius_candidates(other_child, [plausible, implausible, closer_plausible], distances, id_to_index, 3)
    plausible_candidates = filter_plausible_ancestor_candidates(candidates, other_child)

    assert implausible not in plausible_candidates
    assert select_closest_candidate(plausible_candidates, other_child, distances, id_to_index) is closer_plausible
    assert select_biopsy_parent(other_child, [plausible, implausible, closer_plausible], distances, id_to_index, 3) is closer_plausible


def test_radius_candidates_preserve_upper_cell_order():
    first = Genotype([2, 2], 1)
    second = Genotype([2, 2], 2)
    third = Genotype([2, 2], 3)
    child = Genotype([2, 2], 4)
    ids = [1, 2, 3, 4]
    id_to_index = make_id_to_index(ids)
    distances = np.array(
        [
            [0.0, 9.0, 9.0, 1.0],
            [9.0, 0.0, 9.0, 3.0],
            [9.0, 9.0, 0.0, 1.0],
            [1.0, 3.0, 1.0, 0.0],
        ]
    )

    assert find_radius_candidates(child, [first, second, third], distances, id_to_index, 1) == [first, third]


def test_closest_candidate_uses_first_candidate_as_default_tie_breaker():
    first = Genotype([2, 2], 1)
    second = Genotype([2, 2], 2)
    child = Genotype([2, 2], 3)
    ids = [1, 2, 3]
    id_to_index = make_id_to_index(ids)
    distances = np.array(
        [
            [0.0, 4.0, 1.0],
            [4.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    assert select_closest_candidate([first, second], child, distances, id_to_index) is first


def test_closest_candidate_preserves_tie_breaker_order():
    first = Genotype([2, 2], 1)
    second = Genotype([2, 2], 2)
    third = Genotype([2, 2], 3)
    child = Genotype([2, 2], 4)
    ids = [1, 2, 3, 4]
    id_to_index = make_id_to_index(ids)
    distances = np.array(
        [
            [0.0, 9.0, 9.0, 1.0],
            [9.0, 0.0, 9.0, 1.0],
            [9.0, 9.0, 0.0, 2.0],
            [1.0, 1.0, 2.0, 0.0],
        ]
    )
    seen = []

    def record_candidates(candidates, _child, _distances, _id_to_index):
        seen.extend(candidates)
        return candidates[-1]

    assert select_closest_candidate([first, second, third], child, distances, id_to_index, record_candidates) is second
    assert seen == [first, second]


def test_anticentral_candidate_tie_breaker_prefers_larger_distance_sum():
    central = Genotype([2, 2], 1)
    anticentral = Genotype([2, 2], 2)
    child = Genotype([2, 2], 3)
    ids = [1, 2, 3]
    id_to_index = make_id_to_index(ids)
    distances = np.array(
        [
            [0.0, 4.0, 1.0],
            [4.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    # Both candidates are equally close to the child, but candidate 2 has a larger
    # full-matrix distance sum if we add an extra distant reference.
    expanded_ids = [1, 2, 3, 4]
    expanded_id_to_index = make_id_to_index(expanded_ids)
    expanded_distances = np.array(
        [
            [0.0, 4.0, 1.0, 1.0],
            [4.0, 0.0, 1.0, 9.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 9.0, 1.0, 0.0],
        ]
    )

    assert select_first_candidate([central, anticentral], child, distances, id_to_index) is central
    assert (
        select_anticentral_candidate(
            [central, anticentral],
            child,
            expanded_distances,
            expanded_id_to_index,
        )
        is anticentral
    )
    assert (
        select_closest_candidate(
            [central, anticentral],
            child,
            expanded_distances,
            expanded_id_to_index,
            tie_breaker=select_anticentral_candidate,
        )
        is anticentral
    )


def test_central_candidate_tie_breaker_requires_unique_smaller_distance_sum():
    central = Genotype([2, 2], 1)
    peripheral = Genotype([2, 2], 2)
    child = Genotype([2, 2], 3)
    reference = Genotype([2, 2], 4)
    cells = [central, peripheral, child, reference]
    id_to_index = make_id_to_index([cell.cell_id for cell in cells])
    distances = np.array(
        [
            [0.0, 4.0, 1.0, 1.0],
            [4.0, 0.0, 1.0, 9.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 9.0, 1.0, 0.0],
        ]
    )

    assert (
        select_closest_candidate(
            [central, peripheral],
            child,
            distances,
            id_to_index,
            tie_breaker=select_central_candidate,
        )
        is central
    )

    residual_tie = np.array(
        [
            [0.0, 4.0, 1.0],
            [4.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    assert (
        select_closest_candidate(
            [central, peripheral],
            child,
            residual_tie,
            make_id_to_index([1, 2, 3]),
            tie_breaker=select_central_candidate,
        )
        is None
    )


def test_diploid_parsimony_tie_breaker_uses_burden_then_centrality_then_defers():
    diploid = Genotype([2, 2], 1)
    altered = Genotype([3, 3], 2)
    child = Genotype([2, 3], 3)
    reference = Genotype([2, 2], 4)
    id_to_index = make_id_to_index([1, 2, 3, 4])
    distances = np.array(
        [
            [0.0, 4.0, 1.0, 1.0],
            [4.0, 0.0, 1.0, 9.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 9.0, 1.0, 0.0],
        ]
    )

    assert (
        select_closest_candidate(
            [diploid, altered],
            child,
            distances,
            id_to_index,
            tie_breaker=select_diploid_parsimony_candidate,
        )
        is diploid
    )

    equally_altered_central = Genotype([1, 2], 1)
    equally_altered_peripheral = Genotype([3, 2], 2)
    assert (
        select_closest_candidate(
            [equally_altered_central, equally_altered_peripheral],
            child,
            distances,
            id_to_index,
            tie_breaker=select_diploid_parsimony_candidate,
        )
        is equally_altered_central
    )

    residual_tie = np.array(
        [
            [0.0, 4.0, 1.0],
            [4.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    assert (
        select_closest_candidate(
            [equally_altered_central, equally_altered_peripheral],
            child,
            residual_tie,
            make_id_to_index([1, 2, 3]),
            tie_breaker=select_diploid_parsimony_candidate,
        )
        is None
    )


def test_deferred_parent_tie_copies_child_up_and_audits_the_decision():
    parent_a = Genotype([2, 2], 1)
    parent_a.node_id = 1
    parent_b = Genotype([2, 2], 2)
    parent_b.node_id = 2
    child = Genotype([2, 2], 3)
    child.node_id = 3
    cells = [parent_a, parent_b, child]
    tree = nx.DiGraph()
    for cell in cells:
        tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)
    audit = BiopsyGuidedDecisionAudit()

    reconstruct_biopsy_layers(
        [[parent_a, parent_b], [child]],
        tree,
        defaultdict(lambda: None),
        np.array(
            [
                [0.0, 2.0, 1.0],
                [2.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        make_id_to_index([1, 2, 3]),
        radius=2,
        unique_node_counter=itertools.count(start=10),
        config=BiopsyGuidedConfig(
            candidate_tie_breaker=select_deferred_candidate,
            decision_audit=audit,
        ),
    )

    assert tree.has_edge(10, child.node_id)
    assert not tree.has_edge(parent_a.node_id, child.node_id)
    assert not tree.has_edge(parent_b.node_id, child.node_id)
    assert audit.counters["child_decision_count"] == 1
    assert audit.counters["multiple_plausible_parent_count"] == 1
    assert audit.counters["minimum_distance_tie_count"] == 1
    assert audit.counters["tie_deferred_count"] == 1
    assert audit.counters["tie_parent_selected_count"] == 0
    assert audit.counters["selected_parent_count"] == 0
    assert audit.counters["copy_up_count"] == 1
    assert (
        audit.counters["selected_parent_count"]
        + audit.counters["copy_up_count"]
        == audit.counters["child_decision_count"]
    )
    assert (
        audit.counters["tie_parent_selected_count"]
        + audit.counters["tie_deferred_count"]
        == audit.counters["minimum_distance_tie_count"]
    )


def test_same_id_selector_and_deduplication_keep_first_match():
    same_id, plausible, _implausible, child, _other_child = _cells()

    assert select_same_id_candidate([plausible, same_id], child) is same_id
    assert deduplicate_cells_by_cell_id([same_id, child, plausible]) == [same_id, plausible]


def test_copy_missing_parent_adds_upper_cell_and_zero_weight_edge():
    _same_id, _plausible, _implausible, child, _other_child = _cells()
    upper = []
    tree = nx.DiGraph()
    tree.add_node(child.node_id, genome=child.genome, cell_id=child.cell_id)
    node_levels = defaultdict(lambda: None)

    copied = copy_missing_parent_to_upper(
        child,
        upper,
        tree,
        node_levels,
        itertools.count(start=20),
        copied_level=1,
    )

    assert upper == [copied]
    assert copied.cell_id == child.cell_id
    assert copied.node_id == 20
    assert node_levels[copied.node_id] == 1
    assert tree.edges[copied.node_id, child.node_id]["weight"] == 0


def test_initialize_biopsy_tree_records_levels_by_node_id_not_object_identity():
    older = Genotype([2], 7)
    older.node_id = 700
    younger_same_genotype = Genotype([2], 7)
    younger_same_genotype.node_id = 701

    assign_compatible_node_ids([[older], [younger_same_genotype]])
    tree, node_levels = initialize_biopsy_tree(
        [[older], [younger_same_genotype]],
        itertools.count(start=20),
    )

    assert older.node_id != 700
    assert younger_same_genotype.node_id != 701
    assert set(node_levels) == set(tree.nodes)
    assert node_levels[younger_same_genotype.node_id] == 0
    assert node_levels[older.node_id] == 1
    copied_older = Genotype([2], older.cell_id)
    copied_older.node_id = older.node_id
    assert node_levels[copied_older.node_id] == node_levels[older.node_id]


def test_build_final_distance_matrix_uses_cell_ids_not_node_ids():
    a = Genotype([2], 1)
    a.node_id = 100
    b = Genotype([2], 2)
    b.node_id = 200
    ids = [1, 2]
    full = np.array([[0.0, 3.0], [3.0, 0.0]])

    assert np.array_equal(
        build_final_distance_matrix([a, b], full, make_id_to_index(ids)),
        full,
    )
    assert build_final_distance_matrix([a, b], full.astype(int), make_id_to_index(ids)).dtype == float


def test_biopsy_minimum_distance_pair_selector_uses_first_minimum_pair():
    cells = [Genotype([2], i + 1) for i in range(3)]
    distances = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 2.0, 0.0],
        ]
    )
    state = initialize_reconstruction_state(distances, cells, max_id=3, seed=7)

    pair = _minimum_distance_pair_selector(state)

    assert (pair.i, pair.j, pair.score) == (0, 1, 1.0)


def test_biopsy_guided_config_fills_unspecified_strategies_with_defaults():
    config = normalize_biopsy_guided_config(BiopsyGuidedConfig(candidate_selector=select_biopsy_parent))
    defaults = default_biopsy_guided_config()

    assert config.candidate_selector is select_biopsy_parent
    assert config.candidate_tie_breaker is defaults.candidate_tie_breaker
    assert config.level_extender is defaults.level_extender
    assert config.attachment_strategy is defaults.attachment_strategy
    assert config.group_attachment_strategy is not None
    assert config.missing_parent_strategy is defaults.missing_parent_strategy
    assert config.only_nj_final_cell_selector is defaults.only_nj_final_cell_selector


def test_biopsy_subtree_config_fills_missing_engine_blocks_with_defaults():
    pair_selector = make_hybrid_opt_v2_pair_selector()
    config = normalize_biopsy_subtree_config(
        BiopsySubtreeConfig(pair_selector=pair_selector),
        seed=19,
    )

    assert config.pair_selector is pair_selector
    assert config.ancestor_selector is not None
    assert config.merge_strategy is None
    assert config.distance_update is None
    assert config.configure_state is None
    assert config.seed == 19


def test_reconstruct_biopsy_layers_uses_configured_candidate_selector():
    parent_a = Genotype([2, 2], 1)
    parent_a.node_id = 1
    parent_b = Genotype([2, 2], 2)
    parent_b.node_id = 2
    child = Genotype([2, 2], 3)
    child.node_id = 3
    cell_lists = [[parent_a, parent_b], [child]]
    ids = [1, 2, 3]
    id_to_index = make_id_to_index(ids)
    distances = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 2.0, 0.0],
        ]
    )
    tree = nx.DiGraph()
    for cell in [parent_a, parent_b, child]:
        tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)
    node_levels = defaultdict(lambda: None)

    def select_second_parent(_y, upper_cells, _distances, _id_to_index, _radius):
        return upper_cells[1]

    reconstruct_biopsy_layers(
        cell_lists,
        tree,
        node_levels,
        distances,
        id_to_index,
        radius=3,
        unique_node_counter=itertools.count(start=10),
        config=BiopsyGuidedConfig(candidate_selector=select_second_parent),
    )

    assert tree.has_edge(parent_b.node_id, child.node_id)
    assert not tree.has_edge(parent_a.node_id, child.node_id)
    assert tree.edges[parent_b.node_id, child.node_id]["weight"] == 2.0


def test_reconstruct_biopsy_layers_reuses_missing_parent_added_within_level():
    child_a = Genotype([2, 2], 1)
    child_a.node_id = 10
    child_b = Genotype([2, 2], 1)
    child_b.node_id = 11
    cell_lists = [[], [child_a, child_b]]
    distances = np.array([[0.0]])
    id_to_index = make_id_to_index([1])
    tree = nx.DiGraph()
    for cell in [child_a, child_b]:
        tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)

    reconstruct_biopsy_layers(
        cell_lists,
        tree,
        defaultdict(lambda: None),
        distances,
        id_to_index,
        radius=0,
        unique_node_counter=itertools.count(start=20),
    )

    assert [cell.node_id for cell in cell_lists[0]] == [20]
    assert tree.has_edge(20, child_a.node_id)
    assert tree.has_edge(20, child_b.node_id)


def test_biopsy_guided_config_can_use_anticentral_candidate_tie_breaker():
    central = Genotype([2, 2], 1)
    central.node_id = 1
    anticentral = Genotype([2, 2], 2)
    anticentral.node_id = 2
    child = Genotype([2, 2], 3)
    child.node_id = 3
    ids = [1, 2, 3, 4]
    id_to_index = make_id_to_index(ids)
    distances = np.array(
        [
            [0.0, 4.0, 1.0, 1.0],
            [4.0, 0.0, 1.0, 9.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 9.0, 1.0, 0.0],
        ]
    )
    tree = nx.DiGraph()
    for cell in [central, anticentral, child]:
        tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)

    reconstruct_biopsy_layers(
        [[central, anticentral], [child]],
        tree,
        defaultdict(lambda: None),
        distances,
        id_to_index,
        radius=1,
        unique_node_counter=itertools.count(start=10),
        config=BiopsyGuidedConfig(candidate_tie_breaker=select_anticentral_candidate),
    )

    assert tree.has_edge(anticentral.node_id, child.node_id)
    assert not tree.has_edge(central.node_id, child.node_id)


def test_binarized_group_attachment_builds_binary_subtree_under_parent():
    parent = Genotype([2, 2], 1)
    parent.node_id = 1
    child_a = Genotype([2, 2], 2)
    child_a.node_id = 2
    child_b = Genotype([2, 2], 3)
    child_b.node_id = 3
    child_c = Genotype([2, 2], 4)
    child_c.node_id = 4
    cell_lists = [[parent], [child_a, child_b, child_c]]
    ids = [1, 2, 3, 4]
    id_to_index = make_id_to_index(ids)
    distances = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 2.0, 3.0],
            [1.0, 2.0, 0.0, 4.0],
            [1.0, 3.0, 4.0, 0.0],
        ]
    )
    tree = nx.DiGraph()
    for cell in [parent, child_a, child_b, child_c]:
        tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)
    node_levels = defaultdict(lambda: None)

    reconstruct_biopsy_layers(
        cell_lists,
        tree,
        node_levels,
        distances,
        id_to_index,
        radius=1,
        unique_node_counter=itertools.count(start=10),
        config=BiopsyGuidedConfig(
            group_attachment_strategy=make_binarized_group_attachment_strategy(),
        ),
    )

    children = {child_a.node_id, child_b.node_id, child_c.node_id}
    assert tree.out_degree(parent.node_id) == 1
    parent_child = next(tree.successors(parent.node_id))
    assert parent_child not in children
    assert children <= nx.descendants(tree, parent.node_id)
    assert len([node for node in tree.nodes if node not in {1, 2, 3, 4}]) == 2
    assert nx.is_directed_acyclic_graph(tree)


def test_binarized_group_attachment_keeps_single_child_direct():
    parent = Genotype([2, 2], 1)
    parent.node_id = 1
    child = Genotype([2, 2], 2)
    child.node_id = 2
    tree = nx.DiGraph()
    for cell in [parent, child]:
        tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)
    distances = np.array([[0.0, 5.0], [5.0, 0.0]])

    make_binarized_group_attachment_strategy()(
        tree,
        parent,
        [child],
        distances,
        make_id_to_index([1, 2]),
        itertools.count(start=10),
        defaultdict(lambda: None),
        child_level=0,
    )

    assert list(tree.successors(parent.node_id)) == [child.node_id]
    assert tree.edges[parent.node_id, child.node_id]["weight"] == 5.0


def test_binarized_group_attachment_accepts_existing_pair_and_ancestor_blocks():
    parent = Genotype([2, 2], 1)
    parent.node_id = 1
    child_a = Genotype([2, 2], 2)
    child_a.node_id = 2
    child_b = Genotype([2, 2], 3)
    child_b.node_id = 3
    child_c = Genotype([2, 2], 4)
    child_c.node_id = 4
    distances = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 2.0, 5.0],
            [1.0, 2.0, 0.0, 3.0],
            [1.0, 5.0, 3.0, 0.0],
        ]
    )
    tree = nx.DiGraph()
    for cell in [parent, child_a, child_b, child_c]:
        tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)

    make_binarized_group_attachment_strategy(
        BiopsySubtreeConfig(
            pair_selector=make_hybrid_opt_v2_pair_selector(),
            ancestor_selector=more_central_parent_selector_left_tie,
            seed=13,
        )
    )(
        tree,
        parent,
        [child_a, child_b, child_c],
        distances,
        make_id_to_index([1, 2, 3, 4]),
        itertools.count(start=10),
        defaultdict(lambda: None),
        child_level=0,
    )

    assert tree.out_degree(parent.node_id) == 1
    assert {child_a.node_id, child_b.node_id, child_c.node_id} <= nx.descendants(tree, parent.node_id)
    assert nx.is_directed_acyclic_graph(tree)


def test_binarized_group_attachment_accepts_configured_anticentral_v3_blocks():
    parent = Genotype([2, 2], 1)
    parent.node_id = 1
    children = [
        Genotype([2, 2], 2),
        Genotype([2, 2], 3),
        Genotype([2, 2], 4),
    ]
    for child in children:
        child.node_id = child.cell_id

    distances = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 2.0, 5.0],
            [1.0, 2.0, 0.0, 3.0],
            [1.0, 5.0, 3.0, 0.0],
        ]
    )
    tree = nx.DiGraph()
    for cell in [parent, *children]:
        tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)

    make_binarized_group_attachment_strategy(
        pair_selector=make_anticentral_adaptive_v3_pair_selector(),
        ancestor_selector=keep_pair_order_parent_selector,
        configure_state=configure_anticentral_v3_state,
    )(
        tree,
        parent,
        children,
        distances,
        make_id_to_index([1, 2, 3, 4]),
        itertools.count(start=10),
        defaultdict(lambda: None),
        child_level=0,
    )

    assert tree.out_degree(parent.node_id) == 1
    assert {child.node_id for child in children} <= nx.descendants(tree, parent.node_id)
    assert nx.is_directed_acyclic_graph(tree)


def test_clean_anticentral_tie_binarized_preset_keeps_default_subtree_pairing():
    def reconstruct_with(config):
        parent = Genotype([2, 2], 1)
        parent.node_id = 1
        children = [Genotype([2, 2], cell_id) for cell_id in (2, 3, 4)]
        for child in children:
            child.node_id = child.cell_id
        tree = nx.DiGraph()
        for cell in [parent, *children]:
            tree.add_node(cell.node_id, genome=cell.genome, cell_id=cell.cell_id)
        distances = np.ones((4, 4), dtype=float)
        np.fill_diagonal(distances, 0.0)
        reconstruct_biopsy_layers(
            [[parent], children],
            tree,
            defaultdict(lambda: None),
            distances,
            make_id_to_index([1, 2, 3, 4]),
            radius=1,
            unique_node_counter=itertools.count(start=10),
            config=config,
        )
        return tree

    clean = reconstruct_with(
        make_anticentral_tie_binarized_biopsy_guided_config()
    )
    historical_mixed = reconstruct_with(
        make_anticentral_binarized_biopsy_guided_config()
    )

    clean_first_pair_parent = next(
        node
        for node in clean.nodes
        if node >= 10 and set(clean.successors(node)) == {2, 3}
    )
    historical_first_pair_parent = next(
        node
        for node in historical_mixed.nodes
        if node >= 10 and set(historical_mixed.successors(node)) == {2, 4}
    )
    assert clean_first_pair_parent == 10
    assert historical_first_pair_parent == 10
