import itertools
from collections import defaultdict

import networkx as nx
import numpy as np

from reconstructor_biopsy_blocks import (
    BiopsyGuidedConfig,
    build_final_distance_matrix,
    copy_missing_parent_to_upper,
    default_biopsy_guided_config,
    deduplicate_cells_by_cell_id,
    extend_biopsy_levels,
    filter_plausible_ancestor_candidates,
    find_radius_candidates,
    make_id_to_index,
    normalize_biopsy_guided_config,
    reconstruct_biopsy_layers,
    select_anticentral_candidate,
    select_biopsy_parent,
    select_closest_candidate,
    select_first_candidate,
    select_same_id_candidate,
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
    assert node_levels[copied] == 1
    assert tree.edges[copied.node_id, child.node_id]["weight"] == 0


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


def test_biopsy_guided_config_fills_unspecified_strategies_with_defaults():
    config = normalize_biopsy_guided_config(BiopsyGuidedConfig(candidate_selector=select_biopsy_parent))
    defaults = default_biopsy_guided_config()

    assert config.candidate_selector is select_biopsy_parent
    assert config.candidate_tie_breaker is defaults.candidate_tie_breaker
    assert config.level_extender is defaults.level_extender
    assert config.attachment_strategy is defaults.attachment_strategy
    assert config.missing_parent_strategy is defaults.missing_parent_strategy
    assert config.only_nj_final_cell_selector is defaults.only_nj_final_cell_selector


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
