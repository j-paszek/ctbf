import copy
import math

import networkx as nx
import numpy as np
import pytest

from reconstructor import BIOPSY_GUIDED_PRESETS, build_evolution_tree, resolve_biopsy_guided_config
from simulator import Genotype


def _small_biopsy_case():
    c1 = Genotype([2, 0, 1], 1)
    c2 = Genotype([1, 1, 1], 2)
    c3 = Genotype([2, 1, 1], 3)
    c4 = Genotype([1, 2, 0], 4)
    cell_lists = [[c1, c2], [c3, c4]]
    ids = [1, 2, 3, 4]
    dist_matrix = np.array(
        [
            [0.0, 1.0, 1.0, 4.0],
            [1.0, 0.0, 2.0, 4.0],
            [1.0, 2.0, 0.0, 4.0],
            [4.0, 4.0, 4.0, 0.0],
        ],
        dtype=float,
    )
    return cell_lists, ids, dist_matrix


def _assert_valid_rooted_tree(tree, original_ids):
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    assert len(roots) == 1
    assert nx.is_directed_acyclic_graph(tree)
    assert nx.is_weakly_connected(tree)

    for node, indegree in tree.in_degree():
        if node == roots[0]:
            assert indegree == 0
        else:
            assert indegree == 1

    for cell_id in original_ids:
        assert cell_id in tree.nodes

    for _, _, edge_data in tree.edges(data=True):
        weight = edge_data.get("weight", 0.0)
        assert math.isfinite(weight)
        assert weight >= -1e-9


@pytest.mark.parametrize("preset_name", sorted(BIOPSY_GUIDED_PRESETS))
def test_biopsy_guided_presets_return_valid_rooted_trees(preset_name):
    cell_lists, ids, dist_matrix = _small_biopsy_case()

    tree, _, root = build_evolution_tree(
        copy.deepcopy(cell_lists),
        inids=ids,
        indm=dist_matrix,
        r=2,
        biopsy_guided_config=resolve_biopsy_guided_config(preset_name),
        seed=7,
    )

    assert root in tree.nodes
    _assert_valid_rooted_tree(tree, ids)


def test_biopsy_guided_binarized_presets_reduce_direct_parent_fanout():
    cell_lists, ids, dist_matrix = _small_biopsy_case()

    default_tree, _, _ = build_evolution_tree(
        copy.deepcopy(cell_lists),
        inids=ids,
        indm=dist_matrix,
        r=4,
        biopsy_guided_config=resolve_biopsy_guided_config("default"),
        seed=7,
    )
    binarized_tree, _, _ = build_evolution_tree(
        copy.deepcopy(cell_lists),
        inids=ids,
        indm=dist_matrix,
        r=4,
        biopsy_guided_config=resolve_biopsy_guided_config("binarized"),
        seed=7,
    )

    assert default_tree.out_degree(2) == 2
    assert binarized_tree.out_degree(2) == 1
    assert {3, 4} <= nx.descendants(binarized_tree, 2)
