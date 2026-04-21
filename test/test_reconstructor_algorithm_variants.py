import math
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_evaluation.tester import (
    LEGACY_ALGORITHM_NAMES,
    get_algorithms_to_test,
    get_legacy_algorithms_to_test,
)
from ctbs_utils import to_newick
from reconstructor import build_evolution_tree
from simulator import Genotype


def _legacy_algorithms():
    return get_legacy_algorithms_to_test()


def _simple_four_cell_case():
    cells = [
        Genotype([2, 2, 1], 1),
        Genotype([1, 1, 1], 2),
        Genotype([2, 1, 1], 3),
        Genotype([1, 2, 0], 4),
    ]
    ids = [1, 2, 3, 4]
    dist_matrix = np.array(
        [
            [0, 1, 1, 4],
            [1, 0, 2, 4],
            [1, 2, 0, 4],
            [4, 4, 4, 0],
        ],
        dtype=float,
    )
    return cells, ids, dist_matrix


def _three_cell_tie_case():
    cells = [
        Genotype([2, 2, 1], 1),
        Genotype([1, 1, 1], 2),
        Genotype([2, 1, 1], 3),
    ]
    ids = [1, 2, 3]
    dist_matrix = np.array(
        [
            [0, 1, 1],
            [1, 0, 4],
            [1, 4, 0],
        ],
        dtype=float,
    )
    return cells, ids, dist_matrix


def _zero_distance_plausibility_case():
    cells = [
        Genotype([2, 0, 2], 7),
        Genotype([2, 0, 2], 14),
        Genotype([2, 2, 2], 13),
    ]
    ids = [7, 14, 13]
    dist_matrix = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=float,
    )
    return cells, ids, dist_matrix


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


def test_legacy_algorithm_registry_order_is_stable():
    assert [algorithm.__name__ for algorithm in _legacy_algorithms()] == LEGACY_ALGORITHM_NAMES


def test_combined_algorithm_registry_keeps_legacy_prefix_and_unique_names():
    algorithm_names = [algorithm.__name__ for algorithm in get_algorithms_to_test()]

    assert algorithm_names[:len(LEGACY_ALGORITHM_NAMES)] == LEGACY_ALGORITHM_NAMES
    assert len(algorithm_names) == len(set(algorithm_names))


@pytest.mark.parametrize("algorithm", _legacy_algorithms(), ids=lambda algorithm: algorithm.__name__)
@pytest.mark.parametrize(
    "case_factory",
    [_simple_four_cell_case, _three_cell_tie_case, _zero_distance_plausibility_case],
    ids=lambda factory: factory.__name__.removeprefix("_").removesuffix("_case"),
)
def test_legacy_neighbor_joining_variants_return_valid_rooted_trees(algorithm, case_factory):
    cells, ids, dist_matrix = case_factory()

    tree, _, root = build_evolution_tree(
        [cells],
        only_nj=True,
        inids=ids,
        indm=dist_matrix,
        neighbor_joining=algorithm,
        seed=7,
    )

    assert root in tree.nodes
    _assert_valid_rooted_tree(tree, ids)


def test_neighbor_joining_baseline_keeps_known_three_cell_topology():
    baseline = _legacy_algorithms()[0]
    cells, ids, dist_matrix = _three_cell_tie_case()

    tree, _, root = build_evolution_tree(
        [cells],
        only_nj=True,
        inids=ids,
        indm=dist_matrix,
        neighbor_joining=baseline,
        seed=7,
    )

    assert root == 1
    assert to_newick(tree) == "((1:0.0000,2:1.0000)1:0.0000,3:1.0000)1;"
