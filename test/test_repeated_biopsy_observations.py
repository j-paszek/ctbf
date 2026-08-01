import json
import math
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ctbs  # noqa: E402
from algorithm_evaluation.tester import get_algorithms_to_test  # noqa: E402
from ctbs import (  # noqa: E402
    Cnp2CnpFileDistanceProvider,
    DistanceMatrix,
    _compute_distance_matrix,
    default_ctbs_runtime_config,
    unique_cells_by_cell_id,
)
from evaluator import grf_tree, parse_newick_to_nx  # noqa: E402
from evaluator_full import ancestors_unique_restricted_metrics  # noqa: E402
from reconstructor import build_evolution_tree  # noqa: E402
from reconstructor_temporal import uses_ordered_occurrence_input  # noqa: E402
from simulator import Genotype  # noqa: E402


SANITY_DIR = PROJECT_ROOT / "test" / "data" / "sanity"
SANITY_CASE_PATHS = sorted(SANITY_DIR.glob("repeated_cell_id*.json"))
FLAT_INPUT_ALGORITHMS = [
    algorithm
    for algorithm in get_algorithms_to_test()
    if not uses_ordered_occurrence_input(algorithm)
]
ORDERED_OCCURRENCE_ALGORITHMS = [
    algorithm
    for algorithm in get_algorithms_to_test()
    if uses_ordered_occurrence_input(algorithm)
]


def _load_case(path):
    with open(path, "r") as f:
        return json.load(f)


def _case_id(path):
    return path.stem


def _cell_lists(case):
    return [
        [
            Genotype(
                cell["genome"],
                node_id=cell["node_id"],
                generation=cell["generation"],
                cell_id=cell["cell_id"],
            )
            for cell in biopsy["cells"]
        ]
        for biopsy in case["biopsies"]
    ]


def _flatten(cell_lists):
    return [cell for level in cell_lists for cell in level]


def _true_tree(case):
    tree, root = parse_newick_to_nx(case["true_newick"], prefix=case["case_id"])
    for _, data in tree.nodes(data=True):
        if data["cell_id"] is not None:
            data["cell_id"] = int(data["cell_id"])
    return tree, root


def _actual_root(tree):
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    assert len(roots) == 1
    return roots[0]


def _assert_valid_rooted_tree(tree):
    root = _actual_root(tree)
    assert nx.is_directed_acyclic_graph(tree)
    assert nx.is_weakly_connected(tree)

    for node, indegree in tree.in_degree():
        if node == root:
            assert indegree == 0
        else:
            assert indegree == 1

    for _, _, edge_data in tree.edges(data=True):
        weight = edge_data.get("weight", 0.0)
        assert math.isfinite(weight)
        assert weight >= -1e-9


class CapturingDistanceProvider:
    def __init__(self):
        self.cells = None

    def compute(self, cells):
        self.cells = list(cells)
        ids = [cell.get_id() for cell in self.cells]
        return DistanceMatrix(ids=ids, matrix=np.zeros((len(ids), len(ids)), dtype=float))


@pytest.mark.parametrize("case_path", SANITY_CASE_PATHS, ids=_case_id)
def test_repeated_biopsy_observations_keep_raw_count_but_unique_distance_ids(case_path):
    case = _load_case(case_path)
    cell_lists = _cell_lists(case)
    raw_cells = _flatten(cell_lists)

    assert len(raw_cells) == case["expected_raw_observation_count"]
    assert len(raw_cells) >= 3
    assert [cell.cell_id for cell in unique_cells_by_cell_id(raw_cells)] == case["expected_unique_cell_ids"]

    provider = CapturingDistanceProvider()
    distance_matrix = _compute_distance_matrix(
        [raw_cells],
        parallel=False,
        time_collector=None,
        runtime_config=default_ctbs_runtime_config(),
        distance_provider=provider,
    )

    assert [cell.cell_id for cell in provider.cells] == case["expected_unique_cell_ids"]
    assert distance_matrix.ids == case["expected_unique_cell_ids"]
    assert np.array_equal(
        distance_matrix.matrix,
        np.zeros((len(case["expected_unique_cell_ids"]), len(case["expected_unique_cell_ids"]))),
    )


def test_file_distance_provider_returns_zero_matrix_for_single_unique_genotype_without_cnp2cnp():
    cell = Genotype([2, 2], node_id=50, generation=1, cell_id=5)

    distance_matrix = Cnp2CnpFileDistanceProvider(default_ctbs_runtime_config()).compute([cell])

    assert distance_matrix.ids == [5]
    assert np.array_equal(distance_matrix.matrix, np.array([[0.0]]))


def test_pairwise_distance_helper_returns_zero_matrix_for_single_unique_genotype_without_config(monkeypatch):
    def fail_if_called(_runtime_config=None):
        raise AssertionError("single-genotype pairwise distance should not resolve cnp2cnp config")

    monkeypatch.setattr(ctbs, "_coerce_runtime_config", fail_if_called)

    ids, matrix = ctbs.distance_matrix_from_biopsy([Genotype([2, 2], node_id=50, generation=1, cell_id=5)])

    assert ids == [5]
    assert np.array_equal(matrix, np.array([[0.0]]))


@pytest.mark.parametrize("algorithm", FLAT_INPUT_ALGORITHMS, ids=lambda algorithm: algorithm.__name__)
def test_full_cnp_algorithms_accept_two_unique_biopsy_observations(algorithm):
    raw_cells = [
        Genotype([2, 2], node_id=50, generation=1, cell_id=5),
        Genotype([3, 2], node_id=70, generation=2, cell_id=7),
    ]

    tree, _, returned_root = build_evolution_tree(
        [raw_cells],
        only_nj=True,
        inids=[5, 7],
        indm=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        neighbor_joining=algorithm,
        seed=7,
    )

    assert returned_root in tree.nodes
    _assert_valid_rooted_tree(tree)
    reconstructed_labels = {data.get("cell_id") for _, data in tree.nodes(data=True)}
    assert {5, 7} <= reconstructed_labels


@pytest.mark.parametrize("algorithm", get_algorithms_to_test(), ids=lambda algorithm: algorithm.__name__)
def test_biopsy_guided_algorithms_accept_two_raw_observations_with_one_unique_distance_id(algorithm):
    cell_lists = [
        [Genotype([2, 2], node_id=50, generation=1, cell_id=5)],
        [Genotype([2, 2], node_id=51, generation=2, cell_id=5)],
    ]

    tree, _, returned_root = build_evolution_tree(
        cell_lists,
        inids=[5],
        indm=np.array([[0.0]], dtype=float),
        neighbor_joining=algorithm,
        seed=7,
    )

    assert returned_root in tree.nodes
    _assert_valid_rooted_tree(tree)
    reconstructed_labels = {data.get("cell_id") for _, data in tree.nodes(data=True)}
    assert reconstructed_labels == {5}


@pytest.mark.parametrize("case_path", SANITY_CASE_PATHS, ids=_case_id)
@pytest.mark.parametrize("algorithm", FLAT_INPUT_ALGORITHMS, ids=lambda algorithm: algorithm.__name__)
def test_all_full_cnp_algorithms_accept_repeated_biopsy_observations(case_path, algorithm):
    case = _load_case(case_path)
    cell_lists = _cell_lists(case)
    raw_cells = _flatten(cell_lists)
    matrix = case["distance_matrices"]["cnp2cnp"]

    tree, _, returned_root = build_evolution_tree(
        [raw_cells],
        only_nj=True,
        inids=matrix["ids"],
        indm=np.array(matrix["matrix"], dtype=float),
        neighbor_joining=algorithm,
        seed=7,
    )

    assert returned_root in tree.nodes
    _assert_valid_rooted_tree(tree)
    reconstructed_labels = {data.get("cell_id") for _, data in tree.nodes(data=True)}
    assert set(case["expected_unique_cell_ids"]) <= reconstructed_labels

    true_tree, true_root = _true_tree(case)
    similarity = grf_tree(true_tree, true_root, tree, _actual_root(tree))
    assert 0.0 <= similarity <= 1.0

    metrics = ancestors_unique_restricted_metrics(
        true_tree,
        tree,
        restrict_labels=set(case["expected_unique_cell_ids"]),
    )
    adf1 = metrics["F1"]
    assert 0.0 <= adf1 <= 1.0


@pytest.mark.parametrize("case_path", SANITY_CASE_PATHS, ids=_case_id)
@pytest.mark.parametrize(
    "algorithm",
    ORDERED_OCCURRENCE_ALGORITHMS,
    ids=lambda algorithm: algorithm.__name__,
)
def test_temporal_algorithms_preserve_repeated_biopsy_occurrences(case_path, algorithm):
    case = _load_case(case_path)
    cell_lists = _cell_lists(case)
    matrix = case["distance_matrices"]["cnp2cnp"]

    tree, node_levels, returned_root = build_evolution_tree(
        cell_lists,
        inids=matrix["ids"],
        indm=np.array(matrix["matrix"], dtype=float),
        neighbor_joining=algorithm,
        seed=7,
    )

    assert returned_root in tree
    _assert_valid_rooted_tree(tree)
    assert tree.number_of_nodes() == case["expected_raw_observation_count"]
    assert node_levels == nx.get_node_attributes(tree, "biopsy_level")
    assert sorted(data["cell_id"] for _, data in tree.nodes(data=True)) == sorted(
        cell.cell_id
        for level in cell_lists
        for cell in level
    )
