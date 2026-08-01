from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstructor import build_evolution_tree
from reconstructor_biopsy_blocks import copy_reconstruction_cell_lists
from simulator import Genotype


def _cell_lists(source_node_ids):
    return [
        [
            Genotype([2, 2], source_node_ids[0], generation=4, cell_id=5),
            Genotype([3, 2], source_node_ids[1], generation=4, cell_id=7),
        ],
        [
            Genotype([2, 2], source_node_ids[2], generation=8, cell_id=5),
            Genotype([3, 3], source_node_ids[3], generation=8, cell_id=9),
        ],
    ]


def _tree_signature(tree):
    def genome_signature(genome):
        if genome is None:
            return None
        values = np.asarray(genome).tolist()
        return None if values is None else tuple(values)

    nodes = sorted(
        (
            repr(node),
            data.get("cell_id"),
            genome_signature(data.get("genome")),
        )
        for node, data in tree.nodes(data=True)
    )
    edges = sorted(
        (repr(parent), repr(child), float(data.get("weight", 0.0)))
        for parent, child, data in tree.edges(data=True)
    )
    return nodes, edges


def test_reconstruction_boundary_discards_simulator_node_ids_without_mutating_input():
    cells = _cell_lists([100, 200, 300, 400])
    copied = copy_reconstruction_cell_lists(cells)

    assert [[cell.node_id for cell in level] for level in copied] == [[5, 7], [5, 9]]
    assert [[cell.node_id for cell in level] for level in cells] == [[100, 200], [300, 400]]


def test_reconstruction_is_invariant_to_simulator_node_renumbering():
    matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )
    first_tree, _, first_root = build_evolution_tree(
        _cell_lists([100, 200, 300, 400]),
        inids=[5, 7, 9],
        indm=matrix,
        seed=7,
    )
    second_tree, _, second_root = build_evolution_tree(
        _cell_lists([-70, 8000, 17, 42]),
        inids=[5, 7, 9],
        indm=matrix,
        seed=7,
    )

    assert first_root == second_root
    assert _tree_signature(first_tree) == _tree_signature(second_tree)
    assert all(
        "observation_key" not in data
        and "occurrence_kind" not in data
        and "source_observation_key" not in data
        for _, data in first_tree.nodes(data=True)
    )


def test_genotype_contains_only_reconstruction_domain_fields():
    cell = Genotype([2, 2], node_id=100, generation=4, cell_id=5)

    assert not hasattr(cell, "observation_key")
    assert not hasattr(cell, "occurrence_kind")
    assert not hasattr(cell, "source_observation_key")
