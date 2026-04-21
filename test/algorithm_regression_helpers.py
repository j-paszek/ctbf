import copy
import json
from pathlib import Path
import sys

import numpy as np
from networkx.readwrite import json_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_evaluation.tester import get_algorithms_to_test
from ctbs_utils import get_biopsy_nodes_ids, to_newick
from evaluator_full import evaluate_4
from reconstructor import build_evolution_tree
from simulator import CancerCellEvolutionSimulator


def load_frozen_case(path):
    case_path = Path(path)
    with open(case_path, "r") as f:
        case = json.load(f)
    case["_case_dir"] = case_path.parent
    return case


def load_case_simulator(case):
    tree_path = (case["_case_dir"] / case["tree_sample"]).resolve()
    with open(tree_path, "r") as f:
        tree_data = json.load(f)
    tree = json_graph.node_link_graph(tree_data, directed=True, edges="links")
    return CancerCellEvolutionSimulator.from_tree(tree)


def algorithm_by_name():
    return {algorithm.__name__: algorithm for algorithm in get_algorithms_to_test()}


def expand_case_expectations(case):
    for group in case["expectation_groups"]:
        for algorithm_name in group["algorithms"]:
            expectation = dict(group)
            expectation.pop("algorithms")
            expectation["algorithm_name"] = algorithm_name
            yield expectation


def frozen_biopsy_cell_lists(case, simulator):
    cell_lists = []
    for generation in case["biopsy_generations"]:
        node_ids = case["biopsy_node_ids_by_generation"][str(generation)]
        cells = [copy.deepcopy(simulator.genotypes[node_id]) for node_id in node_ids]
        if cells:
            cell_lists.append(cells)
    return cell_lists


def reconstruct_frozen_case(case, algorithm):
    simulator = load_case_simulator(case)
    cell_lists = frozen_biopsy_cell_lists(case, simulator)
    all_in_one_sample = [[copy.deepcopy(cell) for level in cell_lists for cell in level]]
    matrix_spec = case["distance_matrix"]
    ids = matrix_spec["ids"]
    dist_matrix = np.array(matrix_spec["matrix"], dtype=float)
    build_kwargs = {
        "r": case["r_dist"],
        "seed": case["seed"],
        "inids": ids,
        "indm": dist_matrix,
        "neighbor_joining": algorithm,
    }

    rec_tree, _, rec_root = build_evolution_tree(copy.deepcopy(cell_lists), **build_kwargs)
    nj_tree, _, nj_root = build_evolution_tree(
        all_in_one_sample,
        only_nj=True,
        **build_kwargs,
    )
    biopsy_labels = get_biopsy_nodes_ids(rec_tree, nj_tree)
    rec_metrics = evaluate_4(
        simulator.tree,
        rec_tree,
        restrict_labels=biopsy_labels,
    )["ancestors_unique_restricted"]
    nj_metrics = evaluate_4(
        simulator.tree,
        nj_tree,
        restrict_labels=biopsy_labels,
    )["ancestors_unique_restricted"]

    return {
        "rec_root": rec_root,
        "nj_root": nj_root,
        "rec_newick": to_newick(rec_tree),
        "nj_newick": to_newick(nj_tree),
        "rec_ancestors_unique_restricted": {
            "precision": rec_metrics["precision"],
            "F1": rec_metrics["F1"],
        },
        "nj_ancestors_unique_restricted": {
            "precision": nj_metrics["precision"],
            "F1": nj_metrics["F1"],
        },
    }
