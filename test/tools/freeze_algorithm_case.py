#!/usr/bin/env python
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from networkx.readwrite import json_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_evaluation.tester import CONFIG_BY_PROFILE, get_legacy_algorithms_to_test
from ctbs import distance_matrix_from_biopsy, use_cnp2cnp_to_compute_pairwise_distance
from ctbs_utils import get_biopsy_nodes_ids, to_newick
from evaluator_full import ancestors_unique_restricted_metrics
from reconstructor import build_evolution_tree
from simulator import CancerCellEvolutionSimulator


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "data" / "algorithm_cases"
DEFAULT_TREE_SAMPLES_DIR = PROJECT_ROOT / "test" / "data" / "tree_samples"
DEFAULT_BIOPSY_GENERATIONS = [4, 6, 8]


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _node_link_data(tree):
    try:
        return json_graph.node_link_data(tree, edges="links")
    except TypeError:
        return json_graph.node_link_data(tree)


def _write_json(path, data, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_ready(data), f, indent=2)
        f.write("\n")


def _build_simulator(config_path, bedfile, seed):
    if bedfile is None:
        simulator = CancerCellEvolutionSimulator(config_path, seed=seed)
    else:
        simulator = CancerCellEvolutionSimulator(config_path, bedfile, seed=seed)
    simulator.run_simulation()
    return simulator


def _perform_frozen_biopsies(simulator, biopsy_generations, biopsy_size_scalable, seed):
    biopsy_node_ids_by_generation = {}
    cell_lists = []
    for generation in biopsy_generations:
        biopsy = simulator.perform_biopsy(
            generation=generation,
            biopsy_size_scalable=biopsy_size_scalable,
            seed=seed,
        )
        biopsy_node_ids_by_generation[str(generation)] = [cell.node_id for cell in biopsy]
        if biopsy:
            cell_lists.append(biopsy)
    return biopsy_node_ids_by_generation, cell_lists


def _unique_cells_by_cell_id(cell_lists):
    unique = {}
    for cell in [cell for level in cell_lists for cell in level]:
        unique[cell.cell_id] = cell
    return list(unique.values())


def _distance_matrix_from_biopsy_serial(cells):
    n = len(cells)
    ids = [cell.get_id() for cell in cells]
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            input_str = (
                f">{cells[i].get_id()}\n{cells[i].get_cnp()}\n"
                f">{cells[j].get_id()}\n{cells[j].get_cnp()}\n"
            )
            dist = float(use_cnp2cnp_to_compute_pairwise_distance(input_str))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return ids, dist_matrix


def _metric_summary(true_tree, rec_tree, nj_tree):
    biopsy_cell_ids = get_biopsy_nodes_ids(rec_tree, nj_tree)
    rec_restricted = ancestors_unique_restricted_metrics(
        true_tree,
        rec_tree,
        restrict_labels=biopsy_cell_ids,
    )
    nj_restricted = ancestors_unique_restricted_metrics(
        true_tree,
        nj_tree,
        restrict_labels=biopsy_cell_ids,
    )
    return {
        "rec_ancestors_unique_restricted": {
            "precision": rec_restricted["precision"],
            "F1": rec_restricted["F1"],
        },
        "nj_ancestors_unique_restricted": {
            "precision": nj_restricted["precision"],
            "F1": nj_restricted["F1"],
        },
    }


def _algorithm_expectations(case, true_tree, cell_lists):
    expectations_by_output = {}
    ids = case["distance_matrix"]["ids"]
    dist_matrix = np.array(case["distance_matrix"]["matrix"], dtype=float)

    for algorithm in get_legacy_algorithms_to_test():
        build_kwargs = {
            "r": case["r_dist"],
            "seed": case["seed"],
            "inids": ids,
            "indm": dist_matrix,
            "neighbor_joining": algorithm,
        }
        rec_tree, _, rec_root = build_evolution_tree(copy.deepcopy(cell_lists), **build_kwargs)
        all_in_one_sample = [[copy.deepcopy(cell) for level in cell_lists for cell in level]]
        nj_tree, _, nj_root = build_evolution_tree(
            all_in_one_sample,
            only_nj=True,
            **build_kwargs,
        )
        expectation = {
            "rec_root": rec_root,
            "nj_root": nj_root,
            "rec_newick": to_newick(rec_tree),
            "nj_newick": to_newick(nj_tree),
        }
        expectation.update(_metric_summary(true_tree, rec_tree, nj_tree))
        key = json.dumps(_json_ready(expectation), sort_keys=True)
        expectations_by_output.setdefault(key, {"algorithms": [], **expectation})
        expectations_by_output[key]["algorithms"].append(algorithm.__name__)

    return list(expectations_by_output.values())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze a simulator tree, biopsy node IDs, distance matrix, and optional legacy algorithm expectations."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--r", type=int, required=True, dest="r_dist")
    parser.add_argument("--bss", type=float, required=True, dest="biopsy_size_scalable")
    parser.add_argument("--profile", choices=sorted(CONFIG_BY_PROFILE), default="base")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--bedfile", type=Path, default=None)
    parser.add_argument("--biopsy-generation", type=int, action="append", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tree-samples-dir", type=Path, default=DEFAULT_TREE_SAMPLES_DIR)
    parser.add_argument("--no-expectations", action="store_true")
    parser.add_argument(
        "--parallel-distance",
        action="store_true",
        help="Use the existing multiprocessing distance helper instead of the serial fixture-safe path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config if args.config is not None else CONFIG_BY_PROFILE[args.profile]
    biopsy_generations = args.biopsy_generation or DEFAULT_BIOPSY_GENERATIONS
    bss_token = str(args.biopsy_size_scalable).replace(".", "")
    case_id = args.case_id or f"seed{args.seed}_r{args.r_dist}bss{bss_token}"

    simulator = _build_simulator(config_path, args.bedfile, args.seed)
    tree_sample_path = args.tree_samples_dir / f"{args.seed}.json"
    _write_json(tree_sample_path, _node_link_data(simulator.tree), args.overwrite)

    biopsy_node_ids_by_generation, cell_lists = _perform_frozen_biopsies(
        simulator,
        biopsy_generations,
        args.biopsy_size_scalable,
        args.seed,
    )
    unique_cells = _unique_cells_by_cell_id(cell_lists)
    if args.parallel_distance:
        ids, dist_matrix = distance_matrix_from_biopsy(unique_cells)
    else:
        ids, dist_matrix = _distance_matrix_from_biopsy_serial(unique_cells)

    case = {
        "case_id": case_id,
        "tree_sample": str(Path("..") / "tree_samples" / tree_sample_path.name),
        "seed": args.seed,
        "profile": args.profile,
        "config": str(config_path),
        "bedfile": str(args.bedfile) if args.bedfile is not None else None,
        "r_dist": args.r_dist,
        "biopsy_size_scalable": args.biopsy_size_scalable,
        "biopsy_generations": biopsy_generations,
        "biopsy_node_ids_by_generation": biopsy_node_ids_by_generation,
        "distance_matrix": {
            "ids": ids,
            "matrix": dist_matrix,
        },
    }
    if not args.no_expectations:
        case["expectation_groups"] = _algorithm_expectations(case, simulator.tree, cell_lists)

    output_path = args.output_dir / f"{case_id}.json"
    _write_json(output_path, case, args.overwrite)
    print(f"Wrote tree sample: {tree_sample_path}")
    print(f"Wrote algorithm case: {output_path}")


if __name__ == "__main__":
    main()
