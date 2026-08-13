#!/usr/bin/env python
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_evaluation.tester import (  # noqa: E402
    CONFIG_BY_PROFILE,
    DEFAULT_BIOPSY_GENERATIONS,
    LEGACY_ALGORITHM_NAMES,
    build_variant_name,
    get_algorithms_to_test,
    load_seeds,
    select_algorithm_indices,
)
from ctbs import (  # noqa: E402
    compute_symmetric_cnp2cnp_distance,
    distance_matrix_from_cnp2cnp_matrix_mode,
    load_ctbs_runtime_config,
)
from distance_semantics import cnp2cnp_provenance  # noqa: E402
from ctbs_utils import to_newick  # noqa: E402
from evaluator import (  # noqa: E402
    cluster_evaluation_context,
    exact_and_legacy_grf_from_cluster_contexts,
    legacy_set_grf_distance_from_cluster_contexts as evaluator_legacy_set_grf_distance_from_cluster_contexts,
    legacy_set_grf_similarity_from_cluster_contexts as evaluator_legacy_set_grf_similarity_from_cluster_contexts,
)
from evaluator_full import ancestors_unique_restricted_metrics, tree_evaluation_context  # noqa: E402
from reconstructor import build_evolution_tree  # noqa: E402
from simulator import CancerCellEvolutionSimulator, Genotype  # noqa: E402
from simulator_events import count_edge_events  # noqa: E402


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test" / "data" / "algorithm_cases"
PUBLICATION_HEATMAP_ALGORITHM_NAMES = list(LEGACY_ALGORITHM_NAMES)
REFERENCE_ALGORITHM_NAMES = PUBLICATION_HEATMAP_ALGORITHM_NAMES
EXT_GRF_METRIC_FIELD = "ext_grf"
LEGACY_GRF_SET_SIMILARITY_FIELD = "grf_legacy_set_similarity"
ADAPTIVE_RADIUS_MODE = "adaptive_mean_pairwise_input_distance"
DEFAULT_ADAPTIVE_RADIUS_SCALE = 1.0
ADAPTIVE_RADIUS_GRID = (0.5, 0.75, 1.0, 1.25)
LEGACY_VARIANT_PRESETS = {
    "r2bss025": {"r_dist": 2, "biopsy_size_scalable": 0.25, "profile": "base"},
    "r2bss05": {"r_dist": 2, "biopsy_size_scalable": 0.5, "profile": "base"},
    "r2bss075": {"r_dist": 2, "biopsy_size_scalable": 0.75, "profile": "base"},
    "r4bss05": {"r_dist": 4, "biopsy_size_scalable": 0.5, "profile": "base"},
    "r4bss075": {"r_dist": 4, "biopsy_size_scalable": 0.75, "profile": "base"},
    "r4bss05high": {"r_dist": 4, "biopsy_size_scalable": 0.5, "profile": "high"},
    "r4bss05highdm": {"r_dist": 4, "biopsy_size_scalable": 0.5, "profile": "highdm"},
}
ADAPTIVE_RADIUS_VARIANT_PRESETS = {
    "rAbss025": {
        "radius_mode": ADAPTIVE_RADIUS_MODE,
        "adaptive_radius_scale": DEFAULT_ADAPTIVE_RADIUS_SCALE,
        "biopsy_size_scalable": 0.25,
        "profile": "base",
        "source_variant": "r2bss025",
    },
    "rAbss05": {
        "radius_mode": ADAPTIVE_RADIUS_MODE,
        "adaptive_radius_scale": DEFAULT_ADAPTIVE_RADIUS_SCALE,
        "biopsy_size_scalable": 0.5,
        "profile": "base",
        "source_variant": "r2bss05",
    },
    "rAbss075": {
        "radius_mode": ADAPTIVE_RADIUS_MODE,
        "adaptive_radius_scale": DEFAULT_ADAPTIVE_RADIUS_SCALE,
        "biopsy_size_scalable": 0.75,
        "profile": "base",
        "source_variant": "r2bss075",
    },
    "rAbss05high": {
        "radius_mode": ADAPTIVE_RADIUS_MODE,
        "adaptive_radius_scale": DEFAULT_ADAPTIVE_RADIUS_SCALE,
        "biopsy_size_scalable": 0.5,
        "profile": "high",
        "source_variant": "r4bss05high",
    },
    "rAbss05highdm": {
        "radius_mode": ADAPTIVE_RADIUS_MODE,
        "adaptive_radius_scale": DEFAULT_ADAPTIVE_RADIUS_SCALE,
        "biopsy_size_scalable": 0.5,
        "profile": "highdm",
        "source_variant": "r4bss05highdm",
    },
}
VARIANT_PRESETS = {
    **LEGACY_VARIANT_PRESETS,
    **ADAPTIVE_RADIUS_VARIANT_PRESETS,
}
LEGACY_VARIANT_NAMES = list(LEGACY_VARIANT_PRESETS)
ADAPTIVE_RADIUS_VARIANT_NAMES = list(ADAPTIVE_RADIUS_VARIANT_PRESETS)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def node_link_data(tree):
    try:
        return json_graph.node_link_data(tree, edges="links")
    except TypeError:
        return json_graph.node_link_data(tree)


def write_json(path, data, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_ready(data), f, indent=2)
        f.write("\n")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def existing_seeds_for_variant(output_root, variant_name):
    variant_dir = Path(output_root) / variant_name
    if not variant_dir.exists():
        return []
    seeds = []
    for seed_dir in variant_dir.iterdir():
        if seed_dir.is_dir() and seed_dir.name.isdigit() and (seed_dir / "input.json").exists():
            seeds.append(int(seed_dir.name))
    return sorted(seeds)


def source_variant_name(variant_name, variant):
    return variant.get("source_variant", variant_name)


def resolve_variants(selected_variants):
    names = selected_variants or LEGACY_VARIANT_NAMES
    unknown = sorted(set(names) - set(VARIANT_PRESETS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(VARIANT_PRESETS)}")
    return [(name, VARIANT_PRESETS[name]) for name in names]


def is_adaptive_radius_variant(variant):
    return variant.get("radius_mode") == ADAPTIVE_RADIUS_MODE


def mean_pairwise_input_distance(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] < 2:
        return 0.0
    values = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return float(values.mean()) if values.size else 0.0


def adaptive_radius_from_matrix(matrix, scale):
    return float(scale) * mean_pairwise_input_distance(matrix)


def resolve_variant_radius(variant, matrix, adaptive_radius_scale=None):
    if not is_adaptive_radius_variant(variant):
        return variant["r_dist"], None

    scale = (
        float(adaptive_radius_scale)
        if adaptive_radius_scale is not None
        else float(variant.get("adaptive_radius_scale", DEFAULT_ADAPTIVE_RADIUS_SCALE))
    )
    d_mean = mean_pairwise_input_distance(matrix)
    r_dist = float(scale * d_mean)
    return r_dist, {
        "radius_mode": ADAPTIVE_RADIUS_MODE,
        "adaptive_radius_scale": scale,
        "adaptive_radius_distance_mean": d_mean,
        "adaptive_radius_formula": "rA = adaptive_radius_scale * adaptive_radius_distance_mean",
    }


def apply_variant_metadata(input_case, variant_name, variant, adaptive_radius_scale=None):
    matrix = input_case["distance_matrices"]["cnp2cnp"]["matrix"]
    r_dist, radius_metadata = resolve_variant_radius(
        variant,
        matrix,
        adaptive_radius_scale=adaptive_radius_scale,
    )
    input_case["case_id"] = f"{variant_name}_seed{input_case['seed']}"
    input_case["variant"] = variant_name
    input_case["profile"] = variant["profile"]
    input_case["r_dist"] = r_dist
    input_case["biopsy_size_scalable"] = variant["biopsy_size_scalable"]
    for key in [
        "radius_mode",
        "adaptive_radius_scale",
        "adaptive_radius_distance_mean",
        "adaptive_radius_formula",
    ]:
        input_case.pop(key, None)
    if radius_metadata is not None:
        input_case.update(radius_metadata)
    return input_case


def input_case_from_existing(source_input_case, variant_name, variant, adaptive_radius_scale=None):
    input_case = copy.deepcopy(source_input_case)
    return apply_variant_metadata(
        input_case,
        variant_name,
        variant,
        adaptive_radius_scale=adaptive_radius_scale,
    )


def algorithm_by_name():
    return {algorithm.__name__: algorithm for algorithm in get_algorithms_to_test()}


def resolve_reference_algorithm_indices(algorithm_indexes=None, algorithm_names=None):
    names = algorithm_names if algorithm_names is not None else REFERENCE_ALGORITHM_NAMES
    return select_algorithm_indices(
        get_algorithms_to_test(),
        algorithm_indexes=algorithm_indexes,
        algorithm_names=names,
    )


def case_dir(output_root, variant_name, seed):
    return Path(output_root) / variant_name / str(seed)


def input_path(output_root, variant_name, seed):
    return case_dir(output_root, variant_name, seed) / "input.json"


def result_path(output_root, variant_name, seed, mode, algorithm_name):
    return case_dir(output_root, variant_name, seed) / mode / f"{algorithm_name}.json"


def genotype_to_json(cell):
    return {
        "cell_id": cell.cell_id,
        "generation": cell.generation,
        "genome": cell.genome,
    }


def genotypes_from_json(cells):
    return [
        Genotype(
            genome=cell["genome"],
            node_id=cell.get("node_id", cell["cell_id"]),
            generation=cell.get("generation"),
            cell_id=cell.get("cell_id"),
        )
        for cell in cells
    ]


def unique_cells_by_cell_id(cell_lists):
    unique = {}
    for cell in [cell for level in cell_lists for cell in level]:
        if cell.cell_id not in unique:
            unique[cell.cell_id] = cell
    return list(unique.values())


def parse_cnp2cnp_matrix(path):
    with open(path, "r") as f:
        n = int(f.readline())
        ids = []
        matrix = []
        for _ in range(n):
            parts = f.readline().strip().split()
            ids.append(int(parts[0]))
            matrix.append([float(value) for value in parts[1:]])
    return ids, np.array(matrix, dtype=float)


def write_cnp2cnp_input(path, cells):
    with open(path, "w") as f:
        for cell in cells:
            f.write(f">{cell.get_id()}\n")
            f.write(f"{cell.get_cnp()}\n")


def cnp2cnp_distance_matrix(cells):
    ids, matrix, _provenance = cnp2cnp_distance_matrix_with_provenance(cells)
    return ids, matrix


def cnp2cnp_distance_matrix_with_provenance(cells):
    if len(cells) <= 1:
        return (
            [cell.get_id() for cell in cells],
            np.zeros((len(cells), len(cells)), dtype=float),
            cnp2cnp_provenance(None, construction="trivial_singleton"),
        )
    runtime_config = load_ctbs_runtime_config()
    distance_matrix = distance_matrix_from_cnp2cnp_matrix_mode(
        cells,
        runtime_config=runtime_config,
    )
    return distance_matrix.ids, distance_matrix.matrix, distance_matrix.provenance


def legacy_cnp2cnp_ordered_distance_matrix(cells):
    """Reproduce historical one-order matrices; never use for new evidence."""
    if len(cells) <= 1:
        return [cell.get_id() for cell in cells], np.zeros((len(cells), len(cells)), dtype=float)
    cnp2cnp_file = load_ctbs_runtime_config().cnp2cnp_file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_file = tmpdir_path / "biopsy.txt"
        output_file = tmpdir_path / "cnp_distance_matrix.txt"
        write_cnp2cnp_input(input_file, cells)
        subprocess.run(
            [sys.executable, cnp2cnp_file, "-m", "matrix", "-i", str(input_file), "-o", str(output_file)],
            check=True,
        )
        return parse_cnp2cnp_matrix(output_file)


def cnp2cnp_pairwise_distance_matrix(cells):
    n = len(cells)
    ids = [cell.get_id() for cell in cells]
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_symmetric_cnp2cnp_distance(cells[i], cells[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return ids, dist_matrix


def true_tree_distance_matrix(tree, node_ids):
    undirected = tree.to_undirected()
    n = len(node_ids)
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            path = nx.shortest_path(undirected, source=node_ids[i], target=node_ids[j])
            distance = 0
            for u, v in zip(path[:-1], path[1:]):
                edge_data = tree.get_edge_data(u, v) or tree.get_edge_data(v, u) or {}
                events = edge_data.get("events")
                if isinstance(events, (list, tuple)):
                    # CTBF v5: an empty typed list is a zero-event persistence
                    # edge and must not fall through to a unit edge weight.
                    distance += count_edge_events(events)
                elif isinstance(events, str) and events.strip():
                    # Historical frozen cases used nonempty semicolon strings.
                    distance += count_edge_events(events)
                else:
                    # Preserve only the frozen-case helper's historical
                    # unlabeled/empty-string fallback; this is not v2 truth
                    # semantics and is never used by simulator.to_distance_matrix.
                    distance += float(edge_data.get("weight", 1))
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
    return node_ids, dist_matrix


def root_id(tree):
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root, found {len(roots)}")
    return roots[0]


def legacy_set_grf_distance_from_cluster_contexts(
    true_cluster_context,
    reconstructed_cluster_context,
    *,
    jaccard_cache=None,
):
    return evaluator_legacy_set_grf_distance_from_cluster_contexts(
        true_cluster_context,
        reconstructed_cluster_context,
        jaccard_cache=jaccard_cache,
    )


def legacy_set_grf_similarity_from_cluster_contexts(
    true_cluster_context,
    reconstructed_cluster_context,
    *,
    jaccard_cache=None,
):
    return evaluator_legacy_set_grf_similarity_from_cluster_contexts(
        true_cluster_context,
        reconstructed_cluster_context,
        jaccard_cache=jaccard_cache,
    )


def legacy_set_grf_distance_tree(true_tree, true_root, reconstructed_tree, reconstructed_root):
    return legacy_set_grf_distance_from_cluster_contexts(
        cluster_evaluation_context(true_tree, true_root),
        cluster_evaluation_context(reconstructed_tree, reconstructed_root),
    )


def legacy_set_grf_similarity_tree(true_tree, true_root, reconstructed_tree, reconstructed_root):
    return 1 - legacy_set_grf_distance_tree(
        true_tree,
        true_root,
        reconstructed_tree,
        reconstructed_root,
    )


def metric_summary_from_contexts(true_eval_context, reconstructed_eval_context):
    restricted_labels = set(reconstructed_eval_context.labels.values())
    restricted_metrics = ancestors_unique_restricted_metrics(
        true_eval_context,
        reconstructed_eval_context,
        restrict_labels=restricted_labels,
    )
    true_cluster_context = cluster_evaluation_context(true_eval_context)
    reconstructed_cluster_context = cluster_evaluation_context(
        reconstructed_eval_context,
    )
    ext_grf, legacy_grf = exact_and_legacy_grf_from_cluster_contexts(
        true_cluster_context,
        reconstructed_cluster_context,
        jaccard_cache={},
    )
    return {
        "ancestors_unique_restricted": restricted_metrics,
        "grf": 1 - ext_grf,
        EXT_GRF_METRIC_FIELD: ext_grf,
        LEGACY_GRF_SET_SIMILARITY_FIELD: legacy_grf,
    }


def metric_summary(true_tree, reconstructed_tree):
    return metric_summary_from_contexts(
        tree_evaluation_context(true_tree),
        tree_evaluation_context(reconstructed_tree),
    )


def build_simulator(config_path, bedfile, seed):
    simulator = CancerCellEvolutionSimulator(config_path, bedfile, seed=seed) if bedfile else CancerCellEvolutionSimulator(
        config_path,
        seed=seed,
    )
    simulator.run_simulation()
    return simulator


def perform_biopsies(simulator, biopsy_generations, biopsy_size_scalable, seed):
    biopsies = []
    cell_lists = []
    for generation in biopsy_generations:
        biopsy = simulator.perform_biopsy(
            generation=generation,
            biopsy_size_scalable=biopsy_size_scalable,
            seed=seed,
        )
        biopsies.append({
            "generation": generation,
            "cells": [genotype_to_json(cell) for cell in biopsy],
        })
        if biopsy:
            cell_lists.append(biopsy)
    return biopsies, cell_lists


def input_case_from_simulation(variant_name, variant, seed, config_path, bedfile, adaptive_radius_scale=None):
    simulator = build_simulator(config_path, bedfile, seed)
    biopsies, cell_lists = perform_biopsies(
        simulator,
        DEFAULT_BIOPSY_GENERATIONS,
        variant["biopsy_size_scalable"],
        seed,
    )
    unique_cells = unique_cells_by_cell_id(cell_lists)
    cnp_ids, cnp_matrix, cnp_provenance = cnp2cnp_distance_matrix_with_provenance(
        unique_cells
    )
    true_ids, true_matrix = true_tree_distance_matrix(simulator.tree, cnp_ids)
    input_case = {
        "case_id": f"{variant_name}_seed{seed}",
        "variant": variant_name,
        "seed": seed,
        "profile": variant["profile"],
        "config": str(config_path),
        "bedfile": str(bedfile) if bedfile is not None else None,
        "r_dist": variant.get("r_dist"),
        "biopsy_size_scalable": variant["biopsy_size_scalable"],
        "biopsy_generations": DEFAULT_BIOPSY_GENERATIONS,
        "true_tree": node_link_data(simulator.tree),
        "biopsies": biopsies,
        "distance_matrices": {
            "cnp2cnp": {
                "ids": cnp_ids,
                "matrix": cnp_matrix,
                "provenance": cnp_provenance,
            },
            "true_tree": {
                "ids": true_ids,
                "matrix": true_matrix,
            },
        },
    }
    return apply_variant_metadata(
        input_case,
        variant_name,
        variant,
        adaptive_radius_scale=adaptive_radius_scale,
    )


def cell_lists_from_input(input_case):
    cell_lists = []
    for biopsy in input_case["biopsies"]:
        cells = genotypes_from_json(biopsy["cells"])
        if cells:
            cell_lists.append(cells)
    return cell_lists


def true_tree_from_input(input_case):
    return json_graph.node_link_graph(input_case["true_tree"], directed=True, edges="links")


def reconstruction_result(input_case, algorithm, mode):
    cell_lists = cell_lists_from_input(input_case)
    all_in_one_sample = [[copy.deepcopy(cell) for level in cell_lists for cell in level]]
    matrix = input_case["distance_matrices"]["cnp2cnp"]
    build_kwargs = {
        "r": input_case["r_dist"],
        "seed": input_case["seed"],
        "inids": matrix["ids"],
        "indm": np.array(matrix["matrix"], dtype=float),
        "neighbor_joining": algorithm,
    }
    if mode == "full_cnp":
        reconstructed_tree, _, reconstructed_root = build_evolution_tree(
            all_in_one_sample,
            only_nj=True,
            **build_kwargs,
        )
    elif mode == "biopsy_guided_top":
        reconstructed_tree, _, reconstructed_root = build_evolution_tree(
            copy.deepcopy(cell_lists),
            **build_kwargs,
        )
    else:
        raise ValueError(f"Unsupported reconstruction mode: {mode}")

    true_tree = true_tree_from_input(input_case)
    return {
        "case_id": input_case["case_id"],
        "variant": input_case["variant"],
        "seed": input_case["seed"],
        "algorithm": getattr(algorithm, "__name__", str(algorithm)),
        "mode": mode,
        "root": reconstructed_root,
        "newick": to_newick(reconstructed_tree),
        "reconstructed_tree": node_link_data(reconstructed_tree),
        "metrics": metric_summary(true_tree, reconstructed_tree),
    }


def write_seed_case(
    output_root,
    variant_name,
    variant,
    seed,
    algorithms,
    overwrite,
    input_only,
    bedfile=None,
    results_only=False,
    skip_existing=False,
    source_cases_root=None,
    adaptive_radius_scale=None,
):
    config_path = CONFIG_BY_PROFILE[variant["profile"]]
    current_input_path = input_path(output_root, variant_name, seed)
    if results_only:
        input_case = load_json(current_input_path)
    elif source_cases_root is not None:
        if skip_existing and current_input_path.exists():
            input_case = load_json(current_input_path)
        else:
            source_input_path = input_path(
                source_cases_root,
                source_variant_name(variant_name, variant),
                seed,
            )
            input_case = input_case_from_existing(
                load_json(source_input_path),
                variant_name,
                variant,
                adaptive_radius_scale=adaptive_radius_scale,
            )
            write_json(current_input_path, input_case, overwrite)
    else:
        input_case = input_case_from_simulation(
            variant_name,
            variant,
            seed,
            config_path,
            bedfile,
            adaptive_radius_scale=adaptive_radius_scale,
        )
        if not (skip_existing and current_input_path.exists()):
            write_json(current_input_path, input_case, overwrite)

    if input_only:
        return

    for algorithm in algorithms:
        algorithm_name = getattr(algorithm, "__name__", str(algorithm))
        for mode in ["full_cnp", "biopsy_guided_top"]:
            output_path = result_path(output_root, variant_name, seed, mode, algorithm_name)
            if skip_existing and output_path.exists():
                continue
            write_json(output_path, reconstruction_result(input_case, algorithm, mode), overwrite)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze nested algorithm fixtures for benchmark variants and publication heatmap algorithms."
    )
    parser.add_argument("--variant", action="append", choices=sorted(VARIANT_PRESETS),
                        help=(
                            "Benchmark variant to freeze. Can be passed multiple times. "
                            "Defaults to legacy fixed-r variants."
                        ))
    parser.add_argument("--seed", type=int, action="append", default=None,
                        help="Seed to freeze. Can be passed multiple times. Overrides --seeds-file.")
    parser.add_argument("--seeds-file", type=Path, default=PROJECT_ROOT / "test" / "data" / "seeds.json")
    parser.add_argument("--algorithm-index", type=int, action="append", default=None)
    parser.add_argument("--algorithm-name", action="append", default=None,
                        help="Algorithm name to freeze. Defaults to all publication heatmap algorithms.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source-cases-root",
        type=Path,
        default=None,
        help=(
            "Copy frozen input.json data from this cases root before applying the selected "
            "variant metadata. Intended for scratch adaptive-radius cases."
        ),
    )
    parser.add_argument(
        "--adaptive-radius-scale",
        type=float,
        default=None,
        help=(
            "Alpha for adaptive-radius variants. For rA variants, r_dist is set to "
            "alpha times the mean pairwise input distance d."
        ),
    )
    parser.add_argument("--input-only", action="store_true")
    parser.add_argument("--results-only", action="store_true",
                        help="Reuse existing input.json files and write only algorithm result JSON files.")
    parser.add_argument("--existing-seeds", action="store_true",
                        help="Use seeds that already have input.json under each selected variant.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Do not rewrite files that already exist.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of seeds per variant after filtering.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = resolve_variants(args.variant)
    if args.seed is not None:
        fixed_seeds = args.seed
        seeds_by_variant = {name: fixed_seeds for name, _ in variants}
    elif args.existing_seeds:
        seed_root = args.source_cases_root if args.source_cases_root is not None else args.output_root
        seeds_by_variant = {
            name: existing_seeds_for_variant(seed_root, source_variant_name(name, variant))
            for name, variant in variants
        }
    else:
        fixed_seeds = load_seeds(args.seeds_file)
        seeds_by_variant = {name: fixed_seeds for name, _ in variants}
    if args.limit is not None:
        seeds_by_variant = {
            name: seeds[:args.limit]
            for name, seeds in seeds_by_variant.items()
        }
    algorithm_indices = resolve_reference_algorithm_indices(args.algorithm_index, args.algorithm_name)
    algorithms = [get_algorithms_to_test()[index] for index in algorithm_indices]

    print("Variants:", ", ".join(name for name, _ in variants))
    for variant_name, _ in variants:
        seeds = seeds_by_variant[variant_name]
        print(f"Seeds for {variant_name}:", len(seeds), ", ".join(str(seed) for seed in seeds))
    print("Algorithms:", ", ".join(getattr(algorithm, "__name__", str(algorithm)) for algorithm in algorithms))
    print("Output root:", args.output_root)
    if args.source_cases_root is not None:
        print("Source cases root:", args.source_cases_root)
    if args.adaptive_radius_scale is not None:
        print("Adaptive radius scale:", args.adaptive_radius_scale)
    if args.dry_run:
        return

    failures = []
    for variant_name, variant in variants:
        for seed in seeds_by_variant[variant_name]:
            try:
                write_seed_case(
                    args.output_root,
                    variant_name,
                    variant,
                    seed,
                    algorithms,
                    args.overwrite,
                    args.input_only,
                    results_only=args.results_only,
                    skip_existing=args.skip_existing,
                    source_cases_root=args.source_cases_root,
                    adaptive_radius_scale=args.adaptive_radius_scale,
                )
                print(f"Wrote {variant_name}/{seed}")
            except Exception as exc:
                failures.append({"variant": variant_name, "seed": seed, "error": str(exc)})
                print(f"Failed {variant_name}/{seed}: {exc}", file=sys.stderr)
                traceback.print_exc()
                if args.fail_fast:
                    raise

    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure['variant']}/{failure['seed']}: {failure['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
