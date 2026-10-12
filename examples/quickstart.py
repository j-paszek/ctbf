"""JSON-configured simulation, sampling and reconstruction with paper methods M0–M7.

Run from the checkout with ``python -m examples.quickstart --help``.
All scientific operations use the existing public CTBF implementations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

from algorithm_evaluation.v5_algorithm_development_common import (
    ARM_SPEC_BY_ID, CASE_INPUT_SCHEMA_VERSION, reconstruct_development_arm,
)
from ctbs import (
    Cnp2CnpFileDistanceProvider, Cnp2CnpPairwiseDistanceProvider,
    CtbsRuntimeConfig, unique_cells_by_cell_id,
)
from ctbs_utils import to_newick
from evaluation_contract import evaluate_tree_pair_result
from simulator import CancerCellEvolutionSimulator
from trimmed_truth_evaluation import trim_true_tree
from .tree_gallery import write_gallery


DEFAULT_CONFIG = Path(__file__).with_name("quickstart.json")
METHODS = {
    "M0": ("Pooled classical NJ · partial", "classical_partial"),
    "M1": ("Pooled labeled closest-pair · full", "neighbor_joining_baseline"),
    "M2": ("Biopsy-guided, binary anticentral top · partial",
           "biopsy_guided_top_anticentral_binary_adaptive_median_prior_nn_bottom_deferred_tie"),
    "M3": ("Pooled rooted labeled-Q · full", "rooted_labeled_nj"),
    "M4": ("Biopsy-guided, classical NJ top · partial",
           "biopsy_guided_classical_adaptive_median_prior_nn_bottom_deferred_tie"),
    "M5": ("Pooled binary anticentral · full",
           "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony"),
    "M6": ("Biopsy-guided, binary anticentral top · full",
           "biopsy_guided_full_anticentral_binary_adaptive_median_prior_nn_bottom_deferred_tie"),
    "M7": ("Biopsy-guided, rooted labeled-Q top · full",
           "biopsy_guided_full_rooted_labeled_q_adaptive_median_prior_nn_bottom_deferred_tie"),
}
PARTIAL_METHODS = frozenset({"M0", "M2", "M4"})


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not a JSON value: {type(value).__name__}")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=json_value, allow_nan=False)
        stream.write("\n")


def configured_path(value, base_dir):
    """JSON paths are relative to its file; CLI overrides are relative to cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() if base_dir is None else Path(base_dir)) / path


def validate_sampling(sampling):
    if not isinstance(sampling, dict):
        raise ValueError("biopsy_sampling must be an object")
    mode = sampling.get("mode")
    if mode == "fraction" and set(sampling) == {"mode", "fraction", "lower_bound"}:
        fraction, lower = sampling["fraction"], sampling["lower_bound"]
        if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError("biopsy_sampling.fraction must be in (0, 1]")
        if type(lower) is not int or lower < 1:
            raise ValueError("biopsy_sampling.lower_bound must be an integer >= 1")
    elif mode == "fixed" and set(sampling) == {"mode", "size"}:
        if type(sampling["size"]) is not int or sampling["size"] < 1:
            raise ValueError("biopsy_sampling.size must be an integer >= 1")
    else:
        raise ValueError("biopsy_sampling needs mode=fraction with fraction/lower_bound, "
                         "or mode=fixed with size (no other fields)")


def validate_settings(settings, *, base_dir=None):
    expected = {"seed", "biopsy_generations", "biopsy_sampling", "methods",
                "max_distance_profiles", "cnp2cnp_timeout_seconds", "distance_workers",
                "visualize", "write_newick", "partial_ad_f1", "bedfile", "cnp2cnp",
                "output", "simulator"}
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError(f"Configuration must contain exactly: {sorted(expected)}")
    for key, minimum in (("seed", 0), ("distance_workers", 1), ("max_distance_profiles", 2)):
        if type(settings[key]) is not int or settings[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("visualize", "write_newick", "partial_ad_f1"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    for key in ("bedfile", "cnp2cnp", "output"):
        value = settings[key]
        if value is None and key != "output":
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path" + (" or null" if key != "output" else ""))
    validate_sampling(settings["biopsy_sampling"])
    timeout = settings["cnp2cnp_timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("cnp2cnp_timeout_seconds must be finite and positive")
    generations = settings["biopsy_generations"]
    if (not isinstance(generations, list) or len(generations) < 2
            or any(type(g) is not int or g < 0 for g in generations)
            or generations != sorted(set(generations))):
        raise ValueError("biopsy_generations must contain at least two increasing integers >= 0")
    methods = settings["methods"]
    if (not isinstance(methods, list) or not methods
            or any(not isinstance(m, str) or m not in METHODS for m in methods)
            or len(methods) != len(set(methods))):
        raise ValueError("methods must select distinct names from M0, M1, M2, M3, M4, M5, M6, M7")
    if not isinstance(settings["simulator"], dict):
        raise ValueError("simulator must contain the complete simulator JSON object")
    bedfile = None if settings["bedfile"] is None else configured_path(settings["bedfile"], base_dir)
    sim = CancerCellEvolutionSimulator(settings["simulator"], bedfile, seed=settings["seed"])
    if len({b.chromosome for b in sim.genome_bins}) != 1:
        raise ValueError("This runner uses single-chromosome cnp2cnp distances. "
                         "Use one chromosome; concatenating chromosomes would allow invalid cross-boundary events.")
    if generations[-1] > sim.config.number_of_generations:
        raise ValueError("A biopsy generation exceeds NUMBER_OF_GENERATIONS")
    return sim


def simulate(settings, *, base_dir=None):
    sim = validate_settings(settings, base_dir=base_dir)
    sim.run_simulation()
    counts = Counter(cell.generation for cell in sim.genotypes.values())
    biopsies, sampling = [], []
    for generation in settings["biopsy_generations"]:
        available = counts[generation]
        if not available:
            raise ValueError(f"Generation {generation} is empty; no biopsy can be taken")
        policy = settings["biopsy_sampling"]
        requested = (policy["size"] if policy["mode"] == "fixed" else
                     max(policy["lower_bound"], math.ceil(available * policy["fraction"])))
        size = min(available, requested)
        cells = sim.perform_biopsy(generation, biopsy_size=size, seed=settings["seed"])
        biopsies.append(cells)
        sampling.append(dict(generation=generation, available=available, requested=size,
                             selected=len(cells), labels=[c.cell_id for c in cells]))
    return sim, biopsies, sampling


def reconstruct(biopsies, distance, methods, seed):
    # Reuse the benchmark's exact compositions, including M2's top projection.
    # No truth-occurrence ids, node ids or truth tree enter this boundary.
    payload = {
        "schema_version": CASE_INPUT_SCHEMA_VERSION,
        "levels": [dict(biopsy_level=i, generation=int(biopsy[0].generation), states=[
            dict(state_label=c.cell_id, cnp=np.asarray(c.genome).tolist()) for c in biopsy
        ]) for i, biopsy in enumerate(biopsies)],
    }
    trees = {}
    for method in methods:
        tree, _, root, _ = reconstruct_development_arm(
            ARM_SPEC_BY_ID[METHODS[method][1]], payload, distance, reconstruction_seed=seed,
        )
        if not nx.is_arborescence(tree) or tree.in_degree(root) != 0:
            raise ValueError(f"{method} did not produce a rooted tree")
        trees[method] = tree
    return trees


def evaluate(truth, trimmed, trees, labels, *, partial_ad_f1=False):
    scores = {}
    for method, tree in trees.items():
        full = evaluate_tree_pair_result(truth, tree, labels)
        reduced = evaluate_tree_pair_result(trimmed, tree, labels)
        if full["status"] != "success" or reduced["status"] != "success":
            raise ValueError(f"{method} evaluation failed: {full.get('failure')}, {reduced.get('failure')}")
        show_ad = method not in PARTIAL_METHODS or partial_ad_f1
        scores[method] = dict(arm_id=METHODS[method][1],
                              output="partial" if method in PARTIAL_METHODS else "closed-state",
                              grf_full=full["metrics"]["grf"],
                              grf_trimmed=reduced["metrics"]["grf"],
                              ad_f1=full["metrics"]["ad_f1"] if show_ad else None,
                              ad_f1_role=("partial_diagnostic" if method in PARTIAL_METHODS else
                                          "closed_state_endpoint") if show_ad else "not_reported")
    return scores


def write_newick_trees(output, trees):
    """Export labeled topology only; JSON remains the lossless representation."""
    for key, tree in trees.items():
        export = tree.copy()
        for _, attrs in export.nodes(data=True):
            if attrs.get("cell_id") is None:
                attrs["cell_id"] = ""
        newick = to_newick(export, prefer_weight=False, use_events=False, sort_children=True)
        with (output / f"{key}.nwk").open("x", encoding="utf-8") as stream:
            stream.write(newick + "\n")


def run(settings, output=None, cnp2cnp=None, *, base_dir=None):
    # Check the configuration before simulation or any external execution.
    validate_settings(settings, base_dir=base_dir)
    output = (Path(output).expanduser() if output is not None else
              configured_path(settings["output"], base_dir))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    if cnp2cnp is None:
        if settings["cnp2cnp"] is None:
            raise ValueError("Set cnp2cnp in the JSON to your installed cnp2cnp.py path, or supply --cnp2cnp")
        cnp2cnp = configured_path(settings["cnp2cnp"], base_dir)
    cnp2cnp = Path(cnp2cnp).expanduser().resolve()
    if not cnp2cnp.is_file():
        raise FileNotFoundError(f"Install cnp2cnp separately and supply its cnp2cnp.py path: {cnp2cnp}")
    sim, biopsies, sampling = simulate(settings, base_dir=base_dir)
    cells = unique_cells_by_cell_id([c for biopsy in biopsies for c in biopsy])
    if len(cells) > settings["max_distance_profiles"]:
        raise ValueError(f"{len(cells)} distinct profiles exceed max_distance_profiles; "
                         "review the requested workload before raising this limit")
    print(f"Simulation: {len(sim.tree)} vertices; {len(cells)} distinct sampled CNPs", flush=True)
    runtime = CtbsRuntimeConfig(
        in_file_name="unused", out_file_name="unused", sim_dm="unused",
        cnp2cnp_folder=str(cnp2cnp.parent), cnp2cnp_file=str(cnp2cnp),
        true_tree_root_id=0, run_single_test={},
        cnp2cnp_timeout_seconds=settings["cnp2cnp_timeout_seconds"],
        cnp2cnp_capture_limit_bytes=1048576,
    )
    workers = settings["distance_workers"]
    provider = (Cnp2CnpFileDistanceProvider(runtime) if workers == 1 else
                Cnp2CnpPairwiseDistanceProvider(runtime, max_threads=workers))
    distance = provider.compute(cells)
    trees = reconstruct(biopsies, distance, settings["methods"], settings["seed"])
    truth = sim.canonicalized_tree_by_genome()
    labels = [c.cell_id for c in cells]
    trimmed = trim_true_tree(truth, labels)
    scores = evaluate(truth, trimmed, trees, labels, partial_ad_f1=settings["partial_ad_f1"])
    all_trees = {"true_tree": truth, "trimmed_true_tree": trimmed, **trees}
    summary = dict(seed=settings["seed"], truth_vertices=len(truth),
                   trimmed_truth_vertices=len(trimmed), distinct_sampled_profiles=len(cells),
                   sampling=sampling, methods=scores,
                   simulator_provenance=dict(config_sha256=sim.config_sha256, bed_sha256=sim.bed_sha256),
                   distance=dict(name="cnp2cnp d_any", symmetrization="minimum_bidirectional",
                                 construction="matrix" if workers == 1 else "pairwise",
                                 requested_workers=workers,
                                 cnp2cnp_sha256=hashlib.sha256(cnp2cnp.read_bytes()).hexdigest()),
                   numpy_version=np.__version__)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", settings)
    write_json(output / "summary.json", summary)
    write_json(output / "input.json", {"bins": [
        dict(chromosome=b.chromosome, start=b.start, end=b.end) for b in sim.genome_bins
    ], "biopsies": [
        {"generation": generation, "cells": [dict(cell_id=c.cell_id, genome=c.genome) for c in biopsy]}
        for generation, biopsy in zip(settings["biopsy_generations"], biopsies)
    ]})
    # Only CTBF-generated data are exported. No external matrix, log or source is copied.
    write_json(output / "trees.json", {key: nx.node_link_data(tree, edges="edges")
                                       for key, tree in all_trees.items()})
    if settings["write_newick"]:
        write_newick_trees(output, all_trees)
    if settings["visualize"]:
        write_gallery(output, all_trees, biopsies, METHODS, summary)
    for method, row in scores.items():
        ad = "—" if row["ad_f1"] is None else f"{row['ad_f1']:.3f}"
        qualifier = " (partial diagnostic)" if row["ad_f1_role"] == "partial_diagnostic" else ""
        print(f"{method}: GRF(full)={row['grf_full']:.3f}, GRF(trimmed)={row['grf_trimmed']:.3f}, AD-F1={ad}{qualifier}")
    destination = (f"open {output.resolve() / 'index.html'}" if settings["visualize"] else
                   f"results {output.resolve() / 'summary.json'}")
    print(f"complete: {len(all_trees)} trees; {destination}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="JSON containing simulator, biopsy and method settings")
    parser.add_argument("--cnp2cnp", type=Path, help="Override the JSON path to independently installed cnp2cnp.py")
    parser.add_argument("--output", type=Path, help="Override the JSON output directory (must be new)")
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as stream:
        settings = json.load(stream)
    run(settings, args.output, args.cnp2cnp, base_dir=args.config.resolve().parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
