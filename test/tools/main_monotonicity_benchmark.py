#!/usr/bin/env python
import argparse
import copy
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import traceback

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph
from scipy.stats import wilcoxon

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_evaluation.tester import (  # noqa: E402
    CONFIG_BY_PROFILE,
    DEFAULT_SEEDS_FILE,
    load_seeds,
    select_algorithm_indices,
)
from ctbf_constraints import MIN_TOTAL_BIOPSY_CELLS  # noqa: E402
from ctbs_utils import to_newick  # noqa: E402
from reconstructor import build_evolution_tree  # noqa: E402
from reconstructor_registry import get_algorithms_to_test  # noqa: E402
from simulator import CancerCellEvolutionSimulator, Genotype  # noqa: E402
from freeze_algorithm_variant_cases import (  # noqa: E402
    cnp2cnp_distance_matrix,
    genotypes_from_json,
    json_ready,
    metric_summary,
    node_link_data,
    true_tree_distance_matrix,
    unique_cells_by_cell_id,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test" / "data" / "main"
CORPUS_NAME = "main_monotonicity"
DEFAULT_GENOME_LENGTHS = (10,)
DEFAULT_GENERATION_COUNTS = (10,)
DEFAULT_BIOPSY_SIZE_SCALABLES = (0.25, 0.5, 0.75)
DEFAULT_BIOPSY_LEVEL_COUNTS = (2, 3, 4)
DEFAULT_EVENT_PROBS = (0.01, 0.05)
DEFAULT_EVENT_SHAPES = {
    "low": (0.01, 1),
    "high": (0.05, 1),
    "highdm": (0.05, 3),
}
DEFAULT_CORE_ALGORITHM_NAMES = (
    "neighbor_joining_baseline",
    "neighbor_joining_hybrid_opt",
    "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
)
MODES = ("full_cnp", "biopsy_guided_top")
METRICS = ("adf1", "grf")
CHECK_TOLERANCE = 1e-12
CASE_ID_PATTERN = re.compile(
    r"^gl(?P<genome_length>\d+)_"
    r"g(?P<generation_count>\d+)_"
    r"seed(?P<seed>\d+)_"
)
METRIC_FIELDS = {
    "adf1": ("metrics", "ancestors_unique_restricted", "F1"),
    "grf": ("metrics", "grf"),
}
TIMING_REPORT_COLUMNS = [
    "stage",
    "operation",
    "count",
    "input_files",
    "instances",
    "skipped",
    "failed",
    "read_json_seconds",
    "core_seconds",
    "write_json_seconds",
    "total_seconds",
    "algorithm",
    "mode",
    "distance_mode",
]


@dataclass(frozen=True)
class MainCaseSpec:
    genome_length: int
    generation_count: int
    seed: int
    biopsy_size_scalable: float
    biopsy_level_count: int
    general_event_prob: float
    event_shape_label: str
    single_or_multiple_event_prob: float
    duplication_multiplicity: int
    r_dist: float = 4.0


class TimingRecorder:
    def __init__(self):
        self.records = []

    def add(self, stage, operation, **values):
        record = {column: "" for column in TIMING_REPORT_COLUMNS}
        record["stage"] = stage
        record["operation"] = operation
        for key, value in values.items():
            if key in record:
                record[key] = value
        for key in [
            "count",
            "input_files",
            "instances",
            "skipped",
            "failed",
        ]:
            if record[key] == "":
                record[key] = 0
        for key in [
            "read_json_seconds",
            "core_seconds",
            "write_json_seconds",
            "total_seconds",
        ]:
            if record[key] == "":
                record[key] = 0.0
            else:
                record[key] = float(record[key])
        self.records.append(record)

    def write(self, output_root):
        if not self.records:
            return {}
        reports_dir = Path(output_root) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.records, columns=TIMING_REPORT_COLUMNS)
        csv_path = reports_dir / "timing_summary.csv"
        json_path = reports_dir / "timing_summary.json"
        frame.to_csv(csv_path, index=False)
        json_payload = {
            "columns": TIMING_REPORT_COLUMNS,
            "records": self.records,
            "totals_by_stage": (
                frame.groupby("stage", dropna=False)[
                    ["count", "instances", "read_json_seconds", "core_seconds", "write_json_seconds", "total_seconds"]
                ]
                .sum(numeric_only=True)
                .reset_index()
                .to_dict(orient="records")
            ),
        }
        write_json(json_path, json_payload, overwrite=True)
        return {"csv": csv_path, "json": json_path}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_ready(data), f, indent=2)
        f.write("\n")
    return True


def node_link_graph(data):
    try:
        return json_graph.node_link_graph(data, directed=True, edges="links")
    except TypeError:
        return json_graph.node_link_graph(data, directed=True)


def format_bss_token(value):
    return str(value).replace(".", "")


def format_float_token(value):
    return f"{float(value):g}".replace(".", "p")


def case_id(spec):
    return (
        f"gl{spec.genome_length}"
        f"_g{spec.generation_count}"
        f"_seed{spec.seed}"
        f"_bss{format_bss_token(spec.biopsy_size_scalable)}"
        f"_L{spec.biopsy_level_count}"
        f"_x{format_float_token(spec.general_event_prob)}"
        f"_y{format_float_token(spec.single_or_multiple_event_prob)}m{spec.duplication_multiplicity}"
    )


def case_group_parts(spec_or_case_id):
    if isinstance(spec_or_case_id, str):
        match = CASE_ID_PATTERN.match(spec_or_case_id)
        if not match:
            raise ValueError(f"Cannot derive grouped path from case id: {spec_or_case_id}")
        return (
            f"gl{match.group('genome_length')}",
            f"g{match.group('generation_count')}",
            f"seed{match.group('seed')}",
        )
    return (
        f"gl{spec_or_case_id.genome_length}",
        f"g{spec_or_case_id.generation_count}",
        f"seed{spec_or_case_id.seed}",
    )


def case_dir(output_root, spec_or_case_id):
    name = spec_or_case_id if isinstance(spec_or_case_id, str) else case_id(spec_or_case_id)
    return Path(output_root).joinpath(*case_group_parts(spec_or_case_id), name)


def input_path(output_root, spec_or_case_id):
    return case_dir(output_root, spec_or_case_id) / "input.json"


def result_path(output_root, spec_or_case_id, mode, algorithm_name):
    return case_dir(output_root, spec_or_case_id) / mode / f"{algorithm_name}.json"


def stable_rng(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def build_config_snapshot(base_config, spec):
    config = copy.deepcopy(base_config)
    config["genome_length"] = spec.genome_length
    config["NUMBER_OF_GENERATIONS"] = spec.generation_count
    config["GENERAL_EVENT_PROB"] = spec.general_event_prob
    config["GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB"] = spec.single_or_multiple_event_prob
    config["GENERAL_DUPLICATION_MULTIPLICITY"] = spec.duplication_multiplicity
    return config


def build_simulator_from_config(config, seed):
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(json_ready(config), f)
        simulator = CancerCellEvolutionSimulator(config_path, seed=seed)
        simulator.run_simulation()
    return simulator


def genotype_to_json(cell):
    return {
        "node_id": cell.node_id,
        "cell_id": cell.cell_id,
        "generation": cell.generation,
        "genome": cell.genome,
    }


def perform_biopsies(simulator, generations, biopsy_size_scalable, seed):
    biopsies = []
    cell_lists = []
    for index, generation in enumerate(generations, start=1):
        biopsy = simulator.perform_biopsy(
            generation=generation,
            biopsy_size_scalable=biopsy_size_scalable,
            seed=seed,
        )
        biopsies.append({
            "level": f"L{index}",
            "generation": generation,
            "cells": [genotype_to_json(cell) for cell in biopsy],
        })
        if biopsy:
            cell_lists.append(biopsy)
    return biopsies, cell_lists


def latest_biopsy_generations(available_generations, biopsy_level_count):
    return list(available_generations[-biopsy_level_count:])


def empty_distance_matrices():
    return {
        "cnp2cnp": {
            "ids": [],
            "matrix": [],
            "distance_mode": None,
        },
        "true_tree": {
            "ids": [],
            "matrix": [],
        },
    }


def choose_biopsy_generations_with_retry(
    simulator,
    spec,
    *,
    max_retries=100,
    min_total_cells=MIN_TOTAL_BIOPSY_CELLS,
):
    available_generations = list(range(1, spec.generation_count))
    if spec.biopsy_level_count > len(available_generations):
        return {
            "status": "failed",
            "failure_reason": "biopsy_level_count_exceeds_available_generations",
            "available_generations": available_generations,
            "retry_count": 0,
            "random_attempt_count": 0,
            "max_random_attempts": max_retries,
            "selected_generations": [],
            "total_sampled_biopsy_cells": 0,
            "nonempty_biopsy_levels": 0,
            "selection_strategy": "unavailable",
            "fallback_used": False,
            "pre_fallback_selected_generations": [],
            "biopsies": [],
            "cell_lists": [],
        }

    rng = stable_rng("biopsy-generations", spec.generation_count, spec.seed, spec.biopsy_level_count)
    last = None
    random_attempt_count = 0
    for retry_count in range(max_retries):
        selected_generations = sorted(rng.sample(available_generations, spec.biopsy_level_count))
        random_attempt_count = retry_count + 1
        biopsies, cell_lists = perform_biopsies(
            simulator,
            selected_generations,
            spec.biopsy_size_scalable,
            spec.seed,
        )
        total_cells = sum(len(biopsy["cells"]) for biopsy in biopsies)
        last = {
            "status": "ok" if total_cells >= min_total_cells else "failed",
            "failure_reason": None if total_cells >= min_total_cells else "small_biopsy",
            "available_generations": available_generations,
            "retry_count": retry_count,
            "random_attempt_count": random_attempt_count,
            "max_random_attempts": max_retries,
            "selected_generations": selected_generations,
            "total_sampled_biopsy_cells": total_cells,
            "nonempty_biopsy_levels": sum(1 for biopsy in biopsies if biopsy["cells"]),
            "selection_strategy": "random_retry",
            "fallback_used": False,
            "pre_fallback_selected_generations": [],
            "biopsies": biopsies,
            "cell_lists": cell_lists,
        }
        if total_cells >= min_total_cells:
            return last

    fallback_generations = latest_biopsy_generations(
        available_generations,
        spec.biopsy_level_count,
    )
    biopsies, cell_lists = perform_biopsies(
        simulator,
        fallback_generations,
        spec.biopsy_size_scalable,
        spec.seed,
    )
    total_cells = sum(len(biopsy["cells"]) for biopsy in biopsies)
    return {
        "status": "ok" if total_cells >= min_total_cells else "failed",
        "failure_reason": None if total_cells >= min_total_cells else "small_biopsy",
        "available_generations": available_generations,
        "retry_count": max_retries,
        "random_attempt_count": random_attempt_count,
        "max_random_attempts": max_retries,
        "selected_generations": fallback_generations,
        "total_sampled_biopsy_cells": total_cells,
        "nonempty_biopsy_levels": sum(1 for biopsy in biopsies if biopsy["cells"]),
        "selection_strategy": "latest_generations_fallback",
        "fallback_used": True,
        "pre_fallback_selected_generations": (
            [] if last is None else last["selected_generations"]
        ),
        "biopsies": biopsies,
        "cell_lists": cell_lists,
    }


def input_case_from_simulation(spec, base_config, *, max_retries=100):
    config = build_config_snapshot(base_config, spec)
    simulator = build_simulator_from_config(config, spec.seed)
    selection = choose_biopsy_generations_with_retry(
        simulator,
        spec,
        max_retries=max_retries,
    )
    cid = case_id(spec)
    input_case = {
        "case_id": cid,
        "corpus": CORPUS_NAME,
        "status": selection["status"],
        "failure_reason": selection["failure_reason"],
        "genome_length": spec.genome_length,
        "NUMBER_OF_GENERATIONS": spec.generation_count,
        "seed": spec.seed,
        "r_dist": spec.r_dist,
        "biopsy_size_scalable": spec.biopsy_size_scalable,
        "biopsy_level_count": spec.biopsy_level_count,
        "biopsy_generations": selection["selected_generations"],
        "GENERAL_EVENT_PROB": spec.general_event_prob,
        "event_shape_label": spec.event_shape_label,
        "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": spec.single_or_multiple_event_prob,
        "GENERAL_DUPLICATION_MULTIPLICITY": spec.duplication_multiplicity,
        "config_snapshot": config,
        "config_overrides": {
            "genome_length": spec.genome_length,
            "NUMBER_OF_GENERATIONS": spec.generation_count,
            "GENERAL_EVENT_PROB": spec.general_event_prob,
            "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": spec.single_or_multiple_event_prob,
            "GENERAL_DUPLICATION_MULTIPLICITY": spec.duplication_multiplicity,
        },
        "biopsy_selection": {
            "available_generations": selection["available_generations"],
            "retry_count": selection["retry_count"],
            "max_retries": max_retries,
            "random_attempt_count": selection["random_attempt_count"],
            "max_random_attempts": selection["max_random_attempts"],
            "total_sampled_biopsy_cells": selection["total_sampled_biopsy_cells"],
            "nonempty_biopsy_levels": selection["nonempty_biopsy_levels"],
            "min_total_sampled_biopsy_cells": MIN_TOTAL_BIOPSY_CELLS,
            "selection_strategy": selection["selection_strategy"],
            "fallback_used": selection["fallback_used"],
            "pre_fallback_selected_generations": selection["pre_fallback_selected_generations"],
        },
        "true_tree": node_link_data(simulator.tree),
        "biopsies": selection["biopsies"],
    }
    if selection["status"] != "ok":
        input_case["distance_matrices"] = empty_distance_matrices()
        input_case["unique_distance_cell_ids"] = []
    return input_case


def cell_lists_from_input(input_case):
    cell_lists = []
    for biopsy in input_case.get("biopsies", []):
        cells = genotypes_from_json(biopsy["cells"])
        if cells:
            cell_lists.append(cells)
    return cell_lists


def true_tree_from_input(input_case):
    return node_link_graph(input_case["true_tree"])


def l1_distance_matrix(cells):
    ids = [cell.get_id() for cell in cells]
    matrix = np.zeros((len(cells), len(cells)), dtype=float)
    for i, left in enumerate(cells):
        for j in range(i + 1, len(cells)):
            right = cells[j]
            distance = float(np.abs(np.asarray(left.genome) - np.asarray(right.genome)).sum())
            matrix[i, j] = distance
            matrix[j, i] = distance
    return ids, matrix


def compute_case_distances(input_case, *, distance_mode="cnp2cnp"):
    if input_case.get("status") != "ok":
        return input_case
    cell_lists = cell_lists_from_input(input_case)
    unique_cells = unique_cells_by_cell_id(cell_lists)
    if distance_mode == "cnp2cnp":
        cnp_ids, cnp_matrix = cnp2cnp_distance_matrix(unique_cells)
    elif distance_mode == "l1":
        cnp_ids, cnp_matrix = l1_distance_matrix(unique_cells)
    else:
        raise ValueError(f"Unsupported distance mode: {distance_mode}")
    true_ids, true_matrix = true_tree_distance_matrix(true_tree_from_input(input_case), cnp_ids)
    input_case["distance_matrices"] = {
        "cnp2cnp": {
            "ids": cnp_ids,
            "matrix": cnp_matrix,
            "distance_mode": distance_mode,
        },
        "true_tree": {
            "ids": true_ids,
            "matrix": true_matrix,
        },
    }
    input_case["unique_distance_cell_ids"] = cnp_ids
    return input_case


def actual_root(tree):
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root, found {len(roots)}")
    return roots[0]


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
    algorithm_name = getattr(algorithm, "__name__", str(algorithm))
    return {
        "case_id": input_case["case_id"],
        "corpus": CORPUS_NAME,
        "status": "reconstructed",
        "algorithm": algorithm_name,
        "mode": mode,
        "root": reconstructed_root,
        "actual_root": actual_root(reconstructed_tree),
        "newick": to_newick(reconstructed_tree),
        "reconstructed_tree": node_link_data(reconstructed_tree),
    }


def evaluate_result(input_case, result):
    if result.get("status") == "failed":
        return result
    true_tree = true_tree_from_input(input_case)
    reconstructed_tree = node_link_graph(result["reconstructed_tree"])
    result = copy.deepcopy(result)
    result["metrics"] = metric_summary(true_tree, reconstructed_tree)
    result["status"] = "evaluated"
    return result


def metric_value(result, metric):
    value = result
    for key in METRIC_FIELDS[metric]:
        value = value[key]
    return float(value)


def case_parameters(input_case):
    return {
        "case_id": input_case["case_id"],
        "genome_length": input_case["genome_length"],
        "generation_count": input_case["NUMBER_OF_GENERATIONS"],
        "seed": input_case["seed"],
        "biopsy_size_scalable": input_case["biopsy_size_scalable"],
        "biopsy_level_count": input_case["biopsy_level_count"],
        "general_event_prob": input_case["GENERAL_EVENT_PROB"],
        "event_shape_label": input_case["event_shape_label"],
        "single_or_multiple_event_prob": input_case["GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB"],
        "duplication_multiplicity": input_case["GENERAL_DUPLICATION_MULTIPLICITY"],
    }


def collect_result_rows(output_root):
    rows = []
    output_root = Path(output_root)
    if not output_root.exists():
        return pd.DataFrame()
    for input_file in sorted(output_root.glob("**/input.json")):
        input_case = load_json(input_file)
        if input_case.get("status") != "ok":
            continue
        params = case_parameters(input_case)
        for mode in MODES:
            mode_dir = input_file.parent / mode
            if not mode_dir.exists():
                continue
            for result_file in sorted(mode_dir.glob("*.json")):
                result = load_json(result_file)
                if "metrics" not in result:
                    continue
                row = {
                    **params,
                    "mode": mode,
                    "algorithm": result_file.stem,
                    "adf1": metric_value(result, "adf1"),
                    "grf": metric_value(result, "grf"),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _wilcoxon_greater(after, before):
    after = pd.Series(after, dtype=float)
    before = pd.Series(before, dtype=float)
    diffs = after - before
    if len(after) == 0 or np.allclose(diffs, 0):
        return 1.0
    try:
        _, p_value = wilcoxon(after, before, alternative="greater")
    except ValueError:
        p_value = 1.0
    return float(p_value)


def monotonic_summary(rows, *, dimension, values, fixed_columns):
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    summaries = []
    violations = []
    per_seed_columns = fixed_columns + ["seed"]
    for metric in METRICS:
        per_seed_records = []
        for key, group in rows.groupby(per_seed_columns, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            by_value = group.set_index(dimension)[metric]
            if not all(value in by_value.index for value in values):
                continue
            sequence = [float(by_value.loc[value]) for value in values]
            deltas = [sequence[index + 1] - sequence[index] for index in range(len(values) - 1)]
            monotonic = all(delta >= -1e-12 for delta in deltas)
            record = dict(zip(per_seed_columns, key))
            record.update({
                "metric": metric,
                "sequence": sequence,
                "deltas": deltas,
                "monotonic": monotonic,
            })
            per_seed_records.append(record)
            if not monotonic:
                violations.append({
                    **record,
                    "dimension": dimension,
                    "values": list(values),
                })
        if not per_seed_records:
            continue
        per_seed_frame = pd.DataFrame(per_seed_records)
        for key, group in per_seed_frame.groupby(fixed_columns, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            summary = dict(zip(fixed_columns, key))
            summary["metric"] = metric
            summary["dimension"] = dimension
            summary["values"] = list(values)
            summary["n_complete"] = int(len(group))
            summary["monotonic_passes"] = int(group["monotonic"].sum())
            summary["monotonic_failures"] = int((~group["monotonic"]).sum())
            deltas = np.array(group["deltas"].tolist(), dtype=float)
            for index in range(len(values) - 1):
                before_values = [seq[index] for seq in group["sequence"]]
                after_values = [seq[index + 1] for seq in group["sequence"]]
                left = values[index]
                right = values[index + 1]
                summary[f"mean_delta_{left}_to_{right}"] = float(deltas[:, index].mean())
                summary[f"wilcoxon_greater_p_{left}_to_{right}"] = _wilcoxon_greater(after_values, before_values)
            summaries.append(summary)
    return pd.DataFrame(summaries), pd.DataFrame(violations)


def write_metric_heatmaps(rows, reports_dir):
    if rows.empty:
        return []
    matplotlib_cache = Path(tempfile.gettempdir()) / "ctbf_matplotlib_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    written = []
    reports_dir = Path(reports_dir)
    for dimension, values, name in [
        ("biopsy_size_scalable", DEFAULT_BIOPSY_SIZE_SCALABLES, "biopsy_size"),
        ("biopsy_level_count", DEFAULT_BIOPSY_LEVEL_COUNTS, "biopsy_levels"),
    ]:
        for metric in METRICS:
            pivot_rows = rows.copy()
            pivot_rows["algorithm_mode"] = pivot_rows["algorithm"] + " / " + pivot_rows["mode"]
            matrix = pivot_rows.pivot_table(
                index="algorithm_mode",
                columns=dimension,
                values=metric,
                aggfunc="mean",
            )
            matrix = matrix.reindex(columns=list(values)).dropna(how="all")
            if matrix.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(matrix))))
            sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis", ax=ax)
            ax.set_title(f"Mean {metric.upper()} by {name.replace('_', ' ')}")
            ax.set_xlabel(dimension)
            ax.set_ylabel("Algorithm / mode")
            output_file = reports_dir / f"heatmap_mean_{metric}_{name}.png"
            fig.savefig(output_file, dpi=200, bbox_inches="tight")
            plt.close(fig)
            written.append(output_file)
    return written


def write_reports(output_root):
    reports_dir = Path(output_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_result_rows(output_root)
    rows_path = reports_dir / "result_rows.csv"
    rows.to_csv(rows_path, index=False)

    if rows.empty:
        return {"rows": rows_path, "summaries": [], "violations": [], "heatmaps": []}

    biopsy_size_fixed = [
        "genome_length",
        "generation_count",
        "general_event_prob",
        "event_shape_label",
        "single_or_multiple_event_prob",
        "duplication_multiplicity",
        "biopsy_level_count",
        "algorithm",
        "mode",
    ]
    level_fixed = [
        "genome_length",
        "generation_count",
        "general_event_prob",
        "event_shape_label",
        "single_or_multiple_event_prob",
        "duplication_multiplicity",
        "biopsy_size_scalable",
        "algorithm",
        "mode",
    ]
    b_summary, b_violations = monotonic_summary(
        rows,
        dimension="biopsy_size_scalable",
        values=DEFAULT_BIOPSY_SIZE_SCALABLES,
        fixed_columns=biopsy_size_fixed,
    )
    l_summary, l_violations = monotonic_summary(
        rows,
        dimension="biopsy_level_count",
        values=DEFAULT_BIOPSY_LEVEL_COUNTS,
        fixed_columns=level_fixed,
    )
    summary_paths = []
    violation_paths = []
    for name, frame in [
        ("monotonic_biopsy_size.csv", b_summary),
        ("monotonic_biopsy_levels.csv", l_summary),
    ]:
        path = reports_dir / name
        frame.to_csv(path, index=False)
        summary_paths.append(path)
    for name, frame in [
        ("violations_biopsy_size.csv", b_violations),
        ("violations_biopsy_levels.csv", l_violations),
    ]:
        path = reports_dir / name
        frame.to_csv(path, index=False)
        violation_paths.append(path)
    heatmaps = write_metric_heatmaps(rows, reports_dir)
    return {
        "rows": rows_path,
        "summaries": summary_paths,
        "violations": violation_paths,
        "heatmaps": heatmaps,
    }


def _metric_from_metrics_dict(metrics, metric):
    value = metrics
    for key in METRIC_FIELDS[metric][1:]:
        value = value[key]
    return float(value)


def _check_distance_matrix(input_case, input_file):
    errors = []
    distance_matrices = input_case.get("distance_matrices", {})
    if "cnp2cnp" not in distance_matrices:
        return [f"{input_file}: missing distance_matrices.cnp2cnp"]
    if "true_tree" not in distance_matrices:
        errors.append(f"{input_file}: missing distance_matrices.true_tree")

    cell_lists = cell_lists_from_input(input_case)
    expected_ids = [cell.get_id() for cell in unique_cells_by_cell_id(cell_lists)]
    matrix_payload = distance_matrices["cnp2cnp"]
    ids = matrix_payload.get("ids")
    matrix = np.array(matrix_payload.get("matrix", []), dtype=float)
    if ids != expected_ids:
        errors.append(f"{input_file}: cnp2cnp ids {ids} do not match unique biopsy cell ids {expected_ids}")
    if matrix.shape != (len(ids), len(ids)):
        errors.append(f"{input_file}: cnp2cnp matrix shape {matrix.shape} does not match ids length {len(ids)}")
    elif len(ids) > 0:
        if not np.allclose(np.diag(matrix), 0.0):
            errors.append(f"{input_file}: cnp2cnp matrix diagonal is not zero")
        if not np.allclose(matrix, matrix.T):
            errors.append(f"{input_file}: cnp2cnp matrix is not symmetric")

    true_payload = distance_matrices.get("true_tree")
    if true_payload:
        true_ids = true_payload.get("ids")
        true_matrix = np.array(true_payload.get("matrix", []), dtype=float)
        if true_ids != ids:
            errors.append(f"{input_file}: true_tree ids {true_ids} do not match cnp2cnp ids {ids}")
        if true_matrix.shape != (len(ids), len(ids)):
            errors.append(f"{input_file}: true_tree matrix shape {true_matrix.shape} does not match ids length {len(ids)}")
    return errors


def _check_biopsy_order(input_case, input_file):
    errors = []
    generations = input_case.get("biopsy_generations")
    if not isinstance(generations, list):
        errors.append(f"{input_file}: biopsy_generations must be a list")
        generations = []
    elif any(left >= right for left, right in zip(generations, generations[1:])):
        errors.append(
            f"{input_file}: biopsy_generations must be strictly increasing "
            f"ordered distinct levels, got {generations}"
        )
    biopsy_level_count = input_case.get("biopsy_level_count")
    if isinstance(biopsy_level_count, int) and len(generations) != biopsy_level_count:
        errors.append(
            f"{input_file}: {len(generations)} biopsy_generations do not match "
            f"biopsy_level_count {biopsy_level_count}"
        )

    biopsies = input_case.get("biopsies", [])
    if not isinstance(biopsies, list):
        errors.append(f"{input_file}: biopsies must be a list")
        return errors

    if len(biopsies) != len(generations):
        errors.append(
            f"{input_file}: {len(biopsies)} biopsies do not match "
            f"{len(generations)} biopsy_generations"
        )

    for index, biopsy in enumerate(biopsies):
        expected_level = f"L{index + 1}"
        if biopsy.get("level") != expected_level:
            errors.append(
                f"{input_file}: biopsy at position {index} has level "
                f"{biopsy.get('level')!r}, expected {expected_level!r}"
            )

        expected_generation = generations[index] if index < len(generations) else None
        biopsy_generation = biopsy.get("generation")
        if expected_generation is not None and biopsy_generation != expected_generation:
            errors.append(
                f"{input_file}: biopsy {expected_level} generation "
                f"{biopsy_generation!r} does not match selected generation "
                f"{expected_generation!r}"
            )

        for cell in biopsy.get("cells", []):
            if cell.get("generation") != biopsy_generation:
                errors.append(
                    f"{input_file}: cell {cell.get('node_id')!r} in "
                    f"{expected_level} has generation {cell.get('generation')!r}, "
                    f"expected {biopsy_generation!r}"
                )

    return errors


def _check_result_metrics(input_case, result_file):
    result = load_json(result_file)
    if result.get("status") == "failed":
        return [f"{result_file}: reconstructed result status is failed"]
    if "reconstructed_tree" not in result:
        return [f"{result_file}: missing reconstructed_tree"]
    if "metrics" not in result:
        return [f"{result_file}: missing metrics"]

    errors = []
    recomputed = evaluate_result(input_case, result)
    for metric in METRICS:
        stored = metric_value(result, metric)
        current = metric_value(recomputed, metric)
        if abs(stored - current) > CHECK_TOLERANCE:
            errors.append(
                f"{result_file}: stored {metric}={stored} does not match recomputed {current}"
            )
    return errors


def check_corpus(output_root, algorithms=None, modes=None, *, replay_reports=True):
    output_root = Path(output_root)
    modes = modes or list(MODES)
    algorithm_names = None
    if algorithms is not None:
        algorithm_names = [getattr(algorithm, "__name__", str(algorithm)) for algorithm in algorithms]

    input_files = sorted(output_root.glob("**/input.json"))
    errors = []
    checked_inputs = 0
    failed_inputs = 0
    checked_results = 0
    missing_results = 0
    for input_file in input_files:
        input_case = load_json(input_file)
        if input_case.get("corpus") != CORPUS_NAME:
            errors.append(f"{input_file}: corpus is {input_case.get('corpus')!r}, expected {CORPUS_NAME!r}")
        required_fields = [
            "case_id",
            "genome_length",
            "NUMBER_OF_GENERATIONS",
            "seed",
            "biopsy_size_scalable",
            "biopsy_level_count",
            "biopsy_generations",
            "GENERAL_EVENT_PROB",
            "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB",
            "GENERAL_DUPLICATION_MULTIPLICITY",
            "config_snapshot",
            "true_tree",
            "biopsies",
            "biopsy_selection",
        ]
        for field in required_fields:
            if field not in input_case:
                errors.append(f"{input_file}: missing {field}")
        biopsy_level_count = input_case.get("biopsy_level_count")
        if biopsy_level_count not in DEFAULT_BIOPSY_LEVEL_COUNTS:
            errors.append(
                f"{input_file}: biopsy_level_count {biopsy_level_count!r} "
                f"is outside active scope {list(DEFAULT_BIOPSY_LEVEL_COUNTS)}"
            )
        errors.extend(_check_biopsy_order(input_case, input_file))
        if input_case.get("status") != "ok":
            failed_inputs += 1
            distance_matrices = input_case.get("distance_matrices")
            if distance_matrices != empty_distance_matrices():
                errors.append(f"{input_file}: failed input must record empty distance matrices")
            continue

        checked_inputs += 1
        errors.extend(_check_distance_matrix(input_case, input_file))
        for mode in modes:
            mode_dir = input_file.parent / mode
            names = algorithm_names
            if names is None:
                names = sorted(path.stem for path in mode_dir.glob("*.json")) if mode_dir.exists() else []
            for algorithm_name in names:
                result_file = mode_dir / f"{algorithm_name}.json"
                if not result_file.exists():
                    missing_results += 1
                    errors.append(f"{input_file}: missing result {mode}/{algorithm_name}.json")
                    continue
                checked_results += 1
                errors.extend(_check_result_metrics(input_case, result_file))

    report_paths = []
    if replay_reports:
        written = write_reports(output_root)
        for value in [written["rows"], *written["summaries"], *written["violations"], *written["heatmaps"]]:
            path = Path(value)
            report_paths.append(path)
            if not path.exists():
                errors.append(f"{path}: expected report file was not written")

    return {
        "input_files": len(input_files),
        "checked_inputs": checked_inputs,
        "failed_inputs": failed_inputs,
        "checked_results": checked_results,
        "missing_results": missing_results,
        "report_files": len(report_paths),
        "errors": errors,
    }


def resolve_event_shapes(labels):
    labels = labels or list(DEFAULT_EVENT_SHAPES)
    shapes = []
    for label in labels:
        if label not in DEFAULT_EVENT_SHAPES:
            raise ValueError(f"Unknown event shape '{label}'. Available: {sorted(DEFAULT_EVENT_SHAPES)}")
        single_or_multiple, multiplicity = DEFAULT_EVENT_SHAPES[label]
        shapes.append((label, single_or_multiple, multiplicity))
    return shapes


def iter_case_specs(
    *,
    genome_lengths=DEFAULT_GENOME_LENGTHS,
    generation_counts=DEFAULT_GENERATION_COUNTS,
    seeds=(),
    biopsy_size_scalables=DEFAULT_BIOPSY_SIZE_SCALABLES,
    biopsy_level_counts=DEFAULT_BIOPSY_LEVEL_COUNTS,
    event_probs=DEFAULT_EVENT_PROBS,
    event_shapes=None,
    r_dist=4.0,
):
    for genome_length in genome_lengths:
        for generation_count in generation_counts:
            for seed in seeds:
                for biopsy_size_scalable in biopsy_size_scalables:
                    for biopsy_level_count in biopsy_level_counts:
                        for event_prob in event_probs:
                            for shape_label, single_or_multiple, multiplicity in resolve_event_shapes(event_shapes):
                                yield MainCaseSpec(
                                    genome_length=int(genome_length),
                                    generation_count=int(generation_count),
                                    seed=int(seed),
                                    biopsy_size_scalable=float(biopsy_size_scalable),
                                    biopsy_level_count=int(biopsy_level_count),
                                    general_event_prob=float(event_prob),
                                    event_shape_label=shape_label,
                                    single_or_multiple_event_prob=float(single_or_multiple),
                                    duplication_multiplicity=int(multiplicity),
                                    r_dist=float(r_dist),
                                )


def selected_algorithms(algorithm_indexes=None, algorithm_names=None):
    algorithms = get_algorithms_to_test()
    names = algorithm_names
    if names is None and algorithm_indexes is None:
        names = list(DEFAULT_CORE_ALGORITHM_NAMES)
    indices = select_algorithm_indices(
        algorithms,
        algorithm_indexes=algorithm_indexes,
        algorithm_names=names,
    )
    return [algorithms[index] for index in indices]


def selected_case_specs(args):
    seeds = args.seed if args.seed else load_seeds(args.seeds_file)
    specs = list(iter_case_specs(
        genome_lengths=args.genome_length or DEFAULT_GENOME_LENGTHS,
        generation_counts=args.generation_count or DEFAULT_GENERATION_COUNTS,
        seeds=seeds,
        biopsy_size_scalables=args.biopsy_size_scalable or DEFAULT_BIOPSY_SIZE_SCALABLES,
        biopsy_level_counts=args.biopsy_level_count or DEFAULT_BIOPSY_LEVEL_COUNTS,
        event_probs=args.event_prob or DEFAULT_EVENT_PROBS,
        event_shapes=args.event_shape,
        r_dist=args.r_dist,
    ))
    if args.offset:
        specs = specs[args.offset:]
    if args.limit is not None:
        specs = specs[:args.limit]
    return specs


def run_simulate_stage(specs, args, base_config):
    for spec in specs:
        path = input_path(args.output_root, spec)
        if path.exists() and not args.overwrite:
            print(f"exists input {path}")
            continue
        input_case = input_case_from_simulation(
            spec,
            base_config,
            max_retries=args.max_biopsy_generation_retries,
        )
        write_json(path, input_case, overwrite=True)
        print(f"wrote input {path}")


def run_distance_stage(specs, args):
    for spec in specs:
        path = input_path(args.output_root, spec)
        if not path.exists():
            print(f"missing input {path}")
            continue
        input_case = load_json(path)
        if input_case.get("status") != "ok":
            print(f"skip failed input {path}")
            continue
        if input_case.get("distance_matrices") and not args.overwrite:
            print(f"exists distances {path}")
            continue
        input_case = compute_case_distances(input_case, distance_mode=args.distance_mode)
        write_json(path, input_case, overwrite=True)
        print(f"wrote distances {path}")


def run_reconstruct_stage(specs, args, algorithms, modes):
    for spec in specs:
        input_file = input_path(args.output_root, spec)
        if not input_file.exists():
            print(f"missing input {input_file}")
            continue
        input_case = load_json(input_file)
        if input_case.get("status") != "ok" or "cnp2cnp" not in input_case.get("distance_matrices", {}):
            print(f"skip input without distances {input_file}")
            continue
        for algorithm in algorithms:
            algorithm_name = getattr(algorithm, "__name__", str(algorithm))
            for mode in modes:
                output_file = result_path(args.output_root, spec, mode, algorithm_name)
                if output_file.exists() and not args.overwrite:
                    print(f"exists result {output_file}")
                    continue
                try:
                    result = reconstruction_result(input_case, algorithm, mode)
                except Exception as exc:
                    result = {
                        "case_id": input_case["case_id"],
                        "corpus": CORPUS_NAME,
                        "status": "failed",
                        "algorithm": algorithm_name,
                        "mode": mode,
                        "error": str(exc),
                    }
                    if args.fail_fast:
                        raise
                write_json(output_file, result, overwrite=True)
                print(f"wrote result {output_file}")


def run_evaluate_stage(specs, args, algorithms, modes):
    for spec in specs:
        input_file = input_path(args.output_root, spec)
        if not input_file.exists():
            print(f"missing input {input_file}")
            continue
        input_case = load_json(input_file)
        if input_case.get("status") != "ok":
            continue
        for algorithm in algorithms:
            algorithm_name = getattr(algorithm, "__name__", str(algorithm))
            for mode in modes:
                output_file = result_path(args.output_root, spec, mode, algorithm_name)
                if not output_file.exists():
                    continue
                result = load_json(output_file)
                if "metrics" in result and not args.overwrite:
                    print(f"exists metrics {output_file}")
                    continue
                try:
                    result = evaluate_result(input_case, result)
                except Exception as exc:
                    result["status"] = "failed"
                    result["error"] = str(exc)
                    if args.fail_fast:
                        raise
                write_json(output_file, result, overwrite=True)
                print(f"wrote metrics {output_file}")


def run_check_stage(args, algorithms, modes):
    summary = check_corpus(args.output_root, algorithms=algorithms, modes=modes)
    print("Check summary:", json.dumps({key: value for key, value in summary.items() if key != "errors"}, indent=2))
    if summary["errors"]:
        for error in summary["errors"]:
            print(f"CHECK ERROR: {error}")
        raise AssertionError(f"main monotonicity corpus check failed with {len(summary['errors'])} error(s)")


def parse_args():
    parser = argparse.ArgumentParser(description="Build and report the main sampling-monotonicity benchmark corpus.")
    parser.add_argument(
        "--stage",
        action="append",
        choices=["simulate", "distance", "reconstruct", "evaluate", "report", "check", "all"],
        help="Stage to run. Can be passed multiple times. Defaults to all.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_BY_PROFILE["base"])
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS_FILE)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--genome-length", type=int, action="append", default=None)
    parser.add_argument("--generation-count", type=int, action="append", default=None)
    parser.add_argument("--biopsy-size-scalable", type=float, action="append", default=None)
    parser.add_argument("--biopsy-level-count", type=int, action="append", default=None)
    parser.add_argument("--event-prob", type=float, action="append", default=None)
    parser.add_argument("--event-shape", action="append", choices=sorted(DEFAULT_EVENT_SHAPES), default=None)
    parser.add_argument("--r-dist", type=float, default=4.0)
    parser.add_argument("--algorithm-index", type=int, action="append", default=None)
    parser.add_argument("--algorithm-name", action="append", default=None)
    parser.add_argument("--mode", action="append", choices=MODES, default=None)
    parser.add_argument("--distance-mode", choices=["cnp2cnp", "l1"], default="cnp2cnp")
    parser.add_argument("--max-biopsy-generation-retries", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    stages = args.stage or ["all"]
    if "all" in stages:
        stages = ["simulate", "distance", "reconstruct", "evaluate", "report"]
    specs = selected_case_specs(args)
    algorithms = selected_algorithms(args.algorithm_index, args.algorithm_name)
    modes = args.mode or list(MODES)

    print("Output root:", args.output_root)
    print("Cases:", len(specs))
    print("Algorithms:", ", ".join(getattr(algorithm, "__name__", str(algorithm)) for algorithm in algorithms))
    print("Modes:", ", ".join(modes))
    print("Stages:", ", ".join(stages))
    if args.dry_run:
        for spec in specs[:10]:
            print(case_id(spec))
        if len(specs) > 10:
            print(f"... {len(specs) - 10} more")
        return

    with open(args.config, "r") as f:
        base_config = json.load(f)

    for stage in stages:
        try:
            if stage == "simulate":
                run_simulate_stage(specs, args, base_config)
            elif stage == "distance":
                run_distance_stage(specs, args)
            elif stage == "reconstruct":
                run_reconstruct_stage(specs, args, algorithms, modes)
            elif stage == "evaluate":
                run_evaluate_stage(specs, args, algorithms, modes)
            elif stage == "report":
                written = write_reports(args.output_root)
                print("Wrote reports:", written)
            elif stage == "check":
                run_check_stage(args, algorithms, modes)
        except Exception:
            traceback.print_exc()
            if args.fail_fast:
                raise
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
