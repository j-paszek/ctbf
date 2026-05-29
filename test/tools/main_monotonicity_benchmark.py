#!/usr/bin/env python
import argparse
import copy
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
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
    EXT_GRF_METRIC_FIELD,
    LEGACY_GRF_SET_SIMILARITY_FIELD,
    cnp2cnp_distance_matrix,
    genotypes_from_json,
    json_ready,
    legacy_set_grf_similarity_from_cluster_contexts,
    node_link_data,
    root_id,
    unique_cells_by_cell_id,
)
from evaluator import cluster_evaluation_context, ext_grf_from_cluster_counts  # noqa: E402
from evaluator_full import evaluate_4, tree_evaluation_context  # noqa: E402


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
LEGACY_INPUT_FILE_NAME = "input.json"
TRUE_TREE_INPUT_FILE_NAME = "input_truetree.json"
BIOPSY_INPUT_FILE_NAME = "input_biopsy.json"
GENOME_DICT_FILE_NAME = "genome_dict.csv"
INPUT_LAYOUTS = ("legacy", "split")
SPLIT_INPUT_LAYOUT = "split_v1"
DISTANCE_FILE_NAME = "input_dm_cnp.json"
CASE_RESULT_CSV_NAME = "result.csv"
CASE_TIMING_FILE_NAME = "times.csv"
RESULT_ROW_COLUMNS = [
    "case_id",
    "genome_length",
    "generation_count",
    "seed",
    "biopsy_size_scalable",
    "biopsy_level_count",
    "general_event_prob",
    "event_shape_label",
    "single_or_multiple_event_prob",
    "duplication_multiplicity",
    "mode",
    "algorithm",
    "status",
    "adf1",
    "grf",
    "result_file",
    "error",
]
BIOPSY_CELL_SUMMARY_COLUMNS = [
    "generation",
    "bss",
    "level",
    "n",
    "min",
    "max",
    "avg",
    "total",
]
TIMING_REPORT_COLUMNS = [
    "generation_count",
    "stage",
    "operation",
    "scope",
    "count",
    "input_files",
    "instances",
    "skipped",
    "missing",
    "failed",
    "read_json_seconds",
    "core_seconds",
    "write_json_seconds",
    "total_seconds",
    "algorithm",
    "mode",
    "evaluation_method",
    "distance_mode",
]
CASE_TIMING_COLUMNS = [
    "case_id",
    "generation_count",
    "stage",
    "operation",
    "algorithm",
    "mode",
    "evaluation_method",
    "distance_mode",
    "elapsed_seconds",
    "started_at",
    "finished_at",
    "status",
    "input_cell_count",
    "unique_cell_count",
    "tree_node_count",
    "error",
]
CASE_TIMING_KEY_COLUMNS = [
    "case_id",
    "stage",
    "operation",
    "algorithm",
    "mode",
    "evaluation_method",
    "distance_mode",
]
COMMAND_TIMING_COLUMNS = [
    "generation_count",
    "stage",
    "started_at",
    "finished_at",
    "total_seconds",
    "case_count",
    "status",
    "command",
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


@dataclass(frozen=True)
class EvaluationCaseContext:
    input_case: dict
    genome_dict: dict
    true_tree: nx.DiGraph
    true_eval_context: object
    true_root: object
    true_cluster_context: object


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_timing_report_record(record):
    normalized = {column: "" for column in TIMING_REPORT_COLUMNS}
    for key, value in record.items():
        if key in normalized:
            normalized[key] = value
    for key in [
        "generation_count",
        "count",
        "input_files",
        "instances",
        "skipped",
        "missing",
        "failed",
    ]:
        if normalized[key] == "":
            normalized[key] = 0
        normalized[key] = int(normalized[key])
    for key in [
        "read_json_seconds",
        "core_seconds",
        "write_json_seconds",
        "total_seconds",
    ]:
        if normalized[key] == "":
            normalized[key] = 0.0
        normalized[key] = float(normalized[key])
    return normalized


def write_detailed_timing_summary_records(output_root, records, *, generation_count):
    reports_dir = Path(output_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    records = [normalize_timing_report_record(record) for record in records]
    frame = pd.DataFrame(records, columns=TIMING_REPORT_COLUMNS)
    csv_path = detailed_timing_summary_path(output_root, generation_count)
    frame.to_csv(csv_path, index=False)
    return {"csv": csv_path}


class TimingRecorder:
    def __init__(self):
        self.records = []

    def add(self, stage, operation, **values):
        record = {column: "" for column in TIMING_REPORT_COLUMNS}
        record["stage"] = stage
        record["operation"] = operation
        record["scope"] = "stage_total"
        for key, value in values.items():
            if key in record:
                record[key] = value
        self.records.append(normalize_timing_report_record(record))

    def write(self, output_root):
        if not self.records:
            return {}
        generations = sorted({record["generation_count"] for record in self.records if record["generation_count"]})
        if len(generations) != 1:
            return {}
        return write_detailed_timing_summary_records(
            output_root,
            self.records,
            generation_count=generations[0],
        )


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
    return case_dir(output_root, spec_or_case_id) / LEGACY_INPUT_FILE_NAME


def input_truetree_path(output_root, spec_or_case_id):
    return case_dir(output_root, spec_or_case_id) / TRUE_TREE_INPUT_FILE_NAME


def input_biopsy_path(output_root, spec_or_case_id):
    return case_dir(output_root, spec_or_case_id) / BIOPSY_INPUT_FILE_NAME


def genome_dict_path(output_root, spec_or_case_id):
    return case_dir(output_root, spec_or_case_id) / GENOME_DICT_FILE_NAME


def case_timing_path(output_root, spec_or_case_id):
    return case_dir(output_root, spec_or_case_id) / CASE_TIMING_FILE_NAME


def timing_summary_path(output_root):
    return Path(output_root) / "reports" / "timing_summary.csv"


def detailed_timing_summary_path(output_root, generation_count):
    return Path(output_root) / "reports" / f"timing_summary_g{int(generation_count)}.csv"


def distance_path(output_root, spec_or_case_id):
    return case_dir(output_root, spec_or_case_id) / DISTANCE_FILE_NAME


def result_path(output_root, spec_or_case_id, mode, algorithm_name):
    return case_dir(output_root, spec_or_case_id) / mode / f"{algorithm_name}.json"


def case_result_csv_path(output_root, spec_or_case_id):
    return case_dir(output_root, spec_or_case_id) / CASE_RESULT_CSV_NAME


def stable_rng(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def case_timing_record(
    case_or_spec,
    *,
    stage,
    operation,
    elapsed_seconds,
    started_at,
    finished_at,
    status="ok",
    algorithm="",
    mode="",
    evaluation_method="",
    distance_mode="",
    input_cell_count="",
    unique_cell_count="",
    tree_node_count="",
    error="",
):
    if isinstance(case_or_spec, MainCaseSpec):
        cid = case_id(case_or_spec)
        generation_count = case_or_spec.generation_count
    else:
        cid = case_or_spec["case_id"]
        generation_count = case_or_spec["NUMBER_OF_GENERATIONS"]
    return {
        "case_id": cid,
        "generation_count": generation_count,
        "stage": stage,
        "operation": operation,
        "algorithm": algorithm,
        "mode": mode,
        "evaluation_method": evaluation_method,
        "distance_mode": distance_mode,
        "elapsed_seconds": float(elapsed_seconds),
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "input_cell_count": input_cell_count,
        "unique_cell_count": unique_cell_count,
        "tree_node_count": tree_node_count,
        "error": error,
    }


def normalize_case_timing_record(record):
    normalized = {column: "" for column in CASE_TIMING_COLUMNS}
    for key, value in record.items():
        if key in normalized:
            normalized[key] = value
    for key in ["generation_count", "input_cell_count", "unique_cell_count", "tree_node_count"]:
        if normalized[key] == "":
            continue
        normalized[key] = int(normalized[key])
    if normalized["elapsed_seconds"] == "":
        normalized["elapsed_seconds"] = 0.0
    normalized["elapsed_seconds"] = float(normalized["elapsed_seconds"])
    return normalized


def case_timing_key(record):
    return tuple(str(record.get(column, "")) for column in CASE_TIMING_KEY_COLUMNS)


def write_case_timing_records(case_directory, records):
    records = [normalize_case_timing_record(record) for record in records]
    if not records:
        return None
    path = Path(case_directory) / CASE_TIMING_FILE_NAME
    if path.exists():
        existing = [
            normalize_case_timing_record(row)
            for row in pd.read_csv(path, keep_default_na=False).to_dict(orient="records")
        ]
    else:
        existing = []
    replacement_keys = {case_timing_key(record) for record in records}
    kept = [
        record
        for record in existing
        if case_timing_key(record) not in replacement_keys
    ]
    merged = sorted(
        kept + records,
        key=lambda record: (
            int(record.get("generation_count") or 0),
            str(record.get("case_id", "")),
            str(record.get("stage", "")),
            str(record.get("operation", "")),
            str(record.get("algorithm", "")),
            str(record.get("mode", "")),
            str(record.get("evaluation_method", "")),
            str(record.get("distance_mode", "")),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(merged, columns=CASE_TIMING_COLUMNS).to_csv(path, index=False)
    return path


def read_case_timing_records(path):
    if not Path(path).exists():
        return []
    return [
        normalize_case_timing_record(record)
        for record in pd.read_csv(path, keep_default_na=False).to_dict(orient="records")
    ]


def collect_case_timing_records(output_root, specs=None):
    if specs is None:
        paths = sorted(Path(output_root).glob(f"**/{CASE_TIMING_FILE_NAME}"))
    else:
        paths = [
            case_timing_path(output_root, spec)
            for spec in specs
            if case_timing_path(output_root, spec).exists()
        ]
    records = []
    for path in paths:
        records.extend(read_case_timing_records(path))
    return records


def normalize_command_timing_record(record):
    normalized = {column: "" for column in COMMAND_TIMING_COLUMNS}
    for key, value in record.items():
        if key in normalized:
            normalized[key] = value
    if normalized["total_seconds"] == "" and "elapsed_seconds" in record:
        normalized["total_seconds"] = record["elapsed_seconds"]
    for key in ["generation_count", "case_count"]:
        if normalized[key] == "":
            normalized[key] = 0
        normalized[key] = int(normalized[key])
    if normalized["total_seconds"] == "":
        normalized["total_seconds"] = 0.0
    normalized["total_seconds"] = float(normalized["total_seconds"])
    return normalized


def write_command_timing_if_absent(
    output_root,
    *,
    generation_count,
    stage,
    elapsed_seconds,
    case_count,
    started_at,
    finished_at,
    status="ok",
    command="",
):
    path = timing_summary_path(output_root)
    existing = read_command_timing_records(path)
    if any(
        record["generation_count"] == int(generation_count) and record["stage"] == stage
        for record in existing
    ):
        return None
    record = normalize_command_timing_record({
        "generation_count": generation_count,
        "stage": stage,
        "started_at": started_at,
        "finished_at": finished_at,
        "total_seconds": elapsed_seconds,
        "case_count": case_count,
        "status": status,
        "command": command,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(existing + [record], columns=COMMAND_TIMING_COLUMNS).to_csv(path, index=False)
    return path


def read_command_timing_records(path):
    path = Path(path)
    if not path.exists():
        return []
    frame = pd.read_csv(path, keep_default_na=False)
    columns = set(frame.columns)
    current_columns = set(COMMAND_TIMING_COLUMNS)
    legacy_elapsed_columns = (current_columns - {"total_seconds"}) | {"elapsed_seconds"}
    if not (current_columns <= columns or legacy_elapsed_columns <= columns):
        return []
    return [
        normalize_command_timing_record(record)
        for record in frame.to_dict(orient="records")
    ]


def collect_command_timing_records(output_root):
    return read_command_timing_records(timing_summary_path(output_root))


def write_single_stage_command_timing(
    output_root,
    specs,
    stages,
    *,
    elapsed_seconds,
    started_at,
    finished_at,
    status,
    command="",
):
    if status != "ok" or len(stages) != 1:
        return []
    counts_by_generation = {}
    for spec in specs:
        counts_by_generation[spec.generation_count] = counts_by_generation.get(spec.generation_count, 0) + 1
    written = []
    for generation_count, case_count in sorted(counts_by_generation.items()):
        path = write_command_timing_if_absent(
            output_root,
            generation_count=generation_count,
            stage=stages[0],
            elapsed_seconds=elapsed_seconds,
            case_count=case_count,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            command=command,
        )
        if path is not None:
            written.append(path)
    unique_written = []
    seen = set()
    for path in written:
        if path in seen:
            continue
        seen.add(path)
        unique_written.append(path)
    return unique_written


def timing_report_record_from_group(group, *, stage, operation, scope, generation_count=0, **labels):
    record = {
        "generation_count": generation_count,
        "stage": stage,
        "operation": operation,
        "scope": scope,
        "count": len(group),
        "instances": len(group),
        "core_seconds": float(sum(float(row["elapsed_seconds"]) for row in group)),
        "total_seconds": float(sum(float(row["elapsed_seconds"]) for row in group)),
    }
    record.update(labels)
    return record


def timing_report_records_from_artifacts(output_root, specs=None):
    records = []
    allowed_generations = None if specs is None else {spec.generation_count for spec in specs}
    case_records = collect_case_timing_records(output_root, specs=specs)
    groups = {}
    for record in case_records:
        key = (record["generation_count"], record["stage"])
        groups.setdefault(key, []).append(record)
    for (generation_count, stage), group in sorted(groups.items()):
        records.append(timing_report_record_from_group(
            group,
            generation_count=generation_count,
            stage=stage,
            operation="case_timing_total",
            scope="stage_total",
        ))

    operation_groups = {}
    for record in case_records:
        if record["stage"] == "reconstruct":
            key = (
                record["generation_count"],
                record["stage"],
                "reconstruct_by_algorithm",
                "algorithm",
                record["algorithm"],
                "",
                "",
                "",
            )
        elif record["stage"] == "evaluate":
            key = (
                record["generation_count"],
                record["stage"],
                "evaluate_by_metric_method",
                "evaluation_method",
                "",
                "",
                record["evaluation_method"],
                "",
            )
        elif record["stage"] == "distance":
            key = (
                record["generation_count"],
                record["stage"],
                record["operation"],
                "distance_mode",
                "",
                "",
                "",
                record["distance_mode"],
            )
        else:
            key = (
                record["generation_count"],
                record["stage"],
                record["operation"],
                "operation",
                "",
                "",
                "",
                "",
            )
        operation_groups.setdefault(key, []).append(record)
    for (
        generation_count,
        stage,
        operation,
        scope,
        algorithm,
        mode,
        evaluation_method,
        distance_mode,
    ), group in sorted(operation_groups.items()):
        records.append(timing_report_record_from_group(
            group,
            generation_count=generation_count,
            stage=stage,
            operation=operation,
            scope=scope,
            algorithm=algorithm,
            mode=mode,
            evaluation_method=evaluation_method,
            distance_mode=distance_mode,
        ))

    for command_record in collect_command_timing_records(output_root):
        if allowed_generations is not None and command_record["generation_count"] not in allowed_generations:
            continue
        records.append({
            "generation_count": command_record["generation_count"],
            "stage": command_record["stage"],
            "operation": "command_wall_time",
            "scope": "command_stage",
            "count": command_record["case_count"],
            "input_files": command_record["case_count"],
            "instances": 1,
            "core_seconds": command_record["total_seconds"],
            "total_seconds": command_record["total_seconds"],
        })
    return records


def write_timing_report(output_root, specs=None):
    records = timing_report_records_from_artifacts(output_root, specs=specs)
    frame = pd.DataFrame(
        [normalize_timing_report_record(record) for record in records],
        columns=TIMING_REPORT_COLUMNS,
    )
    generation_summaries = {}
    for generation_count in sorted(
        int(value)
        for value in frame["generation_count"].unique()
        if int(value) != 0
    ):
        generation_frame = frame[frame["generation_count"] == generation_count]
        path = detailed_timing_summary_path(output_root, generation_count)
        path.parent.mkdir(parents=True, exist_ok=True)
        generation_frame.to_csv(path, index=False)
        generation_summaries[f"g{generation_count}"] = path
    if not generation_summaries:
        return {}
    return {"generation_summaries": generation_summaries}


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


def empty_distance_payload(input_case=None, *, distance_mode=None):
    input_case = input_case or {}
    return {
        "case_id": input_case.get("case_id"),
        "corpus": input_case.get("corpus", CORPUS_NAME),
        "status": input_case.get("status", "failed"),
        "failure_reason": input_case.get("failure_reason"),
        "distance_mode": distance_mode,
        "unique_distance_cell_ids": [],
        "distance_matrices": {
            "cnp2cnp": {
                "ids": [],
                "matrix": [],
                "distance_mode": distance_mode,
            },
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


def input_case_from_simulation(spec, base_config, *, max_retries=100, timing_records=None):
    config = build_config_snapshot(base_config, spec)
    sim_started_at = utc_timestamp()
    sim_start = time.perf_counter()
    simulator = build_simulator_from_config(config, spec.seed)
    sim_finished_at = utc_timestamp()
    if timing_records is not None:
        timing_records.append(case_timing_record(
            spec,
            stage="simulate",
            operation="simulate_true_tree",
            elapsed_seconds=time.perf_counter() - sim_start,
            started_at=sim_started_at,
            finished_at=sim_finished_at,
            status="ok",
            tree_node_count=len(simulator.tree.nodes),
        ))
    biopsy_started_at = utc_timestamp()
    biopsy_start = time.perf_counter()
    selection = choose_biopsy_generations_with_retry(
        simulator,
        spec,
        max_retries=max_retries,
    )
    biopsy_finished_at = utc_timestamp()
    if timing_records is not None:
        timing_records.append(case_timing_record(
            spec,
            stage="simulate",
            operation="sample_biopsies",
            elapsed_seconds=time.perf_counter() - biopsy_start,
            started_at=biopsy_started_at,
            finished_at=biopsy_finished_at,
            status=selection["status"],
            input_cell_count=selection["total_sampled_biopsy_cells"],
            tree_node_count=len(simulator.tree.nodes),
        ))
    cid = case_id(spec)
    true_tree = (
        simulator.canonicalized_tree_by_genome()
        if hasattr(simulator, "canonicalized_tree_by_genome")
        else simulator.tree
    )
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
        "true_tree": node_link_data(true_tree),
        "biopsies": selection["biopsies"],
    }
    return input_case


def genome_to_key(genome):
    return tuple(json_ready(genome))


def genome_to_csv_value(genome):
    return json.dumps(list(genome_to_key(genome)), separators=(",", ":"))


def collect_genome_records(input_case):
    records = []
    for node in input_case.get("true_tree", {}).get("nodes", []):
        if "genome" in node:
            records.append({
                "source": "true_tree",
                "node_id": node.get("id"),
                "cell_id": node.get("cell_id"),
                "genome": node["genome"],
            })
    for biopsy in input_case.get("biopsies", []):
        for cell in biopsy.get("cells", []):
            if "genome" in cell:
                records.append({
                    "source": f"biopsy:{biopsy.get('level')}",
                    "node_id": cell.get("node_id"),
                    "cell_id": cell.get("cell_id"),
                    "genome": cell["genome"],
                })
    return records


def canonical_cell_id_by_genome_from_input(input_case):
    case = input_case.get("case_id", "<unknown>")
    by_cell_id = {}
    by_genome = {}
    for record in collect_genome_records(input_case):
        cell_id = record.get("cell_id")
        if cell_id is None:
            raise ValueError(
                f"{case}: {record['source']} node {record.get('node_id')!r} "
                "has genome but no cell_id"
            )
        genome_key = genome_to_key(record["genome"])
        if cell_id in by_cell_id and by_cell_id[cell_id] != genome_key:
            raise ValueError(
                f"{case}: cell_id {cell_id!r} maps to multiple genomes; "
                "this is a hard corpus/system invariant error"
            )
        by_cell_id[cell_id] = genome_key
        if genome_key not in by_genome or cell_id < by_genome[genome_key]:
            by_genome[genome_key] = cell_id
    return by_genome


def canonicalize_input_case_cell_ids_by_genome(input_case):
    canonical = canonical_cell_id_by_genome_from_input(input_case)
    normalized = copy.deepcopy(input_case)
    for node in normalized.get("true_tree", {}).get("nodes", []):
        if "genome" in node:
            node["cell_id"] = canonical[genome_to_key(node["genome"])]
    for biopsy in normalized.get("biopsies", []):
        for cell in biopsy.get("cells", []):
            if "genome" in cell:
                cell["cell_id"] = canonical[genome_to_key(cell["genome"])]
    return normalized


def build_genome_dictionary(input_case):
    by_genome = canonical_cell_id_by_genome_from_input(input_case)
    by_cell_id = {}
    for genome_key, cell_id in by_genome.items():
        by_cell_id[cell_id] = genome_key
    return {
        cell_id: list(genome)
        for cell_id, genome in sorted(by_cell_id.items(), key=lambda item: item[0])
    }


def write_genome_dict(path, genome_dict, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"cell_id": cell_id, "genome": genome_to_csv_value(genome)}
        for cell_id, genome in genome_dict.items()
    ]
    pd.DataFrame(rows, columns=["cell_id", "genome"]).to_csv(path, index=False)
    return True


def read_genome_dict(path):
    frame = pd.read_csv(path)
    genome_dict = {}
    genome_to_cell_id = {}
    for row in frame.to_dict(orient="records"):
        cell_id = int(row["cell_id"])
        genome = json.loads(row["genome"])
        genome_key = genome_to_key(genome)
        if cell_id in genome_dict and genome_to_key(genome_dict[cell_id]) != genome_key:
            raise ValueError(f"{path}: cell_id {cell_id!r} maps to multiple genomes")
        if genome_key in genome_to_cell_id and genome_to_cell_id[genome_key] != cell_id:
            raise ValueError(
                f"{path}: genome {list(genome_key)!r} appears under multiple "
                f"cell_id values ({genome_to_cell_id[genome_key]!r}, {cell_id!r})"
            )
        genome_dict[cell_id] = genome
        genome_to_cell_id[genome_key] = cell_id
    return genome_dict


def strip_genomes_from_tree_payload(tree_payload, genome_dict, *, require_all=True):
    stripped = copy.deepcopy(tree_payload)
    for node in stripped.get("nodes", []):
        if "genome" not in node:
            continue
        cell_id = node.get("cell_id")
        if cell_id is None:
            if require_all:
                raise ValueError(f"tree node {node.get('id')!r} has genome but no cell_id")
            continue
        if cell_id not in genome_dict:
            if require_all:
                raise ValueError(
                    f"tree node {node.get('id')!r} has cell_id {cell_id!r} "
                    "missing from genome dictionary"
                )
            continue
        if genome_to_key(node["genome"]) != genome_to_key(genome_dict[cell_id]):
            raise ValueError(
                f"tree node {node.get('id')!r} genome does not match "
                f"genome dictionary entry for cell_id {cell_id!r}"
            )
        del node["genome"]
    return stripped


def hydrate_tree_payload(tree_payload, genome_dict, *, require_all=True):
    hydrated = copy.deepcopy(tree_payload)
    for node in hydrated.get("nodes", []):
        if "genome" in node:
            continue
        cell_id = node.get("cell_id")
        if cell_id is None:
            if require_all:
                raise ValueError(f"tree node {node.get('id')!r} has no genome and no cell_id")
            continue
        if cell_id not in genome_dict:
            if require_all:
                raise ValueError(
                    f"tree node {node.get('id')!r} has cell_id {cell_id!r} "
                    "missing from genome dictionary"
                )
            continue
        node["genome"] = copy.deepcopy(genome_dict[cell_id])
    return hydrated


def strip_genomes_from_biopsies(biopsies, genome_dict):
    stripped = copy.deepcopy(biopsies)
    for biopsy in stripped:
        for cell in biopsy.get("cells", []):
            if "genome" not in cell:
                continue
            cell_id = cell.get("cell_id")
            if cell_id is None:
                raise ValueError(
                    f"biopsy {biopsy.get('level')!r} cell {cell.get('node_id')!r} "
                    "has genome but no cell_id"
                )
            if cell_id not in genome_dict:
                raise ValueError(
                    f"biopsy {biopsy.get('level')!r} cell {cell.get('node_id')!r} "
                    f"has cell_id {cell_id!r} missing from genome dictionary"
                )
            if genome_to_key(cell["genome"]) != genome_to_key(genome_dict[cell_id]):
                raise ValueError(
                    f"biopsy {biopsy.get('level')!r} cell {cell.get('node_id')!r} "
                    f"genome does not match genome dictionary entry for cell_id {cell_id!r}"
                )
            del cell["genome"]
    return stripped


def hydrate_biopsies(biopsies, genome_dict):
    hydrated = copy.deepcopy(biopsies)
    for biopsy in hydrated:
        for cell in biopsy.get("cells", []):
            if "genome" in cell:
                continue
            cell_id = cell.get("cell_id")
            if cell_id is None:
                raise ValueError(
                    f"biopsy {biopsy.get('level')!r} cell {cell.get('node_id')!r} "
                    "has no genome and no cell_id"
                )
            if cell_id not in genome_dict:
                raise ValueError(
                    f"biopsy {biopsy.get('level')!r} cell {cell.get('node_id')!r} "
                    f"has cell_id {cell_id!r} missing from genome dictionary"
                )
            cell["genome"] = copy.deepcopy(genome_dict[cell_id])
    return hydrated


def split_input_payloads(input_case):
    input_case = canonicalize_input_case_cell_ids_by_genome(input_case)
    genome_dict = build_genome_dictionary(input_case)
    biopsy_payload = {
        key: copy.deepcopy(value)
        for key, value in input_case.items()
        if key not in {"true_tree", "biopsies"}
    }
    biopsy_payload.update({
        "input_layout": SPLIT_INPUT_LAYOUT,
        "true_tree_file": TRUE_TREE_INPUT_FILE_NAME,
        "genome_dict_file": GENOME_DICT_FILE_NAME,
        "biopsies": strip_genomes_from_biopsies(input_case.get("biopsies", []), genome_dict),
    })
    true_tree_payload = {
        "case_id": input_case.get("case_id"),
        "corpus": input_case.get("corpus", CORPUS_NAME),
        "input_layout": SPLIT_INPUT_LAYOUT,
        "genome_dict_file": GENOME_DICT_FILE_NAME,
        "true_tree": strip_genomes_from_tree_payload(input_case["true_tree"], genome_dict),
    }
    return true_tree_payload, biopsy_payload, genome_dict


def split_input_exists(case_directory):
    case_directory = Path(case_directory)
    return (
        (case_directory / TRUE_TREE_INPUT_FILE_NAME).exists()
        and (case_directory / BIOPSY_INPUT_FILE_NAME).exists()
        and (case_directory / GENOME_DICT_FILE_NAME).exists()
    )


def input_artifact_exists(case_directory, layout=None):
    case_directory = Path(case_directory)
    if layout == "legacy":
        return (case_directory / LEGACY_INPUT_FILE_NAME).exists()
    if layout == "split":
        return split_input_exists(case_directory)
    return (case_directory / LEGACY_INPUT_FILE_NAME).exists() or split_input_exists(case_directory)


def input_reference_path(case_directory, preferred_layout=None):
    case_directory = Path(case_directory)
    legacy_path = case_directory / LEGACY_INPUT_FILE_NAME
    split_path = case_directory / BIOPSY_INPUT_FILE_NAME
    if preferred_layout == "split" and split_input_exists(case_directory):
        return split_path
    if preferred_layout == "legacy" and legacy_path.exists():
        return legacy_path
    if legacy_path.exists():
        return legacy_path
    if split_input_exists(case_directory):
        return split_path
    return legacy_path


def write_input_case(case_directory, input_case, *, layout="legacy", overwrite=False):
    case_directory = Path(case_directory)
    if layout == "legacy":
        path = case_directory / LEGACY_INPUT_FILE_NAME
        write_json(path, input_case, overwrite=overwrite)
        return [path]
    if layout != "split":
        raise ValueError(f"Unsupported input layout: {layout}")
    true_tree_payload, biopsy_payload, genome_dict = split_input_payloads(input_case)
    paths = [
        case_directory / TRUE_TREE_INPUT_FILE_NAME,
        case_directory / BIOPSY_INPUT_FILE_NAME,
        case_directory / GENOME_DICT_FILE_NAME,
    ]
    if not overwrite and any(path.exists() for path in paths):
        return []
    write_json(paths[0], true_tree_payload, overwrite=True)
    write_json(paths[1], biopsy_payload, overwrite=True)
    write_genome_dict(paths[2], genome_dict, overwrite=True)
    return paths


def load_split_input_case(case_directory):
    case_directory = Path(case_directory)
    biopsy_payload = load_json(case_directory / BIOPSY_INPUT_FILE_NAME)
    true_tree_payload = load_json(case_directory / TRUE_TREE_INPUT_FILE_NAME)
    genome_dict = read_genome_dict(case_directory / GENOME_DICT_FILE_NAME)
    input_case = {
        key: copy.deepcopy(value)
        for key, value in biopsy_payload.items()
        if key not in {"true_tree_file", "genome_dict_file"}
    }
    input_case["true_tree"] = hydrate_tree_payload(
        true_tree_payload["true_tree"],
        genome_dict,
        require_all=True,
    )
    input_case["biopsies"] = hydrate_biopsies(
        biopsy_payload.get("biopsies", []),
        genome_dict,
    )
    input_case["input_layout"] = SPLIT_INPUT_LAYOUT
    return input_case


def load_split_evaluation_input_case(case_directory):
    case_directory = Path(case_directory)
    biopsy_payload = load_json(case_directory / BIOPSY_INPUT_FILE_NAME)
    true_tree_payload = load_json(case_directory / TRUE_TREE_INPUT_FILE_NAME)
    genome_dict = read_genome_dict(case_directory / GENOME_DICT_FILE_NAME)
    input_case = {
        key: copy.deepcopy(value)
        for key, value in biopsy_payload.items()
        if key not in {"true_tree_file", "genome_dict_file", "biopsies"}
    }
    if input_case.get("status") == "ok":
        input_case["true_tree"] = hydrate_tree_payload(
            true_tree_payload["true_tree"],
            genome_dict,
            require_all=True,
        )
    input_case["input_layout"] = SPLIT_INPUT_LAYOUT
    return input_case, genome_dict


def load_input_case(case_directory_or_path, *, preferred_layout=None, hydrate=True):
    path = Path(case_directory_or_path)
    case_directory = path if path.is_dir() else path.parent
    legacy_path = case_directory / LEGACY_INPUT_FILE_NAME
    if preferred_layout == "split" and split_input_exists(case_directory):
        return load_split_input_case(case_directory) if hydrate else load_json(case_directory / BIOPSY_INPUT_FILE_NAME)
    if legacy_path.exists():
        return load_json(legacy_path)
    if split_input_exists(case_directory):
        return load_split_input_case(case_directory) if hydrate else load_json(case_directory / BIOPSY_INPUT_FILE_NAME)
    raise FileNotFoundError(f"No input artifact found in {case_directory}")


def load_evaluation_input_case(case_directory_or_path, *, preferred_layout=None):
    path = Path(case_directory_or_path)
    case_directory = path if path.is_dir() else path.parent
    legacy_path = case_directory / LEGACY_INPUT_FILE_NAME
    if preferred_layout == "split" and split_input_exists(case_directory):
        return load_split_evaluation_input_case(case_directory)
    if legacy_path.exists():
        input_case = load_json(legacy_path)
        try:
            genome_dict = genome_dict_from_input_case(input_case)
        except ValueError:
            genome_dict = None
        return input_case, genome_dict
    if split_input_exists(case_directory):
        return load_split_evaluation_input_case(case_directory)
    raise FileNotFoundError(f"No input artifact found in {case_directory}")


def load_input_case_by_spec(output_root, spec, *, preferred_layout=None, hydrate=True):
    return load_input_case(
        case_dir(output_root, spec),
        preferred_layout=preferred_layout,
        hydrate=hydrate,
    )


def input_case_directories(output_root, specs=None):
    output_root = Path(output_root)
    if specs is not None:
        return [
            case_dir(output_root, spec)
            for spec in specs
            if input_artifact_exists(case_dir(output_root, spec))
        ]
    directories = {path.parent for path in output_root.glob(f"**/{LEGACY_INPUT_FILE_NAME}")}
    directories.update(
        path.parent
        for path in output_root.glob(f"**/{BIOPSY_INPUT_FILE_NAME}")
        if split_input_exists(path.parent)
    )
    return sorted(directories)


def genome_dict_from_input_case(input_case):
    return build_genome_dictionary(input_case)


def strip_reconstructed_result_genomes(result, input_case):
    result = copy.deepcopy(result)
    if result.get("status") == "failed" or "reconstructed_tree" not in result:
        return result
    genome_dict = genome_dict_from_input_case(input_case)
    result["reconstructed_tree"] = strip_genomes_from_tree_payload(
        result["reconstructed_tree"],
        genome_dict,
        require_all=False,
    )
    result["genome_dict_file"] = GENOME_DICT_FILE_NAME
    return result


def cell_lists_from_input(input_case):
    cell_lists = []
    for biopsy in input_case.get("biopsies", []):
        cells = genotypes_from_json(biopsy["cells"])
        if cells:
            cell_lists.append(cells)
    return cell_lists


def true_tree_from_input(input_case):
    return node_link_graph(input_case["true_tree"])


def build_evaluation_case_context(input_case, genome_dict=None):
    true_tree = true_tree_from_input(input_case)
    true_root = root_id(true_tree)
    return EvaluationCaseContext(
        input_case=input_case,
        genome_dict=genome_dict,
        true_tree=true_tree,
        true_eval_context=tree_evaluation_context(true_tree),
        true_root=true_root,
        true_cluster_context=cluster_evaluation_context(true_tree, true_root),
    )


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
        return empty_distance_payload(input_case, distance_mode=distance_mode)
    cell_lists = cell_lists_from_input(input_case)
    unique_cells = unique_cells_by_cell_id(cell_lists)
    if distance_mode == "cnp2cnp":
        cnp_ids, cnp_matrix = cnp2cnp_distance_matrix(unique_cells)
    elif distance_mode == "l1":
        cnp_ids, cnp_matrix = l1_distance_matrix(unique_cells)
    else:
        raise ValueError(f"Unsupported distance mode: {distance_mode}")
    return {
        "case_id": input_case["case_id"],
        "corpus": input_case.get("corpus", CORPUS_NAME),
        "status": "ok",
        "failure_reason": None,
        "distance_mode": distance_mode,
        "biopsy_cell_count": sum(len(level) for level in cell_lists),
        "unique_biopsy_cell_count": len(cnp_ids),
        "unique_distance_cell_ids": cnp_ids,
        "distance_matrices": {
            "cnp2cnp": {
                "ids": cnp_ids,
                "matrix": cnp_matrix,
                "distance_mode": distance_mode,
            },
        },
    }


def actual_root(tree):
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    if len(roots) != 1:
        raise ValueError(f"Tree must have exactly one root, found {len(roots)}")
    return roots[0]


def reconstruction_result(input_case, distance_payload, algorithm, mode):
    cell_lists = cell_lists_from_input(input_case)
    all_in_one_sample = [[copy.deepcopy(cell) for level in cell_lists for cell in level]]
    matrix = distance_payload["distance_matrices"]["cnp2cnp"]
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


def metric_summary_with_timings(
    true_tree,
    reconstructed_tree,
    *,
    true_eval_context=None,
    true_root=None,
    true_cluster_context=None,
):
    timings = []
    adf1_started_at = utc_timestamp()
    adf1_start = time.perf_counter()
    if true_eval_context is None:
        true_eval_context = tree_evaluation_context(true_tree)
    reconstructed_eval_context = tree_evaluation_context(reconstructed_tree)
    restricted_labels = {
        str(data.get("cell_id"))
        for _, data in reconstructed_tree.nodes(data=True)
        if data.get("cell_id") is not None
    }
    metrics = evaluate_4(
        true_eval_context,
        reconstructed_eval_context,
        restrict_labels=restricted_labels,
    )
    timings.append({
        "evaluation_method": "adf1",
        "elapsed_seconds": time.perf_counter() - adf1_start,
        "started_at": adf1_started_at,
        "finished_at": utc_timestamp(),
    })

    grf_started_at = utc_timestamp()
    grf_start = time.perf_counter()
    if true_root is None:
        true_root = root_id(true_tree)
    if true_cluster_context is None:
        true_cluster_context = cluster_evaluation_context(true_tree, true_root)
    reconstructed_root = root_id(reconstructed_tree)
    reconstructed_cluster_context = cluster_evaluation_context(
        reconstructed_tree,
        reconstructed_root,
    )
    jaccard_cache = {}
    ext_grf = ext_grf_from_cluster_counts(
        true_cluster_context.counts,
        reconstructed_cluster_context.counts,
        jaccard_cache=jaccard_cache,
    )
    legacy_grf = legacy_set_grf_similarity_from_cluster_contexts(
        true_cluster_context,
        reconstructed_cluster_context,
        jaccard_cache=jaccard_cache,
    )
    timings.append({
        "evaluation_method": "grf",
        "elapsed_seconds": time.perf_counter() - grf_start,
        "started_at": grf_started_at,
        "finished_at": utc_timestamp(),
    })

    return {
        "ancestors_unique_restricted": metrics["ancestors_unique_restricted"],
        "ancestors_multiset": metrics["ancestors_multiset"],
        "ancestors_unique": metrics["ancestors_unique"],
        "grf": 1 - ext_grf,
        EXT_GRF_METRIC_FIELD: ext_grf,
        LEGACY_GRF_SET_SIMILARITY_FIELD: legacy_grf,
    }, timings


def evaluate_result(input_case, result, metric_timings=None, evaluation_context=None):
    if result.get("status") == "failed":
        return result
    if evaluation_context is None:
        true_tree = true_tree_from_input(input_case)
        try:
            genome_dict = genome_dict_from_input_case(input_case)
        except ValueError:
            genome_dict = None
        true_eval_context = None
        true_root = None
        true_cluster_context = None
    else:
        true_tree = evaluation_context.true_tree
        genome_dict = evaluation_context.genome_dict
        true_eval_context = evaluation_context.true_eval_context
        true_root = evaluation_context.true_root
        true_cluster_context = evaluation_context.true_cluster_context
    reconstructed_payload = result["reconstructed_tree"]
    if genome_dict is not None:
        reconstructed_payload = hydrate_tree_payload(
            reconstructed_payload,
            genome_dict,
            require_all=False,
        )
    reconstructed_tree = node_link_graph(reconstructed_payload)
    result = copy.deepcopy(result)
    metrics, timings = metric_summary_with_timings(
        true_tree,
        reconstructed_tree,
        true_eval_context=true_eval_context,
        true_root=true_root,
        true_cluster_context=true_cluster_context,
    )
    result["metrics"] = metrics
    result["status"] = "evaluated"
    if metric_timings is not None:
        metric_timings.extend(timings)
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


def case_result_record(input_case, result_file, evaluated_result):
    row = {
        **case_parameters(input_case),
        "mode": evaluated_result.get("mode"),
        "algorithm": evaluated_result.get("algorithm", Path(result_file).stem),
        "status": evaluated_result.get("status"),
        "adf1": "",
        "grf": "",
        "result_file": str(Path(result_file).relative_to(Path(result_file).parent.parent)),
        "error": evaluated_result.get("error", ""),
    }
    if evaluated_result.get("status") == "evaluated" and "metrics" in evaluated_result:
        row["adf1"] = metric_value(evaluated_result, "adf1")
        row["grf"] = metric_value(evaluated_result, "grf")
    return row


def write_case_result_file(case_directory, records, *, overwrite=True):
    case_directory = Path(case_directory)
    rows = [{column: record.get(column, "") for column in RESULT_ROW_COLUMNS} for record in records]
    csv_path = case_directory / CASE_RESULT_CSV_NAME
    if csv_path.exists() and not overwrite:
        return csv_path
    case_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_ROW_COLUMNS).to_csv(csv_path, index=False)
    return csv_path


def load_case_result_rows(case_directory):
    case_directory = Path(case_directory)
    csv_path = case_directory / CASE_RESULT_CSV_NAME
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        return frame.to_dict(orient="records")
    return []


def collect_result_rows(output_root):
    rows = []
    output_root = Path(output_root)
    if not output_root.exists():
        return pd.DataFrame()
    for result_csv in sorted(output_root.glob(f"**/{CASE_RESULT_CSV_NAME}")):
        if result_csv.parent.name == "reports":
            continue
        frame = pd.read_csv(result_csv)
        for row in frame.to_dict(orient="records"):
            if row.get("status") != "evaluated" or pd.isna(row.get("adf1")) or pd.isna(row.get("grf")):
                continue
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=RESULT_ROW_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in [
        "genome_length",
        "generation_count",
        "seed",
        "biopsy_level_count",
        "duplication_multiplicity",
    ]:
        if column in frame:
            frame[column] = frame[column].astype(int)
    for column in [
        "biopsy_size_scalable",
        "general_event_prob",
        "single_or_multiple_event_prob",
        "adf1",
        "grf",
    ]:
        if column in frame:
            frame[column] = frame[column].astype(float)
    return frame


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


def selected_biopsy_cell_count(input_case):
    return sum(len(biopsy.get("cells", [])) for biopsy in input_case.get("biopsies", []))


def input_files_for_specs(output_root, specs=None):
    return [input_reference_path(directory) for directory in input_case_directories(output_root, specs)]


def biopsy_summary_label(specs):
    if not specs:
        return None
    generation_counts = sorted({spec.generation_count for spec in specs})
    if len(generation_counts) == 1:
        return f"g{generation_counts[0]}"
    return "selected"


def biopsy_cell_summary(output_root, specs=None):
    rows = []
    for case_directory in input_case_directories(output_root, specs):
        input_case = load_input_case(case_directory, hydrate=False)
        if input_case.get("corpus") != CORPUS_NAME:
            continue
        rows.append({
            "generation": f"g{input_case['NUMBER_OF_GENERATIONS']}",
            "bss": input_case["biopsy_size_scalable"],
            "level": f"L{input_case['biopsy_level_count']}",
            "selected_cells": selected_biopsy_cell_count(input_case),
        })
    if not rows:
        return pd.DataFrame(columns=BIOPSY_CELL_SUMMARY_COLUMNS)
    frame = pd.DataFrame(rows)
    summaries = []
    for key, group in frame.groupby(["generation", "bss", "level"], dropna=False):
        generation, bss, level = key
        values = group["selected_cells"].astype(int)
        summaries.append({
            "generation": generation,
            "bss": float(bss),
            "level": level,
            "n": int(len(values)),
            "min": int(values.min()),
            "max": int(values.max()),
            "avg": float(values.mean()),
            "total": int(values.sum()),
        })
    summary = pd.DataFrame(summaries, columns=BIOPSY_CELL_SUMMARY_COLUMNS)
    summary["_generation_sort"] = summary["generation"].str.replace("^g", "", regex=True).astype(int)
    summary["_bss_sort"] = summary["bss"].astype(float)
    summary["_level_sort"] = summary["level"].str.replace("^L", "", regex=True).astype(int)
    summary = summary.sort_values(["_generation_sort", "_bss_sort", "_level_sort"])
    return summary.drop(columns=["_generation_sort", "_bss_sort", "_level_sort"]).reset_index(drop=True)


def format_biopsy_cell_summary_markdown(summary):
    lines = []
    if summary.empty:
        return "No input cases found.\n"
    for generation, group in summary.groupby("generation", sort=False):
        lines.append(f"## {generation}")
        lines.append("")
        lines.append("| bss | L | n | min | max | avg | total |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in group.to_dict(orient="records"):
            lines.append(
                "| "
                f"{row['bss']:g} | "
                f"{row['level']} | "
                f"{int(row['n']):,} | "
                f"{int(row['min']):,} | "
                f"{int(row['max']):,} | "
                f"{float(row['avg']):,.2f} | "
                f"{int(row['total']):,} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_biopsy_cell_summary(output_root, specs=None, label=None):
    reports_dir = Path(output_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = biopsy_cell_summary(output_root, specs=specs)
    suffix = f"_{label}" if label else ""
    csv_path = reports_dir / f"biopsy_cell_summary{suffix}.csv"
    md_path = reports_dir / f"biopsy_cell_summary{suffix}.md"
    summary.to_csv(csv_path, index=False)
    md_path.write_text(format_biopsy_cell_summary_markdown(summary))
    return {"csv": csv_path, "markdown": md_path}


def write_reports(output_root):
    reports_dir = Path(output_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_result_rows(output_root)
    rows_path = reports_dir / "result_rows.csv"
    rows.to_csv(rows_path, index=False)
    biopsy_summary_paths = write_biopsy_cell_summary(output_root)

    if rows.empty:
        return {
            "rows": rows_path,
            "summaries": [],
            "violations": [],
            "heatmaps": [],
            "biopsy_summary": biopsy_summary_paths,
        }

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
        "biopsy_summary": biopsy_summary_paths,
    }


def _metric_from_metrics_dict(metrics, metric):
    value = metrics
    for key in METRIC_FIELDS[metric][1:]:
        value = value[key]
    return float(value)


def _check_distance_matrix(input_case, distance_payload, distance_file):
    errors = []
    if distance_payload.get("case_id") != input_case.get("case_id"):
        errors.append(
            f"{distance_file}: case_id {distance_payload.get('case_id')!r} "
            f"does not match input {input_case.get('case_id')!r}"
        )
    distance_matrices = distance_payload.get("distance_matrices", {})
    if "cnp2cnp" not in distance_matrices:
        return [f"{distance_file}: missing distance_matrices.cnp2cnp"]

    cell_lists = cell_lists_from_input(input_case)
    expected_ids = [cell.get_id() for cell in unique_cells_by_cell_id(cell_lists)]
    matrix_payload = distance_matrices["cnp2cnp"]
    ids = matrix_payload.get("ids")
    matrix = np.array(matrix_payload.get("matrix", []), dtype=float)
    if ids != expected_ids:
        errors.append(f"{distance_file}: cnp2cnp ids {ids} do not match unique biopsy cell ids {expected_ids}")
    if matrix.shape != (len(ids), len(ids)):
        errors.append(f"{distance_file}: cnp2cnp matrix shape {matrix.shape} does not match ids length {len(ids)}")
    elif len(ids) > 0:
        if not np.allclose(np.diag(matrix), 0.0):
            errors.append(f"{distance_file}: cnp2cnp matrix diagonal is not zero")
        if not np.allclose(matrix, matrix.T):
            errors.append(f"{distance_file}: cnp2cnp matrix is not symmetric")
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


def _check_result_metrics(input_case, result_file, result_row):
    result = load_json(result_file)
    if result.get("status") == "failed":
        return [f"{result_file}: reconstructed result status is failed"]
    if "reconstructed_tree" not in result:
        return [f"{result_file}: missing reconstructed_tree"]

    errors = []
    recomputed = evaluate_result(input_case, result)
    if result_row is None:
        return [f"{result_file}: missing evaluated row in {CASE_RESULT_CSV_NAME}"]
    if result_row.get("status") != "evaluated":
        errors.append(
            f"{result_file}: evaluated row status is {result_row.get('status')!r}, "
            "expected 'evaluated'"
        )
    for metric in METRICS:
        stored_value = result_row.get(metric)
        if stored_value in ("", None) or pd.isna(stored_value):
            errors.append(f"{result_file}: missing {metric} in {CASE_RESULT_CSV_NAME}")
            continue
        stored = float(stored_value)
        current = metric_value(recomputed, metric)
        if abs(stored - current) > CHECK_TOLERANCE:
            errors.append(
                f"{result_file}: stored {metric}={stored} in {CASE_RESULT_CSV_NAME} "
                f"does not match recomputed {current}"
            )
    return errors


def check_corpus(output_root, algorithms=None, modes=None, *, replay_reports=True):
    output_root = Path(output_root)
    modes = modes or list(MODES)
    algorithm_names = None
    if algorithms is not None:
        algorithm_names = [getattr(algorithm, "__name__", str(algorithm)) for algorithm in algorithms]

    case_directories = input_case_directories(output_root)
    errors = []
    checked_inputs = 0
    failed_inputs = 0
    checked_results = 0
    missing_results = 0
    for case_directory in case_directories:
        input_file = input_reference_path(case_directory)
        try:
            input_case = load_input_case(case_directory)
        except Exception as exc:
            errors.append(f"{input_file}: failed to load input artifact: {exc}")
            continue
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
        if "distance_matrices" in input_case or "unique_distance_cell_ids" in input_case:
            errors.append(
                f"{input_file}: distance data must be stored in {DISTANCE_FILE_NAME}, "
                "not input.json"
            )
        biopsy_level_count = input_case.get("biopsy_level_count")
        if biopsy_level_count not in DEFAULT_BIOPSY_LEVEL_COUNTS:
            errors.append(
                f"{input_file}: biopsy_level_count {biopsy_level_count!r} "
                f"is outside active scope {list(DEFAULT_BIOPSY_LEVEL_COUNTS)}"
            )
        errors.extend(_check_biopsy_order(input_case, input_file))
        if input_case.get("status") != "ok":
            failed_inputs += 1
            continue

        checked_inputs += 1
        distance_file = case_directory / DISTANCE_FILE_NAME
        if not distance_file.exists():
            errors.append(f"{input_file}: missing {DISTANCE_FILE_NAME}")
        else:
            errors.extend(_check_distance_matrix(input_case, load_json(distance_file), distance_file))
        evaluated_rows = {
            (str(row.get("mode")), str(row.get("algorithm"))): row
            for row in load_case_result_rows(case_directory)
        }
        for mode in modes:
            mode_dir = case_directory / mode
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
                result_row = evaluated_rows.get((mode, algorithm_name))
                errors.extend(_check_result_metrics(input_case, result_file, result_row))

    report_paths = []
    if replay_reports:
        written = write_reports(output_root)
        for value in [
            written["rows"],
            *written["summaries"],
            *written["violations"],
            *written["heatmaps"],
            *written["biopsy_summary"].values(),
        ]:
            path = Path(value)
            report_paths.append(path)
            if not path.exists():
                errors.append(f"{path}: expected report file was not written")

    return {
        "input_files": len(case_directories),
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


def run_simulate_stage(specs, args, base_config, timing=None):
    stage_start = time.perf_counter()
    core_seconds = 0.0
    write_seconds = 0.0
    completed = 0
    skipped = 0
    failed = 0
    input_layout = getattr(args, "input_layout", "legacy")
    for spec in specs:
        directory = case_dir(args.output_root, spec)
        if input_artifact_exists(directory, layout=input_layout) and not args.overwrite:
            print(f"exists input {input_reference_path(directory, preferred_layout=input_layout)}")
            skipped += 1
            continue
        try:
            core_start = time.perf_counter()
            case_timing_records = []
            input_case = input_case_from_simulation(
                spec,
                base_config,
                max_retries=args.max_biopsy_generation_retries,
                timing_records=case_timing_records,
            )
            core_seconds += time.perf_counter() - core_start
        except Exception:
            failed += 1
            raise
        write_start = time.perf_counter()
        written = write_input_case(
            directory,
            input_case,
            layout=input_layout,
            overwrite=True,
        )
        write_seconds += time.perf_counter() - write_start
        write_case_timing_records(directory, case_timing_records)
        completed += 1
        print(f"wrote input {written[0] if written else directory}")
    if timing is not None:
        timing.add(
            "simulate",
            "simulate_case",
            count=completed,
            input_files=len(specs),
            instances=completed,
            skipped=skipped,
            failed=failed,
            core_seconds=core_seconds,
            write_json_seconds=write_seconds,
            total_seconds=time.perf_counter() - stage_start,
        )


def run_distance_stage(specs, args, timing=None):
    stage_start = time.perf_counter()
    read_seconds = 0.0
    core_seconds = 0.0
    write_seconds = 0.0
    completed = 0
    skipped = 0
    failed = 0
    missing = 0
    input_layout = getattr(args, "input_layout", "legacy")
    for spec in specs:
        directory = case_dir(args.output_root, spec)
        output_file = distance_path(args.output_root, spec)
        if not input_artifact_exists(directory):
            print(f"missing input {input_reference_path(directory)}")
            missing += 1
            continue
        read_start = time.perf_counter()
        input_case = load_input_case(
            directory,
            preferred_layout=input_layout,
        )
        read_seconds += time.perf_counter() - read_start
        if input_case.get("status") != "ok":
            print(f"skip failed input {input_reference_path(directory, preferred_layout=input_layout)}")
            skipped += 1
            continue
        if output_file.exists() and not args.overwrite:
            print(f"exists distances {output_file}")
            skipped += 1
            continue
        try:
            operation_started_at = utc_timestamp()
            core_start = time.perf_counter()
            distance_payload = compute_case_distances(input_case, distance_mode=args.distance_mode)
            core_elapsed = time.perf_counter() - core_start
            operation_finished_at = utc_timestamp()
            core_seconds += core_elapsed
        except Exception:
            failed += 1
            raise
        write_start = time.perf_counter()
        write_json(output_file, distance_payload, overwrite=True)
        write_seconds += time.perf_counter() - write_start
        write_case_timing_records(directory, [case_timing_record(
            input_case,
            stage="distance",
            operation=f"distance_{args.distance_mode}",
            elapsed_seconds=core_elapsed,
            started_at=operation_started_at,
            finished_at=operation_finished_at,
            status=distance_payload["status"],
            distance_mode=args.distance_mode,
            input_cell_count=distance_payload.get("biopsy_cell_count", ""),
            unique_cell_count=distance_payload.get("unique_biopsy_cell_count", ""),
        )])
        completed += 1
        print(f"wrote distances {output_file}")
    if timing is not None:
        timing.add(
            "distance",
            "distance_case",
            count=completed,
            input_files=len(specs),
            instances=completed,
            skipped=skipped,
            missing=missing,
            failed=failed,
            read_json_seconds=read_seconds,
            core_seconds=core_seconds,
            write_json_seconds=write_seconds,
            total_seconds=time.perf_counter() - stage_start,
            distance_mode=args.distance_mode,
        )


def run_reconstruct_stage(specs, args, algorithms, modes, timing=None):
    stage_start = time.perf_counter()
    read_seconds = 0.0
    core_seconds = 0.0
    write_seconds = 0.0
    completed = 0
    skipped = 0
    failed = 0
    missing = 0
    by_algorithm_mode = {}
    by_algorithm = {}
    input_layout = getattr(args, "input_layout", "legacy")
    for spec in specs:
        directory = case_dir(args.output_root, spec)
        input_file = input_reference_path(directory, preferred_layout=input_layout)
        distance_file = distance_path(args.output_root, spec)
        if not input_artifact_exists(directory):
            print(f"missing input {input_file}")
            missing += 1
            continue
        if not distance_file.exists():
            print(f"missing distances {distance_file}")
            missing += 1
            continue
        read_start = time.perf_counter()
        input_case = load_input_case(
            directory,
            preferred_layout=input_layout,
        )
        distance_payload = load_json(distance_file)
        read_seconds += time.perf_counter() - read_start
        if input_case.get("status") != "ok" or "cnp2cnp" not in distance_payload.get("distance_matrices", {}):
            print(f"skip input without distances {input_file}")
            skipped += 1
            continue
        for algorithm in algorithms:
            algorithm_name = getattr(algorithm, "__name__", str(algorithm))
            for mode in modes:
                timing_key = (algorithm_name, mode)
                by_algorithm_mode.setdefault(timing_key, {
                    "count": 0,
                    "core_seconds": 0.0,
                    "write_json_seconds": 0.0,
                    "failed": 0,
                    "skipped": 0,
                })
                by_algorithm.setdefault(algorithm_name, {
                    "count": 0,
                    "core_seconds": 0.0,
                    "write_json_seconds": 0.0,
                    "failed": 0,
                    "skipped": 0,
                })
                output_file = result_path(args.output_root, spec, mode, algorithm_name)
                if output_file.exists() and not args.overwrite:
                    print(f"exists result {output_file}")
                    skipped += 1
                    by_algorithm_mode[timing_key]["skipped"] += 1
                    by_algorithm[algorithm_name]["skipped"] += 1
                    continue
                operation_started_at = utc_timestamp()
                core_start = time.perf_counter()
                try:
                    result = reconstruction_result(input_case, distance_payload, algorithm, mode)
                    if input_case.get("input_layout") == SPLIT_INPUT_LAYOUT:
                        result = strip_reconstructed_result_genomes(result, input_case)
                except Exception as exc:
                    failed += 1
                    by_algorithm_mode[timing_key]["failed"] += 1
                    by_algorithm[algorithm_name]["failed"] += 1
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
                core_elapsed = time.perf_counter() - core_start
                operation_finished_at = utc_timestamp()
                core_seconds += core_elapsed
                by_algorithm_mode[timing_key]["core_seconds"] += core_elapsed
                by_algorithm[algorithm_name]["core_seconds"] += core_elapsed
                write_start = time.perf_counter()
                write_json(output_file, result, overwrite=True)
                write_elapsed = time.perf_counter() - write_start
                write_seconds += write_elapsed
                by_algorithm_mode[timing_key]["write_json_seconds"] += write_elapsed
                by_algorithm[algorithm_name]["write_json_seconds"] += write_elapsed
                completed += 1
                by_algorithm_mode[timing_key]["count"] += 1
                by_algorithm[algorithm_name]["count"] += 1
                tree_node_count = ""
                if result.get("status") == "reconstructed" and "reconstructed_tree" in result:
                    tree_node_count = len(result["reconstructed_tree"].get("nodes", []))
                write_case_timing_records(directory, [case_timing_record(
                    input_case,
                    stage="reconstruct",
                    operation="reconstruct",
                    elapsed_seconds=core_elapsed,
                    started_at=operation_started_at,
                    finished_at=operation_finished_at,
                    status=result.get("status", "failed"),
                    algorithm=algorithm_name,
                    mode=mode,
                    tree_node_count=tree_node_count,
                    error=result.get("error", ""),
                )])
                print(f"wrote result {output_file}")
    if timing is not None:
        total_seconds = time.perf_counter() - stage_start
        timing.add(
            "reconstruct",
            "reconstruct_result",
            count=completed,
            input_files=len(specs),
            instances=completed,
            skipped=skipped,
            missing=missing,
            failed=failed,
            read_json_seconds=read_seconds,
            core_seconds=core_seconds,
            write_json_seconds=write_seconds,
            total_seconds=total_seconds,
        )
        for algorithm_name, values in sorted(by_algorithm.items()):
            timing.add(
                "reconstruct",
                "reconstruct_result_by_algorithm",
                scope="algorithm",
                count=values["count"],
                instances=values["count"],
                skipped=values["skipped"],
                failed=values["failed"],
                core_seconds=values["core_seconds"],
                write_json_seconds=values["write_json_seconds"],
                total_seconds=values["core_seconds"] + values["write_json_seconds"],
                algorithm=algorithm_name,
            )


def run_evaluate_stage(specs, args, algorithms, modes, timing=None):
    stage_start = time.perf_counter()
    read_seconds = 0.0
    core_seconds = 0.0
    write_seconds = 0.0
    completed = 0
    skipped = 0
    failed = 0
    missing = 0
    by_algorithm_mode = {}
    by_evaluation_method = {}
    input_layout = getattr(args, "input_layout", "legacy")
    for spec in specs:
        case_directory = case_dir(args.output_root, spec)
        input_file = input_reference_path(case_directory, preferred_layout=input_layout)
        case_result_csv = case_result_csv_path(args.output_root, spec)
        if not input_artifact_exists(case_directory):
            print(f"missing input {input_file}")
            missing += 1
            continue
        if case_result_csv.exists() and not args.overwrite:
            print(f"exists evaluated results {case_result_csv}")
            skipped += 1
            continue
        read_start = time.perf_counter()
        input_case, genome_dict = load_evaluation_input_case(
            case_directory,
            preferred_layout=input_layout,
        )
        read_seconds += time.perf_counter() - read_start
        if input_case.get("status") != "ok":
            skipped += 1
            continue
        core_start = time.perf_counter()
        evaluation_context = build_evaluation_case_context(
            input_case,
            genome_dict=genome_dict,
        )
        core_seconds += time.perf_counter() - core_start
        case_records = []
        case_timing_records = []
        for algorithm in algorithms:
            algorithm_name = getattr(algorithm, "__name__", str(algorithm))
            for mode in modes:
                timing_key = (algorithm_name, mode)
                by_algorithm_mode.setdefault(timing_key, {
                    "count": 0,
                    "core_seconds": 0.0,
                    "read_json_seconds": 0.0,
                    "write_json_seconds": 0.0,
                    "failed": 0,
                    "skipped": 0,
                    "missing": 0,
                })
                output_file = result_path(args.output_root, spec, mode, algorithm_name)
                if not output_file.exists():
                    missing += 1
                    by_algorithm_mode[timing_key]["missing"] += 1
                    continue
                read_start = time.perf_counter()
                result = load_json(output_file)
                read_elapsed = time.perf_counter() - read_start
                read_seconds += read_elapsed
                by_algorithm_mode[timing_key]["read_json_seconds"] += read_elapsed
                core_start = time.perf_counter()
                try:
                    metric_timings = []
                    evaluated = evaluate_result(
                        input_case,
                        result,
                        metric_timings=metric_timings,
                        evaluation_context=evaluation_context,
                    )
                except Exception as exc:
                    failed += 1
                    by_algorithm_mode[timing_key]["failed"] += 1
                    evaluated = copy.deepcopy(result)
                    evaluated["status"] = "failed"
                    evaluated["error"] = str(exc)
                    metric_timings = []
                    if args.fail_fast:
                        raise
                core_elapsed = time.perf_counter() - core_start
                core_seconds += core_elapsed
                by_algorithm_mode[timing_key]["core_seconds"] += core_elapsed
                case_records.append(case_result_record(input_case, output_file, evaluated))
                for metric_timing in metric_timings:
                    method = metric_timing["evaluation_method"]
                    by_evaluation_method.setdefault(method, {
                        "count": 0,
                        "core_seconds": 0.0,
                    })
                    by_evaluation_method[method]["count"] += 1
                    by_evaluation_method[method]["core_seconds"] += metric_timing["elapsed_seconds"]
                    case_timing_records.append(case_timing_record(
                        input_case,
                        stage="evaluate",
                        operation="evaluate_metric",
                        elapsed_seconds=metric_timing["elapsed_seconds"],
                        started_at=metric_timing["started_at"],
                        finished_at=metric_timing["finished_at"],
                        status=evaluated.get("status", "failed"),
                        algorithm=algorithm_name,
                        mode=mode,
                        evaluation_method=metric_timing["evaluation_method"],
                        error=evaluated.get("error", ""),
                    ))
                completed += 1
                by_algorithm_mode[timing_key]["count"] += 1
        if case_records:
            write_start = time.perf_counter()
            written = write_case_result_file(case_directory, case_records, overwrite=True)
            write_elapsed = time.perf_counter() - write_start
            write_seconds += write_elapsed
            for record in case_records:
                timing_key = (record["algorithm"], record["mode"])
                by_algorithm_mode[timing_key]["write_json_seconds"] += (
                    write_elapsed / len(case_records)
                )
            write_case_timing_records(case_directory, case_timing_records)
            print(f"wrote evaluated results {written}")
    if timing is not None:
        total_seconds = time.perf_counter() - stage_start
        timing.add(
            "evaluate",
            "evaluate_result",
            count=completed,
            input_files=len(specs),
            instances=completed,
            skipped=skipped,
            missing=missing,
            failed=failed,
            read_json_seconds=read_seconds,
            core_seconds=core_seconds,
            write_json_seconds=write_seconds,
            total_seconds=total_seconds,
        )
        for evaluation_method, values in sorted(by_evaluation_method.items()):
            timing.add(
                "evaluate",
                "evaluate_result_by_metric_method",
                scope="evaluation_method",
                count=values["count"],
                instances=values["count"],
                core_seconds=values["core_seconds"],
                total_seconds=values["core_seconds"],
                evaluation_method=evaluation_method,
            )


def run_check_stage(args, algorithms, modes):
    summary = check_corpus(args.output_root, algorithms=algorithms, modes=modes)
    print("Check summary:", json.dumps({key: value for key, value in summary.items() if key != "errors"}, indent=2))
    if summary["errors"]:
        for error in summary["errors"]:
            print(f"CHECK ERROR: {error}")
        raise AssertionError(f"main monotonicity corpus check failed with {len(summary['errors'])} error(s)")


def run_biopsy_summary_stage(specs, args, timing=None):
    stage_start = time.perf_counter()
    core_start = time.perf_counter()
    written = write_biopsy_cell_summary(
        args.output_root,
        specs=specs,
        label=biopsy_summary_label(specs),
    )
    core_seconds = time.perf_counter() - core_start
    if timing is not None:
        timing.add(
            "biopsy_summary",
            "write_biopsy_cell_summary",
            count=1,
            input_files=len(input_files_for_specs(args.output_root, specs)),
            instances=1,
            core_seconds=core_seconds,
            total_seconds=time.perf_counter() - stage_start,
        )
    print("Wrote biopsy-cell summary:", written)


def run_timing_report_stage(specs, args):
    written = write_timing_report(args.output_root, specs=specs)
    print("Wrote timing reports:", written)
    return written


def parse_args():
    parser = argparse.ArgumentParser(description="Build and report the main sampling-monotonicity benchmark corpus.")
    parser.add_argument(
        "--stage",
        action="append",
        choices=[
            "simulate",
            "distance",
            "reconstruct",
            "evaluate",
            "biopsy-summary",
            "timing-report",
            "report",
            "check",
            "all",
        ],
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
    parser.add_argument("--input-layout", choices=INPUT_LAYOUTS, default="legacy")
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

    command_started_at = utc_timestamp()
    command_start = time.perf_counter()
    with open(args.config, "r") as f:
        base_config = json.load(f)

    timing = TimingRecorder()
    exit_code = 0
    for stage in stages:
        try:
            if stage == "simulate":
                run_simulate_stage(specs, args, base_config, timing=timing)
            elif stage == "distance":
                run_distance_stage(specs, args, timing=timing)
            elif stage == "reconstruct":
                run_reconstruct_stage(specs, args, algorithms, modes, timing=timing)
            elif stage == "evaluate":
                run_evaluate_stage(specs, args, algorithms, modes, timing=timing)
            elif stage == "biopsy-summary":
                run_biopsy_summary_stage(specs, args, timing=timing)
            elif stage == "timing-report":
                pass
            elif stage == "report":
                stage_start = time.perf_counter()
                core_start = time.perf_counter()
                written = write_reports(args.output_root)
                core_seconds = time.perf_counter() - core_start
                timing.add(
                    "report",
                    "write_reports",
                    count=1,
                    instances=1,
                    core_seconds=core_seconds,
                    total_seconds=time.perf_counter() - stage_start,
                )
                print("Wrote reports:", written)
            elif stage == "check":
                run_check_stage(args, algorithms, modes)
        except Exception:
            traceback.print_exc()
            if args.fail_fast:
                raise
            exit_code = 1
            break
    command_finished_at = utc_timestamp()
    command_timing_written = write_single_stage_command_timing(
        args.output_root,
        specs,
        stages,
        elapsed_seconds=time.perf_counter() - command_start,
        started_at=command_started_at,
        finished_at=command_finished_at,
        status="ok" if exit_code == 0 else "failed",
        command=" ".join(sys.argv),
    )
    if command_timing_written:
        print("Wrote command timing:", command_timing_written)
    if exit_code == 0:
        timing_written = write_timing_report(args.output_root)
        if timing_written:
            print("Wrote timing reports:", timing_written)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
