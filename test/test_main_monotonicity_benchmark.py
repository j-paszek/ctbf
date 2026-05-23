from pathlib import Path
import importlib.util
import sys

import networkx as nx
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOL_PATH = PROJECT_ROOT / "test" / "tools" / "main_monotonicity_benchmark.py"
spec = importlib.util.spec_from_file_location("main_monotonicity_benchmark", TOOL_PATH)
main_benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_benchmark)

from ctbf_constraints import MIN_BIOPSY_CELLS_FROM_BIOPSY  # noqa: E402
from reconstructor_algorithms import neighbor_joining_baseline  # noqa: E402
from simulator import Genotype  # noqa: E402


def _spec(**overrides):
    values = {
        "genome_length": 10,
        "generation_count": 20,
        "seed": 295,
        "biopsy_size_scalable": 0.5,
        "biopsy_level_count": 3,
        "general_event_prob": 0.05,
        "event_shape_label": "highdm",
        "single_or_multiple_event_prob": 0.05,
        "duplication_multiplicity": 3,
        "r_dist": 4.0,
    }
    values.update(overrides)
    return main_benchmark.MainCaseSpec(**values)


def test_case_id_encodes_main_parameter_tuple():
    assert (
        main_benchmark.case_id(_spec())
        == "gl10_g20_seed295_bss05_L3_x0p05_y0p05m3"
    )


def test_case_dir_groups_by_genome_length_generation_and_seed(tmp_path):
    spec = _spec()
    cid = main_benchmark.case_id(spec)

    assert main_benchmark.case_dir(tmp_path, spec) == tmp_path / "gl10" / "g20" / "seed295" / cid
    assert main_benchmark.case_dir(tmp_path, cid) == tmp_path / "gl10" / "g20" / "seed295" / cid
    assert main_benchmark.input_path(tmp_path, spec) == tmp_path / "gl10" / "g20" / "seed295" / cid / "input.json"


def test_iter_case_specs_uses_cartesian_grid_and_event_shape_values():
    specs = list(main_benchmark.iter_case_specs(
        genome_lengths=[10],
        generation_counts=[10],
        seeds=[1, 2],
        biopsy_size_scalables=[0.25, 0.5],
        biopsy_level_counts=[1],
        event_probs=[0.01],
        event_shapes=["low", "high"],
    ))

    assert len(specs) == 8
    assert {(spec.event_shape_label, spec.single_or_multiple_event_prob, spec.duplication_multiplicity)
            for spec in specs} == {("low", 0.01, 1), ("high", 0.05, 1)}


def test_build_config_snapshot_applies_only_case_overrides():
    base = {
        "genome_length": 99,
        "initial_copies": 2,
        "NUMBER_OF_GENERATIONS": 99,
        "GENERAL_EVENT_PROB": 0.99,
        "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": 0.99,
        "GENERAL_DUPLICATION_MULTIPLICITY": 99,
        "unchanged": "value",
    }

    config = main_benchmark.build_config_snapshot(base, _spec())

    assert config["genome_length"] == 10
    assert config["NUMBER_OF_GENERATIONS"] == 20
    assert config["GENERAL_EVENT_PROB"] == 0.05
    assert config["GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB"] == 0.05
    assert config["GENERAL_DUPLICATION_MULTIPLICITY"] == 3
    assert config["unchanged"] == "value"
    assert base["genome_length"] == 99


class FakeSimulator:
    def __init__(self, population_size):
        self.population_size = population_size
        self.tree = nx.DiGraph()
        self.tree.add_node(0, genome=[2], generation=0, cell_id=0)

    def perform_biopsy(self, generation, biopsy_size=0, biopsy_size_scalable=None, seed=None):
        count = 0 if self.population_size == 0 else max(
            MIN_BIOPSY_CELLS_FROM_BIOPSY,
            int(self.population_size * biopsy_size_scalable),
        )
        return [
            Genotype([2], node_id=generation * 100 + index, generation=generation, cell_id=generation * 100 + index)
            for index in range(count)
        ]


class GenerationSizedFakeSimulator:
    def __init__(self, population_sizes):
        self.population_sizes = population_sizes

    def perform_biopsy(self, generation, biopsy_size=0, biopsy_size_scalable=None, seed=None):
        population_size = self.population_sizes.get(generation, 0)
        count = 0 if population_size == 0 else max(
            MIN_BIOPSY_CELLS_FROM_BIOPSY,
            int(population_size * biopsy_size_scalable),
        )
        return [
            Genotype([2], node_id=generation * 100 + index, generation=generation, cell_id=generation * 100 + index)
            for index in range(count)
        ]


def test_choose_biopsy_generations_is_deterministic_and_records_counts():
    spec = _spec(generation_count=10, biopsy_level_count=2, biopsy_size_scalable=0.25)
    first = main_benchmark.choose_biopsy_generations_with_retry(FakeSimulator(12), spec)
    second = main_benchmark.choose_biopsy_generations_with_retry(FakeSimulator(12), spec)

    assert first["status"] == "ok"
    assert first["selected_generations"] == second["selected_generations"]
    assert first["total_sampled_biopsy_cells"] == 6
    assert first["nonempty_biopsy_levels"] == 2
    assert first["selection_strategy"] == "random_retry"
    assert first["fallback_used"] is False


def test_choose_biopsy_generations_accepts_two_total_sampled_cells():
    spec = _spec(generation_count=10, biopsy_level_count=1, biopsy_size_scalable=0.5)

    selection = main_benchmark.choose_biopsy_generations_with_retry(
        FakeSimulator(4),
        spec,
        max_retries=0,
    )

    assert selection["status"] == "ok"
    assert selection["failure_reason"] is None
    assert selection["total_sampled_biopsy_cells"] == 2


def test_input_case_records_two_cell_minimum(monkeypatch):
    spec = _spec(generation_count=10, biopsy_level_count=1, biopsy_size_scalable=0.5)
    monkeypatch.setattr(
        main_benchmark,
        "build_simulator_from_config",
        lambda config, seed: FakeSimulator(4),
    )

    input_case = main_benchmark.input_case_from_simulation(spec, {}, max_retries=0)

    assert input_case["status"] == "ok"
    assert input_case["biopsy_selection"]["total_sampled_biopsy_cells"] == 2
    assert input_case["biopsy_selection"]["min_total_sampled_biopsy_cells"] == 2


def test_choose_biopsy_generations_falls_back_to_latest_generations_after_retries():
    spec = _spec(
        generation_count=10,
        seed=1,
        biopsy_level_count=1,
        biopsy_size_scalable=0.25,
    )

    selection = main_benchmark.choose_biopsy_generations_with_retry(
        GenerationSizedFakeSimulator({9: 12}),
        spec,
        max_retries=0,
    )

    assert selection["status"] == "ok"
    assert selection["selected_generations"] == [9]
    assert selection["total_sampled_biopsy_cells"] == 3
    assert selection["selection_strategy"] == "latest_generations_fallback"
    assert selection["fallback_used"] is True
    assert selection["pre_fallback_selected_generations"] == [6]


def test_choose_biopsy_generations_marks_small_biopsy_after_max_retries():
    spec = _spec(generation_count=10, biopsy_level_count=1, biopsy_size_scalable=0.25)

    selection = main_benchmark.choose_biopsy_generations_with_retry(
        FakeSimulator(4),
        spec,
        max_retries=2,
    )

    assert selection["status"] == "failed"
    assert selection["failure_reason"] == "small_biopsy"
    assert selection["retry_count"] == 2
    assert selection["selected_generations"] == [9]
    assert selection["total_sampled_biopsy_cells"] == 1
    assert selection["selection_strategy"] == "latest_generations_fallback"
    assert selection["fallback_used"] is True


def test_compute_case_distances_deduplicates_repeated_cell_ids_with_l1_matrix():
    tree = nx.DiGraph()
    tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
    tree.add_node(5, genome=[2, 2], generation=1, cell_id=5)
    tree.add_node(7, genome=[3, 2], generation=2, cell_id=7)
    tree.add_edge(0, 5, events="duplication(pos=0, copies=1)")
    tree.add_edge(5, 7, events="duplication(pos=0, copies=1)")
    input_case = {
        "case_id": "case",
        "status": "ok",
        "true_tree": main_benchmark.node_link_data(tree),
        "biopsies": [
            {"generation": 1, "cells": [{"node_id": 5, "cell_id": 5, "generation": 1, "genome": [2, 2]}]},
            {"generation": 2, "cells": [{"node_id": 50, "cell_id": 5, "generation": 2, "genome": [2, 2]}]},
            {"generation": 3, "cells": [{"node_id": 7, "cell_id": 7, "generation": 3, "genome": [3, 2]}]},
        ],
    }

    updated = main_benchmark.compute_case_distances(input_case, distance_mode="l1")

    assert updated["distance_matrices"]["cnp2cnp"]["ids"] == [5, 7]
    assert np.array_equal(
        updated["distance_matrices"]["cnp2cnp"]["matrix"],
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert updated["unique_distance_cell_ids"] == [5, 7]


def test_compute_case_distances_allows_two_raw_cells_with_one_unique_cell_id():
    tree = nx.DiGraph()
    tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
    tree.add_node(5, genome=[2, 2], generation=1, cell_id=5)
    tree.add_node(50, genome=[2, 2], generation=2, cell_id=5)
    tree.add_edge(0, 5, events="")
    tree.add_edge(5, 50, events="")
    input_case = {
        "case_id": "case",
        "status": "ok",
        "true_tree": main_benchmark.node_link_data(tree),
        "biopsies": [
            {"generation": 1, "cells": [{"node_id": 5, "cell_id": 5, "generation": 1, "genome": [2, 2]}]},
            {"generation": 2, "cells": [{"node_id": 50, "cell_id": 5, "generation": 2, "genome": [2, 2]}]},
        ],
    }

    updated = main_benchmark.compute_case_distances(input_case, distance_mode="l1")

    assert updated["distance_matrices"]["cnp2cnp"]["ids"] == [5]
    assert np.array_equal(
        updated["distance_matrices"]["cnp2cnp"]["matrix"],
        np.array([[0.0]]),
    )
    assert updated["unique_distance_cell_ids"] == [5]


def test_reconstruction_result_handles_two_raw_cells_with_one_unique_cell_id():
    tree = nx.DiGraph()
    tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
    tree.add_node(5, genome=[2, 2], generation=1, cell_id=5)
    tree.add_node(50, genome=[2, 2], generation=2, cell_id=5)
    tree.add_edge(0, 5, events="")
    tree.add_edge(5, 50, events="")
    input_case = {
        "case_id": "case",
        "status": "ok",
        "seed": 7,
        "r_dist": 4,
        "true_tree": main_benchmark.node_link_data(tree),
        "biopsies": [
            {"generation": 1, "cells": [{"node_id": 5, "cell_id": 5, "generation": 1, "genome": [2, 2]}]},
            {"generation": 2, "cells": [{"node_id": 50, "cell_id": 5, "generation": 2, "genome": [2, 2]}]},
        ],
    }
    input_case = main_benchmark.compute_case_distances(input_case, distance_mode="l1")

    result = main_benchmark.reconstruction_result(
        input_case,
        neighbor_joining_baseline,
        "biopsy_guided_top",
    )
    reconstructed_tree = main_benchmark.node_link_graph(result["reconstructed_tree"])

    assert result["status"] == "reconstructed"
    assert result["actual_root"] in reconstructed_tree.nodes
    assert reconstructed_tree.in_degree(result["actual_root"]) == 0


def test_reconstruction_result_has_tree_without_evaluating_metrics():
    tree = nx.DiGraph()
    tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
    tree.add_node(5, genome=[2, 2], generation=1, cell_id=5)
    tree.add_node(7, genome=[3, 2], generation=2, cell_id=7)
    tree.add_edge(0, 5, events="")
    tree.add_edge(5, 7, events="duplication(pos=0, copies=1)")
    input_case = {
        "case_id": "case",
        "status": "ok",
        "seed": 7,
        "r_dist": 4,
        "true_tree": main_benchmark.node_link_data(tree),
        "biopsies": [
            {"generation": 1, "cells": [{"node_id": 5, "cell_id": 5, "generation": 1, "genome": [2, 2]}]},
            {"generation": 2, "cells": [{"node_id": 7, "cell_id": 7, "generation": 2, "genome": [3, 2]}]},
        ],
        "distance_matrices": {
            "cnp2cnp": {
                "ids": [5, 7],
                "matrix": [[0.0, 1.0], [1.0, 0.0]],
            }
        },
    }

    result = main_benchmark.reconstruction_result(input_case, neighbor_joining_baseline, "full_cnp")
    evaluated = main_benchmark.evaluate_result(input_case, result)

    assert result["status"] == "reconstructed"
    assert result["algorithm"] == "neighbor_joining_baseline"
    assert "reconstructed_tree" in result
    assert "metrics" not in result
    assert 0.0 <= evaluated["metrics"]["ancestors_unique_restricted"]["F1"] <= 1.0
    assert 0.0 <= evaluated["metrics"]["grf"] <= 1.0


def _write_tiny_checked_corpus(tmp_path, *, corrupt_grf=False):
    tree = nx.DiGraph()
    tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
    tree.add_node(5, genome=[2, 2], generation=1, cell_id=5)
    tree.add_node(7, genome=[3, 2], generation=2, cell_id=7)
    tree.add_edge(0, 5, events="")
    tree.add_edge(5, 7, events="duplication(pos=0, copies=1)")
    input_case = {
        "case_id": "tiny_case",
        "corpus": main_benchmark.CORPUS_NAME,
        "status": "ok",
        "genome_length": 10,
        "NUMBER_OF_GENERATIONS": 10,
        "seed": 7,
        "r_dist": 4,
        "biopsy_size_scalable": 0.5,
        "biopsy_level_count": 2,
        "biopsy_generations": [1, 2],
        "GENERAL_EVENT_PROB": 0.01,
        "event_shape_label": "low",
        "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": 0.01,
        "GENERAL_DUPLICATION_MULTIPLICITY": 1,
        "config_snapshot": {"genome_length": 10},
        "biopsy_selection": {
            "retry_count": 0,
            "total_sampled_biopsy_cells": 2,
            "nonempty_biopsy_levels": 2,
        },
        "true_tree": main_benchmark.node_link_data(tree),
        "biopsies": [
            {"generation": 1, "cells": [{"node_id": 5, "cell_id": 5, "generation": 1, "genome": [2, 2]}]},
            {"generation": 2, "cells": [{"node_id": 7, "cell_id": 7, "generation": 2, "genome": [3, 2]}]},
        ],
    }
    input_case = main_benchmark.compute_case_distances(input_case, distance_mode="l1")
    result = main_benchmark.reconstruction_result(input_case, neighbor_joining_baseline, "full_cnp")
    result = main_benchmark.evaluate_result(input_case, result)
    if corrupt_grf:
        result["metrics"]["grf"] = 0.0 if result["metrics"]["grf"] != 0.0 else 1.0

    case_dir = tmp_path / "tiny_case"
    main_benchmark.write_json(case_dir / "input.json", input_case, overwrite=True)
    main_benchmark.write_json(
        case_dir / "full_cnp" / "neighbor_joining_baseline.json",
        result,
        overwrite=True,
    )
    return tmp_path


def test_check_corpus_replays_metrics_and_writes_reports(tmp_path):
    root = _write_tiny_checked_corpus(tmp_path)

    summary = main_benchmark.check_corpus(
        root,
        algorithms=[neighbor_joining_baseline],
        modes=["full_cnp"],
    )

    assert summary["errors"] == []
    assert summary["checked_inputs"] == 1
    assert summary["checked_results"] == 1
    assert (root / "reports" / "result_rows.csv").exists()


def test_check_corpus_reports_stale_metric_values(tmp_path):
    root = _write_tiny_checked_corpus(tmp_path, corrupt_grf=True)

    summary = main_benchmark.check_corpus(
        root,
        algorithms=[neighbor_joining_baseline],
        modes=["full_cnp"],
        replay_reports=False,
    )

    assert any("stored grf" in error for error in summary["errors"])


def test_monotonic_summary_reports_passes_and_failures():
    rows = pd.DataFrame([
        {
            "genome_length": 10,
            "generation_count": 10,
            "general_event_prob": 0.01,
            "event_shape_label": "low",
            "single_or_multiple_event_prob": 0.01,
            "duplication_multiplicity": 1,
            "biopsy_level_count": 1,
            "algorithm": "alg",
            "mode": "full_cnp",
            "seed": seed,
            "biopsy_size_scalable": b,
            "adf1": adf1,
            "grf": grf,
        }
        for seed, values in {
            1: [(0.25, 0.1, 0.2), (0.5, 0.2, 0.3), (0.75, 0.3, 0.4)],
            2: [(0.25, 0.3, 0.4), (0.5, 0.2, 0.5), (0.75, 0.4, 0.6)],
        }.items()
        for b, adf1, grf in values
    ])
    fixed = [
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

    summary, violations = main_benchmark.monotonic_summary(
        rows,
        dimension="biopsy_size_scalable",
        values=(0.25, 0.5, 0.75),
        fixed_columns=fixed,
    )

    adf1 = summary[summary["metric"] == "adf1"].iloc[0]
    grf = summary[summary["metric"] == "grf"].iloc[0]
    assert adf1["n_complete"] == 2
    assert adf1["monotonic_passes"] == 1
    assert adf1["monotonic_failures"] == 1
    assert grf["monotonic_passes"] == 2
    assert set(violations["metric"]) == {"adf1"}


def test_selected_algorithms_defaults_to_core_rows():
    names = [algorithm.__name__ for algorithm in main_benchmark.selected_algorithms()]

    assert names == list(main_benchmark.DEFAULT_CORE_ALGORITHM_NAMES)


def test_selected_algorithms_respects_explicit_index_without_adding_core_rows():
    names = [algorithm.__name__ for algorithm in main_benchmark.selected_algorithms(algorithm_indexes=[0])]

    assert names == ["neighbor_joining_baseline"]
