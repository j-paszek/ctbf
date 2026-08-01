from pathlib import Path
import importlib.util
import json
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

from ctbf_constraints import MIN_BIOPSY_CELLS_FROM_BIOPSY, MIN_TOTAL_BIOPSY_CELLS  # noqa: E402
from reconstructor_algorithms import neighbor_joining_baseline  # noqa: E402
from simulator import CancerCellEvolutionSimulator, Genotype  # noqa: E402


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
    assert main_benchmark.input_truetree_path(tmp_path, spec) == tmp_path / "gl10" / "g20" / "seed295" / cid / "input_truetree.json"
    assert main_benchmark.input_biopsy_path(tmp_path, spec) == tmp_path / "gl10" / "g20" / "seed295" / cid / "input_biopsy.json"
    assert main_benchmark.genome_dict_path(tmp_path, spec) == tmp_path / "gl10" / "g20" / "seed295" / cid / "genome_dict.csv"
    assert main_benchmark.distance_path(tmp_path, spec) == tmp_path / "gl10" / "g20" / "seed295" / cid / "input_dm_cnp.json"
    assert main_benchmark.case_result_csv_path(tmp_path, spec) == tmp_path / "gl10" / "g20" / "seed295" / cid / "result.csv"


def test_iter_case_specs_uses_cartesian_grid_and_event_shape_values():
    specs = list(main_benchmark.iter_case_specs(
        genome_lengths=[10],
        generation_counts=[10],
        seeds=[1, 2],
        biopsy_size_scalables=[0.25, 0.5],
        biopsy_level_counts=[2],
        event_probs=[0.01],
        event_shapes=["low", "high"],
    ))

    assert len(specs) == 8
    assert {(spec.event_shape_label, spec.single_or_multiple_event_prob, spec.duplication_multiplicity)
            for spec in specs} == {("low", 0.01, 1), ("high", 0.05, 1)}
    assert {spec.biopsy_level_count for spec in specs} == {2}


def test_default_case_grid_scope_is_curated_l2_to_l4_g10_only():
    specs = list(main_benchmark.iter_case_specs(
        seeds=[1],
        event_shapes=["low"],
        event_probs=[0.01],
    ))

    assert {spec.generation_count for spec in specs} == {10}
    assert {spec.biopsy_level_count for spec in specs} == {2, 3, 4}
    assert all("_L1_" not in main_benchmark.case_id(spec) for spec in specs)


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

    def perform_biopsy(
        self,
        generation,
        biopsy_size=0,
        biopsy_size_scalable=None,
        seed=None,
    ):
        count = 0 if self.population_size == 0 else max(
            MIN_BIOPSY_CELLS_FROM_BIOPSY,
            int(self.population_size * biopsy_size_scalable),
        )
        cells = [
            Genotype([2], node_id=generation * 100 + index, generation=generation, cell_id=generation * 100 + index)
            for index in range(count)
        ]
        for cell in cells:
            if cell.node_id not in self.tree:
                self.tree.add_node(
                    cell.node_id,
                    genome=[2],
                    generation=generation,
                    cell_id=cell.cell_id,
                )
                self.tree.add_edge(0, cell.node_id)
        return cells


class GenerationSizedFakeSimulator:
    def __init__(self, population_sizes):
        self.population_sizes = population_sizes

    def perform_biopsy(
        self,
        generation,
        biopsy_size=0,
        biopsy_size_scalable=None,
        seed=None,
    ):
        population_size = self.population_sizes.get(generation, 0)
        count = 0 if population_size == 0 else max(
            MIN_BIOPSY_CELLS_FROM_BIOPSY,
            int(population_size * biopsy_size_scalable),
        )
        return [
            Genotype([2], node_id=generation * 100 + index, generation=generation, cell_id=generation * 100 + index)
            for index in range(count)
        ]


def test_perform_biopsies_preserves_ordered_distinct_generation_levels():
    generations = [2, 5, 8]

    biopsies, cell_lists = main_benchmark.perform_biopsies(
        GenerationSizedFakeSimulator({2: 4, 5: 4, 8: 4}),
        generations,
        biopsy_size_scalable=0.5,
        seed=11,
    )

    assert [biopsy["level"] for biopsy in biopsies] == ["L1", "L2", "L3"]
    assert [biopsy["generation"] for biopsy in biopsies] == generations
    assert all(
        cell["generation"] == biopsy["generation"]
        for biopsy in biopsies
        for cell in biopsy["cells"]
    )
    assert all(
        "node_id" not in cell
        and "observation_key" not in cell
        and "occurrence_kind" not in cell
        and "source_observation_key" not in cell
        for biopsy in biopsies
        for cell in biopsy["cells"]
    )
    assert [[cell.generation for cell in level] for level in cell_lists] == [
        [2, 2],
        [5, 5],
        [8, 8],
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
    assert first["selected_generations"] == sorted(set(first["selected_generations"]))
    assert [biopsy["generation"] for biopsy in first["biopsies"]] == first["selected_generations"]


def test_choose_biopsy_generations_accepts_two_total_sampled_cells():
    spec = _spec(generation_count=10, biopsy_level_count=2, biopsy_size_scalable=0.5)

    selection = main_benchmark.choose_biopsy_generations_with_retry(
        FakeSimulator(2),
        spec,
        max_retries=1,
    )

    assert selection["status"] == "ok"
    assert selection["failure_reason"] is None
    assert selection["total_sampled_biopsy_cells"] == 2
    assert selection["random_attempt_count"] == 1
    assert selection["max_random_attempts"] == 1


def test_input_case_records_two_cell_minimum(monkeypatch):
    spec = _spec(generation_count=10, biopsy_level_count=2, biopsy_size_scalable=0.5)
    monkeypatch.setattr(
        main_benchmark,
        "build_simulator_from_config",
        lambda config, seed: FakeSimulator(2),
    )

    input_case = main_benchmark.input_case_from_simulation(spec, {}, max_retries=1)

    assert input_case["status"] == "ok"
    assert input_case["biopsy_selection"]["total_sampled_biopsy_cells"] == 2
    assert input_case["biopsy_selection"]["min_total_sampled_biopsy_cells"] == MIN_TOTAL_BIOPSY_CELLS
    assert input_case["biopsy_selection"]["random_attempt_count"] == 1
    assert main_benchmark._check_biopsy_order(input_case, "input_case") == []
    assert [biopsy["generation"] for biopsy in input_case["biopsies"]] == input_case["biopsy_generations"]


def test_choose_biopsy_generations_falls_back_to_latest_generations_after_retries():
    spec = _spec(
        generation_count=10,
        seed=1,
        biopsy_level_count=2,
        biopsy_size_scalable=0.25,
    )

    selection = main_benchmark.choose_biopsy_generations_with_retry(
        GenerationSizedFakeSimulator({8: 4, 9: 4}),
        spec,
        max_retries=1,
    )

    assert selection["status"] == "ok"
    assert selection["selected_generations"] == [8, 9]
    assert selection["total_sampled_biopsy_cells"] == 2
    assert selection["selection_strategy"] == "latest_generations_fallback"
    assert selection["fallback_used"] is True
    assert selection["random_attempt_count"] == 1
    assert selection["retry_count"] == 1
    assert selection["pre_fallback_selected_generations"]
    assert [biopsy["generation"] for biopsy in selection["biopsies"]] == [8, 9]


def test_choose_biopsy_generations_marks_small_biopsy_after_max_retries():
    spec = _spec(generation_count=10, biopsy_level_count=2, biopsy_size_scalable=0.25)

    selection = main_benchmark.choose_biopsy_generations_with_retry(
        GenerationSizedFakeSimulator({9: 4}),
        spec,
        max_retries=2,
    )

    assert selection["status"] == "failed"
    assert selection["failure_reason"] == "small_biopsy"
    assert selection["retry_count"] == 2
    assert selection["random_attempt_count"] == 2
    assert selection["selected_generations"] == [8, 9]
    assert selection["total_sampled_biopsy_cells"] == 1
    assert selection["selection_strategy"] == "latest_generations_fallback"
    assert selection["fallback_used"] is True


def test_check_biopsy_order_rejects_duplicate_and_mismatched_levels():
    input_case = {
        "biopsy_level_count": 2,
        "biopsy_generations": [2, 2],
        "biopsies": [
            {"level": "L1", "generation": 2, "cells": []},
            {"level": "L2", "generation": 2, "cells": []},
        ],
    }

    errors = main_benchmark._check_biopsy_order(input_case, "case/input.json")

    assert any("strictly increasing" in error for error in errors)

    input_case = {
        "biopsy_level_count": 3,
        "biopsy_generations": [2, 5],
        "biopsies": [
            {"level": "L2", "generation": 5, "cells": [{"node_id": 10, "generation": 2}]},
            {"level": "L1", "generation": 2, "cells": []},
        ],
    }

    errors = main_benchmark._check_biopsy_order(input_case, "case/input.json")

    assert any("expected 'L1'" in error for error in errors)
    assert any("does not match selected generation 2" in error for error in errors)
    assert any("cell 10" in error for error in errors)
    assert any("biopsy_level_count 3" in error for error in errors)


def test_choose_biopsy_generations_uses_exactly_100_random_attempts_before_fallback(monkeypatch):
    spec = _spec(generation_count=10, biopsy_level_count=3, biopsy_size_scalable=0.25)
    seen_generations = []

    def always_empty_biopsies(_simulator, generations, _biopsy_size_scalable, _seed):
        seen_generations.append(list(generations))
        return [
            {"level": f"L{index}", "generation": generation, "cells": []}
            for index, generation in enumerate(generations, start=1)
        ], []

    monkeypatch.setattr(main_benchmark, "perform_biopsies", always_empty_biopsies)

    selection = main_benchmark.choose_biopsy_generations_with_retry(
        object(),
        spec,
        max_retries=100,
    )

    assert selection["status"] == "failed"
    assert len(seen_generations) == 101
    assert seen_generations[-1] == [7, 8, 9]
    assert selection["random_attempt_count"] == 100
    assert selection["max_random_attempts"] == 100
    assert selection["retry_count"] == 100


def test_simulator_canonicalizes_recurrent_genome_cell_ids_for_biopsy_and_tree():
    tree = nx.DiGraph()
    tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
    tree.add_node(2, genome=[2, 3], generation=1, cell_id=2)
    tree.add_node(11, genome=[2, 2], generation=2, cell_id=11)
    tree.add_edge(0, 2, events="duplication(pos=1, copies=1)")
    tree.add_edge(2, 11, events="loss(pos=1, copies=-1)")
    simulator = CancerCellEvolutionSimulator.from_tree(tree)

    biopsy = simulator.perform_biopsy(2, biopsy_size_scalable=1.0, seed=1)
    canonical_tree = simulator.canonicalized_tree_by_genome()

    assert [cell.node_id for cell in biopsy] == [11]
    assert [cell.cell_id for cell in biopsy] == [0]
    assert simulator.genotypes[11].cell_id == 11
    assert tree.nodes[11]["cell_id"] == 11
    assert canonical_tree.nodes[11]["cell_id"] == 0


def test_simulator_collapses_duplicate_canonical_genotypes_within_one_biopsy():
    tree = nx.DiGraph()
    tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
    tree.add_node(11, genome=[2, 2], generation=2, cell_id=11)
    tree.add_node(12, genome=[2, 2], generation=2, cell_id=12)
    tree.add_edge(0, 11, events="duplication(pos=1, copies=1);loss(pos=1, copies=-1)")
    tree.add_edge(0, 12, events="duplication(pos=1, copies=1);loss(pos=1, copies=-1)")
    simulator = CancerCellEvolutionSimulator.from_tree(tree)

    biopsy = simulator.perform_biopsy(2, biopsy_size_scalable=1.0, seed=1)

    assert len(biopsy) == 1
    assert biopsy[0].cell_id == 0
    assert biopsy[0].node_id in {11, 12}


def _unique_genome_input_case(case_id="case"):
    tree = nx.DiGraph()
    tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
    tree.add_node(5, genome=[3, 2], generation=1, cell_id=5)
    tree.add_node(7, genome=[3, 3], generation=2, cell_id=7)
    tree.add_edge(0, 5, events="duplication(pos=0, copies=1)")
    tree.add_edge(5, 7, events="duplication(pos=1, copies=1)")
    return {
        "case_id": case_id,
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
            "random_attempt_count": 1,
            "max_random_attempts": 100,
            "total_sampled_biopsy_cells": 2,
            "nonempty_biopsy_levels": 2,
            "min_total_sampled_biopsy_cells": MIN_TOTAL_BIOPSY_CELLS,
        },
        "true_tree": main_benchmark.node_link_data(tree),
        "biopsies": [
            {"level": "L1", "generation": 1, "cells": [{"node_id": 5, "cell_id": 5, "generation": 1, "genome": [3, 2]}]},
            {"level": "L2", "generation": 2, "cells": [{"node_id": 7, "cell_id": 7, "generation": 2, "genome": [3, 3]}]},
        ],
    }


def test_split_input_layout_removes_inline_genomes_and_hydrates_legacy_shape(tmp_path):
    input_case = _unique_genome_input_case()
    case_directory = tmp_path / "case"

    written = main_benchmark.write_input_case(
        case_directory,
        input_case,
        layout="split",
        overwrite=True,
    )

    assert [path.name for path in written] == [
        "input_truetree.json",
        "input_biopsy.json",
        "genome_dict.csv",
    ]
    assert not (case_directory / "input.json").exists()
    true_tree_payload = json.loads((case_directory / "input_truetree.json").read_text())
    biopsy_payload = json.loads((case_directory / "input_biopsy.json").read_text())
    assert all("genome" not in node for node in true_tree_payload["true_tree"]["nodes"])
    assert all(
        "genome" not in cell
        for biopsy in biopsy_payload["biopsies"]
        for cell in biopsy["cells"]
    )
    genome_dict = pd.read_csv(case_directory / "genome_dict.csv")
    assert set(genome_dict["cell_id"]) == {0, 5, 7}

    hydrated = main_benchmark.load_input_case(case_directory)

    assert hydrated["input_layout"] == main_benchmark.SPLIT_INPUT_LAYOUT
    assert hydrated["biopsies"] == input_case["biopsies"]
    assert main_benchmark.node_link_graph(hydrated["true_tree"]).nodes[5]["genome"] == [3, 2]


def test_split_input_layout_reports_hard_cell_id_multiple_genome_invariant_errors():
    input_case = _unique_genome_input_case()
    input_case["biopsies"][0]["cells"][0]["genome"] = [9, 9]

    with pytest.raises(ValueError, match="cell_id 5.*multiple genomes"):
        main_benchmark.build_genome_dictionary(input_case)


def test_split_input_layout_canonicalizes_duplicate_genome_cell_ids(tmp_path):
    input_case = _unique_genome_input_case()
    tree = main_benchmark.node_link_graph(input_case["true_tree"])
    tree.add_node(9, genome=[3, 2], generation=2, cell_id=9)
    tree.add_edge(5, 9, events="loss(pos=1, copies=-1);duplication(pos=1, copies=1)")
    input_case["true_tree"] = main_benchmark.node_link_data(tree)
    input_case["biopsies"][1]["cells"].append(
        {"node_id": 9, "cell_id": 9, "generation": 2, "genome": [3, 2]}
    )

    written = main_benchmark.write_input_case(
        tmp_path / "case",
        input_case,
        layout="split",
        overwrite=True,
    )

    assert [path.name for path in written] == [
        "input_truetree.json",
        "input_biopsy.json",
        "genome_dict.csv",
    ]
    genome_dict = pd.read_csv(tmp_path / "case" / "genome_dict.csv")
    assert set(genome_dict["cell_id"]) == {0, 5, 7}

    true_tree_payload = json.loads((tmp_path / "case" / "input_truetree.json").read_text())
    node9 = next(node for node in true_tree_payload["true_tree"]["nodes"] if node["id"] == 9)
    biopsy_payload = json.loads((tmp_path / "case" / "input_biopsy.json").read_text())
    biopsy9 = next(
        cell
        for biopsy in biopsy_payload["biopsies"]
        for cell in biopsy["cells"]
        if cell["node_id"] == 9
    )
    hydrated = main_benchmark.load_input_case(tmp_path / "case")

    assert node9["cell_id"] == 5
    assert biopsy9["cell_id"] == 5
    assert main_benchmark.node_link_graph(hydrated["true_tree"]).nodes[9]["cell_id"] == 5
    assert main_benchmark.node_link_graph(hydrated["true_tree"]).nodes[9]["genome"] == [3, 2]


def test_build_genome_dictionary_uses_min_cell_id_for_duplicate_genomes():
    input_case = _unique_genome_input_case()
    for node in input_case["true_tree"]["nodes"]:
        if node.get("cell_id") == 7:
            node["genome"] = [3, 2]
    input_case["biopsies"][1]["cells"][0]["genome"] = [3, 2]

    assert main_benchmark.build_genome_dictionary(input_case) == {
        0: [2, 2],
        5: [3, 2],
    }


def test_biopsy_cell_summary_reads_split_inputs_without_legacy_input(tmp_path):
    spec = _spec(generation_count=10, seed=7, biopsy_size_scalable=0.5, biopsy_level_count=2)
    input_case = _unique_genome_input_case(main_benchmark.case_id(spec))
    main_benchmark.write_input_case(
        main_benchmark.case_dir(tmp_path, spec),
        input_case,
        layout="split",
        overwrite=True,
    )

    written = main_benchmark.write_biopsy_cell_summary(
        tmp_path,
        specs=[spec],
        label=main_benchmark.biopsy_summary_label([spec]),
    )

    summary = pd.read_csv(written["csv"])
    assert len(summary) == 1
    assert summary.iloc[0]["generation"] == "g10"
    assert summary.iloc[0]["level"] == "L2"
    assert summary.iloc[0]["total"] == 2


def test_split_input_layout_runs_distance_reconstruct_and_evaluate_stages(tmp_path):
    spec = _spec(generation_count=10, seed=7, biopsy_size_scalable=0.5, biopsy_level_count=2)
    input_case = _unique_genome_input_case(main_benchmark.case_id(spec))
    main_benchmark.write_input_case(
        main_benchmark.case_dir(tmp_path, spec),
        input_case,
        layout="split",
        overwrite=True,
    )

    class Args:
        output_root = tmp_path
        input_layout = "split"
        distance_mode = "l1"
        overwrite = True
        fail_fast = True

    main_benchmark.run_distance_stage([spec], Args())
    main_benchmark.run_reconstruct_stage(
        [spec],
        Args(),
        [neighbor_joining_baseline],
        ["full_cnp"],
    )
    main_benchmark.run_evaluate_stage(
        [spec],
        Args(),
        [neighbor_joining_baseline],
        ["full_cnp"],
    )

    result_file = main_benchmark.result_path(
        tmp_path,
        spec,
        "full_cnp",
        "neighbor_joining_baseline",
    )
    result = json.loads(result_file.read_text())
    assert result["status"] == "reconstructed"
    assert any(
        "genome" not in node
        for node in result["reconstructed_tree"]["nodes"]
        if "cell_id" in node
    )
    rows = pd.read_csv(main_benchmark.case_result_csv_path(tmp_path, spec))
    assert rows.iloc[0]["status"] == "evaluated"
    timing_rows = pd.read_csv(main_benchmark.case_timing_path(tmp_path, spec))
    assert {"distance_l1", "reconstruct", "evaluate_metric"} <= set(timing_rows["operation"])
    assert any(
        str(operation).startswith("evaluate_phase:")
        for operation in timing_rows["operation"]
    )
    assert {"adf1", "grf"} == set(timing_rows[timing_rows["stage"] == "evaluate"]["evaluation_method"])
    reconstruct_rows = timing_rows[timing_rows["stage"] == "reconstruct"]
    assert set(reconstruct_rows["algorithm"]) == {"neighbor_joining_baseline"}
    assert set(reconstruct_rows["mode"]) == {"full_cnp"}

    summary = main_benchmark.check_corpus(
        tmp_path,
        algorithms=[neighbor_joining_baseline],
        modes=["full_cnp"],
        replay_reports=False,
    )
    assert summary["errors"] == []
    assert summary["checked_inputs"] == 1
    assert summary["checked_results"] == 1


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

    distance_payload = main_benchmark.compute_case_distances(input_case, distance_mode="l1")

    assert "distance_matrices" not in input_case
    assert distance_payload["distance_matrices"]["cnp2cnp"]["ids"] == [5, 7]
    assert np.array_equal(
        distance_payload["distance_matrices"]["cnp2cnp"]["matrix"],
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert distance_payload["unique_distance_cell_ids"] == [5, 7]
    assert distance_payload["distance_matrices"]["cnp2cnp"]["provenance"] == {
        "schema_version": "ctbf-distance-provenance-v1",
        "metric": "l1",
        "semantics_version": "ctbf-l1-profile-v1",
        "formula": "sum(abs(u_i-v_i))",
    }


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

    distance_payload = main_benchmark.compute_case_distances(input_case, distance_mode="l1")

    assert distance_payload["distance_matrices"]["cnp2cnp"]["ids"] == [5]
    assert np.array_equal(
        distance_payload["distance_matrices"]["cnp2cnp"]["matrix"],
        np.array([[0.0]]),
    )
    assert distance_payload["unique_distance_cell_ids"] == [5]


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
    distance_payload = main_benchmark.compute_case_distances(input_case, distance_mode="l1")

    result = main_benchmark.reconstruction_result(
        input_case,
        distance_payload,
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
    }
    distance_payload = {
        "case_id": "case",
        "corpus": main_benchmark.CORPUS_NAME,
        "status": "ok",
        "distance_mode": "l1",
        "unique_distance_cell_ids": [5, 7],
        "distance_matrices": {
            "cnp2cnp": {
                "ids": [5, 7],
                "matrix": [[0.0, 1.0], [1.0, 0.0]],
            }
        },
    }

    result = main_benchmark.reconstruction_result(input_case, distance_payload, neighbor_joining_baseline, "full_cnp")
    evaluated = main_benchmark.evaluate_result(input_case, result)

    assert result["status"] == "reconstructed"
    assert result["algorithm"] == "neighbor_joining_baseline"
    assert "reconstructed_tree" in result
    assert "metrics" not in result
    assert 0.0 <= evaluated["metrics"]["ancestors_unique_restricted"]["F1"] <= 1.0
    assert 0.0 <= evaluated["metrics"]["grf"] <= 1.0


def test_split_evaluation_loader_does_not_read_genome_dict(tmp_path, monkeypatch):
    tree = nx.DiGraph()
    tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
    tree.add_node(5, genome=[3, 2], generation=1, cell_id=5)
    tree.add_edge(0, 5, events="")
    input_case = {
        "case_id": "split_case",
        "corpus": main_benchmark.CORPUS_NAME,
        "status": "ok",
        "genome_length": 10,
        "NUMBER_OF_GENERATIONS": 10,
        "seed": 7,
        "r_dist": 4,
        "biopsy_size_scalable": 0.5,
        "biopsy_level_count": 2,
        "biopsy_generations": [1],
        "GENERAL_EVENT_PROB": 0.01,
        "event_shape_label": "low",
        "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": 0.01,
        "GENERAL_DUPLICATION_MULTIPLICITY": 1,
        "config_snapshot": {"genome_length": 10},
        "biopsy_selection": {},
        "true_tree": main_benchmark.node_link_data(tree),
        "biopsies": [
            {"level": "L1", "generation": 1, "cells": [{"node_id": 5, "cell_id": 5, "generation": 1, "genome": [3, 2]}]},
        ],
    }
    case_dir = tmp_path / "case"
    main_benchmark.write_input_case(case_dir, input_case, layout="split", overwrite=True)

    def fail_read_genome_dict(_path):
        raise AssertionError("evaluation loader should not read genome_dict.csv")

    monkeypatch.setattr(main_benchmark, "read_genome_dict", fail_read_genome_dict)

    loaded = main_benchmark.load_evaluation_input_case(case_dir, preferred_layout="split")

    assert "biopsies" not in loaded
    assert loaded["true_tree"]["nodes"]
    assert all("genome" not in node for node in loaded["true_tree"]["nodes"])


def test_evaluate_result_metrics_does_not_copy_full_result_payload():
    tree = nx.DiGraph()
    tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
    tree.add_node(5, genome=[3, 2], generation=1, cell_id=5)
    tree.add_edge(0, 5, events="")
    input_case = {
        "case_id": "case",
        "corpus": main_benchmark.CORPUS_NAME,
        "status": "ok",
        "genome_length": 10,
        "NUMBER_OF_GENERATIONS": 10,
        "seed": 7,
        "r_dist": 4,
        "biopsy_size_scalable": 0.5,
        "biopsy_level_count": 2,
        "biopsy_generations": [1],
        "GENERAL_EVENT_PROB": 0.01,
        "event_shape_label": "low",
        "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": 0.01,
        "GENERAL_DUPLICATION_MULTIPLICITY": 1,
        "true_tree": main_benchmark.node_link_data(tree),
    }
    result = {
        "status": "reconstructed",
        "algorithm": "alg",
        "mode": "full_cnp",
        "reconstructed_tree": main_benchmark.node_link_data(tree),
    }

    metric_only = main_benchmark.evaluate_result_metrics(input_case, result)
    evaluated = main_benchmark.evaluate_result(input_case, result)

    assert "metrics" not in result
    assert metric_only["status"] == "evaluated"
    assert metric_only["metrics"] == evaluated["metrics"]


def _write_tiny_checked_corpus(tmp_path, *, corrupt_grf=False, biopsy_level_count=2):
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
        "biopsy_level_count": biopsy_level_count,
        "biopsy_generations": [1, 2],
        "GENERAL_EVENT_PROB": 0.01,
        "event_shape_label": "low",
        "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": 0.01,
        "GENERAL_DUPLICATION_MULTIPLICITY": 1,
        "config_snapshot": {"genome_length": 10},
        "biopsy_selection": {
            "retry_count": 0,
            "random_attempt_count": 1,
            "max_random_attempts": 100,
            "total_sampled_biopsy_cells": 2,
            "nonempty_biopsy_levels": 2,
            "min_total_sampled_biopsy_cells": MIN_TOTAL_BIOPSY_CELLS,
        },
        "true_tree": main_benchmark.node_link_data(tree),
        "biopsies": [
            {"level": "L1", "generation": 1, "cells": [{"node_id": 5, "cell_id": 5, "generation": 1, "genome": [2, 2]}]},
            {"level": "L2", "generation": 2, "cells": [{"node_id": 7, "cell_id": 7, "generation": 2, "genome": [3, 2]}]},
        ],
    }
    distance_payload = main_benchmark.compute_case_distances(input_case, distance_mode="l1")
    result = main_benchmark.reconstruction_result(
        input_case,
        distance_payload,
        neighbor_joining_baseline,
        "full_cnp",
    )
    evaluated = main_benchmark.evaluate_result(input_case, result)
    result_file = tmp_path / "tiny_case" / "full_cnp" / "neighbor_joining_baseline.json"
    records = [main_benchmark.case_result_record(
        input_case,
        result_file,
        evaluated,
    )]
    if corrupt_grf:
        records[0]["grf"] = 0.0 if records[0]["grf"] != 0.0 else 1.0

    case_dir = tmp_path / "tiny_case"
    main_benchmark.write_json(case_dir / "input.json", input_case, overwrite=True)
    main_benchmark.write_json(case_dir / "input_dm_cnp.json", distance_payload, overwrite=True)
    main_benchmark.write_json(
        case_dir / "full_cnp" / "neighbor_joining_baseline.json",
        result,
        overwrite=True,
    )
    main_benchmark.write_case_result_file(case_dir, records, overwrite=True)
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


def test_run_evaluate_stage_writes_case_results_without_mutating_reconstruction_json(tmp_path):
    spec = _spec(generation_count=10, seed=7, biopsy_size_scalable=0.5, biopsy_level_count=2)
    tree = nx.DiGraph()
    tree.add_node(0, genome=[2, 2], generation=0, cell_id=0)
    tree.add_node(5, genome=[2, 2], generation=1, cell_id=5)
    tree.add_node(7, genome=[3, 2], generation=2, cell_id=7)
    tree.add_edge(0, 5, events="")
    tree.add_edge(5, 7, events="duplication(pos=0, copies=1)")
    input_case = {
        "case_id": main_benchmark.case_id(spec),
        "corpus": main_benchmark.CORPUS_NAME,
        "status": "ok",
        "genome_length": spec.genome_length,
        "NUMBER_OF_GENERATIONS": spec.generation_count,
        "seed": spec.seed,
        "r_dist": spec.r_dist,
        "biopsy_size_scalable": spec.biopsy_size_scalable,
        "biopsy_level_count": spec.biopsy_level_count,
        "biopsy_generations": [1, 2],
        "GENERAL_EVENT_PROB": spec.general_event_prob,
        "event_shape_label": spec.event_shape_label,
        "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": spec.single_or_multiple_event_prob,
        "GENERAL_DUPLICATION_MULTIPLICITY": spec.duplication_multiplicity,
        "config_snapshot": {"genome_length": spec.genome_length},
        "biopsy_selection": {"total_sampled_biopsy_cells": 2},
        "true_tree": main_benchmark.node_link_data(tree),
        "biopsies": [
            {"level": "L1", "generation": 1, "cells": [{"node_id": 5, "cell_id": 5, "generation": 1, "genome": [2, 2]}]},
            {"level": "L2", "generation": 2, "cells": [{"node_id": 7, "cell_id": 7, "generation": 2, "genome": [3, 2]}]},
        ],
    }
    distance_payload = main_benchmark.compute_case_distances(input_case, distance_mode="l1")
    result = main_benchmark.reconstruction_result(
        input_case,
        distance_payload,
        neighbor_joining_baseline,
        "full_cnp",
    )
    input_file = main_benchmark.input_path(tmp_path, spec)
    result_file = main_benchmark.result_path(
        tmp_path,
        spec,
        "full_cnp",
        "neighbor_joining_baseline",
    )
    main_benchmark.write_json(input_file, input_case, overwrite=True)
    main_benchmark.write_json(result_file, result, overwrite=True)
    stored_reconstruction = json.loads(result_file.read_text())

    class Args:
        output_root = tmp_path
        overwrite = True
        fail_fast = True

    main_benchmark.run_evaluate_stage(
        [spec],
        Args(),
        [neighbor_joining_baseline],
        ["full_cnp"],
    )

    assert json.loads(result_file.read_text()) == stored_reconstruction
    assert "metrics" not in json.loads(result_file.read_text())
    result_rows = pd.read_csv(main_benchmark.case_result_csv_path(tmp_path, spec))
    assert len(result_rows) == 1
    assert result_rows.iloc[0]["status"] == "evaluated"
    assert result_rows.iloc[0]["mode"] == "full_cnp"
    assert not (main_benchmark.case_dir(tmp_path, spec) / "result.json").exists()


def test_collect_result_rows_reads_case_result_csv(tmp_path):
    spec = _spec(generation_count=10, seed=8, biopsy_size_scalable=0.5, biopsy_level_count=2)
    case_directory = main_benchmark.case_dir(tmp_path, spec)
    row = {
        **main_benchmark.case_parameters({
            "case_id": main_benchmark.case_id(spec),
            "genome_length": spec.genome_length,
            "NUMBER_OF_GENERATIONS": spec.generation_count,
            "seed": spec.seed,
            "biopsy_size_scalable": spec.biopsy_size_scalable,
            "biopsy_level_count": spec.biopsy_level_count,
            "GENERAL_EVENT_PROB": spec.general_event_prob,
            "event_shape_label": spec.event_shape_label,
            "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": spec.single_or_multiple_event_prob,
            "GENERAL_DUPLICATION_MULTIPLICITY": spec.duplication_multiplicity,
        }),
        "mode": "full_cnp",
        "algorithm": "neighbor_joining_baseline",
        "status": "evaluated",
        "adf1": 0.5,
        "grf": 0.75,
        "result_file": "full_cnp/neighbor_joining_baseline.json",
        "error": "",
    }
    main_benchmark.write_case_result_file(case_directory, [row], overwrite=True)

    rows = main_benchmark.collect_result_rows(tmp_path)

    assert len(rows) == 1
    assert rows.iloc[0]["algorithm"] == "neighbor_joining_baseline"
    assert rows.iloc[0]["adf1"] == pytest.approx(0.5)
    assert not (case_directory / "result.json").exists()


def test_check_corpus_reports_stale_metric_values(tmp_path):
    root = _write_tiny_checked_corpus(tmp_path, corrupt_grf=True)

    summary = main_benchmark.check_corpus(
        root,
        algorithms=[neighbor_joining_baseline],
        modes=["full_cnp"],
        replay_reports=False,
    )

    assert any("stored grf" in error for error in summary["errors"])


def test_check_corpus_rejects_stale_l1_inputs(tmp_path):
    root = _write_tiny_checked_corpus(tmp_path, biopsy_level_count=1)

    summary = main_benchmark.check_corpus(
        root,
        algorithms=[neighbor_joining_baseline],
        modes=["full_cnp"],
        replay_reports=False,
    )

    assert any("biopsy_level_count 1" in error for error in summary["errors"])


def test_curated_main_inputs_use_ordered_distinct_biopsy_generations():
    corpus_root = PROJECT_ROOT / "test" / "data" / "main" / "gl10" / "g10"
    input_files = sorted(corpus_root.glob("seed*/gl10_g10_seed*/input.json"))
    if not input_files:
        pytest.skip("curated main corpus is not present")

    errors = []
    for input_file in input_files:
        input_case = json.loads(input_file.read_text())
        errors.extend(main_benchmark._check_biopsy_order(input_case, input_file))

    assert errors == []


def test_monotonic_summary_reports_passes_and_failures():
    rows = pd.DataFrame([
        {
            "genome_length": 10,
            "generation_count": 10,
            "general_event_prob": 0.01,
            "event_shape_label": "low",
            "single_or_multiple_event_prob": 0.01,
            "duplication_multiplicity": 1,
            "biopsy_level_count": 2,
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


def test_write_reports_writes_biopsy_cell_summary_from_inputs(tmp_path):
    for seed, cell_count in [(1, 2), (2, 4)]:
        spec = _spec(generation_count=10, seed=seed, biopsy_size_scalable=0.25, biopsy_level_count=2)
        cells = [
            {"node_id": index, "cell_id": index, "generation": 1, "genome": [2, 2]}
            for index in range(cell_count)
        ]
        input_case = {
            "case_id": main_benchmark.case_id(spec),
            "corpus": main_benchmark.CORPUS_NAME,
            "status": "ok",
            "genome_length": 10,
            "NUMBER_OF_GENERATIONS": 10,
            "seed": seed,
            "r_dist": 4,
            "biopsy_size_scalable": 0.25,
            "biopsy_level_count": 2,
            "biopsy_generations": [1, 2],
            "GENERAL_EVENT_PROB": 0.01,
            "event_shape_label": "low",
            "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": 0.01,
            "GENERAL_DUPLICATION_MULTIPLICITY": 1,
            "config_snapshot": {"genome_length": 10},
            "biopsy_selection": {"total_sampled_biopsy_cells": cell_count},
            "true_tree": {"directed": True, "multigraph": False, "graph": {}, "nodes": [], "links": []},
            "biopsies": [
                {"level": "L1", "generation": 1, "cells": cells[:1]},
                {"level": "L2", "generation": 2, "cells": cells[1:]},
            ],
        }
        main_benchmark.write_json(main_benchmark.input_path(tmp_path, spec), input_case, overwrite=True)

    written = main_benchmark.write_reports(tmp_path)

    summary = pd.read_csv(written["biopsy_summary"]["csv"])
    row = summary.iloc[0]
    assert row["generation"] == "g10"
    assert row["bss"] == 0.25
    assert row["level"] == "L2"
    assert row["n"] == 2
    assert row["min"] == 2
    assert row["max"] == 4
    assert row["avg"] == pytest.approx(3.0)
    assert row["total"] == 6
    assert (tmp_path / "reports" / "biopsy_cell_summary.md").exists()


def test_write_biopsy_cell_summary_can_scope_to_selected_generation(tmp_path):
    for generation_count, seed, cell_count in [(10, 1, 2), (12, 2, 5)]:
        spec = _spec(
            generation_count=generation_count,
            seed=seed,
            biopsy_size_scalable=0.25,
            biopsy_level_count=2,
        )
        cells = [
            {"node_id": index, "cell_id": index, "generation": 1, "genome": [2, 2]}
            for index in range(cell_count)
        ]
        input_case = {
            "case_id": main_benchmark.case_id(spec),
            "corpus": main_benchmark.CORPUS_NAME,
            "status": "ok",
            "genome_length": 10,
            "NUMBER_OF_GENERATIONS": generation_count,
            "seed": seed,
            "r_dist": 4,
            "biopsy_size_scalable": 0.25,
            "biopsy_level_count": 2,
            "biopsy_generations": [1, 2],
            "GENERAL_EVENT_PROB": 0.01,
            "event_shape_label": "low",
            "GENERAL_SINGLE_OR_MULTIPLE_EVENT_PROB": 0.01,
            "GENERAL_DUPLICATION_MULTIPLICITY": 1,
            "config_snapshot": {"genome_length": 10},
            "biopsy_selection": {"total_sampled_biopsy_cells": cell_count},
            "true_tree": {"directed": True, "multigraph": False, "graph": {}, "nodes": [], "links": []},
            "biopsies": [
                {"level": "L1", "generation": 1, "cells": cells[:1]},
                {"level": "L2", "generation": 2, "cells": cells[1:]},
            ],
        }
        main_benchmark.write_json(main_benchmark.input_path(tmp_path, spec), input_case, overwrite=True)

    g12_spec = _spec(generation_count=12, seed=2, biopsy_size_scalable=0.25, biopsy_level_count=2)
    written = main_benchmark.write_biopsy_cell_summary(
        tmp_path,
        specs=[g12_spec],
        label=main_benchmark.biopsy_summary_label([g12_spec]),
    )

    summary = pd.read_csv(written["csv"])
    assert written["csv"].name == "biopsy_cell_summary_g12.csv"
    assert written["markdown"].name == "biopsy_cell_summary_g12.md"
    assert list(summary["generation"]) == ["g12"]
    assert summary.iloc[0]["total"] == 5


def test_detailed_timing_summary_records_write_generation_csv_without_json(tmp_path):
    records = [
        {
            "generation_count": 14,
            "stage": "distance",
            "operation": "distance_case",
            "scope": "stage_total",
            "count": 2,
            "input_files": 3,
            "instances": 2,
            "skipped": 1,
            "read_json_seconds": 0.1,
            "core_seconds": 0.2,
            "write_json_seconds": 0.3,
            "total_seconds": 0.7,
            "distance_mode": "l1",
        },
        {
            "generation_count": 14,
            "stage": "reconstruct",
            "operation": "reconstruct_by_algorithm",
            "scope": "algorithm",
            "count": 4,
            "instances": 4,
            "core_seconds": 1.0,
            "write_json_seconds": 0.5,
            "total_seconds": 1.5,
            "algorithm": "neighbor_joining_baseline",
        },
    ]

    written = main_benchmark.write_detailed_timing_summary_records(
        tmp_path,
        records,
        generation_count=14,
    )

    assert Path(written["csv"]).name == "timing_summary_g14.csv"
    frame = pd.read_csv(written["csv"])
    assert list(frame.columns) == main_benchmark.TIMING_REPORT_COLUMNS
    assert set(frame["scope"]) == {"stage_total", "algorithm"}
    assert not (tmp_path / "reports" / "timing_summary_g14.json").exists()
    assert not (tmp_path / "reports" / "timing_summary.json").exists()


def test_single_stage_command_timing_appends_once_and_skips_multi_stage(tmp_path):
    specs = [
        _spec(generation_count=14, seed=1),
        _spec(generation_count=14, seed=2),
    ]

    first = main_benchmark.write_single_stage_command_timing(
        tmp_path,
        specs,
        ["distance"],
        elapsed_seconds=1.25,
        started_at="2026-05-26T00:00:00+00:00",
        finished_at="2026-05-26T00:00:02+00:00",
        status="ok",
        command="distance",
    )
    second = main_benchmark.write_single_stage_command_timing(
        tmp_path,
        specs,
        ["distance"],
        elapsed_seconds=9.0,
        started_at="later",
        finished_at="later",
        status="ok",
        command="distance again",
    )
    multi = main_benchmark.write_single_stage_command_timing(
        tmp_path,
        specs,
        ["simulate", "biopsy-summary"],
        elapsed_seconds=3.0,
        started_at="multi",
        finished_at="multi",
        status="ok",
        command="multi",
    )

    assert first == [main_benchmark.timing_summary_path(tmp_path)]
    assert second == []
    assert multi == []
    rows = pd.read_csv(main_benchmark.timing_summary_path(tmp_path))
    assert list(rows.columns) == main_benchmark.COMMAND_TIMING_COLUMNS
    assert len(rows) == 1
    assert rows.iloc[0]["stage"] == "distance"
    assert rows.iloc[0]["case_count"] == 2
    assert rows.iloc[0]["total_seconds"] == pytest.approx(1.25)
    assert not (tmp_path / "reports" / "times_g14.csv").exists()


def test_command_timing_ignores_legacy_detailed_summary_shape(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    pd.DataFrame(
        [
            {
                "generation_count": 10,
                "stage": "simulate",
                "operation": "case_timing_total",
                "scope": "stage_total",
                "total_seconds": 17.0,
            }
        ]
    ).to_csv(main_benchmark.timing_summary_path(tmp_path), index=False)

    written = main_benchmark.write_single_stage_command_timing(
        tmp_path,
        [_spec(generation_count=10, seed=7)],
        ["simulate"],
        elapsed_seconds=12.0,
        started_at="s1",
        finished_at="f1",
        status="ok",
        command="simulate",
    )

    rows = pd.read_csv(main_benchmark.timing_summary_path(tmp_path))
    assert written == [main_benchmark.timing_summary_path(tmp_path)]
    assert list(rows.columns) == main_benchmark.COMMAND_TIMING_COLUMNS
    assert len(rows) == 1
    assert rows.iloc[0]["stage"] == "simulate"
    assert rows.iloc[0]["case_count"] == 1
    assert rows.iloc[0]["total_seconds"] == pytest.approx(12.0)


def test_timing_report_groups_reconstruction_by_algorithm_and_evaluation_by_method(tmp_path):
    spec = _spec(generation_count=14, seed=7, biopsy_size_scalable=0.5, biopsy_level_count=2)
    case_directory = main_benchmark.case_dir(tmp_path, spec)
    main_benchmark.write_case_timing_records(case_directory, [
        main_benchmark.case_timing_record(
            spec,
            stage="reconstruct",
            operation="reconstruct",
            algorithm="alg_a",
            mode="full_cnp",
            elapsed_seconds=1.0,
            started_at="s",
            finished_at="f",
        ),
        main_benchmark.case_timing_record(
            spec,
            stage="reconstruct",
            operation="reconstruct",
            algorithm="alg_a",
            mode="biopsy_guided_top",
            elapsed_seconds=2.0,
            started_at="s",
            finished_at="f",
        ),
        main_benchmark.case_timing_record(
            spec,
            stage="evaluate",
            operation="evaluate_metric",
            algorithm="alg_a",
            mode="full_cnp",
            evaluation_method="adf1",
            elapsed_seconds=0.25,
            started_at="s",
            finished_at="f",
        ),
        main_benchmark.case_timing_record(
            spec,
            stage="evaluate",
            operation="evaluate_metric",
            algorithm="alg_a",
            mode="full_cnp",
            evaluation_method="grf",
            elapsed_seconds=0.75,
            started_at="s",
            finished_at="f",
        ),
        main_benchmark.case_timing_record(
            spec,
            stage="evaluate",
            operation="evaluate_phase:exact_grf_weighted_sums",
            algorithm="alg_a",
            mode="full_cnp",
            evaluation_method="grf",
            elapsed_seconds=0.50,
            started_at="s",
            finished_at="f",
        ),
    ])
    main_benchmark.write_single_stage_command_timing(
        tmp_path,
        [spec],
        ["reconstruct"],
        elapsed_seconds=5.0,
        started_at="s",
        finished_at="f",
        status="ok",
        command="reconstruct",
    )

    written = main_benchmark.write_timing_report(tmp_path, specs=[spec])

    detailed_path = Path(written["generation_summaries"]["g14"])
    rows = pd.read_csv(detailed_path)
    reconstruct = rows[(rows["stage"] == "reconstruct") & (rows["scope"] == "algorithm")].iloc[0]
    assert reconstruct["algorithm"] == "alg_a"
    assert reconstruct["count"] == 2
    assert reconstruct["core_seconds"] == pytest.approx(3.0)
    evaluate = rows[(rows["stage"] == "evaluate") & (rows["scope"] == "evaluation_method")]
    assert set(evaluate["evaluation_method"]) == {"adf1", "grf"}
    assert evaluate.set_index("evaluation_method").loc["grf", "core_seconds"] == pytest.approx(0.75)
    evaluate_total = rows[
        (rows["stage"] == "evaluate") & (rows["scope"] == "stage_total")
    ].iloc[0]
    assert evaluate_total["core_seconds"] == pytest.approx(1.0)
    evaluate_phase = rows[
        (rows["stage"] == "evaluate") & (rows["scope"] == "evaluation_phase")
    ].iloc[0]
    assert evaluate_phase["operation"] == "evaluate_phase:exact_grf_weighted_sums"
    assert evaluate_phase["core_seconds"] == pytest.approx(0.50)
    command = rows[(rows["stage"] == "reconstruct") & (rows["scope"] == "command_stage")].iloc[0]
    assert command["total_seconds"] == pytest.approx(5.0)
    assert detailed_path.name == "timing_summary_g14.csv"
    assert "csv" not in written
    assert not (tmp_path / "reports" / "timing_summary.json").exists()
    assert not (tmp_path / "reports" / "times_g14.csv").exists()


def test_timing_report_preserves_previous_stage_after_next_command_summary(tmp_path):
    specs = [_spec(generation_count=10, seed=7)]

    main_benchmark.write_single_stage_command_timing(
        tmp_path,
        specs,
        ["simulate"],
        elapsed_seconds=12.0,
        started_at="s1",
        finished_at="f1",
        status="ok",
        command="simulate",
    )
    main_benchmark.write_timing_report(tmp_path)

    main_benchmark.write_single_stage_command_timing(
        tmp_path,
        specs,
        ["biopsy-summary"],
        elapsed_seconds=3.0,
        started_at="s2",
        finished_at="f2",
        status="ok",
        command="biopsy-summary",
    )
    main_benchmark.write_timing_report(tmp_path)

    rows = pd.read_csv(main_benchmark.timing_summary_path(tmp_path))

    assert list(rows.columns) == main_benchmark.COMMAND_TIMING_COLUMNS
    assert set(rows["stage"]) == {"simulate", "biopsy-summary"}
    assert rows.set_index("stage").loc["simulate", "total_seconds"] == pytest.approx(12.0)
    assert rows.set_index("stage").loc["biopsy-summary", "total_seconds"] == pytest.approx(3.0)

    detailed_rows = pd.read_csv(main_benchmark.detailed_timing_summary_path(tmp_path, 10))
    assert set(detailed_rows[detailed_rows["scope"] == "command_stage"]["stage"]) == {
        "simulate",
        "biopsy-summary",
    }


def test_selected_algorithms_defaults_to_core_rows():
    names = [algorithm.__name__ for algorithm in main_benchmark.selected_algorithms()]

    assert names == list(main_benchmark.DEFAULT_CORE_ALGORITHM_NAMES)


def test_selected_algorithms_respects_explicit_index_without_adding_core_rows():
    names = [algorithm.__name__ for algorithm in main_benchmark.selected_algorithms(algorithm_indexes=[0])]

    assert names == ["neighbor_joining_baseline"]
