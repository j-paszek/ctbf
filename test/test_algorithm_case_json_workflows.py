import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from networkx.readwrite import json_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from json_case_results import (  # noqa: E402
    DEFAULT_CASES_ROOT,
    load_json,
    pairwise_ranking_from_json,
    variant_seed_dirs,
)
from ctbs_utils import to_newick  # noqa: E402
from evaluator import grf_tree  # noqa: E402
from evaluator_full import evaluate_4  # noqa: E402

TOOL_PATH = PROJECT_ROOT / "test" / "tools" / "freeze_algorithm_variant_cases.py"
spec = importlib.util.spec_from_file_location("freeze_algorithm_variant_cases", TOOL_PATH)
freeze_algorithm_variant_cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze_algorithm_variant_cases)


VARIANT = "r4bss05"
SEED = 295
ALGORITHMS = [
    "neighbor_joining_baseline",
    "neighbor_joining_hybrid_anticentral_adaptive_v3",
    "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
]


def _case_dir():
    return DEFAULT_CASES_ROOT / VARIANT / str(SEED)


def _input_case():
    return load_json(_case_dir() / "input.json")


def _result(mode, algorithm):
    return load_json(_case_dir() / mode / f"{algorithm}.json")


def _tree_from_node_link(data):
    return json_graph.node_link_graph(data, directed=True, edges="links")


def _root_id(tree):
    roots = [node for node, indegree in tree.in_degree() if indegree == 0]
    assert len(roots) == 1
    return roots[0]


def _stored_tree_metric_cases():
    cases = []
    for variant, seed, seed_dir in variant_seed_dirs(DEFAULT_CASES_ROOT):
        input_file = seed_dir / "input.json"
        for mode in ["full_cnp", "biopsy_guided_top"]:
            mode_dir = seed_dir / mode
            if not mode_dir.exists():
                continue
            for result_file in sorted(mode_dir.glob("*.json")):
                cases.append((variant, seed, mode, result_file.stem, input_file, result_file))
    return cases


def _stored_reconstruction_cases():
    return _stored_tree_metric_cases()


def _stored_input_cases():
    return [
        (variant, seed, seed_dir / "input.json")
        for variant, seed, seed_dir in variant_seed_dirs(DEFAULT_CASES_ROOT)
    ]


@pytest.mark.parametrize("mode", ["full_cnp", "biopsy_guided_top"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_json_reconstructed_tree_metrics_match_stored_values(mode, algorithm):
    input_case = _input_case()
    result = _result(mode, algorithm)
    true_tree = _tree_from_node_link(input_case["true_tree"])
    reconstructed_tree = _tree_from_node_link(result["reconstructed_tree"])
    labels = {
        str(data.get("cell_id"))
        for _, data in reconstructed_tree.nodes(data=True)
        if data.get("cell_id") is not None
    }

    metrics = evaluate_4(true_tree, reconstructed_tree, restrict_labels=labels)
    grf = grf_tree(true_tree, _root_id(true_tree), reconstructed_tree, _root_id(reconstructed_tree))

    stored = result["metrics"]["ancestors_unique_restricted"]
    actual = metrics["ancestors_unique_restricted"]
    assert actual["precision"] == pytest.approx(stored["precision"])
    assert actual["recall"] == pytest.approx(stored["recall"])
    assert actual["F1"] == pytest.approx(stored["F1"])
    assert grf == pytest.approx(result["metrics"]["grf"])


def test_json_biopsies_recompute_cnp2cnp_matrix():
    input_case = _input_case()
    cell_lists = freeze_algorithm_variant_cases.cell_lists_from_input(input_case)
    unique_cells = freeze_algorithm_variant_cases.unique_cells_by_cell_id(cell_lists)

    ids, matrix = freeze_algorithm_variant_cases.cnp2cnp_distance_matrix(unique_cells)

    expected = input_case["distance_matrices"]["cnp2cnp"]
    assert ids == expected["ids"]
    assert np.array_equal(matrix, np.array(expected["matrix"], dtype=float))


@pytest.mark.json_full
@pytest.mark.parametrize(
    "case",
    _stored_input_cases(),
    ids=lambda case: f"{case[0]}-{case[1]}",
)
def test_all_json_biopsies_recompute_cnp2cnp_matrix(case):
    _, _, input_file = case
    input_case = load_json(input_file)
    cell_lists = freeze_algorithm_variant_cases.cell_lists_from_input(input_case)
    unique_cells = freeze_algorithm_variant_cases.unique_cells_by_cell_id(cell_lists)

    ids, matrix = freeze_algorithm_variant_cases.cnp2cnp_distance_matrix(unique_cells)

    expected = input_case["distance_matrices"]["cnp2cnp"]
    assert ids == expected["ids"]
    assert np.array_equal(matrix, np.array(expected["matrix"], dtype=float))


def test_json_true_tree_recomputes_true_tree_distance_matrix():
    input_case = _input_case()
    true_tree = freeze_algorithm_variant_cases.true_tree_from_input(input_case)
    expected = input_case["distance_matrices"]["true_tree"]

    ids, matrix = freeze_algorithm_variant_cases.true_tree_distance_matrix(true_tree, expected["ids"])

    assert ids == expected["ids"]
    assert np.array_equal(matrix, np.array(expected["matrix"], dtype=float))


@pytest.mark.json_full
@pytest.mark.parametrize(
    "case",
    _stored_input_cases(),
    ids=lambda case: f"{case[0]}-{case[1]}",
)
def test_all_json_true_tree_recomputes_true_tree_distance_matrix(case):
    _, _, input_file = case
    input_case = load_json(input_file)
    true_tree = freeze_algorithm_variant_cases.true_tree_from_input(input_case)
    expected = input_case["distance_matrices"]["true_tree"]

    ids, matrix = freeze_algorithm_variant_cases.true_tree_distance_matrix(true_tree, expected["ids"])

    assert ids == expected["ids"]
    assert np.array_equal(matrix, np.array(expected["matrix"], dtype=float))


@pytest.mark.parametrize("mode", ["full_cnp", "biopsy_guided_top"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_json_reconstruction_is_deterministic_against_stored_tree(mode, algorithm):
    input_case = _input_case()
    algorithm_callable = freeze_algorithm_variant_cases.algorithm_by_name()[algorithm]

    regenerated = freeze_algorithm_variant_cases.reconstruction_result(input_case, algorithm_callable, mode)
    stored = _result(mode, algorithm)
    stored_tree = _tree_from_node_link(stored["reconstructed_tree"])

    assert regenerated["root"] == stored["root"]
    assert regenerated["newick"] == stored["newick"]
    assert regenerated["newick"] == to_newick(stored_tree)


@pytest.mark.json_full
@pytest.mark.parametrize(
    "case",
    _stored_reconstruction_cases(),
    ids=lambda case: f"{case[0]}-{case[1]}-{case[2]}-{case[3]}",
)
def test_all_json_reconstruction_is_deterministic_against_stored_tree(case):
    _, _, mode, algorithm, input_file, result_file = case
    input_case = load_json(input_file)
    algorithm_callable = freeze_algorithm_variant_cases.algorithm_by_name()[algorithm]

    regenerated = freeze_algorithm_variant_cases.reconstruction_result(input_case, algorithm_callable, mode)
    stored = load_json(result_file)
    stored_tree = _tree_from_node_link(stored["reconstructed_tree"])

    assert regenerated["root"] == stored["root"]
    assert regenerated["newick"] == stored["newick"]
    assert regenerated["newick"] == to_newick(stored_tree)


def test_json_pairwise_ranking_uses_frozen_outputs_without_simulation():
    ranking = pairwise_ranking_from_json(
        DEFAULT_CASES_ROOT,
        VARIANT,
        "full_cnp",
        "3-f1",
        algorithms=ALGORITHMS,
    )

    assert ranking.index.tolist() == ALGORITHMS
    assert set(ranking.columns) == {"wins", "losses", "ties", "score"}
    assert int(ranking["wins"].sum() + ranking["ties"].sum() / 2) == 3


@pytest.mark.json_full
@pytest.mark.parametrize(
    "case",
    _stored_tree_metric_cases(),
    ids=lambda case: f"{case[0]}-{case[1]}-{case[2]}-{case[3]}",
)
def test_all_json_reconstructed_tree_metrics_match_stored_values(case):
    _, _, _, _, input_file, result_file = case
    input_case = load_json(input_file)
    result = load_json(result_file)
    true_tree = _tree_from_node_link(input_case["true_tree"])
    reconstructed_tree = _tree_from_node_link(result["reconstructed_tree"])
    labels = {
        str(data.get("cell_id"))
        for _, data in reconstructed_tree.nodes(data=True)
        if data.get("cell_id") is not None
    }

    metrics = evaluate_4(true_tree, reconstructed_tree, restrict_labels=labels)
    grf = grf_tree(true_tree, _root_id(true_tree), reconstructed_tree, _root_id(reconstructed_tree))

    stored = result["metrics"]["ancestors_unique_restricted"]
    actual = metrics["ancestors_unique_restricted"]
    assert actual["precision"] == pytest.approx(stored["precision"])
    assert actual["recall"] == pytest.approx(stored["recall"])
    assert actual["F1"] == pytest.approx(stored["F1"])
    assert grf == pytest.approx(result["metrics"]["grf"])
