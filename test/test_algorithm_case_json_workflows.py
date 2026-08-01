import importlib.util
import os
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
    algorithms_for_variant,
    load_json,
    pairwise_ranking_from_json,
    variant_seed_dirs,
)
from reconstructor_algorithm_config import resolve_comparison_algorithm_names  # noqa: E402
from ctbs_utils import to_newick  # noqa: E402

TOOL_PATH = PROJECT_ROOT / "test" / "tools" / "freeze_algorithm_variant_cases.py"
spec = importlib.util.spec_from_file_location("freeze_algorithm_variant_cases", TOOL_PATH)
freeze_algorithm_variant_cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze_algorithm_variant_cases)

FAST_BENCHMARK_TOOL_PATH = PROJECT_ROOT / "test" / "tools" / "fast_biopsy_preset_benchmark.py"
fast_spec = importlib.util.spec_from_file_location("fast_biopsy_preset_benchmark", FAST_BENCHMARK_TOOL_PATH)
fast_biopsy_preset_benchmark = importlib.util.module_from_spec(fast_spec)
fast_spec.loader.exec_module(fast_biopsy_preset_benchmark)


VARIANT = "r4bss05"
SEED = 295
ALGORITHMS = freeze_algorithm_variant_cases.PUBLICATION_HEATMAP_ALGORITHM_NAMES
CANONICAL_ALGORITHM_SET = set(ALGORITHMS)


def _case_dir():
    return DEFAULT_CASES_ROOT / VARIANT / str(SEED)


def _input_case():
    return load_json(_case_dir() / "input.json")


def _result(mode, algorithm):
    return load_json(_case_dir() / mode / f"{algorithm}.json")


def _tree_from_node_link(data):
    return json_graph.node_link_graph(data, directed=True, edges="links")


def _assert_reconstruction_matches_stored_observable_tree(regenerated, stored):
    """Compare rooted labeled output without equating graph-local node ids."""
    regenerated_tree = _tree_from_node_link(regenerated["reconstructed_tree"])
    stored_tree = _tree_from_node_link(stored["reconstructed_tree"])

    regenerated_roots = [
        node for node, degree in regenerated_tree.in_degree() if degree == 0
    ]
    stored_roots = [
        node for node, degree in stored_tree.in_degree() if degree == 0
    ]
    assert len(regenerated_roots) == 1
    assert len(stored_roots) == 1

    regenerated_graph_root = regenerated_roots[0]
    stored_graph_root = stored_roots[0]
    regenerated_root_cell_id = regenerated_tree.nodes[regenerated_graph_root].get(
        "cell_id"
    )
    stored_root_cell_id = stored_tree.nodes[stored_graph_root].get("cell_id")
    # CTBF's existing algorithms do not yet expose one uniform root-reference
    # type: legacy directed agglomeration returns root-state cell_id, whereas
    # classical and newer routines may return graph node_id.
    assert regenerated["root"] in (
        regenerated_graph_root,
        regenerated_root_cell_id,
    )
    assert stored["root"] in (stored_graph_root, stored_root_cell_id)

    regenerated_root = regenerated_tree.nodes[regenerated_graph_root]
    stored_root = stored_tree.nodes[stored_graph_root]
    assert regenerated_root.get("cell_id") == stored_root.get("cell_id")
    assert np.array_equal(
        np.asarray(regenerated_root.get("genome")),
        np.asarray(stored_root.get("genome")),
    )

    assert regenerated["newick"] == stored["newick"]
    assert regenerated["newick"] == to_newick(regenerated_tree)
    assert stored["newick"] == to_newick(stored_tree)


def _assert_stored_metrics_match_current(input_case, result):
    true_tree = _tree_from_node_link(input_case["true_tree"])
    reconstructed_tree = _tree_from_node_link(result["reconstructed_tree"])
    current = freeze_algorithm_variant_cases.metric_summary(true_tree, reconstructed_tree)

    stored = result["metrics"]["ancestors_unique_restricted"]
    actual = current["ancestors_unique_restricted"]
    assert actual["precision"] == pytest.approx(stored["precision"])
    assert actual["recall"] == pytest.approx(stored["recall"])
    assert actual["F1"] == pytest.approx(stored["F1"])
    assert current["grf"] == pytest.approx(result["metrics"]["grf"])

    ext_field = freeze_algorithm_variant_cases.EXT_GRF_METRIC_FIELD
    if ext_field in result["metrics"]:
        assert current[ext_field] == pytest.approx(result["metrics"][ext_field])

    legacy_field = freeze_algorithm_variant_cases.LEGACY_GRF_SET_SIMILARITY_FIELD
    if legacy_field in result["metrics"]:
        assert current[legacy_field] == pytest.approx(result["metrics"][legacy_field])


def _stored_tree_metric_cases():
    cases = []
    for variant, seed, seed_dir in variant_seed_dirs(DEFAULT_CASES_ROOT):
        input_file = seed_dir / "input.json"
        for mode in ["full_cnp", "biopsy_guided_top"]:
            mode_dir = seed_dir / mode
            if not mode_dir.exists():
                continue
            for result_file in sorted(mode_dir.glob("*.json")):
                if result_file.stem not in CANONICAL_ALGORITHM_SET:
                    continue
                cases.append((variant, seed, mode, result_file.stem, input_file, result_file))
    return cases


def _stored_reconstruction_cases():
    return _stored_tree_metric_cases()


def _selected_extra_algorithm_names():
    raw = os.environ.get("CTBF_JSON_EXTRA_ALGORITHM_NAMES")
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def _stored_extra_tree_metric_cases():
    selected = set(_selected_extra_algorithm_names())
    if not selected:
        return []

    cases = []
    for variant, seed, seed_dir in variant_seed_dirs(DEFAULT_CASES_ROOT):
        input_file = seed_dir / "input.json"
        for mode in ["full_cnp", "biopsy_guided_top"]:
            mode_dir = seed_dir / mode
            if not mode_dir.exists():
                continue
            for result_file in sorted(mode_dir.glob("*.json")):
                if result_file.stem in CANONICAL_ALGORITHM_SET:
                    continue
                if result_file.stem not in selected:
                    continue
                cases.append((variant, seed, mode, result_file.stem, input_file, result_file))
    return cases


def _stored_input_cases():
    return [
        (variant, seed, seed_dir / "input.json")
        for variant, seed, seed_dir in variant_seed_dirs(DEFAULT_CASES_ROOT)
    ]


def _case_id(case):
    if not isinstance(case, tuple) or len(case) < 4:
        return "no-cases"
    return f"{case[0]}-{case[1]}-{case[2]}-{case[3]}"


def _reconstruct_from_stored_result(input_case, stored_result, mode):
    algorithms = freeze_algorithm_variant_cases.algorithm_by_name()
    algorithm_name = stored_result["algorithm"]
    if algorithm_name in algorithms:
        return freeze_algorithm_variant_cases.reconstruction_result(
            input_case,
            algorithms[algorithm_name],
            mode,
        )

    if stored_result.get("biopsy_guided_preset") is not None:
        return fast_biopsy_preset_benchmark.biopsy_preset_result(
            input_case,
            algorithm_name,
            stored_result["biopsy_guided_preset"],
        )

    raise KeyError(algorithm_name)


@pytest.mark.parametrize("mode", ["full_cnp", "biopsy_guided_top"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_json_reconstructed_tree_metrics_match_stored_values(mode, algorithm):
    input_case = _input_case()
    result = _result(mode, algorithm)
    _assert_stored_metrics_match_current(input_case, result)


def test_json_biopsies_recompute_cnp2cnp_matrix():
    input_case = _input_case()
    cell_lists = freeze_algorithm_variant_cases.cell_lists_from_input(input_case)
    unique_cells = freeze_algorithm_variant_cases.unique_cells_by_cell_id(cell_lists)

    ids, matrix = (
        freeze_algorithm_variant_cases.legacy_cnp2cnp_ordered_distance_matrix(
            unique_cells
        )
    )

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

    ids, matrix = (
        freeze_algorithm_variant_cases.legacy_cnp2cnp_ordered_distance_matrix(
            unique_cells
        )
    )

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
    _assert_reconstruction_matches_stored_observable_tree(regenerated, stored)


@pytest.mark.json_full
@pytest.mark.parametrize(
    "case",
    _stored_reconstruction_cases(),
    ids=_case_id,
)
def test_all_json_reconstruction_is_deterministic_against_stored_tree(case):
    _, _, mode, algorithm, input_file, result_file = case
    input_case = load_json(input_file)
    stored = load_json(result_file)
    regenerated = _reconstruct_from_stored_result(input_case, stored, mode)
    _assert_reconstruction_matches_stored_observable_tree(regenerated, stored)


def test_json_biopsy_preset_reconstruction_is_deterministic_against_stored_tree():
    input_case = load_json(
        DEFAULT_CASES_ROOT / "r4bss05" / "1001" / "input.json"
    )
    stored = load_json(
        DEFAULT_CASES_ROOT
        / "r4bss05"
        / "1001"
        / "biopsy_guided_top"
        / "biopsy_preset_binarized.json"
    )

    regenerated = _reconstruct_from_stored_result(input_case, stored, "biopsy_guided_top")
    _assert_reconstruction_matches_stored_observable_tree(regenerated, stored)


@pytest.mark.parametrize(
    "case",
    _stored_extra_tree_metric_cases(),
    ids=_case_id,
)
def test_selected_extra_json_reconstructed_tree_metrics_match_stored_values(case):
    _, _, _, _, input_file, result_file = case
    input_case = load_json(input_file)
    result = load_json(result_file)
    _assert_stored_metrics_match_current(input_case, result)


@pytest.mark.parametrize(
    "case",
    _stored_extra_tree_metric_cases(),
    ids=_case_id,
)
def test_selected_extra_json_reconstruction_is_deterministic_against_stored_tree(case):
    _, _, mode, _, input_file, result_file = case
    input_case = load_json(input_file)
    stored = load_json(result_file)
    regenerated = _reconstruct_from_stored_result(input_case, stored, mode)
    _assert_reconstruction_matches_stored_observable_tree(regenerated, stored)


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
    assert int(ranking["wins"].sum() + ranking["ties"].sum() / 2) == len(ALGORITHMS) * (len(ALGORITHMS) - 1) // 2


def test_json_algorithm_listing_keeps_biopsy_preset_rows_mode_specific():
    full_cnp_algorithms = algorithms_for_variant(DEFAULT_CASES_ROOT, VARIANT, "full_cnp")
    biopsy_guided_algorithms = algorithms_for_variant(DEFAULT_CASES_ROOT, VARIANT, "biopsy_guided_top")

    preset_rows = {
        "biopsy_preset_default",
        "biopsy_preset_anticentral_tie",
        "biopsy_preset_binarized",
        "biopsy_preset_anticentral_binarized",
    }
    assert preset_rows.isdisjoint(full_cnp_algorithms)
    assert preset_rows.issubset(biopsy_guided_algorithms)


def test_comparison_group_resolves_core_algorithm_names_for_heatmap_filtering():
    assert resolve_comparison_algorithm_names(["recommended_core"]) == [
        "neighbor_joining_classical",
        "rooted_labeled_nj",
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
    ]
    assert resolve_comparison_algorithm_names(["new_alg_comparison"]) == [
        "neighbor_joining_baseline",
        "neighbor_joining_hybrid_opt",
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
        "new_alg",
    ]


@pytest.mark.json_full
@pytest.mark.parametrize(
    "case",
    _stored_tree_metric_cases(),
    ids=_case_id,
)
def test_all_json_reconstructed_tree_metrics_match_stored_values(case):
    _, _, _, _, input_file, result_file = case
    input_case = load_json(input_file)
    result = load_json(result_file)
    _assert_stored_metrics_match_current(input_case, result)
