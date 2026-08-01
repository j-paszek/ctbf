from pathlib import Path
import importlib
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_evaluation.tester import (
    format_algorithm_listing,
    format_run_plan,
    get_legacy_algorithms_to_test,
    select_algorithm_indices,
)


def test_select_algorithm_indices_defaults_to_all_algorithms():
    algorithms = get_legacy_algorithms_to_test()

    assert select_algorithm_indices(algorithms) == list(range(len(algorithms)))


def test_select_algorithm_indices_accepts_indexes_and_names():
    algorithms = get_legacy_algorithms_to_test()

    selected = select_algorithm_indices(
        algorithms,
        algorithm_indexes=[0, 20],
        algorithm_names=["neighbor_joining_hybrid_anticentral_opt"],
    )

    assert selected == [0, 20, 16]


def test_select_algorithm_indices_deduplicates_without_reordering():
    algorithms = get_legacy_algorithms_to_test()

    selected = select_algorithm_indices(
        algorithms,
        algorithm_indexes=[20, 0],
        algorithm_names=["neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony"],
    )

    assert selected == [20, 0]


def test_select_algorithm_indices_rejects_unknown_name():
    algorithms = get_legacy_algorithms_to_test()

    with pytest.raises(ValueError, match="Unknown algorithm name"):
        select_algorithm_indices(algorithms, algorithm_names=["not_an_algorithm"])


def test_select_algorithm_indices_rejects_out_of_range_index():
    algorithms = get_legacy_algorithms_to_test()

    with pytest.raises(ValueError, match="out of range"):
        select_algorithm_indices(algorithms, algorithm_indexes=[len(algorithms)])


def test_benchmark_regression_env_filter_accepts_algorithm_names(monkeypatch):
    monkeypatch.setenv("CTBF_RUN_SLOW_BENCHMARKS", "1")
    monkeypatch.setenv("CTBF_BENCHMARK_ALGORITHM_NAMES", "neighbor_joining_hybrid_anticentral_opt")
    monkeypatch.delenv("CTBF_BENCHMARK_ALGORITHM_INDEXES", raising=False)

    benchmark_regression = importlib.import_module("test_algorithm_benchmark_regression")
    benchmark_regression = importlib.reload(benchmark_regression)

    assert benchmark_regression._selected_algorithm_indexes() == {16}


def test_format_algorithm_listing_marks_legacy_and_experimental_algorithms():
    algorithms = get_legacy_algorithms_to_test()

    def experimental_algorithm():
        return None

    def publication_algorithm():
        return None

    listing = format_algorithm_listing(
        algorithms + [experimental_algorithm, publication_algorithm],
        legacy_count=len(algorithms),
        publication_names={"publication_algorithm"},
    )

    assert "0: neighbor_joining_baseline [legacy]" in listing
    assert "20: neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony [legacy]" in listing
    assert "21: experimental_algorithm [experimental]" in listing
    assert "22: publication_algorithm [publication]" in listing


def test_format_run_plan_shows_resolved_seeds_and_algorithms():
    algorithms = get_legacy_algorithms_to_test()

    plan = format_run_plan(
        variant_name="r4bss05",
        config_path="test/data/config_for_pic.json",
        seeds_source="CLI --seed",
        output_dir="algorithm_evaluation/results/r4bss05",
        seeds=[295],
        algorithms=algorithms,
        selected_indices=[20],
    )

    assert "Variant: r4bss05" in plan
    assert "Seeds source: CLI --seed" in plan
    assert "Seed values: 295" in plan
    assert "Algorithms: [20]" in plan
    assert "20: neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony" in plan
