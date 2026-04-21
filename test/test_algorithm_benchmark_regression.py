import os
from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUN_SLOW_BENCHMARKS = os.environ.get("CTBF_RUN_SLOW_BENCHMARKS") == "1"

from algorithm_evaluation.tester import (  # noqa: E402
    CONFIG_BY_PROFILE,
    DEFAULT_BIOPSY_GENERATIONS,
    get_algorithms_to_test,
    get_root_id,
)
from ctbs import run_single_test  # noqa: E402
from ctbs_utils import get_biopsy_nodes_ids  # noqa: E402
from evaluator import grf_tree  # noqa: E402
from evaluator_full import evaluate_4  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "algorithm_evaluation" / "results"
BENCHMARK_VARIANTS = {
    "r2bss025": {"r_dist": 2, "biopsy_size_scalable": 0.25, "profile": "base"},
    "r2bss05": {"r_dist": 2, "biopsy_size_scalable": 0.5, "profile": "base"},
    "r2bss075": {"r_dist": 2, "biopsy_size_scalable": 0.75, "profile": "base"},
    "r4bss05": {"r_dist": 4, "biopsy_size_scalable": 0.5, "profile": "base"},
    "r4bss075": {"r_dist": 4, "biopsy_size_scalable": 0.75, "profile": "base"},
    "r4bss05high": {"r_dist": 4, "biopsy_size_scalable": 0.5, "profile": "high"},
    "r4bss05highdm": {"r_dist": 4, "biopsy_size_scalable": 0.5, "profile": "highdm"},
}
METRIC_COLUMNS = ["1-precision", "1-f1", "2-precision", "2-f1", "3-precision", "3-f1", "grf"]


def _parse_env_list(name, cast=str):
    value = os.environ.get(name)
    if not value:
        return None
    return {cast(item.strip()) for item in value.split(",") if item.strip()}


def _selected_variants():
    requested = _parse_env_list("CTBF_BENCHMARK_VARIANTS")
    if requested is None:
        return set(BENCHMARK_VARIANTS)
    unknown = requested - set(BENCHMARK_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown CTBF_BENCHMARK_VARIANTS values: {sorted(unknown)}")
    return requested


def _selected_algorithm_indexes():
    requested = _parse_env_list("CTBF_BENCHMARK_ALGORITHM_INDEXES", int)
    if requested is None:
        return set(range(len(get_algorithms_to_test())))
    return requested


def _selected_seeds():
    return _parse_env_list("CTBF_BENCHMARK_SEEDS", int)


def _load_expected_rows(variant_name, algorithm_index):
    rec_path = RESULTS_DIR / variant_name / f"{algorithm_index}rec.csv"
    nj_path = RESULTS_DIR / variant_name / f"{algorithm_index}nj.csv"
    if not rec_path.exists() or not nj_path.exists():
        raise FileNotFoundError(f"Missing expected benchmark CSVs: {rec_path}, {nj_path}")

    rec_by_seed = pd.read_csv(rec_path).set_index("seed")
    nj_by_seed = pd.read_csv(nj_path).set_index("seed")
    common_seeds = sorted(set(rec_by_seed.index) & set(nj_by_seed.index))
    seed_filter = _selected_seeds()
    if seed_filter is not None:
        common_seeds = [seed for seed in common_seeds if seed in seed_filter]

    rows = []
    for seed in common_seeds:
        rows.append(
            {
                "variant_name": variant_name,
                "algorithm_index": algorithm_index,
                "seed": int(seed),
                "expected_rec": rec_by_seed.loc[seed, METRIC_COLUMNS].to_dict(),
                "expected_nj": nj_by_seed.loc[seed, METRIC_COLUMNS].to_dict(),
            }
        )
    return rows


def _benchmark_cases():
    if not RUN_SLOW_BENCHMARKS:
        return [{"skip_id": "slow-benchmarks-disabled"}]

    selected_variants = _selected_variants()
    selected_algorithms = _selected_algorithm_indexes()
    cases = []
    for variant_name in BENCHMARK_VARIANTS:
        if variant_name not in selected_variants:
            continue
        for algorithm_index in range(len(get_algorithms_to_test())):
            if algorithm_index not in selected_algorithms:
                continue
            cases.extend(_load_expected_rows(variant_name, algorithm_index))
    return cases


def _case_id(case):
    if "skip_id" in case:
        return case["skip_id"]
    return f"{case['variant_name']}-alg{case['algorithm_index']}-seed{case['seed']}"


def _current_metrics(case):
    variant = BENCHMARK_VARIANTS[case["variant_name"]]
    algorithm = get_algorithms_to_test()[case["algorithm_index"]]
    true_tree, rec_tree, nj_tree = run_single_test(
        seed=case["seed"],
        config=CONFIG_BY_PROFILE[variant["profile"]],
        bedfile=None,
        biopsy_size_scalable=variant["biopsy_size_scalable"],
        biopsy_generations=DEFAULT_BIOPSY_GENERATIONS,
        r_dist=variant["r_dist"],
        reconstruction_algorithm=algorithm,
    )

    biopsy_cell_ids = get_biopsy_nodes_ids(rec_tree, nj_tree)
    rec_eval = evaluate_4(true_tree, rec_tree, restrict_labels=biopsy_cell_ids)
    nj_eval = evaluate_4(true_tree, nj_tree, restrict_labels=biopsy_cell_ids)

    return {
        "rec": {
            "1-precision": rec_eval["ancestors_multiset"]["precision"],
            "1-f1": rec_eval["ancestors_multiset"]["F1"],
            "2-precision": rec_eval["ancestors_unique"]["precision"],
            "2-f1": rec_eval["ancestors_unique"]["F1"],
            "3-precision": rec_eval["ancestors_unique_restricted"]["precision"],
            "3-f1": rec_eval["ancestors_unique_restricted"]["F1"],
            "grf": grf_tree(true_tree, get_root_id(true_tree), rec_tree, get_root_id(rec_tree)),
        },
        "nj": {
            "1-precision": nj_eval["ancestors_multiset"]["precision"],
            "1-f1": nj_eval["ancestors_multiset"]["F1"],
            "2-precision": nj_eval["ancestors_unique"]["precision"],
            "2-f1": nj_eval["ancestors_unique"]["F1"],
            "3-precision": nj_eval["ancestors_unique_restricted"]["precision"],
            "3-f1": nj_eval["ancestors_unique_restricted"]["F1"],
            "grf": grf_tree(true_tree, get_root_id(true_tree), nj_tree, get_root_id(nj_tree)),
        },
    }


@pytest.mark.slow
@pytest.mark.skipif(
    not RUN_SLOW_BENCHMARKS,
    reason=(
        "Set CTBF_RUN_SLOW_BENCHMARKS=1 to rerun benchmark regression cases. "
        "See test/README_algorithm_benchmark_regression.md for examples."
    ),
)
@pytest.mark.parametrize("case", _benchmark_cases(), ids=_case_id)
def test_legacy_algorithm_benchmark_metrics_match_committed_results(case):
    current = _current_metrics(case)

    for column in METRIC_COLUMNS:
        assert current["rec"][column] == pytest.approx(case["expected_rec"][column], abs=1e-12)
        assert current["nj"][column] == pytest.approx(case["expected_nj"][column], abs=1e-12)
