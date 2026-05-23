from pathlib import Path

from algorithm_evaluation.tester import DEFAULT_SEEDS_FILE, load_seeds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_CASES_ROOT = PROJECT_ROOT / "test" / "data" / "algorithm_cases"


def _frozen_variant_seeds(variant_dir):
    return {
        int(seed_dir.name)
        for seed_dir in variant_dir.iterdir()
        if seed_dir.is_dir()
        and seed_dir.name.isdigit()
        and (seed_dir / "input.json").exists()
    }


def test_default_benchmark_seed_file_has_100_unique_seeds():
    seeds = load_seeds(DEFAULT_SEEDS_FILE)

    assert len(seeds) == 100
    assert len(set(seeds)) == 100


def test_default_benchmark_seed_file_matches_frozen_algorithm_case_seed_set():
    expected = set(load_seeds(DEFAULT_SEEDS_FILE))
    variant_dirs = [
        path
        for path in ALGORITHM_CASES_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]

    assert variant_dirs
    for variant_dir in variant_dirs:
        assert _frozen_variant_seeds(variant_dir) == expected
