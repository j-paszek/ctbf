from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm_evaluation.tester import get_legacy_algorithms_to_test, select_algorithm_indices


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

