from pathlib import Path

import pytest

from algorithm_regression_helpers import (
    algorithm_by_name,
    expand_case_expectations,
    load_frozen_case,
    reconstruct_frozen_case,
)


CASE_PATH = Path(__file__).resolve().parent / "data" / "algorithm_cases" / "seed689_r4bss05.json"
FROZEN_CASE = load_frozen_case(CASE_PATH)


@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
@pytest.mark.parametrize(
    "expectation",
    list(expand_case_expectations(FROZEN_CASE)),
    ids=lambda expectation: expectation["algorithm_name"],
)
def test_seed689_frozen_case_regression_for_legacy_variants(expectation):
    algorithms = algorithm_by_name()
    algorithm = algorithms[expectation["algorithm_name"]]

    actual = reconstruct_frozen_case(FROZEN_CASE, algorithm)

    assert actual["rec_root"] == expectation["rec_root"]
    assert actual["nj_root"] == expectation["nj_root"]
    assert actual["rec_newick"] == expectation["rec_newick"]
    assert actual["nj_newick"] == expectation["nj_newick"]

    for tree_key in ["rec_ancestors_unique_restricted", "nj_ancestors_unique_restricted"]:
        assert actual[tree_key]["precision"] == pytest.approx(expectation[tree_key]["precision"])
        assert actual[tree_key]["F1"] == pytest.approx(expectation[tree_key]["F1"])

