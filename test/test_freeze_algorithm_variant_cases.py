from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOL_PATH = PROJECT_ROOT / "test" / "tools" / "freeze_algorithm_variant_cases.py"
spec = importlib.util.spec_from_file_location("freeze_algorithm_variant_cases", TOOL_PATH)
freeze_algorithm_variant_cases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze_algorithm_variant_cases)

REFERENCE_ALGORITHM_NAMES = freeze_algorithm_variant_cases.REFERENCE_ALGORITHM_NAMES
case_dir = freeze_algorithm_variant_cases.case_dir
genotype_to_json = freeze_algorithm_variant_cases.genotype_to_json
input_path = freeze_algorithm_variant_cases.input_path
json_ready = freeze_algorithm_variant_cases.json_ready
resolve_reference_algorithm_indices = freeze_algorithm_variant_cases.resolve_reference_algorithm_indices
resolve_variants = freeze_algorithm_variant_cases.resolve_variants
result_path = freeze_algorithm_variant_cases.result_path
from simulator import Genotype  # noqa: E402


def test_resolve_variants_defaults_to_all_known_variants():
    variants = resolve_variants(None)

    assert [name for name, _ in variants] == [
        "r2bss025",
        "r2bss05",
        "r2bss075",
        "r4bss05",
        "r4bss075",
        "r4bss05high",
        "r4bss05highdm",
    ]


def test_resolve_variants_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown variants"):
        resolve_variants(["not_a_variant"])


def test_default_reference_algorithm_indices_match_accepted_reference_algorithms():
    indices = resolve_reference_algorithm_indices()

    assert indices == [0, 17, 20]
    assert REFERENCE_ALGORITHM_NAMES == [
        "neighbor_joining_baseline",
        "neighbor_joining_hybrid_anticentral_adaptive_v3",
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
    ]


def test_nested_case_paths_are_stable():
    output_root = Path("test/data/algorithm_cases")

    assert case_dir(output_root, "r4bss05", 295) == Path("test/data/algorithm_cases/r4bss05/295")
    assert input_path(output_root, "r4bss05", 295) == Path("test/data/algorithm_cases/r4bss05/295/input.json")
    assert result_path(
        output_root,
        "r4bss05",
        295,
        "full_cnp",
        "neighbor_joining_baseline",
    ) == Path("test/data/algorithm_cases/r4bss05/295/full_cnp/neighbor_joining_baseline.json")


def test_json_ready_converts_numpy_values():
    data = {
        "int": np.int64(1),
        "float": np.float64(1.5),
        "array": np.array([[1, 2], [3, 4]]),
    }

    assert json_ready(data) == {
        "int": 1,
        "float": 1.5,
        "array": [[1, 2], [3, 4]],
    }


def test_genotype_to_json_stores_reconstructable_cell_data():
    cell = Genotype([2, 0, 2], node_id=14, generation=8, cell_id=7)
    data = genotype_to_json(cell)

    assert data["node_id"] == 14
    assert data["cell_id"] == 7
    assert data["generation"] == 8
    assert np.array_equal(data["genome"], np.array([2, 0, 2]))
