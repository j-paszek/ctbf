from pathlib import Path
import importlib.util
import shutil
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

FAST_BENCHMARK_TOOL_PATH = PROJECT_ROOT / "test" / "tools" / "fast_biopsy_preset_benchmark.py"
fast_spec = importlib.util.spec_from_file_location("fast_biopsy_preset_benchmark", FAST_BENCHMARK_TOOL_PATH)
fast_biopsy_preset_benchmark = importlib.util.module_from_spec(fast_spec)
fast_spec.loader.exec_module(fast_biopsy_preset_benchmark)

OBS_DROPOUT_TOOL_PATH = PROJECT_ROOT / "test" / "tools" / "observation_dropout_stress.py"
obs_spec = importlib.util.spec_from_file_location("observation_dropout_stress", OBS_DROPOUT_TOOL_PATH)
observation_dropout_stress = importlib.util.module_from_spec(obs_spec)
obs_spec.loader.exec_module(observation_dropout_stress)

REFERENCE_ALGORITHM_NAMES = freeze_algorithm_variant_cases.REFERENCE_ALGORITHM_NAMES
case_dir = freeze_algorithm_variant_cases.case_dir
genotype_to_json = freeze_algorithm_variant_cases.genotype_to_json
input_path = freeze_algorithm_variant_cases.input_path
json_ready = freeze_algorithm_variant_cases.json_ready
resolve_reference_algorithm_indices = freeze_algorithm_variant_cases.resolve_reference_algorithm_indices
resolve_variants = freeze_algorithm_variant_cases.resolve_variants
result_path = freeze_algorithm_variant_cases.result_path
from simulator import Genotype  # noqa: E402
from evaluator import parse_newick_to_nx  # noqa: E402


def test_resolve_variants_defaults_to_legacy_fixed_radius_variants():
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


def test_resolve_variants_accepts_adaptive_radius_labels_without_defaulting_to_them():
    variants = resolve_variants(["rAbss025", "rAbss05highdm"])

    assert [name for name, _ in variants] == ["rAbss025", "rAbss05highdm"]
    assert variants[0][1]["radius_mode"] == freeze_algorithm_variant_cases.ADAPTIVE_RADIUS_MODE
    assert variants[0][1]["source_variant"] == "r2bss025"
    assert variants[1][1]["source_variant"] == "r4bss05highdm"


def test_adaptive_radius_input_case_is_derived_from_mean_pairwise_distance():
    source_case = {
        "case_id": "r2bss05_seed123",
        "variant": "r2bss05",
        "seed": 123,
        "profile": "base",
        "r_dist": 2,
        "biopsy_size_scalable": 0.5,
        "distance_matrices": {
            "cnp2cnp": {
                "ids": [1, 2, 3],
                "matrix": [
                    [0.0, 2.0, 4.0],
                    [2.0, 0.0, 6.0],
                    [4.0, 6.0, 0.0],
                ],
            }
        },
    }
    variant = freeze_algorithm_variant_cases.VARIANT_PRESETS["rAbss05"]

    derived = freeze_algorithm_variant_cases.input_case_from_existing(
        source_case,
        "rAbss05",
        variant,
        adaptive_radius_scale=0.75,
    )

    assert derived["case_id"] == "rAbss05_seed123"
    assert derived["variant"] == "rAbss05"
    assert derived["r_dist"] == pytest.approx(3.0)
    assert derived["radius_mode"] == freeze_algorithm_variant_cases.ADAPTIVE_RADIUS_MODE
    assert derived["adaptive_radius_scale"] == pytest.approx(0.75)
    assert derived["adaptive_radius_distance_mean"] == pytest.approx(4.0)
    assert source_case["variant"] == "r2bss05"
    assert source_case["r_dist"] == 2


def test_observation_dropout_stress_perturbs_observed_genomes_consistently_by_cell_id():
    source_case = {
        "case_id": "r2bss05_seed123",
        "variant": "r2bss05",
        "seed": 123,
        "profile": "base",
        "r_dist": 2,
        "biopsy_size_scalable": 0.5,
        "biopsies": [
            {
                "generation": 1,
                "cells": [
                    {"node_id": 1, "cell_id": 10, "generation": 1, "genome": [2, 1, 0]},
                    {"node_id": 2, "cell_id": 11, "generation": 1, "genome": [0, 3, 2]},
                ],
            },
            {
                "generation": 2,
                "cells": [
                    {"node_id": 3, "cell_id": 10, "generation": 2, "genome": [2, 1, 0]},
                ],
            },
        ],
        "distance_matrices": {
            "cnp2cnp": {
                "ids": [10, 11],
                "matrix": [[0.0, 1.0], [1.0, 0.0]],
            }
        },
    }

    perturbed = observation_dropout_stress.perturb_input_case(
        source_case,
        dropout_rate=1.0,
        stress_seed=7,
        distance_mode="l1",
    )

    first_cell = perturbed["biopsies"][0]["cells"][0]
    repeated_cell = perturbed["biopsies"][1]["cells"][0]
    second_cell = perturbed["biopsies"][0]["cells"][1]
    assert first_cell["genome"] == [0, 0, 0]
    assert repeated_cell["genome"] == [0, 0, 0]
    assert second_cell["genome"] == [0, 0, 0]
    assert perturbed["observation_perturbation"]["kind"] == observation_dropout_stress.STRESS_KIND
    assert perturbed["observation_perturbation"]["positive_bins"] == 4
    assert perturbed["observation_perturbation"]["dropped_bins"] == 4
    assert perturbed["distance_matrices"]["cnp2cnp"]["ids"] == [10, 11]
    assert np.array_equal(
        perturbed["distance_matrices"]["cnp2cnp"]["matrix"],
        np.zeros((2, 2)),
    )
    assert perturbed["distance_matrices"]["cnp2cnp"]["provenance"] == {
        "schema_version": "ctbf-distance-provenance-v1",
        "metric": "l1",
        "semantics_version": "ctbf-l1-profile-v1",
        "formula": "sum(abs(u_i-v_i))",
    }
    assert source_case["biopsies"][0]["cells"][0]["genome"] == [2, 1, 0]


def test_default_reference_algorithm_indices_match_accepted_reference_algorithms():
    indices = resolve_reference_algorithm_indices()

    assert indices == list(range(21))
    assert len(REFERENCE_ALGORITHM_NAMES) == 21
    assert REFERENCE_ALGORITHM_NAMES[0] == "neighbor_joining_baseline"
    assert REFERENCE_ALGORITHM_NAMES[17] == "neighbor_joining_hybrid_anticentral_adaptive_v3"
    assert REFERENCE_ALGORITHM_NAMES[20] == "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony"


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


def test_metric_summary_stores_exact_ext_grf_and_legacy_set_similarity():
    true_tree, _ = parse_newick_to_nx("(a,a);", prefix="true")
    reconstructed_tree, _ = parse_newick_to_nx("(a,b);", prefix="reconstructed")

    summary = freeze_algorithm_variant_cases.metric_summary(true_tree, reconstructed_tree)

    assert summary["grf"] == pytest.approx(4 / 9)
    assert summary[freeze_algorithm_variant_cases.EXT_GRF_METRIC_FIELD] == pytest.approx(5 / 9)
    assert summary[freeze_algorithm_variant_cases.LEGACY_GRF_SET_SIMILARITY_FIELD] == pytest.approx(
        31 / 72
    )
    assert summary[freeze_algorithm_variant_cases.LEGACY_GRF_SET_SIMILARITY_FIELD] != pytest.approx(
        summary["grf"]
    )


def test_genotype_to_json_omits_truth_node_id_and_round_trips_from_cell_id():
    cell = Genotype([2, 0, 2], node_id=14, generation=8, cell_id=7)
    data = genotype_to_json(cell)

    assert "node_id" not in data
    assert data["cell_id"] == 7
    assert data["generation"] == 8
    assert np.array_equal(data["genome"], np.array([2, 0, 2]))

    restored = freeze_algorithm_variant_cases.genotypes_from_json([data])[0]
    assert restored.node_id == 7
    assert restored.cell_id == 7
    assert np.array_equal(restored.genome, np.array([2, 0, 2]))


def test_legacy_biopsy_json_retains_node_id_loader_compatibility():
    legacy = {
        "node_id": 14,
        "cell_id": 7,
        "generation": 8,
        "genome": [2, 0, 2],
        "observation_key": "withdrawn-legacy-field",
        "occurrence_kind": "observed",
        "source_observation_key": None,
    }

    restored = freeze_algorithm_variant_cases.genotypes_from_json([legacy])[0]

    assert restored.node_id == 14
    assert restored.cell_id == 7
    assert not hasattr(restored, "observation_key")
    assert not hasattr(restored, "occurrence_kind")
    assert not hasattr(restored, "source_observation_key")


def test_unique_cells_by_cell_id_preserves_first_repeated_observation():
    first = Genotype([2, 2], node_id=50, generation=1, cell_id=5)
    repeated = Genotype([2, 2], node_id=51, generation=2, cell_id=5)
    other = Genotype([3, 2], node_id=70, generation=3, cell_id=7)

    assert freeze_algorithm_variant_cases.unique_cells_by_cell_id([[first], [repeated], [other]]) == [
        first,
        other,
    ]


def test_cnp2cnp_distance_matrix_returns_singleton_zero_without_runtime_config(monkeypatch):
    def fail_if_called():
        raise AssertionError("single-genotype distance matrix should not call cnp2cnp config")

    monkeypatch.setattr(freeze_algorithm_variant_cases, "load_ctbs_runtime_config", fail_if_called)

    ids, matrix = freeze_algorithm_variant_cases.cnp2cnp_distance_matrix(
        [Genotype([2, 2], node_id=50, generation=1, cell_id=5)]
    )

    assert ids == [5]
    assert np.array_equal(matrix, np.array([[0.0]]))

    ids, matrix, provenance = (
        freeze_algorithm_variant_cases.cnp2cnp_distance_matrix_with_provenance(
            [Genotype([2, 2], node_id=50, generation=1, cell_id=5)]
        )
    )
    assert ids == [5]
    assert np.array_equal(matrix, np.array([[0.0]]))
    assert provenance["semantics_version"] == (
        "ctbf-cnp2cnp-any-min-bidirectional-v1"
    )
    assert provenance["construction"] == "trivial_singleton"
    assert provenance["directional_calls_per_unordered_pair"] == 0


def test_fast_biopsy_preset_benchmark_writes_preset_result_from_frozen_input(tmp_path):
    cases_root = tmp_path / "algorithm_cases"
    case_root = cases_root / "r4bss05" / "1001"
    case_root.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / "test" / "data" / "algorithm_cases" / "r4bss05" / "1001" / "input.json",
                    case_root / "input.json")

    written, skipped = fast_biopsy_preset_benchmark.write_case_presets(
        cases_root,
        "r4bss05",
        1001,
        [("biopsy_preset_default", "default")],
        overwrite=False,
        skip_existing=False,
    )

    assert skipped == []
    assert written == [case_root / "biopsy_guided_top" / "biopsy_preset_default.json"]
    result = freeze_algorithm_variant_cases.load_json(written[0])
    assert result["algorithm"] == "biopsy_preset_default"
    assert result["mode"] == "biopsy_guided_top"
    assert result["biopsy_guided_preset"] == "default"
    assert result["neighbor_joining"] == "neighbor_joining_standard"
    assert "reconstructed_tree" in result
    assert "ancestors_unique_restricted" in result["metrics"]
