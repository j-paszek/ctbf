import json
from pathlib import Path

import pytest
import networkx as nx
import numpy as np

from evaluator import (
    GRF_HIGHER_IS_BETTER,
    GRF_METRIC_KIND,
    GRF_METRIC_NAME,
    grf_tree,
)
from evaluator_full import (
    EVALUATE_4_HIGHER_IS_BETTER,
    EVALUATE_4_MODE_SPECS,
    EVALUATE_4_VALUE_DIRECTION,
    evaluate_4,
)
from reconstructor import (
    BIOPSY_GUIDED_PRESETS,
    make_anticentral_binarized_biopsy_guided_config,
    make_anticentral_tie_biopsy_guided_config,
    make_binarized_biopsy_guided_config,
    make_default_biopsy_guided_config,
    resolve_biopsy_guided_config,
)
from reconstructor_algorithm_specs import (
    DISCOVERY_ALGORITHM_SPECS,
    LEGACY_ALGORITHM_SPECS,
    PUBLICATION_ALGORITHM_SPECS,
    ReconstructionAlgorithmSpec,
)
from reconstructor_algorithm_config import (
    ALGORITHM_CONFIG_BY_NAME,
    COMPARISON_GROUPS,
    HIGHLIGHTED_HEATMAP_ALGORITHMS,
    AlgorithmDisplayConfig,
)
from reconstructor_plausibility import (
    is_biologically_plausible_ancestor,
    is_biologically_plausible_pair,
)
from reconstructor_registry import (
    LEGACY_ALGORITHM_NAMES,
    get_algorithms,
    get_discovery_algorithms,
    get_legacy_algorithms,
    get_publication_algorithms,
    resolve_reconstruction_algorithm,
)
from simulator import Genotype


def test_plausibility_public_names_match_expected_rules():
    ancestor = Genotype([2, 0], 1)
    descendant_possible = Genotype([1, 0], 2)
    descendant_impossible = Genotype([1, 1], 3)

    assert is_biologically_plausible_ancestor(ancestor, descendant_possible)
    assert not is_biologically_plausible_ancestor(ancestor, descendant_impossible)
    assert is_biologically_plausible_pair(ancestor, descendant_impossible)


def test_algorithm_specs_are_registry_source_of_truth():
    assert [spec.name for spec in LEGACY_ALGORITHM_SPECS] == LEGACY_ALGORITHM_NAMES
    assert [spec.stable_id for spec in LEGACY_ALGORITHM_SPECS] == LEGACY_ALGORITHM_NAMES
    assert [algorithm.__name__ for algorithm in get_legacy_algorithms()] == LEGACY_ALGORITHM_NAMES
    assert all(isinstance(spec, ReconstructionAlgorithmSpec) for spec in LEGACY_ALGORITHM_SPECS)
    assert LEGACY_ALGORITHM_NAMES[0] == "neighbor_joining_baseline"
    assert LEGACY_ALGORITHM_NAMES[20] == "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony"
    assert [spec.name for spec in PUBLICATION_ALGORITHM_SPECS] == [
        "neighbor_joining_classical",
        "rooted_labeled_nj",
        "temporal_cnp_arborescence",
        "temporal_cnp_arborescence_no_time",
    ]
    assert [algorithm.__name__ for algorithm in get_publication_algorithms()] == [
        "neighbor_joining_classical",
        "rooted_labeled_nj",
        "temporal_cnp_arborescence",
        "temporal_cnp_arborescence_no_time",
    ]
    assert all(spec.legacy is False for spec in PUBLICATION_ALGORITHM_SPECS)
    assert [algorithm.__name__ for algorithm in get_algorithms()][21:] == [
        "new_alg",
        "neighbor_joining_classical",
        "rooted_labeled_nj",
        "temporal_cnp_arborescence",
        "temporal_cnp_arborescence_no_time",
        "temporal_cnp_arborescence_directed",
        "temporal_cnp_arborescence_directed_no_time",
    ]
    assert [spec.name for spec in DISCOVERY_ALGORITHM_SPECS] == [
        "temporal_cnp_arborescence_directed",
        "temporal_cnp_arborescence_directed_no_time",
    ]
    assert [algorithm.__name__ for algorithm in get_discovery_algorithms()] == [
        "temporal_cnp_arborescence_directed",
        "temporal_cnp_arborescence_directed_no_time",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "neighbor_joining_classical",
        "rooted_labeled_nj",
        "temporal_cnp_arborescence",
        "temporal_cnp_arborescence_no_time",
        "temporal_cnp_arborescence_directed",
        "temporal_cnp_arborescence_directed_no_time",
    ],
)
def test_publication_algorithms_resolve_by_stable_name(name):
    assert resolve_reconstruction_algorithm(name).__name__ == name


def _metric_tree(edges):
    tree = nx.DiGraph()
    for parent, child in edges:
        tree.add_node(parent, cell_id=str(parent))
        tree.add_node(child, cell_id=str(child))
        tree.add_edge(parent, child)
    return tree


def test_grf_metadata_identifies_similarity_direction():
    true_tree = _metric_tree([(1, 2), (1, 3)])
    same_tree = _metric_tree([(1, 2), (1, 3)])
    different_tree = _metric_tree([(1, 2), (2, 3)])

    same_score = grf_tree(true_tree, 1, same_tree, 1)
    different_score = grf_tree(true_tree, 1, different_tree, 1)

    assert GRF_METRIC_NAME == "grf"
    assert GRF_METRIC_KIND == "similarity"
    assert GRF_HIGHER_IS_BETTER is True
    assert same_score == pytest.approx(1.0)
    assert different_score < same_score


def test_evaluate_4_metadata_identifies_similarity_modes_and_ad_f1():
    true_tree = _metric_tree([(1, 2), (1, 3)])
    same_tree = _metric_tree([(1, 2), (1, 3)])
    worse_tree = _metric_tree([(1, 2), (2, 3)])

    same_metrics = evaluate_4(true_tree, same_tree, restrict_labels={"1", "2", "3"})
    worse_metrics = evaluate_4(true_tree, worse_tree, restrict_labels={"1", "2", "3"})

    assert set(EVALUATE_4_MODE_SPECS) == set(same_metrics)
    assert EVALUATE_4_MODE_SPECS["ancestors_unique_restricted"]["paper_name"] == (
        "AD-F1 when reading the F1 value"
    )
    assert all(spec["kind"] == "similarity" for spec in EVALUATE_4_MODE_SPECS.values())
    assert all(spec["higher_is_better"] is True for spec in EVALUATE_4_MODE_SPECS.values())
    assert EVALUATE_4_HIGHER_IS_BETTER == {
        "precision": True,
        "recall": True,
        "F1": True,
        "IoU": True,
    }
    assert EVALUATE_4_VALUE_DIRECTION["TP"] == "count"
    assert EVALUATE_4_VALUE_DIRECTION["FP"] == "count"
    assert EVALUATE_4_VALUE_DIRECTION["FN"] == "count"
    assert EVALUATE_4_VALUE_DIRECTION["F1"] == "similarity"
    assert same_metrics["ancestors_unique_restricted"]["F1"] == pytest.approx(1.0)
    assert worse_metrics["ancestors_unique_restricted"]["F1"] < same_metrics["ancestors_unique_restricted"]["F1"]


def test_algorithm_display_config_explains_legacy_and_fast_benchmark_rows():
    expected_names = set(LEGACY_ALGORITHM_NAMES) | {
        "neighbor_joining_classical",
        "rooted_labeled_nj",
        "temporal_cnp_arborescence",
        "temporal_cnp_arborescence_no_time",
        "new_alg",
        "biopsy_preset_default",
        "biopsy_preset_anticentral_tie",
        "biopsy_preset_binarized",
        "biopsy_preset_anticentral_binarized",
    }

    assert expected_names.issubset(ALGORITHM_CONFIG_BY_NAME)
    assert all(isinstance(ALGORITHM_CONFIG_BY_NAME[name], AlgorithmDisplayConfig) for name in expected_names)
    assert ALGORITHM_CONFIG_BY_NAME["neighbor_joining_baseline"].procedure.pair_selection
    assert ALGORITHM_CONFIG_BY_NAME["neighbor_joining_baseline"].summary
    assert ALGORITHM_CONFIG_BY_NAME["neighbor_joining_baseline"].label == (
        "legacy directed closest-pair"
    )
    assert "classical NJ" in ALGORITHM_CONFIG_BY_NAME["neighbor_joining_classical"].label
    assert "no inferred node" in (
        ALGORITHM_CONFIG_BY_NAME["rooted_labeled_nj"].procedure.merge_strategy
    )
    assert COMPARISON_GROUPS["publication"] == (
        "neighbor_joining_classical",
        "rooted_labeled_nj",
        "temporal_cnp_arborescence",
        "temporal_cnp_arborescence_no_time",
    )
    assert COMPARISON_GROUPS["temporal_arborescence_pair"] == (
        "temporal_cnp_arborescence",
        "temporal_cnp_arborescence_no_time",
    )
    assert COMPARISON_GROUPS["cnp2cnp_direction_pair"] == (
        "temporal_cnp_arborescence",
        "temporal_cnp_arborescence_directed",
    )
    assert COMPARISON_GROUPS["cnp2cnp_direction_no_time_pair"] == (
        "temporal_cnp_arborescence_no_time",
        "temporal_cnp_arborescence_directed_no_time",
    )
    assert COMPARISON_GROUPS["historical_legacy"] == tuple(LEGACY_ALGORITHM_NAMES)
    assert COMPARISON_GROUPS["recommended_core"] == (
        "neighbor_joining_classical",
        "rooted_labeled_nj",
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
    )
    assert set(HIGHLIGHTED_HEATMAP_ALGORITHMS) == {"new_alg"}


def test_biopsy_guided_presets_are_resolvable():
    assert set(BIOPSY_GUIDED_PRESETS) == {
        "default",
        "anticentral_tie",
        "binarized",
        "anticentral_binarized",
        "anticentral_tie_binarized",
        "deferred_tie",
        "central_tie",
        "diploid_parsimony_tie",
    }
    assert resolve_biopsy_guided_config(None) is None
    assert resolve_biopsy_guided_config("none") is None
    default_config = resolve_biopsy_guided_config("default")
    assert default_config.level_extender is make_default_biopsy_guided_config().level_extender
    assert default_config.group_attachment_strategy is not None
    assert resolve_biopsy_guided_config("anticentral_tie") == make_anticentral_tie_biopsy_guided_config()
    assert resolve_biopsy_guided_config("binarized").group_attachment_strategy is not None
    assert make_binarized_biopsy_guided_config().group_attachment_strategy is not None
    assert make_anticentral_binarized_biopsy_guided_config().group_attachment_strategy is not None


def test_unknown_biopsy_guided_preset_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown biopsy-guided preset"):
        resolve_biopsy_guided_config("missing")


def test_ctbs_runtime_config_loads_explicit_file(tmp_path):
    import ctbs

    config = ctbs.default_ctbs_runtime_config().as_legacy_dict()
    config["cnp2cnp_FILE"] = "/tmp/custom_cnp2cnp.py"
    config["RUN_SINGLE_TEST"]["seed"] = 123
    config_path = tmp_path / "ctbs_config.json"
    config_path.write_text(json.dumps(config))

    runtime_config = ctbs.load_ctbs_runtime_config(config_path)

    assert runtime_config.cnp2cnp_file == "/tmp/custom_cnp2cnp.py"
    assert runtime_config.run_single_test["seed"] == 123
    legacy_copy = runtime_config.as_legacy_dict()
    legacy_copy["RUN_SINGLE_TEST"]["seed"] = 999
    assert runtime_config.run_single_test["seed"] == 123


def test_ctbs_pairwise_distance_uses_explicit_runtime_config(monkeypatch):
    import ctbs

    calls = []

    def fake_run(args, cwd, capture_output, text, check):
        calls.append(args)
        assert Path(cwd).name.startswith("ctbf-cnp2cnp-pair-")
        return type(
            "Completed",
            (),
            {"stdout": "7", "stderr": "", "returncode": 0},
        )()

    monkeypatch.setattr(ctbs.subprocess, "run", fake_run)
    runtime_config = ctbs.default_ctbs_runtime_config()
    runtime_config = ctbs.CtbsRuntimeConfig(
        in_file_name=runtime_config.in_file_name,
        out_file_name=runtime_config.out_file_name,
        sim_dm=runtime_config.sim_dm,
        cnp2cnp_folder=runtime_config.cnp2cnp_folder,
        cnp2cnp_file="/tmp/runtime_cnp2cnp.py",
        true_tree_root_id=runtime_config.true_tree_root_id,
        run_single_test=runtime_config.run_single_test,
    )

    distance, execution_record = ctbs.use_cnp2cnp_to_compute_pairwise_distance(
        ">1\n2,2\n",
        runtime_config=runtime_config,
        return_execution_record=True,
    )
    assert distance == 7.0
    assert calls[0][1] == str(Path("/tmp/runtime_cnp2cnp.py").resolve())
    assert calls[0][2:6] == ["-m", "dist", "-d", "any"]
    assert execution_record["status"] == "success"
    assert execution_record["returncode"] == 0
    assert execution_record["working_directory"] == "isolated_temporary_directory"
    assert execution_record["stdout"]["preview"] == "7"


def test_ctbs_distance_matrix_validates_id_alignment():
    import ctbs

    with pytest.raises(ValueError, match="2 rows but 1 ids"):
        ctbs.DistanceMatrix(ids=[1], matrix=np.zeros((2, 2)))

    with pytest.raises(ValueError, match="symmetric"):
        ctbs.DistanceMatrix(ids=[1, 2], matrix=np.array([[0.0, 1.0], [2.0, 0.0]]))

    with pytest.raises(ValueError, match="diagonal"):
        ctbs.DistanceMatrix(ids=[1, 2], matrix=np.array([[1.0, 0.0], [0.0, 0.0]]))

    with pytest.raises(ValueError, match="Duplicate"):
        ctbs.DistanceMatrix(ids=[1, 1], matrix=np.zeros((2, 2)))

    with pytest.raises(ValueError, match="nonnegative"):
        ctbs.DistanceMatrix(ids=[1, 2], matrix=np.array([[0.0, -1.0], [-1.0, 0.0]]))

    with pytest.raises(ValueError, match="finite"):
        ctbs.DistanceMatrix(ids=[1, 2], matrix=np.array([[0.0, np.inf], [np.inf, 0.0]]))


def test_ctbs_supplied_distance_provider_returns_validated_matrix():
    import ctbs

    provider = ctbs.SuppliedDistanceProvider(
        ids=[1, 2, 3],
        matrix=np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        ),
    )

    distance_matrix = provider.compute([Genotype([2], 1), Genotype([2], 2), Genotype([2], 3)])

    assert distance_matrix.build_tree_kwargs()["inids"] == [1, 2, 3]
    assert np.array_equal(distance_matrix.build_tree_kwargs()["indm"], provider.matrix)


def test_ctbs_supplied_distance_provider_rejects_missing_observed_label():
    import ctbs

    provider = ctbs.SuppliedDistanceProvider(ids=[1], matrix=np.zeros((1, 1)))

    with pytest.raises(ValueError, match=r"missing=\[2\]"):
        provider.compute([Genotype([2], 1), Genotype([3], 2)])


def test_ctbs_file_distance_provider_uses_validated_two_order_matrix_mode(monkeypatch):
    import ctbs

    runtime_config = ctbs.default_ctbs_runtime_config()
    calls = []

    monkeypatch.setattr(
        ctbs,
        "distance_matrix_from_cnp2cnp_matrix_mode",
        lambda cells, runtime_config: (
            calls.append(("cnp2cnp", len(cells), runtime_config.out_file_name))
            or ctbs.DistanceMatrix(
                ids=[1, 2],
                matrix=np.array([[0.0, 3.0], [3.0, 0.0]]),
                provenance={"semantics_version": "test"},
            )
        ),
    )
    provider = ctbs.Cnp2CnpFileDistanceProvider(runtime_config)
    distance_matrix = provider.compute([Genotype([2], 1), Genotype([2], 2)])

    assert distance_matrix.build_tree_kwargs()["inids"] == [1, 2]
    assert np.array_equal(
        distance_matrix.build_tree_kwargs()["indm"],
        np.array([[0.0, 3.0], [3.0, 0.0]]),
    )
    assert distance_matrix.provenance == {"semantics_version": "test"}
    assert calls == [("cnp2cnp", 2, runtime_config.out_file_name)]


def test_reconstructor_accepts_distance_matrix_adapter(monkeypatch):
    import ctbs
    import reconstructor

    calls = []

    def fake_build_evolution_tree_impl(cell_lists, **kwargs):
        calls.append(kwargs)
        tree = nx.DiGraph()
        tree.add_node(1, cell_id=1, genome=np.array([2]))
        return tree, {}, 1

    monkeypatch.setattr(reconstructor, "build_evolution_tree_impl", fake_build_evolution_tree_impl)
    distance_matrix = ctbs.DistanceMatrix(ids=[1], matrix=np.zeros((1, 1)))

    reconstructor.build_evolution_tree([[Genotype([2], 1)]], distance_matrix=distance_matrix)

    assert calls[0]["dist_matrix_path"] is None
    assert calls[0]["inids"] == [1]
    assert np.array_equal(calls[0]["indm"], np.zeros((1, 1)))


def test_ctbs_passes_biopsy_guided_config_only_to_reconstruction(monkeypatch):
    import ctbs

    calls = []

    def fake_build_evolution_tree(cell_lists, only_nj=False, **kwargs):
        calls.append((only_nj, kwargs.copy()))
        tree = nx.DiGraph()
        tree.add_node(1, cell_id=1, genome=np.array([2]))
        return tree, {}, 1

    monkeypatch.setattr(ctbs, "build_evolution_tree", fake_build_evolution_tree)
    monkeypatch.setattr(ctbs, "grf_tree", lambda *args, **kwargs: 0)

    cells = [Genotype([2], 1), Genotype([2], 2), Genotype([2], 3)]
    config = resolve_biopsy_guided_config("anticentral_tie")

    ctbs._reconstruct_and_evaluate(
        sim=type("Sim", (), {"tree": nx.DiGraph()})(),
        seed=7,
        cell_lists=[cells],
        all_in_one_sample=[cells],
        r_dist=2,
        visualize=False,
        clear_cnps=False,
        parallel=True,
        write_newick=False,
        reconstruction_algorithm=None,
        biopsy_guided_config=config,
        inid=[1, 2, 3],
        indm=np.zeros((3, 3)),
        time_collector=None,
        runtime_config=ctbs.default_ctbs_runtime_config(),
    )

    assert calls[0][0] is True
    assert "biopsy_guided_config" not in calls[0][1]
    assert calls[1][0] is False
    assert calls[1][1]["biopsy_guided_config"] is config
