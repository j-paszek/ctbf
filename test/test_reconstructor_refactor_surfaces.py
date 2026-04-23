import pytest
import networkx as nx
import numpy as np

from reconstructor import (
    BIOPSY_GUIDED_PRESETS,
    make_anticentral_binarized_biopsy_guided_config,
    make_anticentral_tie_biopsy_guided_config,
    make_binarized_biopsy_guided_config,
    make_default_biopsy_guided_config,
    resolve_biopsy_guided_config,
)
from reconstructor_algorithm_specs import LEGACY_ALGORITHM_SPECS, ReconstructionAlgorithmSpec
from reconstructor_plausibility import (
    is_biologically_plausible_ancestor,
    is_biologically_plausible_pair,
)
from reconstructor_registry import LEGACY_ALGORITHM_NAMES, get_legacy_algorithms
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
    assert [algorithm.__name__ for algorithm in get_legacy_algorithms()] == LEGACY_ALGORITHM_NAMES
    assert all(isinstance(spec, ReconstructionAlgorithmSpec) for spec in LEGACY_ALGORITHM_SPECS)


def test_biopsy_guided_presets_are_resolvable():
    assert set(BIOPSY_GUIDED_PRESETS) == {
        "default",
        "anticentral_tie",
        "binarized",
        "anticentral_binarized",
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
    )

    assert calls[0][0] is True
    assert "biopsy_guided_config" not in calls[0][1]
    assert calls[1][0] is False
    assert calls[1][1]["biopsy_guided_config"] is config
