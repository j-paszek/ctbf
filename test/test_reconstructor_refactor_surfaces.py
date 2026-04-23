import pytest

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
    _is_biologically_plausible_ancestor,
    _is_biologically_plausible_pair,
    is_biologically_plausible_ancestor,
    is_biologically_plausible_pair,
)
from reconstructor_registry import LEGACY_ALGORITHM_NAMES, get_legacy_algorithms
from simulator import Genotype


def test_plausibility_public_and_compatibility_names_match():
    ancestor = Genotype([2, 0], 1)
    descendant_possible = Genotype([1, 0], 2)
    descendant_impossible = Genotype([1, 1], 3)

    assert is_biologically_plausible_ancestor(ancestor, descendant_possible)
    assert not is_biologically_plausible_ancestor(ancestor, descendant_impossible)
    assert _is_biologically_plausible_ancestor is is_biologically_plausible_ancestor
    assert _is_biologically_plausible_pair is is_biologically_plausible_pair
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
