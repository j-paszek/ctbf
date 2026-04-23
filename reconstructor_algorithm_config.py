from dataclasses import dataclass

from reconstructor_algorithm_specs import LEGACY_ALGORITHM_SPECS


@dataclass(frozen=True)
class AlgorithmProcedureConfig:
    pair_selection: str
    ancestor_selection: str
    distance_update: str
    merge_strategy: str
    plausibility: str = "none"
    biopsy_guided_preset: str | None = None


@dataclass(frozen=True)
class AlgorithmDisplayConfig:
    name: str
    label: str
    summary: str
    procedure: AlgorithmProcedureConfig
    groups: tuple[str, ...] = ()
    highlight_in_heatmap: bool = False


def _legacy(
    name,
    label,
    summary,
    pair_selection,
    ancestor_selection,
    *,
    distance_update="neighbor_joining",
    merge_strategy="standard_nj_internal_node",
    plausibility="none",
    groups=(),
):
    return AlgorithmDisplayConfig(
        name=name,
        label=label,
        summary=summary,
        procedure=AlgorithmProcedureConfig(
            pair_selection=pair_selection,
            ancestor_selection=ancestor_selection,
            distance_update=distance_update,
            merge_strategy=merge_strategy,
            plausibility=plausibility,
        ),
        groups=("legacy", "publication") + tuple(groups),
    )


def _biopsy_preset(name, label, preset_name, summary, *, groups=()):
    return AlgorithmDisplayConfig(
        name=name,
        label=label,
        summary=summary,
        procedure=AlgorithmProcedureConfig(
            pair_selection="neighbor_joining_standard on final biopsy layer",
            ancestor_selection="biopsy-guided parent selector",
            distance_update="frozen cnp2cnp distance matrix",
            merge_strategy="biopsy-guided tree attachment",
            plausibility="biopsy parent plausibility filter",
            biopsy_guided_preset=preset_name,
        ),
        groups=("biopsy_preset", "fast_benchmark") + tuple(groups),
    )


ALGORITHM_DISPLAY_CONFIGS = [
    _legacy(
        "neighbor_joining_baseline",
        "baseline",
        "Standard neighbor joining baseline used as the reference method.",
        "minimum distance / NJ baseline pair selection",
        "pair-order orientation",
        groups=("recommended_core",),
    ),
    _legacy(
        "neighbor_joining_full_full",
        "full_full",
        "Full-matrix NJ variant with full ancestor-selection information.",
        "full NJ Q-matrix pair selection",
        "full-distance ancestor selector",
    ),
    _legacy(
        "neighbor_joining_full_partial",
        "full_partial",
        "Full-matrix NJ pair selection with partial ancestor-selection information.",
        "full NJ Q-matrix pair selection",
        "partial-distance ancestor selector",
    ),
    _legacy(
        "neighbor_joining_full_cps_full",
        "full_cps_full",
        "Full-matrix CPS variant with full ancestor-selection information.",
        "full CPS pair selection",
        "full-distance ancestor selector",
    ),
    _legacy(
        "neighbor_joining_full_cps_partial",
        "full_cps_partial",
        "Full-matrix CPS variant with partial ancestor-selection information.",
        "full CPS pair selection",
        "partial-distance ancestor selector",
    ),
    _legacy(
        "neighbor_joining_hybrid_full",
        "hybrid_full",
        "Hybrid NJ variant with full ancestor-selection information.",
        "hybrid distance/centrality pair selection",
        "full-distance ancestor selector",
    ),
    _legacy(
        "neighbor_joining_hybrid_partial",
        "hybrid_partial",
        "Hybrid NJ variant with partial ancestor-selection information.",
        "hybrid distance/centrality pair selection",
        "partial-distance ancestor selector",
    ),
    _legacy(
        "neighbor_joining_hybrid_inverse_centrality_full",
        "hybrid_inverse_centrality_full",
        "Hybrid NJ variant using inverse-distance centrality and full ancestor information.",
        "hybrid inverse-distance-centrality pair selection",
        "full-distance ancestor selector",
    ),
    _legacy(
        "neighbor_joining_hybrid_inverse_centrality_partial",
        "hybrid_inverse_centrality_partial",
        "Hybrid NJ variant using inverse-distance centrality and partial ancestor information.",
        "hybrid inverse-distance-centrality pair selection",
        "partial-distance ancestor selector",
    ),
    _legacy(
        "neighbor_joining_adaptive_centrality",
        "adaptive_centrality",
        "Adaptive centrality variant with linear blended centrality.",
        "adaptive linear blended centrality pair selection",
        "more-central parent selector",
    ),
    _legacy(
        "neighbor_joining_adaptive_centrality_nonlinear",
        "adaptive_centrality_nonlinear",
        "Adaptive centrality variant with sigmoid blended centrality.",
        "adaptive sigmoid blended centrality pair selection",
        "more-central parent selector",
    ),
    _legacy(
        "neighbor_joining_adaptive_centrality_reversed",
        "adaptive_centrality_reversed",
        "Adaptive centrality variant that reverses the centrality preference.",
        "reversed adaptive centrality pair selection",
        "less-mixed centrality parent selector",
    ),
    _legacy(
        "neighbor_joining_hybrid_opt",
        "hybrid_opt",
        "Optimized hybrid objective balancing distance and centrality asymmetry.",
        "hybrid optimized pair selection",
        "more-central parent selector",
        groups=("recommended_core",),
    ),
    _legacy(
        "neighbor_joining_hybrid_opt_adaptive",
        "hybrid_opt_adaptive",
        "Adaptive optimized hybrid objective with heterogeneity-adjusted weights.",
        "adaptive hybrid optimized pair selection",
        "more-central parent selector",
    ),
    _legacy(
        "neighbor_joining_hybrid_opt_v2",
        "hybrid_opt_v2",
        "Second optimized hybrid objective using mixed direct/inverse centrality.",
        "hybrid optimized v2 pair selection",
        "more-central parent selector",
    ),
    _legacy(
        "neighbor_joining_hybrid_opt_refined",
        "hybrid_opt_refined",
        "Refined optimized hybrid objective with adjusted centrality handling.",
        "refined hybrid optimized pair selection",
        "more-central parent selector",
    ),
    _legacy(
        "neighbor_joining_hybrid_anticentral_opt",
        "hybrid_anticentral_opt",
        "Anticentral optimized hybrid variant.",
        "anticentral hybrid optimized pair selection",
        "pair-order orientation",
    ),
    _legacy(
        "neighbor_joining_hybrid_anticentral_adaptive_v3",
        "hybrid_anticentral_adaptive_v3",
        "Anticentral adaptive v3 variant with anticentral distance updates.",
        "anticentral adaptive v3 pair selection",
        "pair-order orientation",
        distance_update="anticentral_v3_distance_update",
        merge_strategy="anticentral weighted-copy parent node",
    ),
    _legacy(
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible",
        "hybrid_anticentral_adaptive_v3_plausible",
        "Anticentral adaptive v3 variant with plausibility-aware edge orientation.",
        "anticentral adaptive v3 pair selection",
        "plausible pair-order parent selector",
        distance_update="anticentral_v3_distance_update",
        merge_strategy="anticentral weighted-copy parent node",
        plausibility="ancestor plausibility orientation",
    ),
    _legacy(
        "neighbor_joining_hybrid_anticentral_adaptive_v3_skip_unplausible",
        "hybrid_anticentral_adaptive_v3_skip_unplausible",
        "Anticentral adaptive v3 variant that skips implausible pairs when possible.",
        "anticentral adaptive v3 skip-unplausible pair selection",
        "pair-order orientation",
        distance_update="anticentral_v3_distance_update",
        merge_strategy="anticentral weighted-copy parent node",
        plausibility="pair plausibility filter",
    ),
    _legacy(
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
        "hybrid_anticentral_adaptive_v3_plausible_parsimony",
        "Anticentral adaptive v3 variant with plausible parsimony ancestor selection.",
        "anticentral adaptive v3 pair selection",
        "plausible parsimony parent selector",
        distance_update="anticentral_v3_distance_update",
        merge_strategy="anticentral weighted-copy parent node",
        plausibility="ancestor plausibility plus baseline-CN parsimony",
        groups=("recommended_core",),
    ),
    AlgorithmDisplayConfig(
        name="new_alg",
        label="new_alg",
        summary=(
            "Experimental anticentral reconstruction: anticentral adaptive v3 pair selection, "
            "then plausible ancestor orientation, then centrality fallback."
        ),
        procedure=AlgorithmProcedureConfig(
            pair_selection="anticentral adaptive v3 pair selection",
            ancestor_selection="plausible ancestor selector, then larger-centrality fallback",
            distance_update="anticentral_v3_distance_update",
            merge_strategy="anticentral weighted-copy parent node",
            plausibility="ancestor plausibility first; centrality if plausibility does not decide",
        ),
        groups=("experimental", "new_alg_comparison"),
        highlight_in_heatmap=True,
    ),
    _biopsy_preset(
        "biopsy_preset_default",
        "biopsy preset: default",
        "default",
        "Default biopsy-guided reconstruction preset.",
        groups=("biopsy_preset_comparison",),
    ),
    _biopsy_preset(
        "biopsy_preset_anticentral_tie",
        "biopsy preset: anticentral tie",
        "anticentral_tie",
        "Biopsy-guided preset using anticentrality to break equal-distance parent ties.",
        groups=("biopsy_preset_comparison",),
    ),
    _biopsy_preset(
        "biopsy_preset_binarized",
        "biopsy preset: binarized",
        "binarized",
        "Biopsy-guided preset that binarizes same-parent biopsy attachments.",
        groups=("biopsy_preset_comparison",),
    ),
    _biopsy_preset(
        "biopsy_preset_anticentral_binarized",
        "biopsy preset: anticentral binarized",
        "anticentral_binarized",
        "Biopsy-guided preset combining anticentral tie-breaking and binarized attachments.",
        groups=("biopsy_preset_comparison",),
    ),
]


ALGORITHM_CONFIG_BY_NAME = {
    config.name: config
    for config in ALGORITHM_DISPLAY_CONFIGS
}

COMPARISON_GROUPS = {
    "publication": tuple(spec.name for spec in LEGACY_ALGORITHM_SPECS),
    "recommended_core": (
        "neighbor_joining_baseline",
        "neighbor_joining_hybrid_opt",
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
    ),
    "biopsy_preset_comparison": (
        "neighbor_joining_baseline",
        "neighbor_joining_hybrid_opt",
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
        "biopsy_preset_default",
        "biopsy_preset_anticentral_tie",
        "biopsy_preset_binarized",
        "biopsy_preset_anticentral_binarized",
    ),
    "new_alg_comparison": (
        "neighbor_joining_baseline",
        "neighbor_joining_hybrid_opt",
        "neighbor_joining_hybrid_anticentral_adaptive_v3_plausible_parsimony",
        "new_alg",
    ),
}

HIGHLIGHTED_HEATMAP_ALGORITHMS = tuple(
    config.name
    for config in ALGORITHM_DISPLAY_CONFIGS
    if config.highlight_in_heatmap
)


def algorithm_label(name):
    config = ALGORITHM_CONFIG_BY_NAME.get(name)
    if config is not None:
        return config.label
    return name.replace("neighbor_joining_", "")


def resolve_comparison_algorithm_names(groups=None, names=None):
    selected = []
    for group in groups or ():
        if group not in COMPARISON_GROUPS:
            available = ", ".join(sorted(COMPARISON_GROUPS))
            raise ValueError(f"Unknown comparison group '{group}'. Available groups: {available}")
        selected.extend(COMPARISON_GROUPS[group])
    selected.extend(names or ())
    return list(dict.fromkeys(selected)) if selected else None


__all__ = [
    "ALGORITHM_CONFIG_BY_NAME",
    "ALGORITHM_DISPLAY_CONFIGS",
    "COMPARISON_GROUPS",
    "HIGHLIGHTED_HEATMAP_ALGORITHMS",
    "AlgorithmDisplayConfig",
    "AlgorithmProcedureConfig",
    "algorithm_label",
    "resolve_comparison_algorithm_names",
]
