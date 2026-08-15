from reconstructor_ancestor_selection import keep_pair_order_parent_selector
from reconstructor_anticentral import configure_anticentral_v3_state
from reconstructor_biopsy_blocks import (
    BiopsyGuidedConfig,
    BiopsySubtreeConfig,
    default_biopsy_guided_config,
    make_binarized_group_attachment_strategy,
    select_anticentral_candidate,
    select_central_candidate,
    select_deferred_candidate,
    select_diploid_parsimony_candidate,
)
from reconstructor_pair_selection import make_anticentral_adaptive_v3_pair_selector


def make_default_biopsy_guided_config(*, decision_audit=None):
    return default_biopsy_guided_config(decision_audit=decision_audit)


def make_anticentral_tie_biopsy_guided_config(*, decision_audit=None):
    return BiopsyGuidedConfig(
        candidate_tie_breaker=select_anticentral_candidate,
        decision_audit=decision_audit,
    )


def make_binarized_biopsy_guided_config(subtree_config=None, *, decision_audit=None):
    return BiopsyGuidedConfig(
        group_attachment_strategy=make_binarized_group_attachment_strategy(subtree_config),
        decision_audit=decision_audit,
    )


def make_anticentral_binarized_biopsy_guided_config(*, decision_audit=None):
    return BiopsyGuidedConfig(
        candidate_tie_breaker=select_anticentral_candidate,
        group_attachment_strategy=make_binarized_group_attachment_strategy(
            BiopsySubtreeConfig(
                pair_selector=make_anticentral_adaptive_v3_pair_selector(),
                ancestor_selector=keep_pair_order_parent_selector,
                configure_state=configure_anticentral_v3_state,
            )
        ),
        decision_audit=decision_audit,
    )


def make_anticentral_tie_binarized_biopsy_guided_config(*, decision_audit=None):
    """Clean tie-by-binarization combination using the default subtree solver."""
    return BiopsyGuidedConfig(
        candidate_tie_breaker=select_anticentral_candidate,
        group_attachment_strategy=make_binarized_group_attachment_strategy(),
        decision_audit=decision_audit,
    )


def make_deferred_tie_biopsy_guided_config(*, decision_audit=None):
    return BiopsyGuidedConfig(
        candidate_tie_breaker=select_deferred_candidate,
        decision_audit=decision_audit,
    )


def make_central_tie_biopsy_guided_config(*, decision_audit=None):
    return BiopsyGuidedConfig(
        candidate_tie_breaker=select_central_candidate,
        decision_audit=decision_audit,
    )


def make_diploid_parsimony_tie_biopsy_guided_config(*, decision_audit=None):
    return BiopsyGuidedConfig(
        candidate_tie_breaker=select_diploid_parsimony_candidate,
        decision_audit=decision_audit,
    )


BIOPSY_GUIDED_PRESETS = {
    "default": make_default_biopsy_guided_config,
    "anticentral_tie": make_anticentral_tie_biopsy_guided_config,
    "binarized": make_binarized_biopsy_guided_config,
    "anticentral_binarized": make_anticentral_binarized_biopsy_guided_config,
    "anticentral_tie_binarized": make_anticentral_tie_binarized_biopsy_guided_config,
    "deferred_tie": make_deferred_tie_biopsy_guided_config,
    "central_tie": make_central_tie_biopsy_guided_config,
    "diploid_parsimony_tie": make_diploid_parsimony_tie_biopsy_guided_config,
}


def resolve_biopsy_guided_config(name, *, decision_audit=None):
    if name is None:
        return None
    if isinstance(name, str) and name.strip().lower() in {"", "none"}:
        return None

    preset_name = name.strip() if isinstance(name, str) else name
    if preset_name not in BIOPSY_GUIDED_PRESETS:
        available = ", ".join(sorted(BIOPSY_GUIDED_PRESETS))
        raise ValueError(
            f"Unknown biopsy-guided preset '{name}'. Available options: {available}"
        )
    return BIOPSY_GUIDED_PRESETS[preset_name](decision_audit=decision_audit)


__all__ = [
    "BIOPSY_GUIDED_PRESETS",
    "make_anticentral_binarized_biopsy_guided_config",
    "make_anticentral_tie_binarized_biopsy_guided_config",
    "make_anticentral_tie_biopsy_guided_config",
    "make_binarized_biopsy_guided_config",
    "make_central_tie_biopsy_guided_config",
    "make_default_biopsy_guided_config",
    "make_deferred_tie_biopsy_guided_config",
    "make_diploid_parsimony_tie_biopsy_guided_config",
    "resolve_biopsy_guided_config",
]
