from reconstructor_distance_update import ANTICENTRAL_V3_CONTEXT_KEY
from reconstructor_metrics import normalized_inverse_mean_centrality
from reconstructor_pair_selection import (
    make_anticentral_adaptive_v2_pair_selector,
    make_anticentral_adaptive_v3_pair_selector,
    make_anticentral_adaptive_v3_skip_unplausible_pair_selector,
    make_anticentral_hybrid_opt_pair_selector,
)

def _initial_anticentral_v3_centrality(D):
    return normalized_inverse_mean_centrality(D)


def configure_anticentral_v3_state(state):
    state.context[ANTICENTRAL_V3_CONTEXT_KEY] = _initial_anticentral_v3_centrality(state.D)


__all__ = [
    "configure_anticentral_v3_state",
    "make_anticentral_adaptive_v2_pair_selector",
    "make_anticentral_adaptive_v3_pair_selector",
    "make_anticentral_adaptive_v3_skip_unplausible_pair_selector",
    "make_anticentral_hybrid_opt_pair_selector",
]
