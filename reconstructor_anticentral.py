"""Anticentral state configuration for v3 reconstruction variants."""

from reconstructor_distance_update import ANTICENTRAL_V3_CONTEXT_KEY
from reconstructor_metrics import normalized_inverse_mean_centrality


def _initial_anticentral_v3_centrality(D):
    return normalized_inverse_mean_centrality(D)


def configure_anticentral_v3_state(state):
    state.context[ANTICENTRAL_V3_CONTEXT_KEY] = _initial_anticentral_v3_centrality(state.D)


__all__ = [
    "configure_anticentral_v3_state",
]
