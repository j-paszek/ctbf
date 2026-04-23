import numpy as np

from reconstructor_ancestor_selection import (
    _is_biologically_plausible_ancestor,
)
from reconstructor_distance_update import ANTICENTRAL_V3_CONTEXT_KEY
from reconstructor_engine import Orientation, PairChoice
from reconstructor_pair_selection import (
    _best_pair_from_score_matrix,
    _mixed_direct_inverse_centrality,
    _ordered_pairs_by_score_matrix,
)

def _initial_anticentral_v3_centrality(D):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = 1.0 / (np.mean(D, axis=1) + 1e-9)
    return (c - np.min(c)) / (np.ptp(c) + 1e-12)


def _anticentral_adaptive_v3_score_matrix(D, c, rng, alpha=1.0, beta=1.0, gamma=0.5):
    n = len(D)
    score = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            c_diff = abs(c[i] - c[j])
            adaptive = np.exp(-gamma * c_diff / (np.std(c) + 1e-9))
            anticentral_term = (2 - (c[i] + c[j]))
            score[i, j] = (
                alpha * D[i, j] * (1 + gamma * adaptive)
                - beta * anticentral_term
            )
            score[i, j] += rng.random() * 1e-6
    return score


def configure_anticentral_v3_state(state):
    state.context[ANTICENTRAL_V3_CONTEXT_KEY] = _initial_anticentral_v3_centrality(state.D)


def make_anticentral_adaptive_v3_pair_selector(alpha=1.0, beta=1.0, gamma=0.5):
    def select_pair(state):
        c = state.context[ANTICENTRAL_V3_CONTEXT_KEY]
        score = _anticentral_adaptive_v3_score_matrix(state.D, c, state.rng, alpha, beta, gamma)
        i_best, j_best, _ = _best_pair_from_score_matrix(score)
        return PairChoice(i_best, j_best, score=score[i_best, j_best])

    return select_pair


def make_anticentral_adaptive_v3_skip_unplausible_pair_selector(alpha=1.0, beta=1.0, gamma=0.5):
    def select_pair(state):
        c = state.context[ANTICENTRAL_V3_CONTEXT_KEY]
        score = _anticentral_adaptive_v3_score_matrix(state.D, c, state.rng, alpha, beta, gamma)
        pair_order = _ordered_pairs_by_score_matrix(score)

        for i, j in pair_order:
            a = state.node_list[i]
            b = state.node_list[j]
            can_i_parent_j = _is_biologically_plausible_ancestor(a, b)
            can_j_parent_i = _is_biologically_plausible_ancestor(b, a)

            if can_i_parent_j or can_j_parent_i:
                if can_i_parent_j and not can_j_parent_i:
                    parent_idx, child_idx = i, j
                elif can_j_parent_i and not can_i_parent_j:
                    parent_idx, child_idx = j, i
                else:
                    parent_idx, child_idx = (i, j) if c[i] > c[j] else (j, i)
                return PairChoice(
                    i,
                    j,
                    score=score[i, j],
                    metadata={"orientation": Orientation(parent_idx, child_idx)},
                )

        i, j = pair_order[0]
        parent_idx, child_idx = (i, j) if c[i] > c[j] else (j, i)
        return PairChoice(
            i,
            j,
            score=score[i, j],
            metadata={"orientation": Orientation(parent_idx, child_idx)},
        )

    return select_pair


def make_anticentral_hybrid_opt_pair_selector(alpha=1.0, beta=1.0, lam=1.0, epsilon=1e-6):
    def select_pair(state):
        n = len(state.D)
        c_mix = _mixed_direct_inverse_centrality(state.D, epsilon, lam)

        score = np.full((n, n), -np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = state.D[i, j]
                if not np.isfinite(d_ij):
                    continue
                asym = abs(c_mix[i] - c_mix[j])
                score[i, j] = alpha * d_ij + beta * asym

        i, j, _ = _best_pair_from_score_matrix(score, minimize=False)

        return PairChoice(i, j, score=score[i, j], metadata={"c_mix": c_mix})

    return select_pair


def make_anticentral_adaptive_v2_pair_selector(
    alpha=1.0,
    beta=1.0,
    gamma=0.2,
    lam=1.0,
    epsilon=1e-6,
):
    def select_pair(state):
        n = len(state.D)
        original_n = len(state.D_full)
        a = alpha * (1 + 0.2 * np.tanh(3 * (1 - len(state.node_list) / original_n)))
        b = beta * (1 + 0.3 * np.tanh(2 * (len(state.node_list) / original_n)))
        g = gamma * (0.5 + 0.5 * np.tanh(3 * len(state.node_list) / original_n))

        c_mix = _mixed_direct_inverse_centrality(state.D, epsilon, lam)

        score = np.full((n, n), -np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = state.D[i, j]
                if not np.isfinite(d_ij):
                    continue
                asym = abs(c_mix[i] - c_mix[j])
                score[i, j] = a * d_ij + b * asym - g / (d_ij + epsilon)

        i, j, _ = _best_pair_from_score_matrix(score, minimize=False)

        return PairChoice(i, j, score=score[i, j], metadata={"c_mix": c_mix})

    return select_pair

